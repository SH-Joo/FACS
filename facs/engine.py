"""Explicit training, exact epoch-boundary resume and offline evaluation."""

from __future__ import annotations

from contextlib import nullcontext
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time

import numpy as np
from PIL import Image
import torch
from torchvision.models import ResNet50_Weights

from .data import CV12Dataset, make_loader
from .losses import WeightedLoss
from .metrics import MetricAccumulator, PROTOCOL as METRIC_PROTOCOL
from .model import FACSNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_KIND = "facs-training-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_state_hash(model):
    """Hash names, shapes, dtypes and exact state bytes, including BN buffers."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        digest.update(raw.tobytes())
    return digest.hexdigest()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT/path


def read_config(path):
    config = json.loads(Path(path).read_text())
    expected = {"protocol", "model", "loss", "data", "training", "evaluation"}
    if set(config) != expected:
        raise ValueError(f"Config sections must be {sorted(expected)}")
    training = config["training"]
    if (training["batch_size"] < 1 or training["max_epochs"] < 1 or training["workers"] < 0
            or training["lr"] <= 0 or training["precision"] not in ("fp32", "bf16")
            or training["patience"] < 1 or training["gradient_clip_norm"] <= 0):
        raise ValueError("Invalid training settings")
    if not all(math.isfinite(training[k]) for k in ("lr", "gradient_clip_norm")):
        raise ValueError("Training settings must be finite")
    if config["protocol"]["status"] not in ("diagnostic", "locked"):
        raise ValueError("Protocol status must be diagnostic or locked")
    if training.get("monitor", "val_loss") not in ("val_loss", "val_iou_macro"):
        raise ValueError("Checkpoint monitor must be val_loss or val_iou_macro")
    return config


def seed_everything(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if torch.device(device).type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 requires a CUDA device with bf16 support")
    return torch.autocast("cuda", dtype=torch.bfloat16)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temporary.replace(path)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def capture_rng(generator):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": {
        "algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
        "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gauss": numpy_state[4]},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gauss"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA RNG devices differ")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    generator.set_state(state["loader_generator"].cpu())


def build_loss(config):
    values = config["loss"]
    checkpoint = values.get("edge_checkpoint")
    if values["weights"].get("semeda", 0) > 0 and checkpoint:
        state = torch.load(resolve_path(checkpoint), map_location="cpu", weights_only=True)
        root = resolve_path(config["data"]["root"])
        if (state.get("training_ids") != list(range(1, 9601))
                or state.get("completed_epochs", 0) < 1
                or state.get("manifest_sha256") != sha256(root/"manifests/train.csv")
                or state.get("source_archive_sha256") != json.loads((root/"source_archive.json").read_text())["sha256"]):
            raise ValueError("SEMEDA checkpoint provenance must match all 9600 CV12 train masks")
    return WeightedLoss(values["weights"], soft_parameters=values.get("soft_parameters"),
                        edge_checkpoint=resolve_path(checkpoint) if checkpoint else None,
                        semeda_layer_weights=values.get("semeda_layer_weights", [1, 1, 1]),
                        semeda_reduction=values.get("semeda_reduction", "mean"),
                        dice_smooth=values.get("dice_smooth", 1e-6))


def build_model(config, *, loading=False):
    options = dict(config["model"])
    if loading:
        options["cnn_weights"] = None
    return FACSNet(**options)


def build_optimizer(model, training):
    head_multiplier = training.get("head_lr_multiplier", 1.0)
    encoder_multiplier = training.get("encoder_lr_multiplier", 1.0)
    if any(not math.isfinite(value) or value <= 0 for value in (head_multiplier, encoder_multiplier)):
        raise ValueError("Learning-rate multipliers must be positive and finite")
    if head_multiplier == encoder_multiplier == 1:
        return torch.optim.Adam(model.parameters(), lr=training["lr"])
    groups = {"main":[], "encoder":[], "head":[]}
    for name, parameter in model.named_parameters():
        if name.startswith(("fpcm_tail.seg_head.", "segmentation_head.")):
            group = "head"
        elif name.startswith(("cnn_encoder.", "mix_transformer.")):
            group = "encoder"
        else:
            group = "main"
        groups[group].append(parameter)
    if head_multiplier != 1 and not groups["head"]:
        raise ValueError("Requested head learning rate but no segmentation head found")
    multipliers = {"main":1., "encoder":encoder_multiplier, "head":head_multiplier}
    parameters = [{"params":params, "lr":training["lr"]*multipliers[name], "name":name}
                  for name, params in groups.items() if params]
    return torch.optim.Adam(parameters, lr=training["lr"])


def build_dataset(config, split, *, subset=None, sample_ids=None):
    data = config["data"]
    ids = sample_ids if sample_ids is not None else data.get(f"{split}_ids")
    return CV12Dataset(resolve_path(data["root"]), split, subset=subset, sample_ids=ids,
                       mean=data["mean"], std=data["std"], verify_hashes=data.get("verify_hashes", True),
                       augmentation=data.get("augmentation", "none") if split == "train" else "none")


def environment(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
            "cpu_threads": torch.get_num_threads(),
            "source_hashes": {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in sorted((PROJECT_ROOT/"facs").glob("*.py"))},
            "lock_sha256": sha256(PROJECT_ROOT/"requirements-train.lock")}


def provenance(config, train_dataset, val_dataset, device):
    weight_name = config["model"].get("cnn_weights")
    initialization = {"weights":weight_name,"scope":"CNN ResNet50 encoder"}
    if weight_name:
        weights_url = ResNet50_Weights[weight_name].url
        cached = Path(torch.hub.get_dir())/"checkpoints"/weights_url.rsplit("/",1)[-1]
        initialization.update(url=weights_url, sha256=sha256(cached) if cached.is_file() else None)
    result = {"config_hash": config_hash(config), "environment": environment(device),
              "cnn_initialization":initialization,
              "dataset_archive": json.loads((train_dataset.root/"source_archive.json").read_text()),
              "preparation_summary_sha256": sha256(train_dataset.root/"preparation_summary.json"),
              "train_manifest_sha256": train_dataset.manifest_sha256,
              "val_manifest_sha256": val_dataset.manifest_sha256,
              "all_manifest_sha256": sha256(train_dataset.root/"manifests/all.csv"),
              "test_manifest_sha256": sha256(train_dataset.root/"manifests/test.csv"),
              "train_ids": [int(row["sample_id"]) for row in train_dataset.rows],
              "val_ids": [int(row["sample_id"]) for row in val_dataset.rows]}
    checkpoint = config["loss"].get("edge_checkpoint")
    if config["loss"]["weights"].get("semeda", 0) > 0 and checkpoint:
        result["edge_checkpoint_sha256"] = sha256(resolve_path(checkpoint))
    return result


def run_epoch(model, criterion, loader, *, device, precision, optimizer=None, clip_norm=1.0):
    training = optimizer is not None
    model.train(training)
    criterion.train(training)
    totals, count, steps, iou_sum = {}, 0, 0, 0.0
    bin_sums, bin_counts = {}, {}
    synchronize(device)
    start = time.perf_counter()
    with torch.enable_grad() if training else torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                logits = model(image)

            loss, terms = criterion(logits.float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at IDs {batch['sample_id'].tolist()}")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
                optimizer.step()
            size = image.shape[0]
            count += size
            steps += 1
            for name, value in {"loss": loss, **terms}.items():
                totals[name] = totals.get(name, 0.0)+float(value.detach())*size
            with torch.no_grad():
                predicted = logits > 0
                truth = target.bool()
                intersection = (predicted & truth).sum((1, 2, 3))
                union = (predicted | truth).sum((1, 2, 3))
                ious = torch.where(union > 0, intersection/union.clamp_min(1), 1.)
                iou_sum += float(ious.sum())
                for name, value in zip(batch["thickness_bin"], ious.detach().cpu().tolist()):
                    bin_sums[name] = bin_sums.get(name, 0.0)+value
                    bin_counts[name] = bin_counts.get(name, 0)+1
            if training and steps % 100 == 0:
                print(json.dumps({"training_step": steps, "images": count, "mean_loss": totals["loss"]/count}), flush=True)
    synchronize(device)
    if count != len(loader.dataset):
        raise RuntimeError(f"Incomplete epoch: {count}/{len(loader.dataset)}")
    return {"count": count, "steps": steps, "seconds": time.perf_counter()-start,
            "iou_by_width": {name:{"count":bin_counts[name],"iou_macro":value/bin_counts[name]}
                             for name,value in bin_sums.items()},
            "iou_macro": iou_sum/count, **{name: value/count for name, value in totals.items()}}


def train(config, run_dir, *, device="cuda", resume=None, stop_after_epoch=None):
    if stop_after_epoch is not None and stop_after_epoch < 1:
        raise ValueError("Stop epoch is a positive number of completed epochs")
    training = config["training"]
    seed_everything(training["seed"], training["deterministic"])
    torch.set_num_threads(training.get("cpu_threads", 4))
    generator = torch.Generator().manual_seed(training["seed"])
    train_dataset, val_dataset = build_dataset(config, "train"), build_dataset(config, "val")
    train_loader = make_loader(train_dataset, batch_size=training["batch_size"], workers=training["workers"], shuffle=True, generator=generator)

    val_generator = torch.Generator().manual_seed(training["seed"]+1)
    val_loader = make_loader(val_dataset, batch_size=training["batch_size"], workers=training["workers"], generator=val_generator)
    criterion = build_loss(config).to(device)
    model = build_model(config, loading=resume is not None).to(device)
    optimizer = build_optimizer(model, training)
    monitor = training.get("monitor", "val_loss")
    monitor_mode = "min" if monitor == "val_loss" else "max"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=training["scheduler_patience"],
                                                        factor=training["scheduler_factor"], mode=monitor_mode)
    metadata = provenance(config, train_dataset, val_dataset, device)
    if resume is None:
        metadata["initial_model_state_sha256"] = model_state_hash(model)
    run_dir = Path(run_dir).resolve()
    first_epoch, best_loss, bad_epochs, history = 0, math.inf, 0, []
    best_monitor = math.inf if monitor_mode == "min" else -math.inf
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["kind"] != CHECKPOINT_KIND or checkpoint["metadata"]["config_hash"] != metadata["config_hash"]:
            raise ValueError("Resume checkpoint or config does not match")
        for key in ("train_manifest_sha256", "val_manifest_sha256", "all_manifest_sha256", "test_manifest_sha256", "train_ids", "val_ids", "edge_checkpoint_sha256"):
            if checkpoint["metadata"].get(key) != metadata.get(key):
                raise ValueError(f"Resume data/loss provenance changed: {key}")
        if checkpoint["metadata"]["environment"]["source_hashes"] != metadata["environment"]["source_hashes"]:
            raise ValueError("Source changed since checkpoint; create a separately documented run")
        for key in ("torch", "cuda_runtime", "device", "gpu", "cpu_threads", "lock_sha256"):
            if checkpoint["metadata"]["environment"].get(key) != metadata["environment"].get(key):
                raise ValueError(f"Resume environment changed: {key}")
        if Path(resume).resolve().parent != run_dir or not run_dir.is_dir():
            raise ValueError("Resume must use the checkpoint's existing run directory")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        first_epoch, best_loss, bad_epochs = checkpoint["epoch"]+1, checkpoint["best_loss"], checkpoint["bad_epochs"]
        best_monitor = checkpoint["best_monitor_value"]
        history = checkpoint["history"]

        metadata = checkpoint["metadata"]
        restore_rng(checkpoint["rng"], generator)
        val_generator.set_state(checkpoint["val_generator"].cpu())
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(run_dir/"config.json", config)
        atomic_json(run_dir/"provenance.json", metadata)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stop_reason = "max_epochs"
    try:
        for epoch in range(first_epoch, training["max_epochs"]):
            train_stats = run_epoch(model, criterion, train_loader, device=device, precision=training["precision"], optimizer=optimizer, clip_norm=training["gradient_clip_norm"])
            val_stats = run_epoch(model, criterion, val_loader, device=device, precision=training["precision"])
            monitor_value = val_stats["loss"] if monitor == "val_loss" else val_stats["iou_macro"]
            scheduler.step(monitor_value)
            improved = monitor_value < best_monitor if monitor_mode == "min" else monitor_value > best_monitor
            best_loss = min(best_loss, val_stats["loss"])
            if improved:
                best_monitor, bad_epochs = monitor_value, 0
            else:
                bad_epochs += 1
            row = {"epoch": epoch, "train": train_stats, "val": val_stats,
                   "lr": optimizer.param_groups[0]["lr"], "best_val_loss": best_loss,
                   "learning_rate_groups": {group.get("name","all"):group["lr"] for group in optimizer.param_groups},
                   "monitor":monitor, "best_monitor_value":best_monitor,
                   "peak_memory_mib": torch.cuda.max_memory_allocated(device)/2**20 if torch.device(device).type == "cuda" else None}
            history.append(row)
            checkpoint = {"kind": CHECKPOINT_KIND, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "rng": capture_rng(generator),
                          "val_generator": val_generator.get_state(), "epoch": epoch,
                          "best_loss": best_loss, "bad_epochs": bad_epochs, "history": history,
                          "best_monitor_value":best_monitor,
                          "config": config, "metadata": metadata}
            save_start = time.perf_counter()
            atomic_torch_save(run_dir/"last.pt", checkpoint)
            if improved:

                atomic_torch_save(run_dir/"best.pt", {"kind": "facs-inference-v1", "model": model.state_dict(),
                                                       "config": config, "metadata": metadata, "epoch": epoch})
            row["checkpoint_seconds"] = time.perf_counter()-save_start
            atomic_json(run_dir/"history.json", history)
            print(json.dumps(row), flush=True)
            if bad_epochs >= training["patience"]:
                stop_reason = "early_stopping"
                break
            if stop_after_epoch is not None and epoch+1 >= stop_after_epoch:
                stop_reason = "requested_epoch_boundary"
                break
        result = {"completed_epochs": len(history), "best_val_loss": best_loss,
                  "monitor":monitor, "best_monitor_value":best_monitor,
                  "stop_reason": stop_reason, "test_evaluated": False,
                  "best_checkpoint": str(run_dir/"best.pt"), "last_checkpoint": str(run_dir/"last.pt")}
        atomic_json(run_dir/"result.json", result)
        return result
    except BaseException as error:
        atomic_json(run_dir/"failure.json", {"type": type(error).__name__, "message": str(error),
                                              "completed_epochs": len(history)})
        raise


def evaluate(checkpoint_path, output_dir, *, split="test", device="cuda", subset=None,
             batch_size=12, workers=4, save_predictions=True):
    checkpoint_path, output_dir = Path(checkpoint_path), Path(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") not in (CHECKPOINT_KIND, "facs-inference-v1"):
        raise ValueError("Use a new FACS checkpoint or migrate the legacy checkpoint explicitly")
    config = checkpoint["config"]
    seed_everything(config["training"]["seed"], config["training"]["deterministic"])
    torch.set_num_threads(config["training"].get("cpu_threads", 4))

    evaluation_config = json.loads(json.dumps(config))
    evaluation_config["data"].pop(f"{split}_ids", None)
    dataset = build_dataset(evaluation_config, split, subset=subset)
    expected_manifest = checkpoint["metadata"].get("all_manifest_sha256")
    if expected_manifest and sha256(dataset.root/"manifests/all.csv") != expected_manifest:
        raise ValueError("Evaluation dataset differs from training dataset manifest")
    loader = make_loader(dataset, batch_size=batch_size, workers=workers)
    model = build_model(config, loading=True).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir = output_dir/"predictions"
    if save_predictions:
        prediction_dir.mkdir()
    options = config["evaluation"]
    threshold = options["threshold"]
    if not 0 < threshold < 1:
        raise ValueError("Prediction threshold must be in (0,1)")
    logit_threshold = math.log(threshold/(1-threshold))
    groups = {"all": MetricAccumulator(**{k:v for k,v in options.items() if k != "threshold"})}
    thin_ids = set()
    if split == "test":
        thin_list = dataset.root/"manifests/subsets/thinnest_10pct.txt"
        thin_ids = {int(Path(name).stem) for name in thin_list.read_text().splitlines()}
    rows = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["image"].to(device, non_blocking=True))
            predictions = (logits > logit_threshold).squeeze(1).cpu().numpy()
            targets = batch["target"].squeeze(1).numpy().astype(bool)
            for index, (prediction, target) in enumerate(zip(predictions, targets)):
                sample_id = int(batch["sample_id"][index])
                score = groups["all"].update(sample_id, prediction, target)
                group_names = [batch["thickness_bin"][index]]
                if target.any():
                    group_names.append("nonzero")
                if sample_id in thin_ids:
                    group_names.append("thinnest_10pct")
                for name in group_names:
                    if name not in groups:
                        groups[name] = MetricAccumulator(**{k:v for k,v in options.items() if k != "threshold"})

                    groups[name].ids.add(sample_id)
                    groups[name].rows.append({"sample_id": sample_id, **score})
                rows.append({"sample_id": sample_id, "filename": batch["filename"][index],
                             "thickness_bin": batch["thickness_bin"][index], **score})
                if save_predictions:
                    Image.fromarray(prediction.astype(np.uint8)*255).save(prediction_dir/batch["filename"][index])
    if len(rows) != len(dataset):
        raise RuntimeError("Evaluation sample count differs from manifest")
    with (output_dir/"per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = {"metric_protocol": METRIC_PROTOCOL, "options": options, "split": split,
              "inference_precision": "fp32", "environment": environment(device),
              "subset": subset, "checkpoint_sha256": sha256(checkpoint_path),
              "manifest_sha256": dataset.manifest_sha256,
              "groups": {name: accumulator.summary() for name, accumulator in groups.items()}}
    atomic_json(output_dir/"summary.json", result)
    return result
