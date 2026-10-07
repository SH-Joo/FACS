"""Train-only checks before committing to long benchmark runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .data import make_loader
from .engine import (atomic_json, atomic_torch_save, autocast_context, build_dataset,
                     build_loss, build_model, environment, config_hash, seed_everything,
                     build_optimizer,
                     sha256, synchronize)
from .losses import EdgeNetwork, binary_distribution, semantic_edges, edge_cross_entropy


def train_only_ids(config):
    dataset = build_dataset(config, "train")
    chosen = []
    for group in ("0_2", "4_8", "16_32", "zero"):
        candidate = next((int(r["sample_id"]) for r in dataset.rows if r["thickness_bin"] == group), None)
        if candidate is not None:
            chosen.append(candidate)
    if len(chosen) < 2:
        raise ValueError("Overfit diagnosis needs at least two train samples")
    return chosen


def training_batch_metrics(model, criterion, image, target):
    model.eval()
    with torch.inference_mode():
        logits = model(image)
        loss, terms = criterion(logits, target)
        prediction, truth = logits > 0, target.bool()
        intersection = (prediction & truth).sum((1, 2, 3))
        union = (prediction | truth).sum((1, 2, 3))
        iou = torch.where(union > 0, intersection/union.clamp_min(1), 1.)
        return {"loss": float(loss), "iou_macro": float(iou.mean()),
                "per_image_iou": iou.cpu().tolist(), **{k:float(v) for k,v in terms.items()}}


def overfit(config, output_dir, *, device="cuda", steps=128, sample_ids=None):
    if steps < 1:
        raise ValueError("Overfit steps must be positive")
    training = config["training"]
    seed_everything(training["seed"], training["deterministic"])
    torch.set_num_threads(training.get("cpu_threads", 4))
    ids = sample_ids or train_only_ids(config)
    dataset = build_dataset(config, "train", sample_ids=ids)
    ids = [int(row["sample_id"]) for row in dataset.rows]
    image = torch.stack([dataset[i]["image"] for i in range(len(dataset))]).to(device)
    target = torch.stack([dataset[i]["target"] for i in range(len(dataset))]).to(device)
    model, criterion = build_model(config).to(device), build_loss(config).to(device)
    optimizer = build_optimizer(model, training)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    initial = training_batch_metrics(model, criterion, image, target)
    run_environment = environment(device)
    atomic_json(output_dir/"config.json", config)
    history = []
    synchronize(device)
    start = time.perf_counter()
    for step in range(1, steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, training["precision"]):
            logits = model(image)
        loss, terms = criterion(logits.float(), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite overfit loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 16 == 0 or step == steps:
            stats = {"step": step, "optimization_loss": float(loss.detach()),
                     **training_batch_metrics(model, criterion, image, target)}
            history.append(stats)
            print(json.dumps(stats), flush=True)
    synchronize(device)
    final = history[-1]
    report = {"purpose": "Fit a fixed train-only batch; check optimizer, logits, labels and normalization",
              "sample_ids": ids, "config_hash": config_hash(config), "environment": run_environment,
              "steps": steps, "initial": initial, "final": final, "history": history,
              "seconds": time.perf_counter()-start,
              "gate": {"loss_ratio_max": 0.3, "iou_macro_min": 0.8,
                       "passed": final["loss"] <= 0.3*initial["loss"] and final["iou_macro"] >= 0.8},
              "validation_or_test_used": False}
    atomic_json(output_dir/"overfit_report.json", report)
    atomic_torch_save(output_dir/"diagnostic_weights.pt", {"kind":"facs-overfit-diagnostic-v1", "model":model.state_dict(), "config":config})
    return report


def benchmark(config, output_dir, *, device="cuda", batch_sizes=(2, 12), warmup=2, steps=5):
    if warmup < 1 or steps < 2 or any(batch < 1 for batch in batch_sizes):
        raise ValueError("Benchmark needs warmup and at least two measured steps")
    training = config["training"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(training.get("cpu_threads", 4))
    run_environment = environment(device)
    measurements = []
    for batch_size in batch_sizes:
        seed_everything(training["seed"], training["deterministic"])
        torch.set_num_threads(training.get("cpu_threads", 4))
        dataset = build_dataset(config, "train")
        if batch_size > len(dataset):
            raise ValueError("Benchmark batch exceeds selected train population")
        image = torch.stack([dataset[i]["image"] for i in range(batch_size)]).to(device)
        target = torch.stack([dataset[i]["target"] for i in range(batch_size)]).to(device)
        model, criterion = build_model(config).to(device), build_loss(config).to(device)
        optimizer = build_optimizer(model, training)
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for step in range(warmup+steps):
            synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, training["precision"]):
                logits = model(image)
            loss, _ = criterion(logits.float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite benchmark loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            synchronize(device)
            if step >= warmup:
                times.append(time.perf_counter()-start)
        mean = sum(times)/len(times)
        row = {"batch_size": batch_size, "precision": training["precision"], "step_seconds": times,
               "mean_step_seconds": mean, "images_per_second": batch_size/mean,
               "train_epoch_compute_seconds_9600": math.ceil(9600/batch_size)*mean,
               "peak_memory_mib": torch.cuda.max_memory_allocated(device)/2**20 if torch.device(device).type == "cuda" else None,
               "parameter_count": sum(p.numel() for p in model.parameters())}
        measurements.append(row)
        print(json.dumps(row), flush=True)
        del optimizer, criterion, model, image, target, logits, loss
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    result = {"purpose":"GPU compute measurement on train images only; excludes loading, validation and checkpoint IO",
              "config_hash":config_hash(config), "warmup_steps":warmup, "measurements":measurements,
              "environment":run_environment, "test_used":False}
    atomic_json(output_dir/"benchmark.json", result)
    return result


def pretrain_edge(config, output_dir, *, device="cuda", epochs=5, batch_size=64, noise_sigma=0.1):
    if epochs < 1 or batch_size < 1 or not math.isfinite(noise_sigma) or noise_sigma < 0:
        raise ValueError("Invalid edge pretraining settings")
    training = config["training"]
    seed_everything(training["seed"], training["deterministic"])
    torch.set_num_threads(training.get("cpu_threads", 4))

    edge_config = json.loads(json.dumps(config))
    edge_config["data"].pop("train_ids", None)
    edge_config["data"]["augmentation"] = "none"
    dataset = build_dataset(edge_config, "train")
    generator = torch.Generator().manual_seed(training["seed"])
    loader = make_loader(dataset, batch_size=batch_size, workers=training["workers"], shuffle=True, generator=generator)
    network = EdgeNetwork().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    run_environment = environment(device)
    history = []
    for epoch in range(epochs):
        total_loss, seen = 0.0, 0
        tp, fp, fn = 0, 0, 0
        synchronize(device)
        start = time.perf_counter()
        network.train()
        for batch in loader:
            target = batch["target"].to(device, non_blocking=True)
            distribution = binary_distribution(target)
            noisy = (distribution+torch.randn_like(distribution)*noise_sigma).softmax(1)
            edges = semantic_edges(target)
            optimizer.zero_grad(set_to_none=True)
            logits = network(noisy)
            loss = edge_cross_entropy(logits, edges)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite SEMEDA pretraining loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            prediction = logits.detach().argmax(1).bool()
            truth = edges.bool()
            tp += int((prediction & truth).sum())
            fp += int((prediction & ~truth).sum())
            fn += int((~prediction & truth).sum())
            total_loss += float(loss.detach())*target.shape[0]
            seen += target.shape[0]
        synchronize(device)
        if seen != 9600:
            raise RuntimeError("Edge pretraining must use all 9600 train masks")
        row = {"epoch":epoch, "count":seen, "loss":total_loss/seen,
               "online_edge_dice_micro":2*tp/max(2*tp+fp+fn, 1), "seconds":time.perf_counter()-start}
        history.append(row)
        print(json.dumps(row), flush=True)
    checkpoint = {"kind":"semeda-edge-v1", "model":network.state_dict(), "training_split":"train",
                  "training_ids":[int(r["sample_id"]) for r in dataset.rows],
                  "completed_epochs":epochs, "noise_sigma":noise_sigma, "history":history,
                  "manifest_sha256":dataset.manifest_sha256,
                  "source_archive_sha256":json.loads((dataset.root/"source_archive.json").read_text())["sha256"],
                  "border_policy":"replicate labels for eight-neighbour semantic edges",
                  "input_perturbation":"softmax(onehot + Gaussian(0, noise_sigma))",
                  "embedding":"pre-activation convolution outputs", "environment":run_environment}
    path = output_dir/"edge.pt"
    atomic_torch_save(path, checkpoint)
    report = {k:v for k,v in checkpoint.items() if k != "model"}
    report["checkpoint_sha256"] = sha256(path)
    report["historical_weights_recovered"] = False
    atomic_json(output_dir/"edge_pretraining.json", report)
    return report
