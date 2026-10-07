#!/usr/bin/env python3
"""Export a checkpoint with BatchNorm statistics recalculated on training images."""

import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.optim.swa_utils import update_bn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from facs.data import make_loader
from facs.engine import (atomic_json, atomic_torch_save, build_dataset, build_model,
                         config_hash, environment, seed_everything, sha256)


def recalibrate(checkpoint_path, output, *, device="cpu", sample_count=9600, seed=8106,
                batch_size=12, workers=0, cpu_threads=2):
    if sample_count < batch_size or sample_count > 9600 or sample_count % batch_size:
        raise ValueError("Use a positive train count ≤9600, divisible by batch size")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    pinned = output/"input.pt"
    os.link(Path(checkpoint_path).resolve(), pinned)
    checkpoint = torch.load(pinned, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint["kind"] not in ("facs-training-v1", "facs-inference-v1"):
        raise ValueError("Use a strict migrated or v2 checkpoint")
    seed_everything(seed, True)
    torch.set_num_threads(cpu_threads)
    config = copy.deepcopy(checkpoint["config"])
    calibration_config = copy.deepcopy(config)
    calibration_config["data"]["augmentation"] = "none"
    calibration_config["data"].pop("train_ids", None)
    full = build_dataset(calibration_config, "train")
    if len(full) != 9600 or [int(row["sample_id"]) for row in full.rows] != list(range(1, 9601)):
        raise ValueError("Calibration requires the original CV12 train split")
    generator = torch.Generator().manual_seed(seed)
    selected = (torch.randperm(9600, generator=generator)[:sample_count]+1).tolist()
    dataset = build_dataset(calibration_config, "train", sample_ids=selected)
    loader = make_loader(dataset, batch_size=batch_size, workers=workers, shuffle=True, generator=generator)
    model = build_model(config, loading=True).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    batchnorm = {name: module for name, module in model.named_modules()
                 if isinstance(module, nn.modules.batchnorm._BatchNorm)}
    allowed = {f"{name}.{suffix}" for name in batchnorm
               for suffix in ("running_mean", "running_var", "num_batches_tracked")}
    seen = []
    provenance = {"purpose": "Train-only BatchNorm buffer recalibration; no optimization or weight averaging",
                  "state": "running", "source_checkpoint": str(Path(checkpoint_path).resolve()),
                  "source_checkpoint_sha256": sha256(pinned), "source_epoch_zero_based": checkpoint["epoch"],
                  "script_sha256": sha256(__file__), "pytorch_update_bn_source": inspect.getsource(update_bn),
                  "sample_count": sample_count, "sample_ids": sorted(selected), "split": "train",
                  "augmentation": "none", "batch_size": batch_size, "workers": workers, "seed": seed,
                  "precision": "fp32", "device": device, "cpu_threads": cpu_threads,
                  "scope": "full_train" if sample_count == 9600 else "train_subset_diagnostic",
                  "test_or_val_used_for_calibration": False, "batchnorm_module_count": len(batchnorm),
                  "train_manifest_sha256": dataset.manifest_sha256,
                  "environment": environment(device), "weights_changed": False,
                  "inference_mode": "eval", "evaluation_profile": config["evaluation"]}
    atomic_json(output/"calibration.json", provenance)

    def images():
        for index, batch in enumerate(loader):
            seen.extend(int(value) for value in batch["sample_id"])
            if index % 8 == 0:
                print(json.dumps({"calibration_batch": index+1, "seen_images": len(seen),
                                  "total_images": sample_count}), flush=True)
            yield batch["image"]

    start = time.perf_counter()
    update_bn(images(), model, device=device)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    provenance["seconds"] = time.perf_counter()-start
    if sorted(seen) != sorted(selected):
        raise ValueError("Calibration coverage differs")
    if model.training:
        raise RuntimeError("Calibration did not restore eval mode")
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if state.keys() != checkpoint["model"].keys():
        raise RuntimeError("Calibration changed state keys")
    changed = []
    for name, value in state.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite calibration state: {name}")
        if not torch.equal(value, checkpoint["model"][name]):
            if name not in allowed:
                raise RuntimeError(f"Calibration changed a non-BatchNorm tensor: {name}")
            changed.append(name)
    for name in batchnorm:
        if int(state[f"{name}.num_batches_tracked"]) != sample_count//batch_size:
            raise RuntimeError(f"Incorrect calibration batch count: {name}")
    config["protocol"]["id"] += "-train-bn"
    config["protocol"]["status"] = "locked" if sample_count == 9600 else "diagnostic"
    config["protocol"]["purpose"] = (
        "Full-training-split BatchNorm recalibration for inference"
        if sample_count == 9600 else "Training-subset BatchNorm diagnostic")
    config["protocol"]["choices"].append(
        f"Post-training BN recalibration: {sample_count} original train images, seed {seed}, batch {batch_size}, no augmentation")
    metadata = copy.deepcopy(checkpoint["metadata"])
    metadata["parent_config_hash"] = metadata.get("config_hash")
    metadata["config_hash"] = config_hash(config)
    provenance.update(state="complete", completed_images=len(seen),
                      changed_state_keys=changed, only_batchnorm_buffers_changed=True)
    metadata["batchnorm_calibration"] = copy.deepcopy(provenance)
    export = output/"checkpoint.pt"
    atomic_torch_save(export, {"kind": "facs-inference-v1", "epoch": checkpoint["epoch"],
                              "model": state, "config": config, "metadata": metadata})
    provenance["output_checkpoint"] = str(export)
    provenance["output_checkpoint_sha256"] = sha256(export)
    atomic_json(output/"calibration.json", provenance)
    print(json.dumps({"output": str(export), "seconds": provenance["seconds"],
                      "changed_bn_buffers": len(changed), "sample_count": sample_count,
                      "only_batchnorm_buffers_changed": True}), flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=9600)
    parser.add_argument("--seed", type=int, default=8106)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    recalibrate(args.checkpoint, args.output_dir, device=args.device, sample_count=args.sample_count,
                seed=args.seed, batch_size=args.batch_size, workers=args.workers, cpu_threads=args.cpu_threads)


if __name__ == "__main__":
    main()
