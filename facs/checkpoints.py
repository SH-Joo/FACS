"""Strict, audited import of local Lightning checkpoints, preserving old inference."""

from pathlib import Path

import torch

from .engine import atomic_torch_save, config_hash, environment, sha256
from .model import FACSNet


def migrate_legacy(checkpoint_path, config, output_path):
    if config["model"]["norm_affine"] or config["model"]["filter_mode"] != "normalized":
        raise ValueError("Legacy inference requires fixed affine norms and uploaded normalized FPCM")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = source["state_dict"]
    model = FACSNet(**config["model"])
    model.load_state_dict(state, strict=True)
    metadata = {"config_hash":config_hash(config), "source_checkpoint_sha256":sha256(checkpoint_path),
                "source_state_key_count":len(state), "legacy_epoch":source.get("epoch"),
                "legacy_global_step":source.get("global_step"), "strict_load":True,
                "norm_behavior":"per-channel LayerNorm with fixed gamma=1, beta=0",
                "filter_behavior":"uploaded normalized radial Gaussian; learned cutoff retained",
                "trained_after_conversion":False, "environment":environment("cpu")}
    converted = {"kind":"facs-inference-v1", "model":state, "config":config,
                 "metadata":metadata, "epoch":source.get("epoch")}
    atomic_torch_save(output_path, converted)
    return {"checkpoint":str(output_path), "metadata":metadata}
