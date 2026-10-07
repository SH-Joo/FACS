"""Manifest-driven paired loading; deterministic validation and original split IDs."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


LEGACY_MEAN = (0.51789941, 0.51360926, 0.547762)
LEGACY_STD = (0.1812099, 0.17746663, 0.20386334)


class CV12Dataset(Dataset):
    def __init__(self, root, split, *, subset=None, sample_ids=None,
                 mean=LEGACY_MEAN, std=LEGACY_STD, augmentation="none", verify_hashes=True):
        self.root = Path(root).resolve()
        if split not in ("train", "val", "test") or (subset is not None and split != "test"):
            raise ValueError("Use train/val/test; width subsets are test-only")
        if augmentation not in ("none", "dihedral") or (split != "train" and augmentation != "none"):
            raise ValueError("Validation and test must use deterministic transforms")
        if (len(mean) != 3 or len(std) != 3
                or any(not math.isfinite(v) for v in (*mean, *std)) or any(s <= 0 for s in std)):
            raise ValueError("RGB normalization requires three means and positive standard deviations")
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.augmentation, self.verify_hashes = augmentation, verify_hashes
        manifests = self.root / "manifests"
        path = manifests/f"{split}.csv" if subset is None else manifests/"subsets"/f"{subset}.csv"
        if subset is not None and (Path(subset).name != subset or subset in (".", "..")):
            raise ValueError("Invalid subset name")
        self.manifest_path = path
        self.manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("Dataset manifest is empty or has duplicate IDs")
        for row in rows:
            if row["split"] != split or int(Path(row["filename"]).stem) != int(row["sample_id"]):
                raise ValueError("Wrong split or sample name in manifest")
            for key, kind in (("image_path", "IMG"), ("gt_path", "GT")):
                expected = Path(split)/kind/row["filename"]
                if Path(row[key]) != expected or not (self.root/expected).is_file():
                    raise ValueError(f"Missing or unexpected paired path: {row[key]}")
        if sample_ids is not None:
            requested = {int(i) for i in sample_ids}
            if len(requested) != len(sample_ids) or not requested <= {int(r["sample_id"]) for r in rows}:
                raise ValueError("Requested samples are missing or repeated")
            rows = [r for r in rows if int(r["sample_id"]) in requested]
            if not rows:
                raise ValueError("Sample selection is empty")
        self.rows = sorted(rows, key=lambda row: int(row["sample_id"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_bytes = (self.root/row["image_path"]).read_bytes()
        gt_bytes = (self.root/row["gt_path"]).read_bytes()
        if self.verify_hashes:
            for raw, key in ((image_bytes, "image_sha256"), (gt_bytes, "gt_sha256")):
                if hashlib.sha256(raw).hexdigest() != row[key]:
                    raise ValueError(f"Dataset changed after preparation: {row['filename']}, {key}")
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = np.array(image.convert("RGB"), copy=True)
        with Image.open(io.BytesIO(gt_bytes)) as image:
            mask = np.array(image.convert("L"), copy=True)
        expected_shape = (int(row["height"]), int(row["width"]))
        if mask.shape != expected_shape or rgb.shape != (*expected_shape, 3) or expected_shape != (256, 256):
            raise ValueError("CV12 image and GT must remain aligned 256x256")
        if not set(np.unique(mask).tolist()) <= {0, 1, 255}:
            raise ValueError("Unexpected GT encoding")
        image = torch.from_numpy(rgb).permute(2, 0, 1).float()/255
        target = torch.from_numpy(mask > 0).float()[None]
        if self.augmentation == "dihedral":
            rotation = int(torch.randint(0, 4, ()).item())
            image, target = torch.rot90(image, rotation, (-2, -1)), torch.rot90(target, rotation, (-2, -1))
            if torch.rand(()) < 0.5:
                image, target = image.flip(-1), target.flip(-1)
        return {"image": (image-self.mean)/self.std, "target": target,
                "sample_id": int(row["sample_id"]), "filename": row["filename"],
                "thickness_bin": row["thickness_bin"]}


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, *, batch_size, workers=4, shuffle=False, generator=None):
    if batch_size < 1 or workers < 0:
        raise ValueError("Invalid batch size or worker count")
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                      shuffle=shuffle, drop_last=False, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=generator, persistent_workers=False)
