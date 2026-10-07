"""Reconstruct CV12 width subsets without changing the source images or GT.

Width = foreground area / Zhang-Suen skeleton pixel count, in the supplied
256 x 256 masks. This reproduces all six counts in paper Table 3 and the
archived positive-test thinnest and thickest deciles. Run from any directory:

    .venv/bin/python scripts/prepare_cv12.py --archive CrackVision12K.zip
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import tempfile
import zipfile
import zlib

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION
import skimage
from skimage.morphology import skeletonize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "split_dataset_final"
PROTOCOL = "cv12-area-over-zhang-skeleton-v1"
SPLITS = {"train": (1, 9600), "val": (9601, 10800), "test": (10801, 12000)}
PAPER_COUNTS = {
    "0_2": 63, "2_4": 64, "4_8": 141, "8_16": 252,
    "16_32": 330, "thick": 132,
}
BIN_NAMES = ("zero", *PAPER_COUNTS)
FIELDS = (
    "sample_id", "split", "filename", "image_path", "gt_path", "height",
    "width", "image_mode", "gt_mode", "mask_values", "area_pixels",
    "skeleton_pixels", "width_px", "thickness_bin", "image_sha256", "gt_sha256",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def thickness_bin(width: float) -> str:
    """Bins are left-open, right-closed; zero is reserved for an empty GT."""
    if not math.isfinite(width) or width < 0:
        raise ValueError(f"Invalid crack width: {width}")
    if width == 0:
        return "zero"
    for upper, name in zip((2, 4, 8, 16, 32), PAPER_COUNTS):
        if width <= upper:
            return name
    return "thick"


def measure_width(mask: np.ndarray) -> tuple[int, int, float]:
    """Count centerline pixels, including diagonals as one pixel, without cleanup."""
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Width measurement requires a 2D boolean mask")
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0, 0, 0.0
    length = int(np.count_nonzero(skeletonize(mask, method="zhang")))
    if length == 0:
        raise ValueError("Nonempty GT produced an empty skeleton")
    return area, length, area / length


def select_decile(rows: list[dict], *, thickest: bool = False) -> list[dict]:
    """Select ceil(10% of positive test images), resolving width ties by ID."""
    ordered = sorted(
        (row for row in rows if row["area_pixels"] > 0),
        key=lambda row: (row["width_px"], row["sample_id"]),
    )
    count = (len(ordered) + 9) // 10
    if count == 0:
        return []
    selected = ordered[-count:] if thickest else ordered[:count]
    return sorted(selected, key=lambda row: row["sample_id"])


def archive_members(archive: Path) -> dict[str, dict]:
    members = {}
    with zipfile.ZipFile(archive) as container:
        for info in container.infolist():
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename)
            if (
                path.is_absolute() or ".." in path.parts or "\\" in info.filename
                or len(path.parts) < 3 or path.parts[0] != ARCHIVE_ROOT
                or ((info.external_attr >> 16) & 0o170000) == 0o120000
            ):
                raise ValueError(f"Unsupported archive member: {info.filename}")
            relative = str(PurePosixPath(*path.parts[1:]))
            if relative in members:
                raise ValueError(f"Duplicate archive path: {relative}")
            members[relative] = {
                "archive_path": info.filename, "size": info.file_size, "crc32": info.CRC,
            }
    if not members:
        raise ValueError("Empty dataset archive")
    return members


def extract_archive(archive: Path, output: Path, members: dict, digest: str) -> None:
    """Stage new extraction atomically; never overwrite an existing dataset."""
    marker = output / "source_archive.json"
    if output.exists():
        if not marker.is_file():
            raise ValueError(f"Existing directory lacks source metadata: {output}")
        source = json.loads(marker.read_text())
        if source["sha256"] != digest:
            raise ValueError("Existing dataset came from a different archive")
        for relative, info in members.items():
            path = output / relative
            if not path.is_file() or path.is_symlink() or path.stat().st_size != info["size"]:
                raise ValueError(f"Missing, changed or incomplete source file: {relative}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-extract-", dir=output.parent))
    try:
        with zipfile.ZipFile(archive) as container:
            for index, (relative, info) in enumerate(members.items(), 1):
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with container.open(info["archive_path"]) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                if index % 5000 == 0:
                    print(f"Extracted {index}/{len(members)} source files", flush=True)
        (staging / "source_archive.json").write_text(json.dumps({
            "archive_name": archive.name, "sha256": digest,
            "archive_bytes": archive.stat().st_size, "archive_root": ARCHIVE_ROOT,
            "extracted_files": len(members),
        }, indent=2) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise


def checked_bytes(root: Path, relative: str, members: dict) -> bytes:
    data = (root / relative).read_bytes()
    info = members[relative]
    if len(data) != info["size"] or zlib.crc32(data) != info["crc32"]:
        raise ValueError(f"Source differs from archive: {relative}")
    return data


def index_split(root: Path, split: str, expected_ids: set[int]) -> list[str]:
    directories = {kind: root / split / kind for kind in ("IMG", "GT")}
    names = {}
    for kind, directory in directories.items():
        entries = list(directory.iterdir())
        if any(not p.is_file() or p.is_symlink() or p.suffix != ".png" for p in entries):
            raise ValueError(f"Unexpected files or folders: {directory}")
        names[kind] = {p.name for p in entries}
    if names["IMG"] != names["GT"]:
        raise ValueError(f"Image/GT pairing mismatch in {split}")
    ids = [int(Path(name).stem) for name in names["IMG"]]
    if len(set(ids)) != len(ids) or set(ids) != expected_ids:
        raise ValueError(f"Unexpected or duplicate sample IDs in {split}")
    return sorted(names["IMG"], key=lambda name: int(Path(name).stem))


def inspect_pair(task: tuple[Path, str, str, dict]) -> dict:
    root, split, name, members = task
    image_path, gt_path = f"{split}/IMG/{name}", f"{split}/GT/{name}"
    image_bytes = checked_bytes(root, image_path, members)
    gt_bytes = checked_bytes(root, gt_path, members)
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        image_size, image_mode = image.size, image.mode
    with Image.open(io.BytesIO(gt_bytes)) as gt:
        gt_mode = gt.mode
        values = np.array(gt.convert("L"))
        gt_size = gt.size
    if image_size != gt_size or gt_size != (256, 256):
        raise ValueError(f"Expected aligned 256x256 image and GT: {split}/{name}")
    unique = np.unique(values).tolist()
    if not set(unique).issubset({0, 1, 255}):
        raise ValueError(f"Unexpected binary GT values {unique}: {gt_path}")
    area, length, width = measure_width(values > 0)
    return dict(
        sample_id=int(Path(name).stem), split=split, filename=name,
        image_path=image_path, gt_path=gt_path, height=gt_size[1], width=gt_size[0],
        image_mode=image_mode, gt_mode=gt_mode, mask_values=json.dumps(unique),
        area_pixels=area, skeleton_pixels=length, width_px=width,
        thickness_bin=thickness_bin(width),
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        gt_sha256=hashlib.sha256(gt_bytes).hexdigest(),
    )


def make_groups(test_rows: list[dict]) -> dict[str, list[dict]]:
    groups = {name: [r for r in test_rows if r["thickness_bin"] == name] for name in BIN_NAMES}
    groups["nonzero"] = [r for r in test_rows if r["area_pixels"] > 0]
    groups["thinnest_10pct"] = select_decile(test_rows)
    groups["thickest_10pct"] = select_decile(test_rows, thickest=True)
    return groups


def validate_reference(root: Path, members: dict, groups: dict[str, list[dict]]) -> dict:
    counts = {name: len(groups[name]) for name in PAPER_COUNTS}
    if counts != PAPER_COUNTS:
        raise ValueError(f"Table 3 mismatch: expected {PAPER_COUNTS}, got {counts}")
    comparisons = {}
    test_by_name = {r["filename"]: r for name in BIN_NAMES for r in groups[name]}
    for archived, generated in (
        ("test_nonzero", "nonzero"), ("test_thin", "thinnest_10pct"),
        ("test_thick", "thickest_10pct"),
    ):
        for kind in ("GT", "IMG"):
            prefix = f"{archived}/{kind}/"
            paths = [relative for relative in members if relative.startswith(prefix)]
            if not paths:
                comparisons[f"{archived}/{kind}"] = {"present": False}
                continue
            archived_names = {Path(relative).name for relative in paths}
            generated_names = {r["filename"] for r in groups[generated]}
            if archived_names != generated_names:
                raise ValueError(f"Archived membership differs: {archived}/{kind}")
            for relative in paths:
                row = test_by_name[Path(relative).name]
                expected = row["gt_sha256" if kind == "GT" else "image_sha256"]
                if hashlib.sha256(checked_bytes(root, relative, members)).hexdigest() != expected:
                    raise ValueError(f"Archived copy differs from original test: {relative}")
            comparisons[f"{archived}/{kind}"] = {
                "present": True, "count": len(paths), "membership_matches": True,
                "file_bytes_match": True,
            }
    return comparisons


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        f.write(text)
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict], fields=FIELDS) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def materialize_groups(root: Path, groups: dict[str, list[dict]]) -> None:
    """Relative symlinks keep masks byte-identical and avoid dataset copies."""
    for kind in ("IMG", "GT"):
        parent = root / "split" / kind
        if parent.exists() and {p.name for p in parent.iterdir()} - set(groups):
            raise ValueError(f"Unexpected existing subset folders: {parent}")
        for name, rows in groups.items():
            directory = parent / name
            expected = {r["filename"] for r in rows}
            if directory.exists() and {p.name for p in directory.iterdir()} - expected:
                raise ValueError(f"Existing subset has unexpected membership: {directory}")
            for row in rows:
                dest = directory / row["filename"]
                source = root / "test" / kind / row["filename"]
                if os.path.lexists(dest) and (not dest.is_symlink() or dest.resolve() != source.resolve()):
                    raise ValueError(f"Refusing to overwrite subset entry: {dest}")
    for kind in ("IMG", "GT"):
        for name, rows in groups.items():
            directory = root / "split" / kind / name
            directory.mkdir(parents=True, exist_ok=True)
            for row in rows:
                dest = directory / row["filename"]
                source = root / "test" / kind / row["filename"]
                if not os.path.lexists(dest):
                    dest.symlink_to(os.path.relpath(source, directory))


def duplicate_report(rows: list[dict]) -> tuple[list[dict], dict]:
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["image_sha256"]].append(row)
    duplicate_rows, cross_split = [], Counter()
    for digest, group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        splits = sorted({r["split"] for r in group})
        for i, left in enumerate(splits):
            for right in splits[i + 1:]:
                cross_split[f"{left}--{right}"] += 1
        duplicate_rows.extend({
            "image_sha256": digest, "sample_id": r["sample_id"], "split": r["split"],
            "filename": r["filename"], "group_size": len(group),
            "distinct_gt_hashes": len({g["gt_sha256"] for g in group}),
        } for r in group)
    return duplicate_rows, {
        "unique_image_files": len(by_hash), "duplicate_sample_count": len(duplicate_rows),
        "cross_split_duplicate_hash_groups": dict(cross_split),
    }


def prepare(archive: Path, output: Path, workers: int) -> dict:
    if skimage.__version__ != "0.25.2":
        raise ValueError("Install requirements-data.txt to pin the skeleton implementation")
    archive, output = archive.resolve(), output.resolve()
    if workers < 1:
        raise ValueError("workers must be at least 1")
    members = archive_members(archive)
    digest = file_sha256(archive)
    extract_archive(archive, output, members, digest)
    tasks = []
    for split, (start, end) in SPLITS.items():
        for name in index_split(output, split, set(range(start, end + 1))):
            tasks.append((output, split, name, members))
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, row in enumerate(executor.map(inspect_pair, tasks), 1):
            rows.append(row)
            if index % 1000 == 0:
                print(f"Validated and measured {index}/{len(tasks)} image/GT pairs", flush=True)
    by_split = {split: [r for r in rows if r["split"] == split] for split in SPLITS}
    groups = make_groups(by_split["test"])
    reference = validate_reference(output, members, groups)
    materialize_groups(output, groups)
    manifests = output / "manifests"
    atomic_csv(manifests / "all.csv", rows)
    for split, items in by_split.items():
        atomic_csv(manifests / f"{split}.csv", items)
    for name, items in groups.items():
        atomic_csv(manifests / "subsets" / f"{name}.csv", items)
        atomic_text(manifests / "subsets" / f"{name}.txt", "".join(r["filename"] + "\n" for r in items))
    duplicates, duplicate_summary = duplicate_report(rows)
    atomic_csv(manifests / "duplicate_images.csv", duplicates, (
        "image_sha256", "sample_id", "split", "filename", "group_size", "distinct_gt_hashes",
    ))
    ordered = sorted(groups["nonzero"], key=lambda r: (r["width_px"], r["sample_id"]))
    thin_count = len(groups["thinnest_10pct"])
    thick_count = len(groups["thickest_10pct"])
    summary = {
        "protocol": PROTOCOL, "source_archive_sha256": digest,
        "preparation_script_sha256": file_sha256(Path(__file__)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_image": skimage.__version__, "pillow": PILLOW_VERSION},
        "width_definition": "area_pixels / skeleton_pixels",
        "skeleton": "skimage.morphology.skeletonize(method='zhang')",
        "mask_binarization": "GT > 0, accepted values 0/1/255",
        "resolution": [256, 256], "resize": False, "morphological_cleanup": False,
        "diagonal_length": "one per skeleton pixel, no sqrt(2) correction",
        "split_counts": {split: len(items) for split, items in by_split.items()},
        "bin_counts_by_split": {split: dict(Counter(r["thickness_bin"] for r in items)) for split, items in by_split.items()},
        "test_subset_counts": {name: len(items) for name, items in groups.items()},
        "table3_counts_match": True, "archived_subset_checks": reference,
        "thinnest_10pct": {"population": "982 positive test images",
                           "selection": "ceil(n/10), ordered by (width_px, sample_id)",
                           "count": thin_count, "max_width_px": ordered[thin_count - 1]["width_px"],
                           "next_width_px": ordered[thin_count]["width_px"]},
        "thickest_10pct": {"count": thick_count, "min_width_px": ordered[-thick_count]["width_px"]},
        "test_negative_images": len(groups["zero"]), "duplicates": duplicate_summary,
        "storage": "original PNG bytes; subset IMG/GT are relative symlinks",
        "training_started": False,
    }
    atomic_text(output / "preparation_summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "splits": summary["split_counts"],
                      "test_subsets": summary["test_subset_counts"],
                      "paper_counts_match": True, "duplicates": duplicate_summary}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=PROJECT_ROOT / "CrackVision12K.zip")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "CrackVision12K")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    try:
        prepare(args.archive, args.output, args.workers)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Dataset preparation failed: {error}\n")


if __name__ == "__main__":
    main()
