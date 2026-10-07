"""Scientific and file-integrity checks for reconstructed width subsets."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from scripts.prepare_cv12 import (
    BIN_NAMES, archive_members, checked_bytes, extract_archive, index_split,
    inspect_pair, make_groups, materialize_groups, measure_width, select_decile,
    thickness_bin,
)


class WidthTests(unittest.TestCase):
    def test_one_pixel_horizontal_line(self):
        mask = np.zeros((16, 32), dtype=bool)
        mask[8, 4:24] = True
        self.assertEqual(measure_width(mask), (20, 20, 1.0))

    def test_diagonal_pixels_are_not_euclidean_length(self):
        mask = np.eye(16, dtype=bool)
        self.assertEqual(measure_width(mask), (16, 16, 1.0))

    def test_thick_region_is_actually_thinned(self):
        mask = np.zeros((32, 64), dtype=bool)
        mask[12:19, 5:55] = True
        area, length, width = measure_width(mask)
        self.assertEqual(area, 350)
        self.assertLess(length, area)
        self.assertGreater(width, 6)
        self.assertLess(width, 9)

    def test_empty_has_zero_width(self):
        self.assertEqual(measure_width(np.zeros((16, 16), dtype=bool)), (0, 0, 0.0))

    def test_invalid_measurement_inputs(self):
        for mask in [np.zeros((2, 2), dtype=np.uint8), np.zeros((1, 2, 2), dtype=bool)]:
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                with self.assertRaises(ValueError):
                    measure_width(mask)

    def test_bin_boundaries(self):
        cases = [(0, "zero"), (1, "0_2"), (2, "0_2"), (2.001, "2_4"),
                 (4, "2_4"), (4.001, "4_8"), (8, "4_8"), (8.001, "8_16"),
                 (16, "8_16"), (16.001, "16_32"), (32, "16_32"), (32.001, "thick")]
        for width, expected in cases:
            with self.subTest(width=width):
                self.assertEqual(thickness_bin(width), expected)
        for invalid in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                thickness_bin(invalid)

    def test_positive_decile_excludes_negatives_and_uses_ceiling(self):
        rows = [dict(sample_id=i, area_pixels=10, width_px=float(i)) for i in range(1, 12)]
        rows += [dict(sample_id=100 + i, area_pixels=0, width_px=0.0) for i in range(20)]
        self.assertEqual([r["sample_id"] for r in select_decile(rows)], [1, 2])
        self.assertEqual([r["sample_id"] for r in select_decile(rows, thickest=True)], [10, 11])
        self.assertEqual(select_decile(rows[11:]), [])

    def test_decile_ties_use_id_not_input_order(self):
        rows = [dict(sample_id=i, area_pixels=10, width_px=2.0) for i in range(11, 0, -1)]
        self.assertEqual([r["sample_id"] for r in select_decile(rows)], [1, 2])

    def test_exclusive_bins_cover_each_image_once(self):
        widths = [0, 1, 3, 6, 12, 24, 40]
        rows = [dict(sample_id=i, area_pixels=int(w > 0), width_px=w,
                     thickness_bin=thickness_bin(w)) for i, w in enumerate(widths)]
        groups = make_groups(rows)
        self.assertEqual(sorted(r["sample_id"] for b in BIN_NAMES for r in groups[b]), list(range(7)))
        self.assertEqual(len(groups["nonzero"]), 6)
        self.assertEqual(len(groups["thinnest_10pct"]), 1)


class FileIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture_archive(self, *, invalid_values=False, mismatched_size=False):
        fixture = self.root / "fixture"
        image = fixture / "test" / "IMG" / "10801.png"
        gt = fixture / "test" / "GT" / "10801.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        gt.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (256, 256)).save(image)
        mask = np.zeros((128 if mismatched_size else 256, 256), dtype=np.uint8)
        mask[8, 4:24] = 2 if invalid_values else 255
        Image.fromarray(mask).save(gt)
        archive = self.root / "fixture.zip"
        with zipfile.ZipFile(archive, "w") as container:
            for p in (image, gt):
                container.write(p, "split_dataset_final/" + str(p.relative_to(fixture)))
        return archive

    def test_archive_path_escape_is_rejected(self):
        archive = self.root / "invalid.zip"
        with zipfile.ZipFile(archive, "w") as container:
            container.writestr("split_dataset_final/../../outside.png", b"x")
        with self.assertRaises(ValueError):
            archive_members(archive)

    def test_extract_is_idempotent_and_does_not_overwrite_source(self):
        archive = self.fixture_archive()
        members = archive_members(archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        output = self.root / "data"
        extract_archive(archive, output, members, digest)
        path = output / "test" / "GT" / "10801.png"
        before = path.stat().st_mtime_ns
        extract_archive(archive, output, members, digest)
        self.assertEqual(before, path.stat().st_mtime_ns)
        with self.assertRaises(ValueError):
            extract_archive(archive, output, members, "another-archive")

    def test_pixel_measurement_and_pair_checks(self):
        archive = self.fixture_archive()
        members = archive_members(archive)
        output = self.root / "fixture"
        row = inspect_pair((output, "test", "10801.png", members))
        self.assertEqual((row["area_pixels"], row["skeleton_pixels"], row["width_px"]), (20, 20, 1.0))
        self.assertEqual(index_split(output, "test", {10801}), ["10801.png"])
        (output / "test" / "GT" / "10801.png").unlink()
        with self.assertRaises(ValueError):
            index_split(output, "test", {10801})

    def test_nonbinary_gt_is_rejected(self):
        archive = self.fixture_archive(invalid_values=True)
        with self.assertRaisesRegex(ValueError, "binary GT"):
            inspect_pair((self.root / "fixture", "test", "10801.png", archive_members(archive)))

    def test_shape_mismatch_is_rejected(self):
        archive = self.fixture_archive(mismatched_size=True)
        with self.assertRaisesRegex(ValueError, "aligned 256x256"):
            inspect_pair((self.root / "fixture", "test", "10801.png", archive_members(archive)))

    def test_changed_source_bytes_are_rejected(self):
        archive = self.fixture_archive()
        members = archive_members(archive)
        path = self.root / "fixture" / "test" / "GT" / "10801.png"
        path.write_bytes(b"not-the-original-mask")
        with self.assertRaisesRegex(ValueError, "differs from archive"):
            checked_bytes(self.root / "fixture", "test/GT/10801.png", members)

    def test_relative_links_are_idempotent(self):
        source = self.root / "test" / "GT" / "10801.png"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"GT")
        image = self.root / "test" / "IMG" / "10801.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image")
        groups = {"0_2": [dict(filename="10801.png")]}
        materialize_groups(self.root, groups)
        dest = self.root / "split" / "GT" / "0_2" / "10801.png"
        self.assertTrue(dest.is_symlink())
        self.assertFalse(dest.readlink().is_absolute())
        self.assertEqual(dest.read_bytes(), b"GT")
        before = dest.lstat().st_mtime_ns
        materialize_groups(self.root, groups)
        self.assertEqual(before, dest.lstat().st_mtime_ns)

    def test_existing_regular_subset_file_is_not_overwritten(self):
        dest = self.root / "split" / "GT" / "0_2" / "10801.png"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"preserve-me")
        with self.assertRaises(ValueError):
            materialize_groups(self.root, {"0_2": [dict(filename="10801.png")]})
        self.assertEqual(dest.read_bytes(), b"preserve-me")

    def test_legacy_testmode_paths_match_generated_bin_names(self):
        source = Path(__file__).resolve().parents[1] / "code" / "src" / "data_init.py"
        spec = importlib.util.spec_from_file_location("data_init", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for mode, name in enumerate(("0_2", "2_4", "4_8", "8_16", "16_32", "thick", "zero"), 1):
            img, gt = module.CrackVision12K(SimpleNamespace(testmode=mode), "test", str(self.root))
            self.assertEqual(Path(img), self.root / "CrackVision12K" / "split" / "IMG" / name)
            self.assertEqual(Path(gt), self.root / "CrackVision12K" / "split" / "GT" / name)


if __name__ == "__main__":
    unittest.main()
