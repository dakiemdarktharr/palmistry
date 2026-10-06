"""Regression checks for data integrity and bounded archive decoding."""
import csv
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zlib
import numpy as np

from flask import Flask
from PIL import Image

import prototype_common as common
from prototype_fixture import generate_fixture
from palm_keypoints.web import register_keypoint_ui
from test_keypoints import fixture
import giao_dien_ui as ui


class AuditRegressions(unittest.TestCase):
    def test_health_endpoint_identifies_the_installation(self):
        response = ui.app.test_client().get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["service"], "palmistry")
        self.assertEqual(response.json["project_root"], str(ui.PROJECT_DIR.resolve()))

    def test_compressed_checkpoint_expansion_is_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "large.npz"
            np.savez_compressed(path, weights=np.zeros(4096, dtype=np.uint8))
            self.assertLess(path.stat().st_size, 1024)
            with self.assertRaisesRegex(ValueError, "Expanded checkpoint"):
                common.validate_npz_size(path, 1024)

    def test_malformed_checkpoint_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.npz"
            path.write_bytes(b"not a zip")
            with self.assertRaisesRegex(ValueError, "valid NumPy"):
                common.validate_npz_size(path, 1024)

    def test_atomic_json_preserves_previous_file_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            common.write_json(path, {"revision": 1})
            with mock.patch.object(Path, "replace", side_effect=OSError("disk unavailable")):
                with self.assertRaises(OSError):
                    common.write_json(path, {"revision": 2})
            self.assertEqual(json.loads(path.read_text()), {"revision": 1})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_nonfinite_json_does_not_destroy_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            common.write_json(path, {"ok": True})
            with self.assertRaises(ValueError):
                common.write_json(path, {"loss": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})

    def test_oversized_archive_header_rejected_before_cv_decode(self):
        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        header = struct.pack(">IIBBBBB", 10001, 5000, 8, 2, 0, 0, 0)
        raw = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", b"") + chunk(b"IEND", b"")
        with mock.patch.object(common.cv2, "imdecode", side_effect=AssertionError("must not allocate pixels")):
            with self.assertRaisesRegex(ValueError, "image_pixels_limit"):
                common.decode_archive_image(raw, 50_000_000)

    def test_archive_image_valid_and_invalid_headers(self):
        data = io.BytesIO()
        Image.new("RGB", (20, 10), "red").save(data, format="PNG")
        self.assertEqual(common.decode_archive_image(data.getvalue(), 200).shape, (10, 20, 3))
        with self.assertRaisesRegex(ValueError, "invalid_image"):
            common.decode_archive_image(b"not an image", 200)

    def test_group_ids_cannot_hide_reencoded_cross_split_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            generate_fixture(root)
            with (root / "labels.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            train = next(row for row in rows if row["split"] == "train")
            held = next(row for row in rows if row["split"] == "test")
            # Different encodings and different metadata, but the same pixels.
            with Image.open(root / train["image_path"]) as image:
                image.save(root / held["image_path"], compress_level=0)
            self.assertNotEqual(common.sha256(root / train["image_path"]), common.sha256(root / held["image_path"]))
            with self.assertRaisesRegex(ValueError, "leakage"):
                common.read_manifest(root, root / "labels.csv")

    def test_bad_project_does_not_block_healthy_project(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root / "artifacts/keypoints/healthy", 3)
            broken = root / "artifacts/keypoints/broken"
            broken.mkdir()
            (broken / "project.json").write_text("{truncated")
            app = Flask(__name__)
            register_keypoint_ui(app, root, "token", "nonce")
            response = app.test_client().get("/api/keypoints/state")
            self.assertEqual(response.status_code, 200)
            self.assertEqual([p["name"] for p in response.json["projects"]], ["healthy"])
            self.assertEqual([p["name"] for p in response.json["project_errors"]], ["broken"])


if __name__ == "__main__":
    unittest.main()
