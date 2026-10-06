import copy
import csv
import pickle
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import prototype_common as common
import prototype_fixture
import palmistry_strict_auto_onefile as pipeline
import giao_dien_ui as ui


class AdditionalContracts(unittest.TestCase):
    def test_missing_validation_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prototype_fixture.generate_fixture(root)
            rows = common.read_manifest(root, root / "labels.csv")
            for row in rows:
                if row["split"] == "val":
                    row["split"] = "train"
            with (root / "labels.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Non-empty train, val, and test"):
                common.read_manifest(root, root / "labels.csv")

    def test_malformed_training_mask_rejected_before_loss(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prototype_fixture.generate_fixture(root)
            cv2.imwrite(str(root / "images/case_0/mask.png"), np.full((128, 128), 8, np.uint8))
            with self.assertRaisesRegex(ValueError, "Unknown semantic"):
                common.read_manifest(root, root / "labels.csv")

    def test_failed_worker_invalidates_previous_snapshot(self):
        engine = ui.LiveEngine()
        engine.ready = True
        engine.latest_frame = np.zeros((32, 32, 3), np.uint8)
        engine.snapshot = {"frame_id": 0, "metrics": {"old": 1}, "data": {}, "reading": "old", "overlay": engine.latest_frame}
        with mock.patch.object(engine, "run_predict", side_effect=RuntimeError("failed")):
            engine.predict_loop()
        self.assertEqual(engine.payload()["metrics"], {})
        self.assertIsNone(engine.snapshot)
        self.assertFalse(engine.ready)

    def test_latest_frame_slot_drops_intermediate_frames(self):
        engine = ui.LiveEngine()
        engine.ready = True
        engine.latest_frame = np.ones((32, 32, 3), np.uint8)
        seen = []
        def inference(frame):
            seen.append(int(frame[0, 0, 0]))
            if len(seen) == 1:
                # Producer delivers many frames during inference; only the newest remains.
                with engine.state_lock:
                    for i in range(2, 100):
                        engine.latest_frame = np.full((32, 32, 3), i, np.uint8)
                        engine.frame_id = i
            else:
                engine.stop_event.set()
            return np.zeros((32, 32), np.uint8), frame, {}, {}, ""
        with mock.patch.object(engine, "run_predict", side_effect=inference), mock.patch.object(ui, "PREDICT_EVERY_SECONDS", 0.001):
            engine.predict_loop()
        self.assertEqual(seen, [1, 99])

    def test_local_rate_limit(self):
        ui._requests.clear()
        ui._requests.extend([time.monotonic()] * 300)
        try:
            self.assertEqual(ui.app.test_client().get("/api/status").status_code, 429)
        finally:
            ui._requests.clear()

    def test_unknown_retention_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            config = common.load_config()
            config["retention_seconds"] = 10
            path = Path(temp) / "config.json"
            common.write_json(path, config)
            with self.assertRaisesRegex(ValueError, "immediate"):
                common.load_config(path)

    @unittest.skipUnless(pipeline.TORCH_AVAILABLE, "PyTorch absent: checkpoint path gated off")
    def test_pickle_object_not_allowed_by_weights_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "object.pt"
            pipeline.torch.save({"unexpected_object": mock.sentinel.untrusted}, path)
            with self.assertRaises(pickle.UnpicklingError):
                pipeline.safe_checkpoint(path, "cpu", common.sha256(path))


if __name__ == "__main__":
    unittest.main()
