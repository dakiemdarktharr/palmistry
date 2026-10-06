import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

import prototype_common as common
import palmistry_strict_auto_onefile as pipeline
import tien_xu_ly as converter
import giao_dien_ui as ui
from prototype_fixture import generate_fixture


class FixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.fixture = cls.root / "fixture"
        cls.report = generate_fixture(cls.fixture)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def args(self, *extra):
        args = pipeline.make_parser().parse_args([
            "predict", "--image", str(self.fixture / "input.png"), "--out_size", "128",
            "--min_short_side", "128", "--min_image_quality", "0",
            "--out_json", str(self.root / "prediction.json"), "--out_mask", str(self.root / "mask.png"),
            "--out_overlay", str(self.root / "overlay.png"), *extra])
        args.print_reading = False
        return args

    def predict(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.predict_image(args)


class DataTests(FixtureCase):
    def test_fixture_deterministic_expected_output(self):
        second = generate_fixture(self.root / "second")
        self.assertEqual(second, self.report)
        self.assertEqual(self.report["expected"]["0"]["mask_sha256"], "b16176904cc84a5e35cc09a50da52e11c7ff4fc0da62daf57eb302926988a888")
        self.assertEqual(self.report["expected"]["0"]["rgb_sha256"], "67bd9dab6604e4663d9ee1fdcdf746a8615f89ee40e45990dd37d83bfbab7e27")

    def test_preprocessing_shape_ids_and_determinism(self):
        rgb = common.read_rgb(self.fixture / "input.png")
        crop, mask, report = pipeline.create_strict_pseudo_mask(rgb, 128)
        self.assertEqual(crop.shape, (128, 128, 3))
        common.validate_mask(mask, (128, 128))
        self.assertTrue(np.array_equal(mask, pipeline.create_strict_pseudo_mask(rgb, 128)[1]))
        self.assertEqual(report["inference_source"], "classical CV fallback")

    def test_antialiasing_regression(self):
        mask = np.zeros((64, 64), np.uint8)
        converter.draw_line_mask(mask, [(2, 2), (55, 43)], 4, 1)
        self.assertEqual(set(np.unique(mask)), {0, 4})

    def test_two_point_palm_rejected_without_mutation(self):
        mask = np.zeros((64, 64), np.uint8)
        with self.assertRaisesRegex(ValueError, "invalid_palm_polygon"):
            converter.create_palm_area(mask, [(1, 1), (40, 40)], 64, 64)
        self.assertFalse(mask.any())

    def test_collinear_palm_rejected(self):
        with self.assertRaises(ValueError):
            converter.create_palm_area(np.zeros((64, 64), np.uint8), [(1, 1), (20, 20), (40, 40)], 64, 64)

    def test_duplicate_filename_hash(self):
        self.assertNotEqual(converter.safe_stem(Path("a/hand.jpg")), converter.safe_stem(Path("b/hand.jpg")))
        self.assertEqual(converter.safe_stem(Path("a/hand.jpg")), converter.safe_stem(Path("a/hand.jpg")))

    def test_converter_collision_fails_without_overwrite(self):
        source = self.root / "convert_source"
        source.mkdir(exist_ok=True)
        Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(source / "hand.png")
        (source / "data.yaml").write_text("names: [life]\n")
        (source / "hand.txt").write_text("0 0.5 0.5 0.6 0.6 0.2 0.2 0.7 0.3 0.5 0.8 0.6 0.9\n")
        output = self.root / "converted"
        with self.assertRaisesRegex(ValueError, "training blocked"):
            converter.convert_dataset(source, output, 64, 2, 1, False)
        image = next((output / "images").glob("*.jpg"))
        original = image.read_bytes()
        with self.assertRaises(FileExistsError):
            converter.convert_dataset(source, output, 64, 2, 1, False)
        self.assertEqual(original, image.read_bytes())

    def test_class_map_compatibility(self):
        self.assertEqual(pipeline.CLASS_MAP, converter.MASK_CLASS_NAMES)
        self.assertEqual(converter.class_id_to_mask_class(0, {0: "minor_or_unknown_line"}), 5)
        self.assertEqual(pipeline.CLASS_MAP, ui.class_map_from_bundle())
        common.check_class_map(self.fixture / "class_map.json")
        bad = self.root / "legacy.json"
        bad.write_text(json.dumps({5: "fate_line"}))
        with self.assertRaises(ValueError):
            common.check_class_map(bad)

    def test_legacy_class5_requires_explicit_mode(self):
        source = self.root / "fate"
        source.mkdir(exist_ok=True)
        (source / "data.yaml").write_text("names: [fate]\n")
        with self.assertRaisesRegex(ValueError, "fate-v0"):
            converter.convert_dataset(source, self.root / "fate_out", 64, 2, 1, False)

    def test_deterministic_split_integrity(self):
        rows = common.read_manifest(self.fixture, self.fixture / "labels.csv")
        first = pipeline.split_rows(rows)
        self.assertEqual(first, pipeline.split_rows(list(reversed(rows))))
        self.assertEqual({r["split"] for r in first}, {"train", "val", "test"})
        for group in {r["group_id"] for r in first}:
            self.assertEqual(len({r["split"] for r in first if r["group_id"] == group}), 1)

    def test_group_leakage_subject_source_and_near_duplicate(self):
        rows = common.read_manifest(self.fixture, self.fixture / "labels.csv")
        train = next(r for r in rows if r["split"] == "train")
        test = next(r for r in rows if r["split"] == "test")
        for key in ("subject_id", "source_group", "sha256", "phash"):
            with self.subTest(key=key):
                altered = copy.deepcopy(rows)
                for row in altered:
                    if row["image_path"] == test["image_path"]:
                        row[key] = train[key]
                with self.assertRaises(ValueError):
                    common.group_rows(altered, assign=False)

    def test_missing_subjects_mark_fallback(self):
        rows = common.read_manifest(self.fixture, self.fixture / "labels.csv")
        for row in rows:
            row.pop("subject_id")
        self.assertTrue(all("not subject-independent" in r["split_basis"] for r in common.group_rows(rows)))

    def test_empty_dataset(self):
        with self.assertRaisesRegex(ValueError, "Empty"):
            pipeline.split_rows([])
        with self.assertRaisesRegex(ValueError, "Empty"):
            pipeline.build_pseudomask_dataset([], self.root / "empty")

    def test_malformed_masks(self):
        for mask in (None, np.zeros((3, 3, 3), np.uint8), np.full((3, 3), max(common.CLASS_MAP)+1, np.uint8), np.zeros((3, 3), float)):
            with self.subTest(mask=str(type(mask))), self.assertRaises(ValueError):
                common.validate_mask(mask)

    def test_invalid_size_and_parameters(self):
        for size in (-16, 0, 31, 33, 4096):
            with self.assertRaises(ValueError):
                common.validate_size(size)
        for flag, value in (("--out_size", "33"), ("--read_threshold", "nan"), ("--min_short_side", "0")):
            with self.assertRaises(ValueError):
                pipeline.validate_parameters(self.args(flag, value))

    def test_confusion_metrics_known_answer(self):
        target = np.array([[0, 1], [1, 2]], np.uint8)
        pred = np.array([[0, 1], [2, 2]], np.uint8)
        result = common.metrics_from_confusion(common.confusion_matrix(pred, target))
        self.assertEqual(result["per_class"]["palm_area"]["iou"], 0.5)
        self.assertAlmostEqual(result["per_class"]["palm_area"]["dice"], 2/3)
        self.assertIsNone(result["per_class"]["heart_line"]["iou"])

    def test_remote_download_disabled(self):
        with mock.patch.object(pipeline.requests, "Session") as session:
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                pipeline.download_commons_images(self.root)
            session.assert_not_called()

    def test_bounded_image_decode(self):
        bad = self.root / "oversized.png"
        with bad.open("wb") as f:
            f.truncate(common.MAX_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "20 MiB"):
            common.read_rgb(bad)


class PredictionTests(FixtureCase):
    def test_bad_image_writes_json(self):
        args = self.args("--allow-classical-fallback", "--image", str(self.root / "missing.png"))
        out = self.predict(args)
        self.assertEqual(out["main_reason"], "BAD_READ")
        self.assertEqual(json.loads(Path(args.out_json).read_text(encoding="utf-8"))["request_id"], out["request_id"])
        self.assertFalse(out["model_used"])

    def test_missing_checkpoint_does_not_fallback(self):
        args = self.args("--checkpoint", str(self.root / "missing.pt"), "--allow-classical-fallback")
        out = self.predict(args)
        self.assertEqual(out["status"], "error")
        self.assertFalse(out["model_used"])
        self.assertFalse(Path(args.out_mask).exists())

    def test_explicit_fallback_required(self):
        out = self.predict(self.args())
        self.assertEqual(out["status"], "error")
        self.assertIn("--allow-classical-fallback", out["error"])

    def test_classical_contract_and_stale_cleanup(self):
        args = self.args("--allow-classical-fallback")
        out = self.predict(args)
        self.assertTrue(out["segmentation_available"])
        self.assertEqual(out["inference_source"], "classical CV fallback")
        self.assertEqual(out["success_marker"], out["request_id"])
        common.validate_mask(cv2.imread(args.out_mask, 0))
        args.image = str(self.root / "missing.png")
        out = self.predict(args)
        self.assertIsNone(out["success_marker"])
        self.assertFalse(Path(args.out_mask).exists())
        self.assertFalse(Path(args.out_overlay).exists())

    def test_invalid_predict_parameter_still_writes_json(self):
        args = self.args("--allow-classical-fallback", "--out_size", "33")
        self.assertEqual(self.predict(args)["status"], "error")
        self.assertTrue(Path(args.out_json).exists())

    def test_input_cannot_be_overwritten(self):
        args = self.args("--allow-classical-fallback")
        before = Path(args.image).read_bytes()
        args.out_mask = args.image
        self.assertEqual(self.predict(args)["status"], "error")
        self.assertEqual(before, Path(args.image).read_bytes())


class UITests(unittest.TestCase):
    def setUp(self):
        self.engine = ui.LiveEngine()
        ui._requests.clear()

    def sample(self):
        return {"frame_id": 7, "mask": np.ones((32, 32), np.uint8),
                "overlay": np.full((32, 32, 3), 123, np.uint8),
                "metrics": {"life_line": {"present": True, "length_norm": 0.5}},
                "reading": "geometry", "data": {"request_id": "seven"}}

    def test_live_overlay_and_manual_lock_snapshot(self):
        self.engine.snapshot = self.sample()
        self.engine.latest_frame = np.zeros((32, 32, 3), np.uint8)
        self.assertTrue(np.all(self.engine.get_display_frame() == 123))
        self.assertTrue(self.engine.lock_now())
        self.engine.snapshot["metrics"]["life_line"]["length_norm"] = 99
        self.engine.snapshot["overlay"][:] = 0
        self.assertEqual(self.engine.payload()["metrics"]["life_line"]["length_norm"], 0.5)
        self.assertTrue(np.all(self.engine.get_display_frame() == 123))
        self.assertEqual(self.engine.locked_snapshot["frame_id"], 7)
        self.assertTrue(np.all(self.engine.locked_snapshot["mask"] == 1))

    def test_lock_without_prediction(self):
        self.assertFalse(self.engine.lock_now())
        self.assertFalse(self.engine.locked)

    def test_payload_copy(self):
        self.engine.snapshot = self.sample()
        payload = self.engine.payload()
        payload["metrics"].clear()
        self.assertTrue(self.engine.payload()["metrics"])

    def test_missing_checkpoint_gate(self):
        self.engine.ready = False
        with mock.patch.object(self.engine, "run_predict") as run:
            self.engine.predict_loop()
            run.assert_not_called()

    def test_failed_inference_never_reuses_and_cleans(self):
        self.engine.checkpoint = Path("explicit.pt")
        roots = []
        def fail(cmd, **kwargs):
            root = Path(cmd[cmd.index("--out_json")+1]).parent
            roots.append(root)
            (root / "result.json").write_text('{"success_marker": "old"}')
            return subprocess.CompletedProcess(cmd, 1, "", "failure")
        with mock.patch.object(ui.subprocess, "run", side_effect=fail):
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    self.engine.run_predict(np.zeros((32, 32, 3), np.uint8))
        self.assertNotEqual(roots[0], roots[1])
        self.assertTrue(all(not root.exists() for root in roots))

    def test_timeout_cleanup(self):
        self.engine.checkpoint = Path("explicit.pt")
        roots = []
        def timeout(cmd, **kwargs):
            roots.append(Path(cmd[cmd.index("--out_json")+1]).parent)
            raise subprocess.TimeoutExpired(cmd, 0.01)
        with mock.patch.object(ui.subprocess, "run", side_effect=timeout), self.assertRaises(subprocess.TimeoutExpired):
            self.engine.run_predict(np.zeros((32, 32, 3), np.uint8))
        self.assertFalse(roots[0].exists())

    def test_success_cleanup_and_request_identity(self):
        self.engine.checkpoint = Path("explicit.pt")
        roots = []
        def succeed(cmd, **kwargs):
            root = Path(cmd[cmd.index("--out_json")+1]).parent
            roots.append(root)
            identity = cmd[cmd.index("--request-id")+1]
            (root / "result.json").write_text(json.dumps({"request_id": identity, "success_marker": identity, "segmentation_available": True}))
            cv2.imwrite(str(root / "mask.png"), np.ones((32, 32), np.uint8))
            cv2.imwrite(str(root / "overlay.png"), np.full((32, 32, 3), 50, np.uint8))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        with mock.patch.object(ui.subprocess, "run", side_effect=succeed):
            mask, overlay, data, _, _ = self.engine.run_predict(np.zeros((64, 64, 3), np.uint8))
        self.assertFalse(roots[0].exists())
        self.assertEqual(mask.shape, overlay.shape[:2])
        self.assertEqual(mask.shape, (32, 32))  # Crop coordinates are not stretched to raw frame.

    def test_stale_marker_rejected_even_on_exit_zero(self):
        self.engine.checkpoint = Path("explicit.pt")
        def stale(cmd, **kwargs):
            Path(cmd[cmd.index("--out_json")+1]).write_text('{"request_id":"old","success_marker":"old","segmentation_available":true}')
            return subprocess.CompletedProcess(cmd, 0, "", "")
        with mock.patch.object(ui.subprocess, "run", side_effect=stale), self.assertRaisesRegex(RuntimeError, "marker"):
            self.engine.run_predict(np.zeros((32, 32, 3), np.uint8))

    def test_reset_discards_inflight_result(self):
        self.engine.ready = True
        self.engine.latest_frame = np.zeros((32, 32, 3), np.uint8)
        def inference(frame):
            self.engine.reset()
            self.engine.stop_event.set()
            sample = self.sample()
            return sample["mask"], sample["overlay"], sample["data"], sample["metrics"], sample["reading"]
        with mock.patch.object(self.engine, "run_predict", side_effect=inference):
            self.engine.predict_loop()
        self.assertIsNone(self.engine.snapshot)

    def test_thread_shutdown_with_fake_camera(self):
        capture = mock.Mock()
        capture.read.return_value = (True, np.zeros((32, 32, 3), np.uint8))
        self.engine.cap = capture
        self.engine.ready = False
        self.engine.start()
        time.sleep(0.08)
        self.assertTrue(self.engine.shutdown())
        capture.release.assert_called()
        switching = ui.LiveEngine()
        switching.cap = mock.Mock()
        def switched_read():
            switching.set_camera(1)
            switching.stop_event.set()
            return True, np.ones((32, 32, 3), np.uint8)
        switching.cap.read.side_effect = switched_read
        switching.camera_loop()
        self.assertIsNone(switching.latest_frame, "Old-camera frame must not publish after a camera switch")

    def test_zip_traversal_archive_is_never_extracted(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "malicious.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../escape.txt", "bad")
            with mock.patch.object(ui, "find_bundle_zip", return_value=archive), self.assertRaisesRegex(ValueError, "disabled"):
                ui.extract_bundle()
            self.assertFalse((Path(temp) / "escape.txt").exists())

    def test_flask_loopback_and_csrf(self):
        self.assertEqual(ui.HOST, "127.0.0.1")
        client = ui.app.test_client()
        self.assertEqual(client.get("/api/status").status_code, 200)
        for route in ("/api/status", "/video_feed", "/api/reset", "/api/lock", "/api/camera/0"):
            self.assertEqual(client.open(route, method="POST" if route not in ("/api/status", "/video_feed") else "GET", environ_overrides={"REMOTE_ADDR": "10.0.0.5"}).status_code, 403)
        self.assertEqual(client.get("/api/status", headers={"Host": "evil.test"}).status_code, 403)
        self.assertEqual(client.post("/api/reset").status_code, 403)
        self.assertEqual(client.post("/api/reset", headers={"X-Palm-CSRF": ui.CSRF_TOKEN}).status_code, 200)
        self.assertEqual(client.post("/api/reset", headers={"X-Palm-CSRF": ui.CSRF_TOKEN, "Origin": "https://evil.test"}).status_code, 403)

    def test_secure_headers_and_metadata_rendering(self):
        response = ui.app.test_client().get("/")
        self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertNotIn(b".innerHTML", response.data)
        self.assertNotIn(b"onclick=", response.data)

    def test_network_mode_disabled(self):
        with mock.patch.object(ui, "USE_NGROK", True), self.assertRaises(RuntimeError):
            ui.start_ngrok()


@unittest.skipUnless(pipeline.TORCH_AVAILABLE, "PyTorch missing; core CV fallback covered separately")
class ModelTests(FixtureCase):
    def setUp(self):
        pipeline.torch.set_num_threads(1)

    def test_model_output_shape(self):
        model = pipeline.SmallUNet(base=24).eval()
        with pipeline.torch.no_grad():
            output = model(pipeline.torch.zeros((1, 3, 32, 32)))
        self.assertEqual(tuple(output.shape), (1, 6, 32, 32))

    def test_non_augmented_validation(self):
        ds = pipeline.PalmMaskDataset(self.fixture, self.fixture / "labels.csv", "val", 32, False)
        with mock.patch.object(pipeline, "augment_image_mask", side_effect=AssertionError("augmented validation")):
            first, second = ds[0], ds[0]
        self.assertTrue(pipeline.torch.equal(first[0], second[0]))
        with self.assertRaises(ValueError):
            pipeline.PalmMaskDataset(self.fixture, self.fixture / "labels.csv", "val", 32, True)

    def test_training_partial_accumulation_steps_and_safe_reload(self):
        torch = pipeline.torch
        args = pipeline.make_parser().parse_args(["train", "--data_root", str(self.fixture), "--labels_csv", str(self.fixture / "labels.csv"),
            "--out_dir", str(self.root / "smoke"), "--input_size", "32", "--epochs", "1", "--batch_size", "1", "--grad_accum", "2", "--model_size", "tiny"])
        steps = []
        class CountingAdam(torch.optim.AdamW):
            def step(self, *a, **kw):
                steps.append(1)
                return super().step(*a, **kw)
        batches = [(torch.zeros(1, 3, 32, 32), torch.ones(1, 32, 32, dtype=torch.long)) for _ in range(3)]
        with mock.patch.object(pipeline, "DataLoader", side_effect=[batches, batches[:1]]), mock.patch.object(torch.optim, "AdamW", CountingAdam), contextlib.redirect_stdout(io.StringIO()):
            pipeline.train_model(args)
        self.assertEqual(len(steps), 2, "3 batches with accumulation=2 must step twice")
        checkpoint = self.root / "smoke/checkpoints/best.pt"
        loaded, size, _ = pipeline.load_model_for_predict(checkpoint, torch.device("cpu"), common.sha256(checkpoint))
        self.assertEqual(size, 32)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            pipeline.load_model_for_predict(checkpoint, torch.device("cpu"), "0"*64)
        with self.assertRaisesRegex(ValueError, "explicit"):
            pipeline.load_model_for_predict(checkpoint, torch.device("cpu"), "")

    def test_checkpoint_schema_rejected(self):
        path = self.root / "invalid.pt"
        pipeline.torch.save({"class_map": {5: "fate_line"}}, path)
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            pipeline.safe_checkpoint(path, "cpu", common.sha256(path))


class CommandTests(unittest.TestCase):
    def test_syntax_and_imports(self):
        for path in Path(__file__).resolve().parents[1].glob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_cli_help_and_doctor(self):
        for command in (["--help"], ["doctor"]):
            result = subprocess.run([sys.executable, "palmistry_strict_auto_onefile.py", *command], capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Torch:", result.stdout)


if __name__ == "__main__":
    unittest.main()
