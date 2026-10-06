import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from PIL import Image

import giao_dien_ui as ui


@unittest.skipIf(ui.rarfile is None, "rarfile dependency is unavailable")
class ArchiveImportTests(unittest.TestCase):
    def setUp(self):
        encoded = io.BytesIO()
        Image.new("RGB", (32, 32), (120, 80, 60)).save(encoded, format="PNG")
        self.raw = encoded.getvalue()

    def test_rar_branch_decodes_and_saves_image(self):
        raw = self.raw
        info = type("Info", (), {
            "filename": "nested/palm.png",
            "file_size": len(raw),
            "isdir": lambda self: False,
        })()
        archive = type("Archive", (), {
            "infolist": lambda self: [info],
            "read": lambda self, item: raw,
            "close": lambda self: None,
        })()
        root = Path(tempfile.mkdtemp(prefix="palmistry-rar-test-"))
        patches = [
            mock.patch.object(ui, "PROJECT_DIR", root),
            mock.patch.object(ui, "DATASET_IMAGES_DIR", root / "dataset" / "images"),
            mock.patch.object(ui, "IMPORT_REPORT_DIR", root / "dataset" / "imports"),
            mock.patch.object(ui.rarfile, "is_rarfile", return_value=True),
            mock.patch.object(ui.rarfile, "tool_setup", return_value=mock.Mock(check=lambda: True)),
            mock.patch.object(ui.rarfile, "RarFile", return_value=archive),
        ]
        for patcher in patches:
            patcher.start()
        try:
            report = ui.import_archive_to_dataset(io.BytesIO(b"fake-rar"), "sample.rar")
            self.assertEqual(report["archive_type"], "rar")
            self.assertEqual(report["saved_count"], 1)
            self.assertTrue((root / report["saved_files"][0]).is_file())
        finally:
            for patcher in reversed(patches):
                patcher.stop()
            shutil.rmtree(root, ignore_errors=True)

    def test_rar_backend_is_detectable(self):
        try:
            ui.rarfile.tool_setup()
        except ui.rarfile.RarCannotExec:
            self.skipTest("Optional external RAR backend is not installed")
        self.assertTrue(ui.rarfile.tool_setup().check())

    def test_real_rar_uses_batch_extraction_path(self):
        raw = self.raw
        info = type("Info", (), {
            "filename": "nested/palm.png",
            "file_size": len(raw),
            "isdir": lambda self: False,
        })()

        class Archive:
            _file_parser = object()

            def infolist(self):
                return [info]

            def needs_password(self):
                return False

            def close(self):
                return None

        root = Path(tempfile.mkdtemp(prefix="palmistry-rar-batch-test-"))
        archive_path = root / "sample.rar"
        archive_path.write_bytes(b"fake-rar")

        def fake_batch(_archive_path, extract_dir, member_names):
            self.assertEqual(member_names, ["nested/palm.png"])
            source = Path(extract_dir) / "nested" / "palm.png"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(raw)
            return "fake-batch"

        patches = [
            mock.patch.object(ui, "PROJECT_DIR", root),
            mock.patch.object(ui, "DATASET_DIR", root / "dataset"),
            mock.patch.object(ui, "DATASET_IMAGES_DIR", root / "dataset" / "images"),
            mock.patch.object(ui, "IMPORT_REPORT_DIR", root / "dataset" / "imports"),
            mock.patch.object(ui, "_open_archive_path", return_value=(Archive(), "rarfile")),
            mock.patch.object(ui, "_run_rar_batch_extract", side_effect=fake_batch),
        ]
        for patcher in patches:
            patcher.start()
        try:
            report = ui._import_archive_path_to_dataset(archive_path, "sample.rar")
            self.assertEqual(report["backend"], "fake-batch")
            self.assertEqual(report["saved_count"], 1)
        finally:
            for patcher in reversed(patches):
                patcher.stop()
            shutil.rmtree(root, ignore_errors=True)
    def test_archive_status_http_success_while_running(self):
        job_id = "status-test"
        with ui._archive_upload_lock:
            ui._archive_jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "finished_epoch": 0,
                "error": "",
            }
        try:
            response = ui.app.test_client().get(
                "/api/dataset/import-archive/status",
                query_string={"job_id": job_id},
                base_url="http://127.0.0.1",
            )
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["status"], "running")
        finally:
            with ui._archive_upload_lock:
                ui._archive_jobs.pop(job_id, None)

    def test_four_gb_archive_is_within_default_upload_policy(self):
        root = Path(tempfile.mkdtemp(prefix="palmistry-large-policy-test-"))
        patcher = mock.patch.object(ui, "ARCHIVE_UPLOAD_DIR", root / "uploads")
        patcher.start()
        try:
            meta = ui._new_archive_upload("large.rar", 4 * 1024 * 1024 * 1024)
            self.assertEqual(meta["size"], 4 * 1024 * 1024 * 1024)
            expanded_16_1gb = int(16.1 * 1024 * 1024 * 1024)
            self.assertGreaterEqual(ui.MAX_ARCHIVE_UNCOMPRESSED_BYTES, expanded_16_1gb)
            part_path, meta_path = ui._archive_upload_paths(meta["upload_id"])
            part_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        finally:
            patcher.stop()
            shutil.rmtree(root, ignore_errors=True)

    def test_chunked_upload_imports_in_background(self):
        raw = self.raw
        payload = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("nested/palm.png", raw)
        data = payload.getvalue()
        root = Path(tempfile.mkdtemp(prefix="palmistry-chunk-test-"))
        patches = [
            mock.patch.object(ui, "PROJECT_DIR", root),
            mock.patch.object(ui, "DATASET_DIR", root / "dataset"),
            mock.patch.object(ui, "DATASET_IMAGES_DIR", root / "dataset" / "images"),
            mock.patch.object(ui, "IMPORT_REPORT_DIR", root / "dataset" / "imports"),
            mock.patch.object(ui, "TMP_DIR", root / ".web_live_tmp"),
            mock.patch.object(ui, "ARCHIVE_UPLOAD_DIR", root / ".web_live_tmp" / "archive_uploads"),
        ]
        for patcher in patches:
            patcher.start()
        try:
            client = ui.app.test_client()
            headers = {"X-Palm-CSRF": ui.CSRF_TOKEN}
            response = client.post(
                "/api/dataset/import-archive/init",
                json={"filename": "sample.zip", "size": len(data)},
                headers=headers,
                base_url="http://127.0.0.1",
            )
            self.assertEqual(response.status_code, 201)
            upload_id = response.get_json()["upload_id"]
            chunk_size = 113
            for offset in range(0, len(data), chunk_size):
                response = client.post(
                    "/api/dataset/import-archive/chunk",
                    data=data[offset:offset + chunk_size],
                    headers={
                        **headers,
                        "Content-Type": "application/octet-stream",
                        "X-Archive-Upload-ID": upload_id,
                        "X-Archive-Upload-Offset": str(offset),
                    },
                    base_url="http://127.0.0.1",
                )
                self.assertEqual(response.status_code, 200)
            response = client.post(
                "/api/dataset/import-archive/complete",
                json={"upload_id": upload_id},
                headers=headers,
                base_url="http://127.0.0.1",
            )
            self.assertEqual(response.status_code, 202)
            job_id = response.get_json()["job_id"]
            import time
            result = None
            for _ in range(100):
                result = client.get(
                    "/api/dataset/import-archive/status",
                    query_string={"job_id": job_id},
                    headers=headers,
                    base_url="http://127.0.0.1",
                ).get_json()
                if result["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.02)
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(result["result"]["saved_count"], 1)
        finally:
            for patcher in reversed(patches):
                patcher.stop()
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
