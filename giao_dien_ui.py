import os
import sys
import json
import time
import math
import shutil
import zipfile
try:
    import rarfile
except Exception:
    rarfile = None
import threading
import subprocess
import tempfile
import uuid
import copy
import secrets
import re
import csv
import html
from urllib.parse import quote
from functools import wraps
from prototype_common import CLASS_MAP, validate_mask, load_config, sha256, decode_archive_image, write_json
from pathlib import Path
from collections import deque

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, Response, jsonify, send_from_directory, request, abort
from werkzeug.exceptions import RequestEntityTooLarge


PROJECT_DIR = Path(__file__).resolve().parent

PORT = 8501
HOST = "127.0.0.1"
CONFIG = load_config(os.environ.get("PALM_CONFIG"))

USE_NGROK = False  # Public networking intentionally unavailable.

WEBCAM_INDEX = 0
USB_CAMERA_INDEX = 1
CAMERA_INDEX = WEBCAM_INDEX

DISPLAY_FPS = 25
JPEG_QUALITY = 80

PREDICT_EVERY_SECONDS = float(CONFIG["prediction_interval_seconds"])
PREDICT_TIMEOUT_SECONDS = float(CONFIG["prediction_timeout_seconds"])
READ_THRESHOLD = float(CONFIG["read_threshold"])

LOCK_ON_FIRST_GOOD_OVERLAY = True
GOOD_OVERLAY_MIN_LINES = 2

STABLE_SECONDS = 3.0
STABILITY_TOLERANCE = 0.10

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FLIP_CAMERA = True

CHECKPOINT_PATH = os.environ.get("PALM_CHECKPOINT", CONFIG["checkpoint"])
CHECKPOINT_SHA256 = os.environ.get("PALM_CHECKPOINT_SHA256", CONFIG["checkpoint_sha256"])
CODE_PATH = PROJECT_DIR / "palmistry_strict_auto_onefile.py"

BUNDLE_ZIP_EDA = PROJECT_DIR / "palmistry_colab_bundle_eda.zip"
BUNDLE_ZIP_BASIC = PROJECT_DIR / "palmistry_colab_bundle.zip"

_dataset_dir_env = os.environ.get("PALM_DATASET_DIR", "").strip()
DATASET_DIR = Path(_dataset_dir_env or str(PROJECT_DIR / "dataset")).expanduser()
if not DATASET_DIR.is_absolute():
    DATASET_DIR = PROJECT_DIR / DATASET_DIR
DATASET_IMAGES_DIR = DATASET_DIR / "images"
IMPORT_REPORT_DIR = DATASET_DIR / "imports"


def _env_int(name, default):
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(1, value)


# Large archives are uploaded in chunks from the browser. These limits are
# configurable so a machine with more or less disk can choose its own policy.
MAX_ARCHIVE_UPLOAD_BYTES = _env_int("PALM_MAX_ARCHIVE_BYTES", 8 * 1024 * 1024 * 1024)
MAX_ZIP_UPLOAD_BYTES = MAX_ARCHIVE_UPLOAD_BYTES
MAX_ARCHIVE_UNCOMPRESSED_BYTES = _env_int("PALM_MAX_ARCHIVE_EXPANDED_BYTES", 32 * 1024 * 1024 * 1024)
MAX_ZIP_UNCOMPRESSED_BYTES = MAX_ARCHIVE_UNCOMPRESSED_BYTES
MAX_ARCHIVE_MEMBER_BYTES = _env_int("PALM_MAX_ARCHIVE_MEMBER_BYTES", 100 * 1024 * 1024)
MAX_ZIP_MEMBER_BYTES = MAX_ARCHIVE_MEMBER_BYTES
MAX_ARCHIVE_MEMBERS = _env_int("PALM_MAX_ARCHIVE_MEMBERS", 20000)
MAX_ZIP_MEMBERS = MAX_ARCHIVE_MEMBERS
ARCHIVE_UPLOAD_CHUNK_BYTES = _env_int("PALM_ARCHIVE_UPLOAD_CHUNK_BYTES", 16 * 1024 * 1024)
ARCHIVE_UPLOAD_TTL_SECONDS = _env_int("PALM_ARCHIVE_UPLOAD_TTL_SECONDS", 6 * 60 * 60)
MAX_ARCHIVE_IMAGE_PIXELS = _env_int("PALM_MAX_ARCHIVE_IMAGE_PIXELS", 50_000_000)
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

BUNDLE_DIR = PROJECT_DIR / ".web_bundle"
TMP_DIR = PROJECT_DIR / ".web_live_tmp"
_archive_upload_dir_env = os.environ.get("PALM_ARCHIVE_UPLOAD_DIR", "").strip()
ARCHIVE_UPLOAD_DIR = Path(
    _archive_upload_dir_env or str(TMP_DIR / "archive_uploads")
).expanduser()
if not ARCHIVE_UPLOAD_DIR.is_absolute():
    ARCHIVE_UPLOAD_DIR = PROJECT_DIR / ARCHIVE_UPLOAD_DIR
STATIC_DIR = PROJECT_DIR / ".web_static"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
PIPELINE_MAX_REVIEW_COUNT = 3000
PIPELINE_JOB_LOG_LIMIT = 120



LINE_CLASSES = {
    2: {"name": "Sinh đạo", "key": "life_line", "color": (0, 0, 255)},
    3: {"name": "Trí đạo", "key": "head_line", "color": (0, 255, 0)},
    4: {"name": "Tâm đạo", "key": "heart_line", "color": (255, 0, 0)},
    6: {"name": "Simian", "key": "simian_line", "color": (255, 230, 0)},
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_ARCHIVE_UPLOAD_BYTES


@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(_error):
    return jsonify({
        "ok": False,
        "error": (
            f"Request quá lớn. Hãy dùng upload theo phần; giới hạn archive là "
            f"{_format_bytes(MAX_ARCHIVE_UPLOAD_BYTES)}."
        ),
    }), 413


CSRF_TOKEN = secrets.token_urlsafe(32)
CSP_NONCE = secrets.token_urlsafe(24)
from palm_photo_ui import register_photo_ui
register_photo_ui(app, project_dir=PROJECT_DIR, checkpoint=CHECKPOINT_PATH,
                  checkpoint_sha256=CHECKPOINT_SHA256, csrf=CSRF_TOKEN, nonce=CSP_NONCE)
from palm_keypoints.web import register_keypoint_ui
register_keypoint_ui(app, PROJECT_DIR, CSRF_TOKEN, CSP_NONCE,
                     legacy_busy=lambda: _pipeline_job.get('status') in ('running','starting'))
_requests = deque()
_request_lock = threading.Lock()
_pipeline_lock = threading.RLock()
_pipeline_job = {
    "job_id": "",
    "kind": "",
    "status": "idle",
    "run_name": "",
    "bootstrap_name": "",
    "started_at": "",
    "finished_at": "",
    "returncode": None,
    "log": [],
    "error": "",
    "result": {},
}
_archive_upload_lock = threading.RLock()
_archive_jobs = {}
_archive_import_execution_lock = threading.Lock()


def _safe_pipeline_name(value, fallback):
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("._-")[:64]
    return value or fallback


def _artifact_run_dir(run_name):
    safe_name = _safe_pipeline_name(run_name, "")
    if not safe_name or safe_name != str(run_name):
        raise ValueError("Tên run chỉ được dùng chữ, số, gạch dưới và gạch ngang.")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = (ARTIFACTS_DIR / safe_name).resolve()
    if not path.is_relative_to(ARTIFACTS_DIR.resolve()):
        raise ValueError("Run directory không hợp lệ.")
    return path


def _persist_pipeline_state():
    path = ARTIFACTS_DIR / ".ui_jobs" / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _pipeline_lock:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_pipeline_job, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def _restore_pipeline_state():
    path = ARTIFACTS_DIR / ".ui_jobs" / "state.json"
    try:
        restored = json.loads(path.read_text(encoding="utf-8"))
        if restored.get("status") == "running":
            restored.update(status="failed", error="App đã khởi động lại giữa chừng. Bấm chuẩn bị queue để chạy lại; run cũ được giữ.")
        with _pipeline_lock:
            _pipeline_job.update(restored)
    except (OSError, ValueError, TypeError):
        pass


def _pipeline_snapshot():
    with _pipeline_lock:
        snapshot = copy.deepcopy(_pipeline_job)
    snapshot["running"] = snapshot["status"] == "running"
    if snapshot.get("run_name"):
        snapshot["review_url"] = f"/pipeline_review/{quote(snapshot['run_name'])}"
    if snapshot.get("bootstrap_name"):
        snapshot["bootstrap_url"] = f"/pipeline_artifacts/{quote(snapshot['bootstrap_name'])}/bootstrap_summary.json"
    return snapshot


def _append_pipeline_log(job_id, line):
    line = str(line).rstrip()
    if not line:
        return
    with _pipeline_lock:
        if _pipeline_job.get("job_id") != job_id:
            return
        _pipeline_job.setdefault("log", []).append(line[-2000:])
        _pipeline_job["log"] = _pipeline_job["log"][-PIPELINE_JOB_LOG_LIMIT:]


def _run_pipeline_command(job_id, command, run_name, bootstrap_name, summary_path):
    # Keep the target output directory empty until the child script starts;
    # both pipeline.py and bootstrap_review.py reject pre-populated outputs.
    log_path = ARTIFACTS_DIR / ".ui_jobs" / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    returncode = -1
    try:
        env = os.environ.copy()
        env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        _append_pipeline_log(job_id, "$ " + " ".join(str(part) for part in command))
        with subprocess.Popen(
            command,
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) as process:
            with log_path.open("a", encoding="utf-8") as log:
                for line in process.stdout:
                    _append_pipeline_log(job_id, line)
                    log.write(line)
            returncode = process.wait()
        # Persist the complete UI log only after the child has finished, so it
        # cannot make a freshly-created output directory look non-empty.
        if summary_path and log_path.is_file():
            target_log = Path(summary_path).parent / "ui_job.log"
            target_log.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(log_path, target_log)
        result = load_json(summary_path) if summary_path else {}
        with _pipeline_lock:
            if _pipeline_job.get("job_id") == job_id:
                _pipeline_job["returncode"] = returncode
                _pipeline_job["result"] = result
                _pipeline_job["status"] = "completed" if returncode == 0 else "failed"
                _pipeline_job["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                if returncode != 0:
                    _pipeline_job["error"] = (result.get("error") if isinstance(result, dict) else "") or next((line for line in reversed(_pipeline_job["log"]) if line.strip()), f"Process exited {returncode}")
    except Exception as exc:
        _append_pipeline_log(job_id, repr(exc))
        with _pipeline_lock:
            if _pipeline_job.get("job_id") == job_id:
                _pipeline_job["returncode"] = returncode
                _pipeline_job["status"] = "failed"
                _pipeline_job["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                _pipeline_job["error"] = repr(exc)

    finally:
        _persist_pipeline_state()


def _start_pipeline_command(kind, command, run_name, bootstrap_name, summary_path):
    with _pipeline_lock:
        if _pipeline_job.get("status") == "running":
            raise RuntimeError("Một phase pipeline khác đang chạy.")
        job_id = uuid.uuid4().hex
        _pipeline_job.clear()
        _pipeline_job.update({
            "job_id": job_id,
            "kind": kind,
            "status": "running",
            "run_name": run_name,
            "bootstrap_name": bootstrap_name,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "finished_at": "",
            "returncode": None,
            "log": [],
            "error": "",
            "result": {},
        })
    thread = threading.Thread(
        target=_run_pipeline_command,
        args=(job_id, command, run_name, bootstrap_name, summary_path),
        daemon=True,
        name=f"palmistry-pipeline-{kind}",
    )
    thread.start()
    _persist_pipeline_state()
    return _pipeline_snapshot()


@app.before_request
def local_only():
    if request.remote_addr not in ("127.0.0.1", "::1") or request.host.split(":")[0] not in ("localhost", "127.0.0.1"):
        abort(403)
    if request.headers.get("Sec-Fetch-Site") == "cross-site":
        abort(403)
    # A large archive may legitimately require hundreds of sequential chunks.
    # Chunk requests are authenticated and offset-checked separately, so they
    # must not consume the normal UI request-rate budget.
    if request.path != "/api/dataset/import-archive/chunk" and not request.path.startswith(("/pipeline_artifacts/", "/keypoints/assets/")):
        with _request_lock:
            now = time.monotonic()
            while _requests and now - _requests[0] > 60:
                _requests.popleft()
            if len(_requests) >= 300:
                abort(429)
            _requests.append(now)
    if request.method == "POST":
        if not secrets.compare_digest(request.headers.get("X-Palm-CSRF", ""), CSRF_TOKEN):
            abort(403)
        origin = request.headers.get("Origin")
        if origin and origin != request.host_url.rstrip("/"):
            abort(403)


@app.errorhandler(OSError)
def handle_storage_error(error):
    app.logger.exception("Storage operation failed")
    return jsonify({"ok": False, "error": "Không đọc/ghi được dữ liệu. Kiểm tra dung lượng và quyền thư mục; ảnh gốc vẫn được giữ. " + str(error)}), 507


@app.after_request
def secure_headers(response):
    response.headers["Content-Security-Policy"] = f"default-src 'self'; script-src 'nonce-{CSP_NONCE}'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


def load_json(path):
    path = Path(path)

    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def find_checkpoint():
    # No recursive discovery of pickle-capable files.
    if not CHECKPOINT_PATH:
        return None
    path = Path(CHECKPOINT_PATH)
    path = path if path.is_absolute() else PROJECT_DIR / path
    return path if path.is_file() else None

def find_bundle_zip():
    # Legacy diagnostic only. Archives are never imported implicitly at startup.
    if BUNDLE_ZIP_EDA.exists():
        return BUNDLE_ZIP_EDA
    if BUNDLE_ZIP_BASIC.exists():
        return BUNDLE_ZIP_BASIC
    zips = list(PROJECT_DIR.glob("*bundle*.zip"))
    return zips[0] if zips else None


def extract_bundle():
    # Explicit upload through /api/dataset/import-archive is the only archive import path.
    if find_bundle_zip() is not None:
        raise ValueError("Automatic archive extraction disabled; use the explicit upload button.")
    return None


def dataset_image_count():
    if not DATASET_IMAGES_DIR.exists():
        return 0
    return sum(
        1 for item in DATASET_IMAGES_DIR.rglob("*")
        if item.is_file() and item.suffix.lower() in ALLOWED_IMAGE_EXTS
    )


def _project_relative(path):
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def _safe_dataset_stem(name):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name or "upload").stem)
    stem = stem.strip("._-")[:80]
    return stem or "upload"


def _safe_dataset_part(name):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "group"))
    value = value.strip("._-")[:60]
    return value or "group"


def _safe_zip_member_name(name):
    normalized = str(name or "").replace("\\", "/")
    if not normalized or normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        return None
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    return "/".join(parts)


def _archive_info_is_dir(info):
    checker = getattr(info, "isdir", None)
    if callable(checker):
        return bool(checker())
    checker = getattr(info, "is_dir", False)
    return bool(checker() if callable(checker) else checker)


def _format_bytes(value):
    value = float(max(0, value))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def _archive_upload_id(value):
    value = str(value or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,80}", value):
        raise ValueError("Upload ID không hợp lệ.")
    return value


def _archive_upload_paths(upload_id):
    upload_id = _archive_upload_id(upload_id)
    root = ARCHIVE_UPLOAD_DIR.resolve()
    root.mkdir(parents=True, exist_ok=True)
    part = (root / f"{upload_id}.part").resolve()
    meta = (root / f"{upload_id}.json").resolve()
    if not part.is_relative_to(root) or not meta.is_relative_to(root):
        raise ValueError("Đường dẫn upload không hợp lệ.")
    return part, meta


def _purge_stale_archive_uploads():
    now = time.time()
    if not ARCHIVE_UPLOAD_DIR.exists():
        return
    for meta_path in ARCHIVE_UPLOAD_DIR.glob("*.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            created = float(meta.get("created_at", 0))
            upload_id = meta_path.stem
            if now - created > ARCHIVE_UPLOAD_TTL_SECONDS:
                part_path, _ = _archive_upload_paths(upload_id)
                part_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                for completed_path in ARCHIVE_UPLOAD_DIR.glob(f"{upload_id}.*"):
                    try:
                        completed_path.unlink(missing_ok=True)
                    except OSError:
                        pass
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            try:
                meta_path.unlink(missing_ok=True)
            except OSError:
                pass


def _read_member_limited(archive, info):
    """Read one member without allowing a misleading header to cause RAM growth."""
    opener = getattr(archive, "open", None)
    if not callable(opener):
        try:
            raw = archive.read(info)
        except Exception:
            return None, "unreadable_or_encrypted_member"
        if len(raw) > MAX_ARCHIVE_MEMBER_BYTES:
            return None, "member_size_limit"
        return raw, None
    try:
        source = opener(info)
    except Exception:
        return None, "unreadable_or_encrypted_member"
    chunks = []
    total = 0
    try:
        while True:
            chunk = source.read(min(1024 * 1024, MAX_ARCHIVE_MEMBER_BYTES + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_ARCHIVE_MEMBER_BYTES:
                return None, "member_size_limit"
            chunks.append(chunk)
    except Exception:
        return None, "unreadable_or_encrypted_member"
    finally:
        try:
            source.close()
        except Exception:
            pass
    return b"".join(chunks), None


def _archive_progress(callback, progress, message):
    if callback is None:
        return
    try:
        callback(max(0.0, min(1.0, float(progress))), str(message))
    except Exception:
        # Progress reporting must never abort an import.
        pass


def _rar_batch_backend():
    """Return the configured RAR executable and its batch extraction mode."""
    if rarfile is None:
        raise ValueError("Chưa có hỗ trợ RAR.")
    setup = rarfile.tool_setup()
    setup_data = getattr(setup, "setup", {}) or {}
    open_cmd = tuple(setup_data.get("open_cmd", ()) or ())
    tool_var = str(open_cmd[0]) if open_cmd else ""
    configured = getattr(rarfile, tool_var, None)
    if not configured:
        raise ValueError("Không xác định được backend giải nén RAR.")
    executable = shutil.which(str(configured)) or str(configured)
    if tool_var == "BSDTAR_TOOL":
        return executable, "bsdtar", True
    if tool_var == "SEVENZIP_TOOL":
        return executable, "7z", True
    if tool_var == "UNRAR_TOOL":
        return executable, "unrar", False
    raise ValueError("Backend RAR hiện tại không hỗ trợ giải nén hàng loạt.")


def _run_rar_batch_extract(archive_path, extract_dir, member_names):
    """Extract selected RAR members in one backend process.

    rarfile.open() starts a backend process for every member. That is unusable
    for archives containing thousands of images, so the real archive path uses
    one batch extraction into an isolated temporary directory.
    """
    if not member_names:
        return "batch (không có ảnh hợp lệ)"
    executable, backend_name, supports_list = _rar_batch_backend()
    extract_dir = Path(extract_dir).resolve()
    list_path = None
    try:
        command = [executable]
        if backend_name == "bsdtar":
            command.extend(["-x", "-f", str(Path(archive_path).resolve()), "-C", str(extract_dir)])
            list_path = extract_dir.parent / f".palmistry-rar-members-{uuid.uuid4().hex}.txt"
            list_path.write_text("\n".join(member_names), encoding="utf-8")
            command.extend(["-T", str(list_path)])
        elif backend_name == "7z":
            command.extend(["x", "-y", f"-o{extract_dir}", str(Path(archive_path).resolve())])
            list_path = extract_dir.parent / f".palmistry-rar-members-{uuid.uuid4().hex}.txt"
            list_path.write_text("\n".join(member_names), encoding="utf-8")
            command.append(f"-i@{list_path}")
        else:
            command.extend(["x", "-idq", "-y", str(Path(archive_path).resolve()), str(extract_dir)])
        completed = subprocess.run(
            command,
            cwd=str(extract_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=1800,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Backend RAR quá 30 phút. Thử xuất lại thành ZIP hoặc chia archive thành các phần nhỏ độc lập.") from exc
    except OSError as exc:
        raise ValueError("Không chạy được backend RAR; hãy cài 7-Zip, unrar hoặc bsdtar.") from exc
    finally:
        if list_path is not None:
            try:
                list_path.unlink(missing_ok=True)
            except OSError:
                pass
    if completed.returncode != 0:
        detail = (completed.stdout or "").strip().splitlines()
        detail_text = detail[-1][:300] if detail else "không có chi tiết từ backend"
        lowered = detail_text.lower()
        if "password" in lowered or "encrypted" in lowered:
            raise ValueError("RAR có mật khẩu hoặc entry được mã hóa; hãy giải mã archive trước khi upload.")
        raise ValueError(f"Backend {backend_name} không thể giải nén RAR: {detail_text}")
    return f"{backend_name} (batch)"


def _read_path_limited(path):
    """Read an extracted member with the same bounded-memory policy as archive.open."""
    chunks = []
    total = 0
    try:
        with Path(path).open("rb") as source:
            while True:
                chunk = source.read(min(1024 * 1024, MAX_ARCHIVE_MEMBER_BYTES + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_ARCHIVE_MEMBER_BYTES:
                    return None, "member_size_limit"
                chunks.append(chunk)
    except OSError:
        return None, "unreadable_member"
    return b"".join(chunks), None


def _import_rar_batch(archive_path, infos, import_id, progress_callback=None):
    """Fast, bounded RAR import for real archives with many members."""
    for info in infos:
        name = str(getattr(info, "filename", ""))
        normalized = _safe_zip_member_name(name)
        is_link = getattr(info, "is_symlink", lambda: False)()
        if (normalized is None or is_link or getattr(info, "file_redir", None)
                or any(ch in name for ch in ("\x00", "\n", "\r", ":", "*", "?", "[", "]"))
                or any(part.endswith((" ", ".")) for part in normalized.split("/"))):
            raise ValueError("RAR chứa đường dẫn/link không an toàn. Xuất lại chỉ các ảnh thường vào ZIP rồi nhập lại.")
    extract_dir = Path(tempfile.mkdtemp(prefix="palmistry-rar-extract-", dir=str(DATASET_IMAGES_DIR.parent)))
    extract_root = extract_dir.resolve()
    saved = []
    saved_destinations = []
    skipped = []
    expanded_seen = 0
    eligible = []
    seen_names = set()
    try:
        for info in infos:
            raw_name = str(getattr(info, "filename", ""))
            member_name = _safe_zip_member_name(raw_name)
            if _archive_info_is_dir(info) or member_name is None:
                skipped.append({"name": raw_name, "reason": "unsafe_path_or_directory"})
                continue
            suffix = Path(member_name).suffix.lower()
            if suffix not in ALLOWED_IMAGE_EXTS:
                skipped.append({"name": member_name, "reason": "unsupported_extension"})
                continue
            member_size = int(getattr(info, "file_size", 0) or 0)
            if member_size < 1 or member_size > MAX_ARCHIVE_MEMBER_BYTES:
                skipped.append({"name": member_name, "reason": "member_size_limit"})
                continue
            if member_name in seen_names:
                skipped.append({"name": member_name, "reason": "duplicate_member"})
                continue
            seen_names.add(member_name)
            eligible.append((member_name, suffix, member_size))

        _archive_progress(progress_callback, 0.04, f"Đang giải nén RAR hàng loạt ({len(eligible)} ảnh)...")
        backend = _run_rar_batch_extract(archive_path, extract_dir, [item[0] for item in eligible])
        _archive_progress(progress_callback, 0.35, f"Đã giải nén xong; đang kiểm tra {len(eligible)} ảnh...")
        root_resolved = extract_root.resolve()
        total = len(eligible)
        for index, (member_name, suffix, member_size) in enumerate(eligible, start=1):
            source = extract_dir.joinpath(*member_name.split("/"))
            try:
                source_resolved = source.resolve(strict=False)
                if source.is_symlink() or not source.is_file() or not source_resolved.is_relative_to(root_resolved):
                    skipped.append({"name": member_name, "reason": "unsafe_extracted_path"})
                    continue
                actual_size = source.stat().st_size
                if actual_size < 1 or actual_size > MAX_ARCHIVE_MEMBER_BYTES:
                    skipped.append({"name": member_name, "reason": "member_size_limit"})
                    continue
                raw, read_error = _read_path_limited(source)
                if read_error:
                    skipped.append({"name": member_name, "reason": read_error})
                    continue
                expanded_seen += len(raw or b"")
                if expanded_seen > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ValueError(
                        f"Dung lượng thực tế đã giải nén vượt giới hạn {_format_bytes(MAX_ARCHIVE_UNCOMPRESSED_BYTES)}."
                    )
                try:
                    image = decode_archive_image(raw, MAX_ARCHIVE_IMAGE_PIXELS)
                except ValueError as exc:
                    skipped.append({"name": member_name, "reason": str(exc)})
                    continue
                canonical_suffix = ".jpg" if suffix in {".jpg", ".jpeg"} else suffix
                member_parts = member_name.split("/")
                parent_dir = DATASET_IMAGES_DIR.joinpath(*[_safe_dataset_part(part) for part in member_parts[:-1]])
                parent_dir.mkdir(parents=True, exist_ok=True)
                stem = _safe_dataset_stem(member_parts[-1])
                destination = parent_dir / f"{stem}_{import_id[:10]}_{len(saved) + 1:04d}{canonical_suffix}"
                shutil.move(str(source), str(destination))
                saved_destinations.append(destination)
                saved.append(_project_relative(destination))
            finally:
                # Accepted files were moved; rejected files can be removed now.
                if source.exists() and not source.is_symlink():
                    try:
                        source.unlink(missing_ok=True)
                    except OSError:
                        pass
            if index == 1 or index == total or index % 50 == 0:
                _archive_progress(
                    progress_callback,
                    0.35 + 0.64 * (index / max(1, total)),
                    f"Đang kiểm tra ảnh {index}/{total}...",
                )
        return saved, saved_destinations, skipped, expanded_seen, backend
    except Exception:
        for destination in saved_destinations:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if extract_dir.resolve().parent != DATASET_IMAGES_DIR.parent.resolve():
            raise ValueError("Unexpected extraction cleanup directory")
        shutil.rmtree(extract_dir, ignore_errors=True)


def _open_archive_path(archive_path, archive_type):
    if archive_type == "zip":
        if not zipfile.is_zipfile(archive_path):
            raise ValueError("File tải lên không phải ZIP hợp lệ.")
        return zipfile.ZipFile(archive_path), "python-zipfile"
    if rarfile is None:
        raise ValueError("Chưa có hỗ trợ RAR. Hãy cài rarfile và 7-Zip/unrar/bsdtar.")
    if not rarfile.is_rarfile(archive_path):
        raise ValueError("File tải lên không phải RAR hợp lệ.")
    try:
        if not rarfile.tool_setup().check():
            raise RuntimeError("RAR backend unavailable")
    except Exception as exc:
        raise ValueError("Không có backend giải nén RAR. Hãy cài 7-Zip, unrar hoặc bsdtar rồi thử lại.") from exc
    try:
        return rarfile.RarFile(archive_path), "rarfile + external RAR backend"
    except Exception as exc:
        raise ValueError("Không mở được RAR. Hãy cài 7-Zip, unrar hoặc bsdtar rồi thử lại.") from exc


def _import_archive_path_to_dataset(archive_path, original_name, progress_callback=None):
    """Import a completed local archive without making a second 4GB copy."""
    DATASET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    IMPORT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = Path(archive_path).resolve()
    archive_name = Path(original_name or archive_path.name).name
    archive_suffix = Path(archive_name).suffix.lower()
    if archive_suffix not in {".zip", ".rar"}:
        raise ValueError("Chỉ chấp nhận file .zip hoặc .rar.")
    if archive_suffix == ".rar" and re.search(r"\.part\d+\.rar$", archive_name, re.IGNORECASE):
        raise ValueError(
            "RAR nhiều phần cần đầy đủ các file .part*.rar trong cùng thư mục; "
            "hãy đóng gói thành một RAR duy nhất trước khi upload."
        )
    if not archive_path.is_file():
        raise FileNotFoundError("Archive tạm không tồn tại.")
    archive_size = archive_path.stat().st_size
    if archive_size < 1:
        raise ValueError("File tải lên rỗng.")
    if archive_size > MAX_ARCHIVE_UPLOAD_BYTES:
        raise ValueError(
            f"Archive {_format_bytes(archive_size)} vượt giới hạn {_format_bytes(MAX_ARCHIVE_UPLOAD_BYTES)}."
        )

    import_id = uuid.uuid4().hex
    saved = []
    saved_destinations = []
    skipped = []
    archive = None
    started = time.monotonic()
    expanded_seen = 0
    try:
        archive, backend = _open_archive_path(archive_path, archive_suffix[1:])
        needs_password = getattr(archive, "needs_password", None)
        if callable(needs_password) and needs_password():
            raise ValueError("RAR có mật khẩu; hãy giải mã archive trước khi upload.")
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"Archive có quá nhiều mục (tối đa {MAX_ARCHIVE_MEMBERS}).")
        declared_uncompressed = sum(max(0, int(getattr(info, "file_size", 0) or 0)) for info in infos)
        if declared_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"Tổng dung lượng giải nén {_format_bytes(declared_uncompressed)} vượt giới hạn "
                f"{_format_bytes(MAX_ARCHIVE_UNCOMPRESSED_BYTES)}."
            )
        free_bytes = shutil.disk_usage(DATASET_IMAGES_DIR).free
        extraction_reserve = min(MAX_ARCHIVE_UNCOMPRESSED_BYTES, declared_uncompressed) + 256 * 1024 * 1024
        if free_bytes < extraction_reserve:
            raise ValueError(
                f"Ổ đĩa chỉ còn {_format_bytes(free_bytes)} nhưng archive cần khoảng "
                f"{_format_bytes(extraction_reserve)} cho dữ liệu giải nén. "
                "Hãy giải phóng dung lượng hoặc đặt PALM_ARCHIVE_UPLOAD_DIR và PALM_DATASET_DIR sang ổ khác."
            )

        fast_rar_result = None
        if archive_suffix == ".rar" and getattr(archive, "_file_parser", None) is not None:
            fast_rar_result = _import_rar_batch(archive_path, infos, import_id, progress_callback)
            saved, saved_destinations, skipped, expanded_seen, backend = fast_rar_result
        for info in ([] if fast_rar_result is not None else infos):
            member_name = _safe_zip_member_name(getattr(info, "filename", ""))
            if _archive_info_is_dir(info) or member_name is None:
                skipped.append({"name": str(getattr(info, "filename", "")), "reason": "unsafe_path_or_directory"})
                continue
            suffix = Path(member_name).suffix.lower()
            if suffix not in ALLOWED_IMAGE_EXTS:
                skipped.append({"name": member_name, "reason": "unsupported_extension"})
                continue
            member_size = int(getattr(info, "file_size", 0) or 0)
            if member_size < 1 or member_size > MAX_ARCHIVE_MEMBER_BYTES:
                skipped.append({"name": member_name, "reason": "member_size_limit"})
                continue
            raw, read_error = _read_member_limited(archive, info)
            if read_error:
                skipped.append({"name": member_name, "reason": read_error})
                continue
            expanded_seen += len(raw or b"")
            if expanded_seen > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"Dung lượng thực tế đã giải nén vượt giới hạn {_format_bytes(MAX_ARCHIVE_UNCOMPRESSED_BYTES)}."
                )
            try:
                image = decode_archive_image(raw, MAX_ARCHIVE_IMAGE_PIXELS)
            except ValueError as exc:
                skipped.append({"name": member_name, "reason": str(exc)})
                continue

            canonical_suffix = ".jpg" if suffix in {".jpg", ".jpeg"} else suffix
            member_parts = member_name.split("/")
            parent_dir = DATASET_IMAGES_DIR.joinpath(*[_safe_dataset_part(part) for part in member_parts[:-1]])
            parent_dir.mkdir(parents=True, exist_ok=True)
            stem = _safe_dataset_stem(member_parts[-1])
            destination = parent_dir / f"{stem}_{import_id[:10]}_{len(saved) + 1:04d}{canonical_suffix}"
            destination.write_bytes(raw)
            saved_destinations.append(destination)
            saved.append(_project_relative(destination))
    except Exception:
        for destination in saved_destinations:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        if archive is not None:
            try:
                archive.close()
            except Exception:
                pass

    report = {
        "ok": True,
        "import_id": import_id,
        "source_name": archive_name,
        "archive_type": archive_suffix[1:],
        "backend": backend,
        "archive_size_bytes": archive_size,
        "declared_uncompressed_bytes": declared_uncompressed,
        "expanded_bytes_processed": expanded_seen,
        "saved_count": len(saved),
        "skipped_count": len(skipped),
        "saved_files": saved,
        "skipped": skipped[:200],
        "dataset_dir": _project_relative(DATASET_IMAGES_DIR),
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    report_path = IMPORT_REPORT_DIR / f"import_{import_id}.json"
    report["report_path"] = _project_relative(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def import_archive_to_dataset(file_stream, original_name):
    """Backward-compatible stream importer; chunked UI imports use the path variant."""
    archive_name = Path(original_name or "upload.zip").name
    archive_suffix = Path(archive_name).suffix.lower()
    if archive_suffix not in {".zip", ".rar"}:
        raise ValueError("Chỉ chấp nhận file .zip hoặc .rar.")
    with tempfile.TemporaryDirectory(prefix="palmistry_archive_") as temp_dir:
        archive_path = Path(temp_dir) / ("upload" + archive_suffix)
        total_bytes = 0
        with archive_path.open("wb") as handle:
            while True:
                chunk = file_stream.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_ARCHIVE_UPLOAD_BYTES:
                    raise ValueError(
                        f"Archive vượt giới hạn {_format_bytes(MAX_ARCHIVE_UPLOAD_BYTES)}."
                    )
                handle.write(chunk)
        return _import_archive_path_to_dataset(archive_path, archive_name)


def import_zip_to_dataset(file_stream, original_name):
    """Backward-compatible ZIP-only wrapper used by older callers/tests."""
    if not str(original_name or "").lower().endswith(".zip"):
        raise ValueError("Chỉ chấp nhận file .zip.")
    return import_archive_to_dataset(file_stream, original_name)

def _archive_job_snapshot(job_id):
    with _archive_upload_lock:
        job = _archive_jobs.get(str(job_id))
        return copy.deepcopy(job) if job is not None else None


def _archive_job_update(job_id, **updates):
    with _archive_upload_lock:
        job = _archive_jobs.get(str(job_id))
        if job is not None:
            job.update(updates)


def _run_archive_import_job(job_id, upload_id, archive_path, archive_name):
    _archive_job_update(
        job_id,
        status="queued",
        phase="queued",
        progress=0.0,
        message="Đang chờ lượt giải nén...",
    )
    try:
        # A single worker prevents two multi-GB archives from saturating the
        # disk and makes progress reporting predictable.
        with _archive_import_execution_lock:
            _archive_job_update(
                job_id,
                status="running",
                phase="extracting",
                progress=0.01,
                message="Đang chuẩn bị giải nén...",
                started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            )
            progress_callback = lambda progress, message: _archive_job_update(
                job_id,
                progress=progress,
                message=message,
            )
            report = _import_archive_path_to_dataset(
                archive_path,
                archive_name,
                progress_callback=progress_callback,
            )
            _archive_job_update(
                job_id,
                status="completed",
                phase="completed",
                progress=1.0,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                finished_epoch=time.time(),
                result=report,
                error="",
                message=f"Đã nhập {report.get('saved_count', 0)} ảnh.",
            )
    except Exception as exc:
        _archive_job_update(
            job_id,
            status="failed",
            phase="failed",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            finished_epoch=time.time(),
            result={},
            error=str(exc) or repr(exc),
            message=f"Lỗi: {str(exc) or repr(exc)}",
        )
    finally:
        try:
            Path(archive_path).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            _, meta_path = _archive_upload_paths(upload_id)
            meta_path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass

def _new_archive_upload(filename, size):
    archive_name = Path(filename or "").name
    suffix = Path(archive_name).suffix.lower()
    if suffix not in {".zip", ".rar"}:
        raise ValueError("Chỉ chấp nhận file .zip hoặc .rar.")
    try:
        size = int(size)
    except (TypeError, ValueError):
        raise ValueError("Kích thước archive không hợp lệ.")
    if size < 1:
        raise ValueError("File tải lên rỗng.")
    if size > MAX_ARCHIVE_UPLOAD_BYTES:
        raise ValueError(
            f"Archive {_format_bytes(size)} vượt giới hạn {_format_bytes(MAX_ARCHIVE_UPLOAD_BYTES)}."
        )

    _purge_stale_archive_uploads()
    ARCHIVE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(ARCHIVE_UPLOAD_DIR).free
    reserve = min(MAX_ARCHIVE_UNCOMPRESSED_BYTES, size + 512 * 1024 * 1024)
    if free_bytes < reserve + 256 * 1024 * 1024:
        raise ValueError(
            f"Ổ đĩa chỉ còn {_format_bytes(free_bytes)}; cần ít nhất "
            f"{_format_bytes(reserve + 256 * 1024 * 1024)} để nhận archive an toàn."
        )

    upload_id = secrets.token_urlsafe(24)
    part_path, meta_path = _archive_upload_paths(upload_id)
    meta = {
        "upload_id": upload_id,
        "filename": archive_name,
        "suffix": suffix,
        "size": size,
        "received": 0,
        "created_at": time.time(),
    }
    with _archive_upload_lock:
        part_path.open("wb").close()
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _read_archive_upload_meta(upload_id):
    _, meta_path = _archive_upload_paths(upload_id)
    if not meta_path.is_file():
        raise FileNotFoundError("Phiên upload không tồn tại hoặc đã hết hạn.")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Metadata phiên upload bị hỏng.") from exc
    if float(meta.get("created_at", 0)) + ARCHIVE_UPLOAD_TTL_SECONDS < time.time():
        part_path, _ = _archive_upload_paths(upload_id)
        part_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        raise ValueError("Phiên upload đã hết hạn; hãy chọn lại file.")
    return meta


def _prune_archive_jobs():
    cutoff = time.time() - ARCHIVE_UPLOAD_TTL_SECONDS
    with _archive_upload_lock:
        for job_id, job in list(_archive_jobs.items()):
            if job.get("status") in {"completed", "failed"} and float(job.get("finished_epoch", 0) or 0) < cutoff:
                _archive_jobs.pop(job_id, None)


def class_map_from_bundle():
    return dict(CLASS_MAP)

def read_mask(path):
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return validate_mask(arr)

def estimate_line_metrics(mask_path, frame_shape=None):
    from palm_geometry import resolve_transverse_lines
    from palmistry_strict_auto_onefile import line_feature_from_mask
    mask = read_mask(mask_path)
    features = resolve_transverse_lines({info["key"]:line_feature_from_mask(mask, cls) for cls, info in LINE_CLASSES.items()})
    return {name: {**item, "present":bool(item.get("detected")),
                    "continuity":item.get("principal_path_coverage",0.0)} for name,item in features.items()}


def metrics_vector(metrics):
    vals = []

    for key in ["life_line", "head_line", "heart_line", "simian_line"]:
        m = metrics.get(key, {})

        if m.get("present"):
            vals.extend([
                float(m.get("length_norm", 0.0)),
                float(m.get("continuity", 0.0)),
                float(m.get("center_x", 0.0)),
                float(m.get("center_y", 0.0)),
            ])
        else:
            vals.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(vals, dtype=np.float32)


def is_stable(history):
    if len(history) < 3:
        return False

    arr = np.array([metrics_vector(m) for m in history], dtype=np.float32)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    valid = mean > 0.03

    if valid.sum() == 0:
        return False

    relative_change = std[valid] / np.maximum(mean[valid], 1e-6)

    return bool(np.percentile(relative_change, 80) < STABILITY_TOLERANCE)


def find_reading_text(data):
    keys = {
        "reading_vi",
        "vietnamese_reading",
        "phan_tich_tieng_viet",
        "summary_vi",
        "reading",
        "interpretation",
    }

    found = []

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() in keys and isinstance(v, str) and v.strip():
                    found.append(v.strip())

                walk(v)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

    if found:
        return found[0]

    return ""


def _safe_float(obj, key, default=0.0):
    try:
        return float(obj.get(key, default))
    except Exception:
        return float(default)


def fallback_reading(metrics):
    return "Geometry only; no personality, health or future inference.\n" + "\n".join(
        f"{name}: normalized length {item.get('length_norm', 0):.3f}, continuity {item.get('continuity', 0):.3f}"
        for name, item in metrics.items() if item.get("present"))

def overlay_mask_on_frame(frame, mask):
    if frame is None:
        return None

    if mask is None:
        return frame.copy()

    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

    result = frame.copy()

    for cls, info in LINE_CLASSES.items():
        color = np.array(info["color"], dtype=np.uint8)
        area = mask == cls

        if area.sum() == 0:
            continue

        result[area] = (0.45 * result[area] + 0.55 * color).astype(np.uint8)

    return result


def draw_hud(frame, text, color=(0, 255, 255)):
    out = frame.copy()

    cv2.rectangle(out, (18, 18), (620, 76), (20, 20, 20), -1)

    cv2.putText(
        out,
        text,
        (34, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        color,
        2,
        cv2.LINE_AA,
    )

    return out


def synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.state_lock:
            return method(self, *args, **kwargs)
    return wrapped


class LiveEngine:
    def __init__(self):
        self.checkpoint = find_checkpoint()
        self.cap = None

        self.camera_index = CAMERA_INDEX
        self.requested_camera_index = CAMERA_INDEX
        self.camera_name = "Webcam"

        self.state_lock = threading.RLock()
        self.frame_lock = self.state_lock
        self.stop_event = threading.Event()
        self.generation = 0
        self.frame_id = 0
        self.snapshot = None
        self.locked_snapshot = None
        self.threads = []
        self.ready = False
        self.readiness_reason = "Preview only: explicit checkpoint and trusted SHA-256 required."
        if self.checkpoint and CHECKPOINT_SHA256:
            try:
                from palmistry_strict_auto_onefile import NUMPY_MODEL_AVAILABLE
                self.ready = NUMPY_MODEL_AVAILABLE and self.checkpoint.suffix.lower() == ".npz" and sha256(self.checkpoint).lower() == CHECKPOINT_SHA256.lower()
                self.readiness_reason = "Model ready (NumPy checkpoint)." if self.ready else "Preview only: NumPy checkpoint hash mismatch or unsupported format."
            except (OSError, ValueError) as e:
                self.readiness_reason = f"Preview only: {e}"
        self.latest_frame = None

        self.latest_mask = None
        self.latest_overlay_exact = None
        self.latest_metrics = {}
        self.latest_json = {}
        self.latest_reading = ""

        self.status = self.readiness_reason
        self.last_predict_time = 0.0

        self.locked = False
        self.locked_frame = None
        self.locked_metrics = {}
        self.locked_reading = ""
        self.lock_start = None
        self.history = deque(maxlen=12)

        self.stop = False
        self.predicting = False

    def release_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = None

    def open_camera(self, index):
        self.release_camera()

        if os.name == "nt":
            backends = [
                ("CAP_DSHOW", cv2.CAP_DSHOW),
                ("CAP_MSMF", cv2.CAP_MSMF),
                ("DEFAULT", 0),
            ]
        else:
            backends = [("DEFAULT", 0)]

        for backend_name, backend in backends:
            print(f"Đang mở camera {index} bằng {backend_name}")

            if backend == 0:
                cap = cv2.VideoCapture(index)
            else:
                cap = cv2.VideoCapture(index, backend)

            if not cap.isOpened():
                cap.release()
                continue

            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, DISPLAY_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            good_frame = False

            for _ in range(30):
                ret, frame = cap.read()

                if ret and frame is not None:
                    good_frame = True
                    break

                time.sleep(0.05)

            if good_frame:
                with self.state_lock:
                    if self.stop_event.is_set():
                        cap.release()
                        return False
                    self.cap = cap
                    self.camera_index = index
                    self.camera_name = "Webcam" if index == WEBCAM_INDEX else "Camera USB" if index == USB_CAMERA_INDEX else f"Camera {index}"
                    self.status = f"{self.camera_name} sẵn sàng"
                    print(self.status)
                return True

            cap.release()

        with self.state_lock:
            self.status = f"Không nhận được hình từ camera {index}"
            print(self.status)
        return False

    @synchronized
    def set_camera(self, index):
        if not 0 <= int(index) <= 16:
            raise ValueError("Camera index must be 0..16.")
        self.requested_camera_index = int(index)
        self.reset()
        self.latest_frame = None
        self.status = f"Switching camera to {index}"

    def camera_loop(self):
        try:
            while not self.stop_event.is_set():
                with self.state_lock:
                    index = self.requested_camera_index
                    locked = self.locked
                    generation = self.generation
                    capture = self.cap
                if self.cap is None or self.camera_index != index:
                    if not self.open_camera(index):
                        self.stop_event.wait(1.0)
                    continue
                if locked:
                    self.stop_event.wait(0.05)
                    continue
                ret, frame = capture.read()
                if ret and frame is not None:
                    if FLIP_CAMERA:
                        frame = cv2.flip(frame, 1)
                    with self.state_lock:
                        if generation == self.generation and index == self.requested_camera_index and not self.stop_event.is_set():
                            self.latest_frame = frame.copy()
                            self.frame_id += 1
                else:
                    with self.state_lock:
                        self.status = "Camera has no frame. " + self.readiness_reason
                self.stop_event.wait(1 / max(1, DISPLAY_FPS))
        finally:
            self.release_camera()

    def predict_loop(self):
        # Readiness gate prevents repeated failures when no checkpoint is available.
        if not self.ready:
            return
        last_frame_id = -1
        while not self.stop_event.is_set():
            with self.state_lock:
                frame = None
                if not self.locked and self.latest_frame is not None and self.frame_id != last_frame_id:
                    frame = self.latest_frame.copy()
                    frame_id, generation = self.frame_id, self.generation
                    self.predicting = True
            if frame is None:
                self.stop_event.wait(0.05)
                continue
            last_frame_id = frame_id
            try:
                mask, overlay, data, metrics, reading = self.run_predict(frame)
                with self.state_lock:
                    if generation == self.generation and not self.locked and not self.stop_event.is_set():
                        self.snapshot = {"frame_id": frame_id, "mask": mask.copy(), "overlay": overlay.copy(),
                                         "metrics": copy.deepcopy(metrics), "reading": reading, "data": copy.deepcopy(data)}
                        self.latest_mask, self.latest_overlay_exact = mask, overlay
                        self.latest_json, self.latest_metrics, self.latest_reading = data, metrics, reading
                        self.status = "Valid prediction for frame " + str(frame_id)
                        self.update_lock(overlay, metrics, reading)
            except Exception as e:
                with self.state_lock:
                    if generation == self.generation and not self.locked:
                        self.snapshot = None
                        self.latest_mask = self.latest_overlay_exact = None
                        self.latest_metrics, self.latest_json, self.latest_reading = {}, {}, ""
                        self.status = f"No valid prediction: {e}"
                        # A bad checkpoint is not retried continuously. Restart after repair.
                        self.ready = False
                        self.readiness_reason = self.status
                return
            finally:
                with self.state_lock:
                    self.predicting = False
                    self.last_predict_time = time.time()
            self.stop_event.wait(PREDICT_EVERY_SECONDS)

    def run_predict(self, frame):
        if self.checkpoint is None:
            raise FileNotFoundError("No explicit checkpoint configured.")
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        # TemporaryDirectory is unique, outside the repository, and cleaned on every exit.
        with tempfile.TemporaryDirectory(prefix="palm-request-") as temp:
            root = Path(temp)
            frame_path, out_json = root / "frame.png", root / "result.json"
            out_mask, out_overlay = root / "mask.png", root / "overlay.png"
            if not cv2.imwrite(str(frame_path), frame):
                raise OSError("Could not write temporary camera frame.")
            cmd = [sys.executable, str(CODE_PATH), "--config", str(Path(os.environ.get("PALM_CONFIG", PROJECT_DIR / "prototype_config.json")).resolve()),
                   "predict", "--checkpoint", str(self.checkpoint), "--checkpoint-sha256", CHECKPOINT_SHA256,
                   "--request-id", request_id, "--image", str(frame_path), "--out_json", str(out_json),
                   "--out_mask", str(out_mask), "--out_overlay", str(out_overlay), "--read_threshold", str(READ_THRESHOLD)]
            cmd += ["--mirrored", "yes" if FLIP_CAMERA else "no"]
            env = os.environ.copy()
            env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
            result = subprocess.run(cmd, cwd=str(PROJECT_DIR), capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=PREDICT_TIMEOUT_SECONDS, env=env)
            if result.returncode:
                raise RuntimeError(f"Inference exited {result.returncode}; all request artifacts discarded.")
            data = load_json(out_json)
            if data.get("request_id") != request_id or data.get("success_marker") != request_id or not data.get("segmentation_available"):
                raise RuntimeError("Missing/mismatched request success marker; artifacts discarded.")
            mask = read_mask(out_mask)
            overlay = cv2.imread(str(out_overlay))
            if overlay is None or mask.shape != overlay.shape[:2]:
                raise ValueError("Missing or inconsistent prediction artifacts.")
            # Both mask and overlay remain in the model crop coordinate system.
            metrics = {name: {"length_norm":item["length_norm"], "center_x":item["center_x"], "center_y":item["center_y"], "present":True, "continuity":item.get("principal_path_coverage",0.0)}
                       for name,item in data.get("accepted_lines",{}).items()}
            data["subprocess_end_to_end_seconds"] = time.perf_counter() - started
            return mask.copy(), overlay.copy(), data, metrics, fallback_reading(metrics)

    @synchronized
    def update_lock(self, exact_overlay, metrics, reading):
        if self.snapshot is None or self.locked:
            return
        present = sum(bool(item.get("present")) for item in metrics.values())
        if present < GOOD_OVERLAY_MIN_LINES:
            self.history.clear()
            self.lock_start = None
            return
        if LOCK_ON_FIRST_GOOD_OVERLAY:
            self.lock_now()
            return
        self.history.append(copy.deepcopy(metrics))
        if not is_stable(self.history):
            self.lock_start = None
        elif self.lock_start is None:
            self.lock_start = time.time()
        elif time.time() - self.lock_start >= STABLE_SECONDS:
            self.lock_now()

    @synchronized
    def get_display_frame(self):
        snapshot = self.locked_snapshot if self.locked else self.snapshot
        if snapshot is not None:
            return snapshot["overlay"].copy()
        if self.latest_frame is not None:
            return draw_hud(self.latest_frame.copy(), "PREVIEW - NO VALID PREDICTION", (255, 255, 255))
        return draw_hud(np.zeros((480, 640, 3), dtype=np.uint8), "NO CAMERA FRAME", (0, 0, 255))

    def get_jpeg(self):
        frame = self.get_display_frame()

        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )

        if not ok:
            return b""

        return buffer.tobytes()

    @synchronized
    def reset(self):
        self.generation += 1  # Discard results that started before reset/camera switch.
        self.locked = False
        self.locked_snapshot = self.snapshot = None
        self.locked_frame = self.latest_mask = self.latest_overlay_exact = None
        self.locked_metrics, self.latest_metrics, self.latest_json = {}, {}, {}
        self.locked_reading = self.latest_reading = ""
        self.lock_start = None
        self.history.clear()
        self.status = "No valid prediction. " + self.readiness_reason

    @synchronized
    def lock_now(self):
        if self.locked:
            return True
        if self.snapshot is None:
            self.status = "No valid prediction to lock."
            return False
        self.locked_snapshot = copy.deepcopy(self.snapshot)
        self.locked_frame = self.locked_snapshot["overlay"].copy()
        self.locked_metrics = copy.deepcopy(self.locked_snapshot["metrics"])
        self.locked_reading = self.locked_snapshot["reading"]
        self.locked = True
        self.status = "Locked prediction for frame " + str(self.locked_snapshot["frame_id"])
        return True

    @synchronized
    def payload(self):
        snapshot = self.locked_snapshot if self.locked else self.snapshot
        return copy.deepcopy({"status": self.status, "camera": self.camera_index, "camera_name": self.camera_name,
            "requested_camera": self.requested_camera_index, "predicting": self.predicting, "locked": self.locked,
            "ready": self.ready, "readiness_reason": self.readiness_reason,
            "frame_id": snapshot["frame_id"] if snapshot else None,
            "metrics": snapshot["metrics"] if snapshot else {}, "reading": snapshot["reading"] if snapshot else "",
            "prediction": snapshot["data"] if snapshot else {}, "display_fps": DISPLAY_FPS,
            "predict_every": PREDICT_EVERY_SECONDS, "stable_seconds": STABLE_SECONDS})

    def start(self):
        self.threads = [threading.Thread(target=self.camera_loop, daemon=True),
                        threading.Thread(target=self.predict_loop, daemon=True)]
        for thread in self.threads:
            thread.start()

    def shutdown(self):
        self.stop = True
        self.stop_event.set()
        self.release_camera()
        for thread in self.threads:
            thread.join(timeout=PREDICT_TIMEOUT_SECONDS + 2)
        with self.state_lock:
            self.reset()
            self.latest_frame = None
        return all(not thread.is_alive() for thread in self.threads)


engine = LiveEngine()


def make_split_chart(split_counts):
    if not split_counts:
        return None

    names = list(split_counts.keys())
    values = list(split_counts.values())

    plt.figure(figsize=(7, 4))
    plt.bar(names, values)
    plt.title("Phân bố train / valid / test")
    plt.xlabel("Split")
    plt.ylabel("Số ảnh")
    plt.grid(axis="y", alpha=0.25)

    out = STATIC_DIR / "split_chart.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()

    return out.name


def _read_csv_rows(path):
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def _float_column(rows, key):
    values = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def make_metric_charts():
    history_path = BUNDLE_DIR / "logs" / "history.csv"
    if not history_path.exists() or plt is None:
        return []
    rows = _read_csv_rows(history_path)
    epochs = _float_column(rows, "epoch")
    if not epochs:
        return []
    charts = []
    loss_cols = [c for c in ("train_loss", "val_loss") if any(c in row for row in rows)]
    dice_cols = [c for c in ("dice_life_line", "dice_head_line", "dice_heart_line", "dice_simian_line", "dice_main_lines_mean", "dice_palm_area") if any(c in row for row in rows)]
    switch_epoch = min(79, int(max(epochs)))
    for columns, title, ylabel, filename in ((loss_cols, "Loss theo epoch", "Loss", "loss_chart.png"), (dice_cols, "Dice theo epoch", "Dice", "dice_chart.png")):
        if not columns:
            continue
        plt.figure(figsize=(9, 4))
        for column in columns:
            values = _float_column(rows, column)
            if values:
                plt.plot(epochs[:len(values)], values, label=column)
        plt.axvline(switch_epoch, linestyle="--", linewidth=2)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(alpha=0.25)
        out = STATIC_DIR / filename
        plt.tight_layout()
        plt.savefig(out, dpi=140)
        plt.close()
        charts.append(out.name)
    return charts

def make_heatmaps():
    eval_csv = BUNDLE_DIR / "eval_sample" / "eval_sample.csv"

    if not eval_csv.exists():
        return []

    class_map = class_map_from_bundle()
    eval_rows = _read_csv_rows(eval_csv)

    if not eval_rows or "eval_mask_path" not in eval_rows[0]:
        return []

    masks = [BUNDLE_DIR / row["eval_mask_path"] for row in eval_rows if row.get("eval_mask_path")]
    outputs = []

    for cls in [2, 3, 4]:
        acc = None
        used = 0

        for p in masks[:300]:
            if not p.exists():
                continue

            m = read_mask(p)

            if m is None:
                continue

            m = np.where(m <= 4, m, 0)
            b = (m == cls).astype(np.float32)

            if acc is None:
                acc = np.zeros_like(b, dtype=np.float32)

            if b.shape != acc.shape:
                b = cv2.resize(
                    b,
                    (acc.shape[1], acc.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            acc += b
            used += 1

        if acc is None or used == 0:
            continue

        if acc.max() > 0:
            acc = acc / acc.max()

        img = (acc * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        name = f"heatmap_{class_map.get(cls, cls)}.jpg"
        out = STATIC_DIR / name
        cv2.imwrite(str(out), img)
        outputs.append(name)

    return outputs


def make_pixel_distribution():
    eval_csv = BUNDLE_DIR / "eval_sample" / "eval_sample.csv"

    if not eval_csv.exists():
        return None

    class_map = class_map_from_bundle()
    eval_rows = _read_csv_rows(eval_csv)

    if not eval_rows or "eval_mask_path" not in eval_rows[0]:
        return None

    masks = [BUNDLE_DIR / row["eval_mask_path"] for row in eval_rows if row.get("eval_mask_path")]
    rows = []

    for cls, name in class_map.items():
        if int(cls) > 4:
            continue

        count = 0

        for p in masks[:300]:
            if p.exists():
                m = read_mask(p)

                if m is not None:
                    m = np.where(m <= 4, m, 0)
                    count += int((m == int(cls)).sum())

        rows.append({"class": name, "pixels": count})

    if not rows:
        return None

    plt.figure(figsize=(8, 4))
    plt.bar([row["class"] for row in rows], [row["pixels"] for row in rows])
    plt.title("Pixel distribution trong mask thật")
    plt.xlabel("Class")
    plt.ylabel("Pixel count")
    plt.xticks(rotation=25)
    plt.grid(axis="y", alpha=0.25)

    out = STATIC_DIR / "pixel_distribution.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()

    return out.name


def prepare_eda_assets():
    return {"warning": "EDA assets are optional. Use the explicit ZIP/RAR upload button to import images into dataset/images.", "charts": [], "heatmaps": [], "sample_overlays": []}


EDA_ASSETS = prepare_eda_assets()


def file_size_mb(path):
    if path is None:
        return "Không có"

    path = Path(path)

    if not path.exists():
        return "Không có"

    return f"{path.stat().st_size / 1024 / 1024:.2f} MB"


def model_info_html():
    return "<p>Reproducible computer-vision prototype with a local camera preview. Use the ZIP/RAR upload card to import valid images into <code>dataset/images</code>; every import writes a JSON report under <code>dataset/imports</code>. Model inference uses an explicit NumPy .npz checkpoint and trusted SHA-256; PyTorch and scikit-learn are not used. Geometry does not predict personality, health or the future.</p>"


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Palmistry Live Camera</title>
    <style>
        body {
            margin: 0;
            background: #0f1117;
            color: #f5f5f5;
            font-family: Arial, sans-serif;
        }

        .top {
            padding: 18px 26px;
            background: #151923;
            border-bottom: 1px solid #293041;
        }

        .title {
            font-size: 28px;
            font-weight: 800;
        }

        .tabs {
            display: flex;
            background: #11151d;
            border-bottom: 1px solid #293041;
        }

        .tab-btn {
            padding: 14px 24px;
            cursor: pointer;
            color: #bfc5d2;
            border-right: 1px solid #293041;
            font-weight: 700;
        }

        .tab-btn.active {
            background: #202638;
            color: #ffffff;
        }

        .tab-content {
            display: none;
            padding: 22px;
        }

        .tab-content.active {
            display: block;
        }

        .live-layout {
            display: grid;
            grid-template-columns: minmax(600px, 1fr) 430px;
            gap: 22px;
            align-items: start;
        }

        .card {
            background: #171b26;
            border: 1px solid #2a3244;
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.22);
        }

        .video-card {
            padding: 12px;
        }

        #video {
            width: 100%;
            border-radius: 12px;
            background: #000000;
        }

        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: #263149;
            color: #dce7ff;
            margin-right: 6px;
            margin-bottom: 6px;
            font-size: 13px;
        }

        button {
            border: 0;
            border-radius: 10px;
            padding: 10px 14px;
            background: #3b82f6;
            color: white;
            font-weight: 700;
            cursor: pointer;
            margin-right: 8px;
            margin-bottom: 8px;
        }

        button.secondary {
            background: #374151;
        }

        button.green {
            background: #16a34a;
        }

        button.purple {
            background: #7c3aed;
        }

        input {
            background: #0f172a;
            color: #f5f5f5;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 10px;
            width: 70px;
            margin-right: 8px;
        }
        .file-input {
            width: 100%;
            box-sizing: border-box;
            margin-bottom: 10px;
        }

        .metric-box {
            background: #111827;
            border: 1px solid #2d3748;
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 10px;
        }

        .metric-name {
            font-weight: 800;
            color: #d7e8ff;
        }

        .small {
            color: #aeb4c4;
            font-size: 13px;
            line-height: 1.45;
        }

        .reading {
            line-height: 1.55;
            color: #f3f4f6;
            white-space: pre-line;
        }

        .table {
            width: 100%;
            border-collapse: collapse;
            color: #e5e7eb;
            font-size: 14px;
        }

        .table td, .table th {
            border-bottom: 1px solid #2a3244;
            padding: 9px;
            text-align: left;
        }

        .grid2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }

        img {
            max-width: 100%;
            border-radius: 12px;
        }

        .image-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }
    </style>
</head>
<body>
    <div class="top">
        <div class="title">Palmistry · Keypoint &amp; B-spline</div><p><a href="/keypoints">Mở pipeline mới: gán keypoint → train NumPy → pseudo-label → nhận diện</a></p>
    </div>

    <div class="tabs">
        <div class="tab-btn active" id="tab-live">Live Camera + Prediction</div>
        <div class="tab-btn" id="tab-info">Model Info + EDA</div>
    </div>

    <div id="live" class="tab-content active">
        <div class="live-layout">
            <div class="card video-card">
                <img id="video" src="/video_feed">
            </div>

            <div>
                <div class="card">
                    <h2>Camera (model segmentation cũ)</h2><p><a href="/keypoints">Nhận diện ảnh bằng model keypoint mới</a> · <a href="/analyze">Mở bộ phân tích mask cũ</a></p>
                    <button class="green" id="cam-zero">Webcam</button>
                    <button class="purple" id="cam-one">Camera USB</button>
                    <br>
                    <input id="customCamera" type="number" min="0" value="0">
                    <button class="secondary" id="cam-custom">Chọn camera index</button>
                    <p class="small">Webcam thường là 0. Camera USB thường là 1, 2 hoặc 3.</p>
                </div>

                <div class="card">
                    <h2>Nhập ảnh từ ZIP hoặc RAR</h2>
                    <p class="small">Chọn file .zip hoặc .rar (hỗ trợ archive nhiều GB). Ứng dụng upload theo từng phần, sau đó giải nén ảnh hợp lệ vào <code>dataset/images</code>.</p>
                    <input id="datasetZip" class="file-input" type="file" accept=".zip,.rar,application/zip,application/x-rar-compressed">
                    <button class="green" id="importArchive">Nhận ZIP/RAR và nhập ảnh</button>
                    <progress id="datasetImportProgress" value="0" max="1" hidden style="width:100%; margin:6px 0 10px;"></progress>
                    <div id="datasetImportStatus" class="small">Đang đọc trạng thái dataset...</div>
                </div>

                <div class="card">
                    <h2>Pipeline mask cũ (legacy)</h2><p><a href="/keypoints">Dùng pipeline keypoint Heart / Head / Life / Fate mới</a></p>
                    <p class="small">Sau khi nhập ZIP/RAR, hệ thống lọc ảnh trùng và chuẩn bị 100 mask chờ duyệt. Các ảnh còn lại được xử lý sau khi có seed đã duyệt. Mở trang review để đánh dấu approved/rejected, rồi bootstrap pseudo-label và train model cuối.</p>
                    <label class="small">Tên run:
                        <input id="pipelineRunName" value="pipeline_ui_001" maxlength="64">
                    </label>
                    <br>
                    <label class="small">Số mask cần duyệt:
                        <input id="pipelineReviewCount" type="number" min="1" max="3000" value="100">
                    </label>
                    <br>
                    <button class="purple" id="prepareReview">Chuẩn bị queue review</button>
                    <button id="migrateReview">Tạo bản sao queue v2 (simian)</button>
                    <br><br>
                    <label class="small">Confidence pseudo-label:
                        <input id="pipelineConfidence" type="number" min="0" max="1" step="0.01" value="0.90">
                    </label>
                    <br>
                    <label class="small">Tên output bootstrap:
                        <input id="pipelineBootstrapName" value="bootstrap_ui_001" maxlength="64">
                    </label>
                    <br>
                    <button class="green" id="bootstrapTrain">Bootstrap + train model cuối</button>
                    <div id="pipelineStatus" class="small">Pipeline đang chờ...</div>
                    <a id="pipelineReviewLink" href="#" target="_blank" rel="noopener" hidden>Mở trang duyệt mask</a>
                </div>

                <div class="card">
                    <h2>Điều khiển</h2>
                    <button id="reset-result">Đo lại</button>
                    <button class="secondary" id="lock-result">Khóa khung hình</button>
                    <div id="badges"></div>
                </div>

                <div class="card">
                    <h2>Số liệu đường chỉ tay</h2>
                    <div id="metrics">Đang chờ model...</div>
                </div>

                <div class="card">
                    <h2>Kết quả dự đoán</h2>
                    <div id="reading" class="reading">Đưa lòng bàn tay vào camera.</div>
                </div>
            </div>
        </div>
    </div>

    <div id="info" class="tab-content">
        {INFO_HTML}
    </div>

    <script nonce="{CSP_NONCE}">
        function showTab(id, btn) {
            document.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(x => x.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            btn.classList.add('active');
        }

        function metricBox(name, m) {
            if (!m || !m.present) {
                return `
                    <div class="metric-box">
                        <div class="metric-name">${name}</div>
                        <div class="small">Chưa nhận diện rõ</div>
                    </div>
                `;
            }

            return `
                <div class="metric-box">
                    <div class="metric-name">${name}</div>
                    <div>Độ dài chuẩn hóa: <b>${m.length_norm}</b></div>
                    <div>Độ liền mạch: <b>${m.continuity}</b></div>
                    <div>Góc chính: <b>${m.angle_deg}°</b></div>
                    <div>Số đoạn: <b>${m.components}</b></div>
                </div>
            `;
        }

        async function updateStatus() {
            const res = await fetch('/api/status');
            const data = await res.json();

            document.getElementById('badges').textContent =
                data.camera_name + ' | ' + data.status + ' | ' + data.readiness_reason;
            document.getElementById('metrics').textContent = JSON.stringify(data.metrics || {}, null, 2);
            document.getElementById('reading').textContent = data.reading || "No valid prediction.";
        }

        async function resetResult() {
            await fetch('/api/reset', {method: 'POST', headers: {'X-Palm-CSRF': '{CSRF_TOKEN}'}});
            await updateStatus();
        }

        async function lockResult() {
            await fetch('/api/lock', {method: 'POST', headers: {'X-Palm-CSRF': '{CSRF_TOKEN}'}});
            await updateStatus();
        }

        async function setCamera(index) {
            await fetch('/api/camera/' + index, {method: 'POST', headers: {'X-Palm-CSRF': '{CSRF_TOKEN}'}});
            await updateStatus();
        }

        async function setCustomCamera() {
            const index = document.getElementById("customCamera").value;
            await setCamera(index);
        }

        async function updateDatasetStatus() {
            const status = document.getElementById('datasetImportStatus');
            try {
                const res = await fetch('/api/dataset/status');
                const data = await res.json();
                status.textContent = data.ok
                    ? 'Dataset: ' + data.image_count + ' ảnh trong ' + data.dataset_dir
                    : (data.error || 'Không đọc được dataset.');
            } catch (error) {
                status.textContent = 'Không kết nối được tới dataset.';
            }
        }

        function formatArchiveBytes(value) {
            const units = ['B', 'KB', 'MB', 'GB', 'TB'];
            let n = Number(value) || 0;
            let i = 0;
            while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
            return n.toFixed(i === 0 ? 0 : 1) + ' ' + units[i];
        }

        async function responseData(res) {
            const data = await res.json().catch(() => ({}));
            if (!res.ok || data.ok === false) {
                throw new Error(data.error || ('HTTP ' + res.status));
            }
            return data;
        }

        async function importArchive() {
            const input = document.getElementById('datasetZip');
            const button = document.getElementById('importArchive');
            const progress = document.getElementById('datasetImportProgress');
            const status = document.getElementById('datasetImportStatus');
            if (!input.files.length) {
                status.textContent = 'Hãy chọn một file ZIP hoặc RAR trước.';
                return;
            }

            const file = input.files[0];
            let uploadId = null;
            let queued = false;
            button.disabled = true;
            progress.hidden = false;
            progress.value = 0;
            try {
                status.textContent = 'Đang tạo phiên upload cho ' + formatArchiveBytes(file.size) + '...';
                const initRes = await fetch('/api/dataset/import-archive/init', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'X-Palm-CSRF': '{CSRF_TOKEN}'},
                    body: JSON.stringify({filename: file.name, size: file.size})
                });
                const init = await responseData(initRes);
                uploadId = init.upload_id;
                const chunkSize = Number(init.chunk_size) || (16 * 1024 * 1024);

                for (let offset = 0; offset < file.size; offset += chunkSize) {
                    const end = Math.min(file.size, offset + chunkSize);
                    const chunk = file.slice(offset, end);
                    let chunkData = null;
                    let lastError = null;
                    for (let attempt = 0; attempt < 3; attempt += 1) {
                        try {
                            const chunkRes = await fetch('/api/dataset/import-archive/chunk', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/octet-stream',
                                    'X-Palm-CSRF': '{CSRF_TOKEN}',
                                    'X-Archive-Upload-ID': uploadId,
                                    'X-Archive-Upload-Offset': String(offset)
                                },
                                body: chunk
                            });
                            const chunkBody = await chunkRes.json().catch(() => ({}));
                            if (!chunkRes.ok || chunkBody.ok === false) {
                                // If the server committed the chunk but the response
                                // was lost, accept its expected offset and continue.
                                if (chunkRes.status === 409 && Number(chunkBody.expected_offset) === end) {
                                    chunkData = {received: end};
                                    break;
                                }
                                throw new Error(chunkBody.error || ('HTTP ' + chunkRes.status));
                            }
                            chunkData = chunkBody;
                            break;
                        } catch (error) {
                            lastError = error;
                            if (attempt < 2) {
                                await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
                            }
                        }
                    }
                    if (!chunkData) throw lastError || new Error('Không upload được một phần archive.');
                    progress.value = chunkData.received / file.size;
                    status.textContent = 'Đã upload ' + formatArchiveBytes(chunkData.received) + ' / ' + formatArchiveBytes(file.size);
                }

                status.textContent = 'Đã upload xong. Đang đưa archive vào hàng đợi giải nén...';
                const completeRes = await fetch('/api/dataset/import-archive/complete', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'X-Palm-CSRF': '{CSRF_TOKEN}'},
                    body: JSON.stringify({upload_id: uploadId})
                });
                const complete = await responseData(completeRes);
                queued = true;
                const jobId = complete.job_id;
                progress.value = 0;

                while (true) {
                    await new Promise(resolve => setTimeout(resolve, 1500));
                    const pollRes = await fetch('/api/dataset/import-archive/status?job_id=' + encodeURIComponent(jobId));
                    const job = await responseData(pollRes);
                    const jobProgress = Number(job.progress);
                    if (Number.isFinite(jobProgress)) {
                        progress.value = Math.max(0, Math.min(1, jobProgress));
                    }
                    if (job.status === 'queued') {
                        status.textContent = job.message || 'Đang chờ lượt giải nén...';
                    } else if (job.status === 'running') {
                        status.textContent = job.message || 'Đang giải nén và kiểm tra ảnh...';
                    } else if (job.status === 'failed') {
                        throw new Error(job.error || job.message || 'Giải nén archive thất bại.');
                    } else if (job.status === 'completed') {
                        const result = job.result || {};
                        progress.value = 1;
                        status.textContent = 'Đã nhập ' + (result.saved_count || 0) + ' ảnh; bỏ qua ' + (result.skipped_count || 0) + ' mục.';
                        await updateDatasetStatus();
                        break;
                    }
                }            } catch (error) {
                progress.value = 0;
                status.textContent = 'Lỗi: ' + (error.message || error);
                if (uploadId && !queued) {
                    fetch('/api/dataset/import-archive/cancel', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json', 'X-Palm-CSRF': '{CSRF_TOKEN}'},
                        body: JSON.stringify({upload_id: uploadId})
                    }).catch(() => {});
                }
            } finally {
                button.disabled = false;
                input.value = '';
            }
        }

        function pipelineStatusText(data) {
            const result = data.result || {};
            if (data.status === 'running') {
                const last = (data.log || []).slice(-1)[0] || 'đang xử lý...';
                return 'Đang chạy ' + data.kind + ' | ' + last;
            }
            if (data.status === 'failed') {
                return 'Lỗi pipeline: ' + (data.error || 'xem log chi tiết');
            }
            if (data.status === 'completed') {
                const summary = result.review || result.generated || result.training || {};
                const count = summary.queue_count ?? summary.generated_count ?? summary.accepted;
                return 'Đã hoàn tất ' + data.kind + (count !== undefined ? ' | số lượng: ' + count : '');
            }
            return 'Pipeline đang chờ...';
        }

        let pipelineStateRestored = false;
        async function updatePipelineStatus() {
            const status = document.getElementById('pipelineStatus');
            const link = document.getElementById('pipelineReviewLink');
            try {
                const res = await fetch('/api/pipeline/status');
                const data = await res.json();
                if (!pipelineStateRestored && data.run_name) {
                    document.getElementById('pipelineRunName').value = data.run_name;
                    if (data.bootstrap_name) document.getElementById('pipelineBootstrapName').value = data.bootstrap_name;
                    pipelineStateRestored = true;
                }
                status.textContent = pipelineStatusText(data);
                if (data.review_url && data.kind === 'prepare_review' && data.status === 'completed') {
                    link.href = data.review_url;
                    link.hidden = false;
                } else {
                    link.hidden = true;
                }
            } catch (error) {
                status.textContent = 'Không đọc được trạng thái pipeline.';
            }
        }

        async function prepareReview() {
            const status = document.getElementById('pipelineStatus');
            const runName = document.getElementById('pipelineRunName').value;
            const reviewCount = Number(document.getElementById('pipelineReviewCount').value);
            status.textContent = 'Đang khởi chạy preprocessing và review queue...';
            try {
                const res = await fetch('/api/pipeline/prepare-review', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'X-Palm-CSRF': '{CSRF_TOKEN}'},
                    body: JSON.stringify({run_name: runName, review_count: reviewCount})
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.ok) throw new Error(data.error || 'Không thể chạy pipeline.');
                document.getElementById('pipelineRunName').value = data.run_name;
                status.textContent = 'Đã xếp phase preprocessing vào nền.';
                await updatePipelineStatus();
            } catch (error) {
                status.textContent = 'Lỗi: ' + error.message;
            }
        }

        async function bootstrapTrain() {
            const status = document.getElementById('pipelineStatus');
            const runName = document.getElementById('pipelineRunName').value;
            const confidence = Number(document.getElementById('pipelineConfidence').value);
            status.textContent = 'Đang khởi chạy bootstrap và train model cuối...';
            try {
                const res = await fetch('/api/pipeline/bootstrap', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'X-Palm-CSRF': '{CSRF_TOKEN}'},
                    body: JSON.stringify({run_name: runName, bootstrap_name: document.getElementById('pipelineBootstrapName').value, confidence: confidence})
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.ok) throw new Error(data.error || 'Không thể chạy bootstrap.');
                document.getElementById('pipelineBootstrapName').value = data.bootstrap_name;
                status.textContent = 'Đã xếp bootstrap/train vào nền.';
                await updatePipelineStatus();
            } catch (error) {
                status.textContent = 'Lỗi: ' + error.message;
            }
        }

        document.addEventListener("keydown", function(e) {
            if (e.key === "r" || e.key === "R") {
                resetResult();
            }
        });

        document.getElementById('tab-live').addEventListener('click', function() { showTab('live', this); });
        document.getElementById('tab-info').addEventListener('click', function() { showTab('info', this); });
        document.getElementById('cam-zero').addEventListener('click', function() { setCamera(0); });
        document.getElementById('cam-one').addEventListener('click', function() { setCamera(1); });
        document.getElementById('cam-custom').addEventListener('click', function() { setCustomCamera(); });
        document.getElementById('reset-result').addEventListener('click', function() { resetResult(); });
        document.getElementById('lock-result').addEventListener('click', function() { lockResult(); });
        document.getElementById('importArchive').addEventListener('click', function() { importArchive(); });
        document.getElementById('migrateReview').addEventListener('click', async function(){
            const status=document.getElementById('pipelineStatus');this.disabled=true;
            status.textContent='Đang tạo bản sao queue v2...';
            try{const response=await fetch('/api/pipeline/migrate-review',{method:'POST',headers:{'Content-Type':'application/json','X-Palm-CSRF':'{CSRF_TOKEN}'},body:JSON.stringify({run_name:document.getElementById('pipelineRunName').value})});
                const data=await response.json();if(!response.ok||!data.ok)throw Error(data.error||'Không nâng cấp được queue.');
                document.getElementById('pipelineRunName').value=data.run_name;await updatePipelineStatus();
            }catch(error){status.textContent=error.message;}finally{this.disabled=false;}
        });
        document.getElementById('prepareReview').addEventListener('click', function() { prepareReview(); });
        document.getElementById('bootstrapTrain').addEventListener('click', function() { bootstrapTrain(); });
        setInterval(updateStatus, 500);
        setInterval(updatePipelineStatus, 1000);
        updateStatus();
        updateDatasetStatus();
        updatePipelineStatus();
    </script>
</body>
</html>
"""


def html_page():
    return HTML_TEMPLATE.replace("{INFO_HTML}", model_info_html()).replace("{CSRF_TOKEN}", CSRF_TOKEN).replace("{CSP_NONCE}", CSP_NONCE)


@app.route("/")
def index():
    return html_page()


@app.route("/video_feed")
def video_feed():
    def generate():
        frame_delay = 1.0 / max(1, DISPLAY_FPS)

        while not engine.stop_event.is_set():
            start = time.time()

            frame = engine.get_jpeg()

            yield b"--frame\r\n"
            yield b"Content-Type: image/jpeg\r\n"
            yield b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
            yield b"Pragma: no-cache\r\n"
            yield b"Expires: 0\r\n\r\n"
            yield frame + b"\r\n"

            elapsed = time.time() - start
            sleep_time = max(0.001, frame_delay - elapsed)
            time.sleep(sleep_time)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def _read_review_rows(run_name):
    run_dir = _artifact_run_dir(run_name)
    review_csv = run_dir / "review" / "review.csv"
    if not review_csv.is_file():
        raise FileNotFoundError("Chưa có review queue. Hãy chạy Chuẩn bị queue review trước.")
    with review_csv.open(newline="", encoding="utf-8") as handle:
        return run_dir, review_csv, list(csv.DictReader(handle))


def _write_review_rows(review_csv, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp_path = review_csv.with_name(review_csv.name + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, review_csv)


def _review_asset_url(run_name, relative_path):
    run_part = quote(str(run_name), safe="")
    asset_part = quote(str(relative_path).replace("\\", "/"), safe="/")
    return f"/pipeline_artifacts/{run_part}/01_preprocessed_dataset/{asset_part}"


@app.route("/pipeline_artifacts/<run_name>/<path:name>")
def pipeline_artifact(run_name, name):
    run_dir = _artifact_run_dir(run_name)
    target = (run_dir / Path(name)).resolve()
    if not target.is_relative_to(run_dir.resolve()):
        abort(403)
    if not target.is_file():
        abort(404)
    if request.args.get("view") == "mask":
        mask = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
        try:
            validate_mask(mask)
        except (ValueError, TypeError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 422
        # Display-only RGB palette. Semantic PNG values remain untouched.
        palette = np.array([[0,0,0],[70,160,75],[255,80,80],[60,155,255],[255,90,220],[255,210,40],[0,230,230]], dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", palette[mask][:,:,::-1])
        if not ok:
            raise OSError("Không tạo được bản xem màu của mask.")
        return Response(encoded.tobytes(), mimetype="image/png")
    return send_from_directory(str(run_dir), Path(name).as_posix())


@app.route("/pipeline_review/<run_name>")
def pipeline_review_page(run_name):
    try:
        run_dir, review_csv, rows = _read_review_rows(run_name)
    except (ValueError, FileNotFoundError) as exc:
        return f"<h1>Review unavailable</h1><p>{html.escape(str(exc))}</p>", 404
    run_part = quote(run_name, safe="")
    page_size = 40
    page_count = max(1, (len(rows) + page_size - 1) // page_size)
    try:
        page_number = min(page_count, max(1, int(request.args.get("page", 1))))
    except (ValueError, TypeError):
        page_number = 1
    navigation = "<p>" + " | ".join(
        f"<a href='/pipeline_review/{run_part}?page={i}'>Trang {i}</a>" for i in range(1, page_count + 1)
    ) + f"</p><p>{len(rows)} mask; trang {page_number}/{page_count}. Lưu từng dòng sau khi kiểm tra.</p>"
    html_rows = []
    for row in rows[(page_number - 1) * page_size:page_number * page_size]:
        review_id = html.escape(str(row.get("review_id", "")), quote=True)
        status = str(row.get("review_status") or "pending").lower()
        if status not in ("pending", "approved", "rejected"):
            status = "pending"
        image_url = _review_asset_url(run_name, row.get("image_path", ""))
        mask_url = _review_asset_url(run_name, row.get("mask_path", "")) + "?view=mask"
        overlay_url = _review_asset_url(run_name, row.get("overlay_path", ""))
        notes = html.escape(str(row.get("review_notes", "")), quote=True)
        options = "".join(
            f"<option value='{value}'{' selected' if value == status else ''}>{value}</option>"
            for value in ("pending", "approved", "rejected")
        )
        html_rows.append(
            "<tr data-review-id='{}'><td>{}</td><td><img loading='lazy' decoding='async' src='{}'></td>"
            "<td><img loading='lazy' decoding='async' src='{}'></td><td><img loading='lazy' decoding='async' src='{}'></td><td>"
            "<select class='review-status'>{}</select><br>"
            "<input class='review-notes' value='{}' placeholder='Ghi chú'> "
            "<button class='save-review'>Lưu</button><span class='save-result'></span>"
            "</td></tr>".format(
                review_id, review_id, image_url, mask_url, overlay_url, options, notes
            )
        )
    page = (
        "<!doctype html><meta charset='utf-8'><title>Palmistry review</title>"
        "<style>body{font:14px Arial;background:#111;color:#eee;margin:20px}"
        "table{border-collapse:collapse;width:100%}td{border:1px solid #444;padding:8px;vertical-align:top}"
        "img{width:220px;height:220px;object-fit:contain;background:#222;border-radius:8px}"
        "input,select,button{padding:7px;margin-top:6px}.save-result{margin-left:6px;color:#9f9}"
        "a{color:#9fe3ff}</style><h1>Duyệt mask</h1>"
        "<p>Run: <code>__RUN_ESC__</code>. Kiểm tra ảnh, mask và overlay; nếu cần sửa mask PNG "
        "bằng công cụ annotation bên ngoài rồi tải lại trang. Class IDs v2 là 0..6; class 6 = simian, class 5 vẫn là đường phụ/chưa rõ.</p>"
        "<p>Màu mask: xanh lá = lòng bàn tay; đỏ = life; xanh dương = head; hồng = heart; vàng = đường phụ/chưa rõ; xanh cyan = simian. Bấm ảnh để xem lớn. Đây là mask gợi ý, cần kiểm tra đường trace trước khi approved.</p>"
        "<p>Chỉ <b>approved</b> được dùng làm seed. <a href='/'>Quay lại app</a></p>"
        "__NAV__<table><tr><th>ID</th><th>Ảnh</th><th>Mask</th><th>Overlay</th><th>Trạng thái/Ghi chú</th></tr>__ROWS__</table>__NAV__"
        "<script nonce='__NONCE__'>"
        "document.querySelectorAll('td img').forEach(img=>{const a=document.createElement('a');a.href=img.src;a.target='_blank';a.rel='noopener';img.replaceWith(a);a.appendChild(img);});"
        "document.querySelectorAll('.save-review').forEach(function(button){button.addEventListener('click',async function(){"
        "const row=button.closest('tr');const result=row.querySelector('.save-result');result.textContent=' đang lưu...';"
        "try{const response=await fetch('/api/pipeline/review/update',{method:'POST',headers:{'Content-Type':'application/json','X-Palm-CSRF':'__CSRF__'},"
        "body:JSON.stringify({run_name:'__RUN_JS__',review_id:row.dataset.reviewId,status:row.querySelector('.review-status').value,notes:row.querySelector('.review-notes').value})});"
        "const data=await response.json().catch(()=>({}));if(!response.ok||!data.ok)throw new Error(data.error||'save failed');"
        "result.textContent=' đã lưu';}catch(error){result.textContent=' lỗi: '+error.message;}})});"
        "</script>"
    )
    page = (
        page.replace("__RUN_ESC__", html.escape(run_name, quote=True))
        .replace("__ROWS__", "".join(html_rows))
        .replace("__NAV__", navigation)
        .replace("__NONCE__", CSP_NONCE)
        .replace("__CSRF__", CSRF_TOKEN)
        .replace("__RUN_JS__", run_name)
    )
    return page


@app.route("/api/pipeline/status")
def api_pipeline_status():
    return jsonify(_pipeline_snapshot())


@app.route("/api/pipeline/prepare-review", methods=["POST"])
def api_pipeline_prepare_review():
    payload = request.get_json(silent=True) or {}
    try:
        count = int(payload.get("review_count", 100))
        if not 1 <= count <= PIPELINE_MAX_REVIEW_COUNT:
            raise ValueError("Số mask review phải trong khoảng 1..3000.")
        if dataset_image_count() <= 0:
            raise ValueError("Dataset đang rỗng. Hãy nhận ZIP/RAR trước.")
        requested_name = str(payload.get("run_name") or "").strip()
        run_name = _safe_pipeline_name(
            requested_name,
            "pipeline_ui_" + time.strftime("%Y%m%d_%H%M%S"),
        )
        run_dir = _artifact_run_dir(run_name)
        if run_dir.exists() and any(run_dir.iterdir()):
            run_name = run_name[:40] + "_" + uuid.uuid4().hex[:8]
            run_dir = _artifact_run_dir(run_name)
        command = [
            sys.executable,
            str(PROJECT_DIR / "pipeline.py"),
            "--input_dir", str(DATASET_IMAGES_DIR),
            "--run_dir", str(run_dir),
            "--prepare_review",
            "--review_count", str(count),
        ]
        snapshot = _start_pipeline_command(
            "prepare_review", command, run_name, "", run_dir / "pipeline_summary.json"
        )
        return jsonify({"ok": True, **snapshot})
    except (ValueError, FileExistsError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409 if isinstance(exc, (FileExistsError, RuntimeError)) else 400


@app.post("/api/pipeline/migrate-review")
def api_migrate_review():
    import pipeline
    payload = request.get_json(silent=True) or {}
    try:
        with _pipeline_lock:
            if _pipeline_job.get("status") == "running":
                raise RuntimeError("Đang chạy pipeline; hãy đợi hoàn tất trước khi nâng cấp queue.")
            source = _artifact_run_dir(str(payload.get("run_name") or ""))
            target_name = source.name[:40] + "_v2_" + uuid.uuid4().hex[:8]
            target = _artifact_run_dir(target_name)
            summary = pipeline.migrate_review_v2(source,target)
            _pipeline_job.update(status="completed",kind="prepare_review",run_name=target_name,
                                 bootstrap_name="",result=summary,error="",log=["Đã sao chép queue sang v2; cần duyệt lại simian, queue v1 vẫn được giữ."])
        _persist_pipeline_state()
        return jsonify(ok=True,**_pipeline_snapshot())
    except (ValueError,FileNotFoundError,RuntimeError) as exc:
        return jsonify(ok=False,error=str(exc)),400


@app.route("/api/pipeline/bootstrap", methods=["POST"])
def api_pipeline_bootstrap():
    payload = request.get_json(silent=True) or {}
    try:
        requested_run = str(payload.get("run_name") or "").strip()
        if not requested_run:
            with _pipeline_lock:
                requested_run = _pipeline_job.get("run_name", "")
        run_name = _safe_pipeline_name(requested_run, "")
        run_dir = _artifact_run_dir(run_name)
        review_csv = run_dir / "review" / "review.csv"
        if not review_csv.is_file():
            raise FileNotFoundError("Chưa có review.csv. Hãy chạy queue review và duyệt mask trước.")
        bootstrap_name = _safe_pipeline_name(
            str(payload.get("bootstrap_name") or "").strip(),
            run_name + "_bootstrap_001",
        )
        if bootstrap_name == run_name:
            raise ValueError("Tên bootstrap phải khác tên run.")
        bootstrap_dir = _artifact_run_dir(bootstrap_name)
        if bootstrap_dir.exists() and any(bootstrap_dir.iterdir()):
            bootstrap_name = bootstrap_name[:40] + "_" + uuid.uuid4().hex[:8]
            bootstrap_dir = _artifact_run_dir(bootstrap_name)
        import bootstrap_review
        bootstrap_review.preflight(run_dir)
        confidence = float(payload.get("confidence", 0.90))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence phải trong khoảng 0..1.")
        command = [
            sys.executable,
            str(PROJECT_DIR / "bootstrap_review.py"),
            "--run_dir", str(run_dir),
            "--out_dir", str(bootstrap_dir),
            "--confidence", str(confidence),
            "--min_line_ratio", "0.005",
            "--train_final",
            "--device", "cpu",
        ]
        snapshot = _start_pipeline_command(
            "bootstrap_train", command, run_name, bootstrap_name,
            bootstrap_dir / "bootstrap_summary.json",
        )
        return jsonify({"ok": True, **snapshot})
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except (ValueError, FileExistsError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409 if isinstance(exc, (FileExistsError, RuntimeError)) else 400


@app.route("/api/pipeline/review/update", methods=["POST"])
def api_pipeline_review_update():
    payload = request.get_json(silent=True) or {}
    try:
        run_name = _safe_pipeline_name(str(payload.get("run_name") or "").strip(), "")
        review_id = str(payload.get("review_id") or "").strip()
        status = str(payload.get("status") or "pending").strip().lower()
        notes = str(payload.get("notes") or "")[:1000]
        if not review_id:
            raise ValueError("Thiếu review_id.")
        if status not in ("pending", "approved", "rejected"):
            raise ValueError("Trạng thái phải là pending, approved hoặc rejected.")
        with _pipeline_lock:
            _, review_csv, rows = _read_review_rows(run_name)
            matches = [row for row in rows if str(row.get("review_id", "")) == review_id]
            if len(matches) != 1:
                raise ValueError("Không tìm thấy review_id hoặc review_id bị trùng.")
            matches[0]["review_status"] = status
            matches[0]["review_notes"] = notes
            _write_review_rows(review_csv, rows)
        return jsonify({"ok": True, "review_id": review_id, "review_status": status})
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/dataset/status")
def api_dataset_status():
    return jsonify({
        "ok": True,
        "image_count": dataset_image_count(),
        "dataset_dir": _project_relative(DATASET_IMAGES_DIR),
    })


@app.route("/api/dataset/import-archive/init", methods=["POST"])
def api_dataset_import_archive_init():
    data = request.get_json(silent=True) or {}
    try:
        meta = _new_archive_upload(data.get("filename"), data.get("size"))
    except (ValueError, OSError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({
        "ok": True,
        "upload_id": meta["upload_id"],
        "filename": meta["filename"],
        "size": meta["size"],
        "received": 0,
        "chunk_size": ARCHIVE_UPLOAD_CHUNK_BYTES,
        "max_size": MAX_ARCHIVE_UPLOAD_BYTES,
    }), 201


@app.route("/api/dataset/import-archive/chunk", methods=["POST"])
def api_dataset_import_archive_chunk():
    upload_id = request.headers.get("X-Archive-Upload-ID", "")
    try:
        upload_id = _archive_upload_id(upload_id)
        offset = int(request.headers.get("X-Archive-Upload-Offset", ""))
        if offset < 0:
            raise ValueError("Offset upload không hợp lệ.")
        if request.content_length is not None and request.content_length > ARCHIVE_UPLOAD_CHUNK_BYTES:
            return jsonify({
                "ok": False,
                "error": f"Mỗi phần upload tối đa {_format_bytes(ARCHIVE_UPLOAD_CHUNK_BYTES)}."
            }), 413
        data = request.get_data(cache=False)
        if not data:
            raise ValueError("Chunk upload rỗng.")
        if len(data) > ARCHIVE_UPLOAD_CHUNK_BYTES:
            return jsonify({
                "ok": False,
                "error": f"Mỗi phần upload tối đa {_format_bytes(ARCHIVE_UPLOAD_CHUNK_BYTES)}."
            }), 413
        part_path, meta_path = _archive_upload_paths(upload_id)
        with _archive_upload_lock:
            meta = _read_archive_upload_meta(upload_id)
            expected = int(meta.get("received", 0))
            disk_received = part_path.stat().st_size if part_path.exists() else 0
            if disk_received != expected:
                raise ValueError("File tạm không đồng bộ; hãy hủy phiên và chọn lại archive.")
            if offset != expected:
                return jsonify({
                    "ok": False,
                    "error": f"Chunk sai vị trí; server đang chờ offset {expected}.",
                    "expected_offset": expected,
                }), 409
            total = expected + len(data)
            if total > int(meta["size"]):
                raise ValueError("Tổng dữ liệu vượt kích thước archive đã khai báo.")
            with part_path.open("ab") as handle:
                handle.write(data)
            meta["received"] = total
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return jsonify({
            "ok": True,
            "upload_id": upload_id,
            "received": total,
            "size": int(meta["size"]),
            "complete": total == int(meta["size"]),
        })
    except (ValueError, FileNotFoundError, OSError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/dataset/import-archive/complete", methods=["POST"])
def api_dataset_import_archive_complete():
    data = request.get_json(silent=True) or {}
    try:
        upload_id = _archive_upload_id(data.get("upload_id"))
        meta = _read_archive_upload_meta(upload_id)
        part_path, meta_path = _archive_upload_paths(upload_id)
        received = int(meta.get("received", 0))
        expected_size = int(meta.get("size", 0))
        actual_size = part_path.stat().st_size if part_path.exists() else 0
        if received != expected_size or actual_size != expected_size:
            return jsonify({
                "ok": False,
                "error": (
                    f"Upload chưa hoàn tất ({_format_bytes(actual_size)}/"
                    f"{_format_bytes(expected_size)})."
                ),
                "received": actual_size,
                "size": expected_size,
            }), 409
        archive_path = part_path.with_name(f"{upload_id}{meta['suffix']}")
        with _archive_upload_lock:
            part_path.replace(archive_path)
            meta["status"] = "queued"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            job_id = uuid.uuid4().hex
            _archive_jobs[job_id] = {
                "job_id": job_id,
                "upload_id": upload_id,
                "filename": meta["filename"],
                "status": "queued",
                "phase": "queued",
                "progress": 0.0,
                "message": "Đang chờ lượt giải nén...",
                "started_at": "",
                "finished_at": "",
                "finished_epoch": 0,
                "error": "",
                "result": {},
            }
        thread = threading.Thread(
            target=_run_archive_import_job,
            args=(job_id, upload_id, archive_path, meta["filename"]),
            daemon=True,
            name=f"palmistry-archive-{job_id[:8]}",
        )
        thread.start()
        return jsonify({
            "ok": True,
            "status": "queued",
            "job_id": job_id,
            "status_url": f"/api/dataset/import-archive/status?job_id={quote(job_id)}",
        }), 202
    except (ValueError, FileNotFoundError, OSError, KeyError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/dataset/import-archive/status")
def api_dataset_import_archive_status():
    _prune_archive_jobs()
    job_id = request.args.get("job_id", "")
    snapshot = _archive_job_snapshot(job_id)
    if snapshot is None:
        return jsonify({"ok": False, "error": "Job import không tồn tại hoặc đã hết hạn."}), 404
    # HTTP success means the status request succeeded; job completion is separate.
    snapshot["ok"] = True
    return jsonify(snapshot)


@app.route("/api/dataset/import-archive/cancel", methods=["POST"])
def api_dataset_import_archive_cancel():
    try:
        upload_id = _archive_upload_id((request.get_json(silent=True) or {}).get("upload_id"))
        part_path, meta_path = _archive_upload_paths(upload_id)
        with _archive_upload_lock:
            part_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        return jsonify({"ok": True})
    except (ValueError, OSError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/dataset/import-archive", methods=["POST"])
@app.route("/api/dataset/import-zip", methods=["POST"])
def api_dataset_import_archive():
    # Compatibility endpoint for scripts; the UI uses the chunked endpoints above.
    uploaded = request.files.get("zip_file") or request.files.get("archive_file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"ok": False, "error": "Hãy chọn một file ZIP hoặc RAR."}), 400
    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in {".zip", ".rar"}:
        return jsonify({"ok": False, "error": "Chỉ chấp nhận file .zip hoặc .rar."}), 400
    try:
        report = import_archive_to_dataset(uploaded.stream, uploaded.filename)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify(report)


# Keep the old function name as a compatibility alias for integrations.
api_dataset_import_zip = api_dataset_import_archive


@app.route("/api/status")
def api_status():
    return jsonify(engine.payload())


@app.get("/api/health")
def api_health():
    return jsonify(ok=True, service="palmistry", project_root=str(PROJECT_DIR.resolve()))


@app.route("/api/reset", methods=["POST"])
def api_reset():
    engine.reset()
    return jsonify({"ok": True})


@app.route("/api/lock", methods=["POST"])
def api_lock():
    ok = engine.lock_now()
    return jsonify({"ok": ok}), 200 if ok else 409


@app.route("/api/camera/<int:index>", methods=["POST"])
def api_camera(index):
    if not 0 <= index <= 16:
        abort(400)
    engine.set_camera(index)
    return jsonify({"ok": True, "camera": index})


@app.route("/static_generated/<path:name>")
def static_generated(name):
    return send_from_directory(STATIC_DIR, name)


def start_ngrok():
    if USE_NGROK:
        raise RuntimeError("Network mode is deliberately unavailable; loopback only.")

def main():
    _restore_pipeline_state()
    print("=" * 80)
    print("PALMISTRY LIVE CAMERA WEB GUI")
    print("=" * 80)
    print("Project dir:", PROJECT_DIR)
    print("Code predict:", CODE_PATH)
    print("Checkpoint:", engine.checkpoint)
    print("Webcam index:", WEBCAM_INDEX)
    print("USB camera index:", USB_CAMERA_INDEX)
    print("Localhost:", f"http://127.0.0.1:{PORT}")
    print("Display FPS:", DISPLAY_FPS)
    print("Predict interval:", PREDICT_EVERY_SECONDS)
    print("=" * 80)

    start_ngrok()
    engine.start()
    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
    finally:
        if not engine.shutdown():
            print("Warning: camera backend did not stop within timeout; hardware shutdown remains unverified.")


if __name__ == "__main__":
    main()
