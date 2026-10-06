"""Shared contracts for the reproducible computer-vision prototype."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import subprocess
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
CLASS_DOCUMENT = json.loads((ROOT / "class_map.v2.json").read_text(encoding="utf-8"))
CLASS_VERSION = CLASS_DOCUMENT["version"]
CLASS_MAP = {int(k): v for k, v in CLASS_DOCUMENT["classes"].items()}
MODEL_VERSION = "numpy-diagonal-gaussian-pixel-v2"
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 16_000_000


def validate_size(size):
    if not isinstance(size, int) or size < 32 or size > 2048 or size % 16:
        raise ValueError("Image size must be 32..2048 and divisible by 16 (e.g. 128, 256, 512).")
    return size


def validate_mask(mask, shape=None):
    if mask is None or mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Semantic mask must be a single-channel integer array.")
    if mask.size == 0 or (shape is not None and mask.shape != tuple(shape)):
        raise ValueError("Mask shape does not match the image.")
    if not np.isin(mask, list(CLASS_MAP)).all():
        raise ValueError(f"Unknown semantic class IDs: {np.unique(mask).tolist()}")
    return mask


def read_rgb(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Image exceeds 20 MiB compressed limit.")
    with Image.open(path) as im:
        w, h = im.size
        if w * h > MAX_PIXELS or max(w, h) > 8192:
            raise ValueError("Image exceeds 16 megapixels or 8192 pixels per dimension.")
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()


def decode_archive_image(raw, max_pixels):
    """Check the encoded dimensions before OpenCV allocates the pixel buffer."""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > max_pixels:
                raise ValueError("image_pixels_limit")
    except Image.DecompressionBombError as exc:
        raise ValueError("image_pixels_limit") from exc
    except (OSError, SyntaxError) as exc:
        raise ValueError("invalid_image") from exc
    try:
        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("invalid_image") from exc
    if decoded is None:
        raise ValueError("invalid_image")
    if decoded.shape[0] * decoded.shape[1] > max_pixels:
        raise ValueError("image_pixels_limit")
    return decoded


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_npz_size(path, max_bytes):
    """Bound both the stored archive and expanded arrays before np.load."""
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("Checkpoint exceeds size limit.")
    try:
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > max_bytes:
                raise ValueError("Expanded checkpoint exceeds size limit.")
    except zipfile.BadZipFile as exc:
        raise ValueError("Checkpoint is not a valid NumPy NPZ archive.") from exc


def write_json(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_class_map(root):
    write_json(Path(root) / "class_map.json", CLASS_DOCUMENT)


def check_class_map(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("version") != CLASS_VERSION or doc.get("classes") != CLASS_DOCUMENT["classes"]:
        raise ValueError("Incompatible/unversioned class map. V2 adds simian (6); prepare a new v2 review run or explicitly migrate a copy. Legacy fate-v0 data still requires explicit conversion.")


def perceptual_hash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    coeff = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8].ravel()[1:]
    bits = coeff > np.median(coeff)
    return format(sum(int(v) << i for i, v in enumerate(bits)), "016x")


def group_rows(rows, seed=42, assign=True):
    """Conservatively union subject, hand/source, exact and pHash neighbours.

    Folder fallback cannot establish subject independence. Seven exact hash bands
    shortlist Hamming-radius-6 neighbours without an artificial sample cap.
    """
    if not rows:
        raise ValueError("Empty dataset: supply local labeled images; no training fallback is permitted.")
    rows = [dict(r) for r in sorted(rows, key=lambda r: str(r.get("source_path", r["image_path"])))]
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        parents[find(b)] = find(a)

    seen = {}
    hash_representatives = {}
    bands = [{} for _ in range(7)]
    for i, row in enumerate(rows):
        source = str(row.get("source_group") or Path(row.get("source_path", row["image_path"])).parent)
        row["source_group"] = source
        if not row.get("phash") or not row.get("sha256"):
            raise ValueError("Every row needs phash and sha256 for leakage auditing.")
        keys = [("source", source), ("sha256", row["sha256"])]
        if row.get("subject_id"):
            keys.append(("subject", str(row["subject_id"])))
            keys.append(("hand", str(row["subject_id"]) + ":" + str(row.get("hand_id", "unknown"))))
        for key in keys:
            if key in seen:
                union(i, seen[key])
            seen[key] = i
        value = int(row["phash"], 16)
        if value in hash_representatives:
            union(i, hash_representatives[value])
        else:
            candidates = set()
            for band in range(7):
                candidates.update(bands[band].get((value >> (9 * band)) & 511, ()))
            for previous in candidates:
                if (value ^ previous).bit_count() <= 6:
                    union(i, hash_representatives[previous])
            hash_representatives[value] = i
            for band in range(7):
                bands[band].setdefault((value >> (9 * band)) & 511, set()).add(value)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    grouped = list(groups.values())
    if len(grouped) < 3:
        raise ValueError(f"Need at least 3 independent source/perceptual groups; found {len(grouped)}. Do not split duplicates to force validation.")
    random.Random(seed).shuffle(grouped)
    n_test = max(1, len(grouped) // 10)
    n_val = max(1, len(grouped) // 10)
    for i, group in enumerate(grouped):
        split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        identity = hashlib.sha256("\n".join(str(r["image_path"]) for r in group).encode()).hexdigest()[:16]
        if not assign and len({r.get("split") for r in group}) != 1:
            raise ValueError("Group leakage across splits (subject/source/exact/perceptual duplicate).")
        for row in group:
            row["group_id"] = identity
            row["split_basis"] = "subject+source+perceptual" if row.get("subject_id") else "source-folder+perceptual; not subject-independent"
            if assign:
                row["split"] = split
    if {r.get("split") for r in rows} != {"train", "val", "test"}:
        raise ValueError("Non-empty train, val, and test splits are required.")
    return rows


def split_summary(rows):
    return {s: {"samples": sum(r["split"] == s for r in rows),
                "groups": len({r["group_id"] for r in rows if r["split"] == s})}
            for s in ("train", "val", "test")}


def read_manifest(root, csv_path):
    root = Path(root).resolve()
    check_class_map(root / "class_map.json")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty manifest: supply labeled images; no training fallback is permitted.")

    # Keep the supplied partition and group provenance, while independently
    # checking identity and actual image bytes. CSV metadata is not an audit.
    explicit_audit = all(
        str(row.get("group_id") or "").strip() and str(row.get("split_basis") or "").strip()
        for row in rows
    )
    groups_to_split = {}
    audit_keys_to_split = {}
    seen_images = set()
    for row in rows:
        for key in ("image_path", "mask_path"):
            if not row.get(key):
                raise ValueError(f"Manifest thiếu {key}.")
            path = (root / row[key]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Manifest path escapes dataset root.")
        image_path = root / row["image_path"]
        rgb = read_rgb(image_path)
        validate_mask(cv2.imread(str(root / row["mask_path"]), cv2.IMREAD_UNCHANGED), rgb.shape[:2])
        image_key = str(image_path.resolve())
        if image_key in seen_images:
            raise ValueError(f"Duplicate image_path in manifest: {row['image_path']}")
        seen_images.add(image_key)
        # Audit actual training bytes, not untrusted hash fields in a CSV.
        row["sha256"] = sha256(image_path)
        row["phash"] = perceptual_hash(rgb)
        if explicit_audit:
            split = str(row.get("split") or "").strip()
            group_id = str(row.get("group_id") or "").strip()
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid split in audited manifest: {split!r}")
            previous = groups_to_split.setdefault(group_id, split)
            if previous != split:
                raise ValueError("Group leakage across explicit manifest splits.")
            source_group = str(row.get("source_group") or "").strip()
            subject_id = str(row.get("subject_id") or "").strip()
            for key_name, key_value in (("source", source_group), ("sha256", row["sha256"]), ("subject", subject_id)):
                if not key_value:
                    continue
                audit_key = (key_name, key_value)
                previous = audit_keys_to_split.setdefault(audit_key, split)
                if previous != split:
                    raise ValueError(f"Leakage across explicit manifest splits for {key_name}.")

    if explicit_audit:
        if set(str(row.get("split") or "").strip() for row in rows) != {"train", "val", "test"}:
            raise ValueError("Non-empty train, val, and test splits are required.")
        if len(groups_to_split) < 3:
            raise ValueError("Need at least 3 independent source/perceptual groups.")
        # A supplied group_id is provenance, not proof of independence. Audit
        # the actual encoded images too, including cross-split near duplicates.
        group_rows(rows, assign=False)
        return rows
    return group_rows(rows, assign=False)


def confusion_matrix(pred, target):
    validate_mask(target)
    validate_mask(pred, target.shape)
    classes = len(CLASS_MAP)
    return np.bincount((target.astype(np.int64) * classes + pred).ravel(), minlength=classes*classes).reshape(classes, classes)


def metrics_from_confusion(cm):
    result = {}
    for c, name in CLASS_MAP.items():
        tp = int(cm[c, c]); actual = int(cm[c].sum()); predicted = int(cm[:, c].sum())
        union = actual + predicted - tp
        result[name] = {
            "iou": tp / union if union else None,
            "dice": 2 * tp / (actual + predicted) if actual + predicted else None,
            "precision": tp / predicted if predicted else None,
            "recall": tp / actual if actual else None,
            "f1": 2 * tp / (actual + predicted) if actual + predicted else None,
            "support_pixels": actual,
        }
    present = [r["iou"] for r in result.values() if r["iou"] is not None]
    return {"per_class": result, "mean_iou": float(np.mean(present)) if present else None,
            "confusion_matrix": cm.tolist(), "pixel_count": int(cm.sum()),
            "absent_class_policy": "undefined denominators are null; macro IoU includes nonempty unions"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

def provenance(args, manifest=None):
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = "unavailable", None
    return {"config": vars(args), "seed": getattr(args, "seed", 42), "model_version": MODEL_VERSION,
            "class_map_version": CLASS_VERSION, "git_revision": revision, "git_dirty": dirty,
            "manifest_sha256": sha256(manifest) if manifest else None,
            "source_sha256": {p.name: sha256(p) for p in sorted(ROOT.glob("*.py"))},
            "numpy": np.__version__, "opencv": cv2.__version__,
            "score_semantics": "uncalibrated heuristic ranking; not probability"}


def load_config(path=None):
    config = json.loads((Path(path) if path else ROOT / "prototype_config.json").read_text(encoding="utf-8"))
    if config.get("retention_seconds") != 0:
        raise ValueError("Only immediate temporary-frame cleanup (retention_seconds=0) is supported.")
    for key in ("prediction_interval_seconds", "prediction_timeout_seconds"):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive and finite.")
    return config
