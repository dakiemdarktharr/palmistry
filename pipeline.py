from __future__ import annotations

import argparse
import os
import csv
import html
import json
import re
import shutil
import zipfile
try:
    import rarfile
except Exception:
    rarfile = None
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import palmistry_strict_auto_onefile as core
from prototype_common import decode_archive_image, write_json

def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _format_bytes(value: int) -> str:
    value = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


MAX_ARCHIVE_BYTES = _env_int("PALM_MAX_ARCHIVE_BYTES", 8 * 1024 * 1024 * 1024)
MAX_ZIP_BYTES = MAX_ARCHIVE_BYTES
MAX_ARCHIVE_UNCOMPRESSED_BYTES = _env_int("PALM_MAX_ARCHIVE_EXPANDED_BYTES", 32 * 1024 * 1024 * 1024)
MAX_ZIP_UNCOMPRESSED_BYTES = MAX_ARCHIVE_UNCOMPRESSED_BYTES
MAX_ARCHIVE_MEMBER_BYTES = _env_int("PALM_MAX_ARCHIVE_MEMBER_BYTES", 100 * 1024 * 1024)
MAX_ZIP_MEMBER_BYTES = MAX_ARCHIVE_MEMBER_BYTES
MAX_ARCHIVE_MEMBERS = _env_int("PALM_MAX_ARCHIVE_MEMBERS", 20000)
MAX_ZIP_MEMBERS = MAX_ARCHIVE_MEMBERS
MAX_ARCHIVE_IMAGE_PIXELS = _env_int("PALM_MAX_ARCHIVE_IMAGE_PIXELS", 50_000_000)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_member_name(name: str) -> Optional[str]:
    normalized = str(name or "").replace("\\", "/")
    if not normalized or normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        return None
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    return "/".join(parts)


def safe_component(name: str, fallback: str = "image") -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or ""))
    value = value.strip("._-")[:80]
    return value or fallback


def output_path(raw_dir: Path, member_name: str, index: int) -> Path:
    parts = member_name.split("/")
    parents = [safe_component(part, "group") for part in parts[:-1]]
    folder = raw_dir.joinpath(*parents) if parents else raw_dir
    folder.mkdir(parents=True, exist_ok=True)
    stem = safe_component(Path(parts[-1]).stem)
    suffix = Path(parts[-1]).suffix.lower()
    destination = folder / f"{stem}_{index:06d}{suffix}"
    if destination.exists():
        destination = folder / f"{stem}_{index:06d}_{core.sha256_file(destination)[:8]}{suffix}"
    return destination


def _read_archive_member(archive, info):
    opener = getattr(archive, "open", None)
    if not callable(opener):
        try:
            raw = archive.read(info)
        except Exception:
            return None, "unreadable_member"
        return (None, "member_size_limit") if len(raw) > MAX_ARCHIVE_MEMBER_BYTES else (raw, None)
    try:
        source = opener(info)
    except Exception:
        return None, "unreadable_member"
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
        return None, "unreadable_member"
    finally:
        try:
            source.close()
        except Exception:
            pass
    return b"".join(chunks), None


def collect_zip(zip_path: Path, raw_dir: Path, max_images: Optional[int]) -> Tuple[List[Path], Dict[str, Any]]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_path.resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP không tồn tại: {zip_path}")
    if zip_path.stat().st_size > MAX_ZIP_BYTES:
        raise ValueError(f"ZIP vượt quá giới hạn {_format_bytes(MAX_ZIP_BYTES)}.")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("File đầu vào không phải ZIP hợp lệ.")

    paths: List[Path] = []
    skipped: List[Dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_ZIP_MEMBERS:
            raise ValueError(f"ZIP có quá nhiều mục (tối đa {MAX_ZIP_MEMBERS}).")
        expanded = sum(max(0, int(info.file_size)) for info in infos)
        if expanded > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise ValueError(f"Tổng dung lượng giải nén vượt quá giới hạn {_format_bytes(MAX_ZIP_UNCOMPRESSED_BYTES)}.")
        reserve = min(MAX_ZIP_UNCOMPRESSED_BYTES, expanded) + 256 * 1024 * 1024
        free_bytes = shutil.disk_usage(raw_dir).free
        if free_bytes < reserve:
            raise ValueError(
                f"Ổ đĩa chỉ còn {_format_bytes(free_bytes)} nhưng cần khoảng "
                f"{_format_bytes(reserve)} cho dữ liệu giải nén."
            )

        for info in infos:
            member = safe_member_name(info.filename)
            if info.is_dir() or member is None:
                skipped.append({"name": info.filename, "reason": "unsafe_path_or_directory"})
                continue
            suffix = Path(member).suffix.lower()
            if suffix not in core.IMAGE_EXTS:
                skipped.append({"name": info.filename, "reason": "unsupported_extension"})
                continue
            if info.file_size < 1 or info.file_size > MAX_ZIP_MEMBER_BYTES:
                skipped.append({"name": info.filename, "reason": "member_size_limit"})
                continue
            raw, read_error = _read_archive_member(archive, info)
            if read_error:
                skipped.append({"name": info.filename, "reason": read_error})
                continue
            try:
                decoded = decode_archive_image(raw, MAX_ARCHIVE_IMAGE_PIXELS)
            except ValueError as exc:
                skipped.append({"name": info.filename, "reason": str(exc)})
                continue
            if max_images is not None and len(paths) >= max_images:
                skipped.append({"name": info.filename, "reason": "max_images"})
                continue
            destination = output_path(raw_dir, member, len(paths))
            destination.write_bytes(raw)
            paths.append(destination)

    return paths, {
        "source_type": "zip",
        "source": str(zip_path),
        "archive_entries": len(infos),
        "selected_images": len(paths),
        "skipped_entries": len(skipped),
        "skipped": skipped[:200],
        "source_groups": len({str(path.parent) for path in paths}),
    }


def _archive_info_is_dir(info):
    checker = getattr(info, "isdir", None)
    if callable(checker):
        return bool(checker())
    checker = getattr(info, "is_dir", False)
    return bool(checker() if callable(checker) else checker)


def collect_rar(rar_path: Path, raw_dir: Path, max_images: Optional[int]) -> Tuple[List[Path], Dict[str, Any]]:
    """Read a RAR archive through rarfile and the installed RAR backend."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rar_path = rar_path.resolve()
    if not rar_path.is_file():
        raise FileNotFoundError(f"RAR không tồn tại: {rar_path}")
    if rar_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError(f"RAR vượt quá giới hạn {_format_bytes(MAX_ARCHIVE_BYTES)}.")
    if rarfile is None:
        raise ValueError("Chưa có hỗ trợ RAR. Hãy cài rarfile và 7-Zip/unrar/bsdtar.")
    if not rarfile.is_rarfile(rar_path):
        raise ValueError("File đầu vào không phải RAR hợp lệ.")
    try:
        if not rarfile.tool_setup().check():
            raise RuntimeError("RAR backend unavailable")
    except Exception as exc:
        raise ValueError("Không có backend giải nén RAR. Hãy cài 7-Zip, unrar hoặc bsdtar rồi thử lại.") from exc

    paths: List[Path] = []
    skipped: List[Dict[str, str]] = []
    try:
        archive = rarfile.RarFile(rar_path)
    except Exception as exc:
        raise ValueError("Không mở được RAR. Hãy cài 7-Zip, unrar hoặc bsdtar rồi thử lại.") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"RAR có quá nhiều mục (tối đa {MAX_ARCHIVE_MEMBERS}).")
        expanded = sum(max(0, int(getattr(info, "file_size", 0) or 0)) for info in infos)
        if expanded > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ValueError(f"Tổng dung lượng giải nén của RAR vượt quá giới hạn {_format_bytes(MAX_ARCHIVE_UNCOMPRESSED_BYTES)}.")
        reserve = min(MAX_ARCHIVE_UNCOMPRESSED_BYTES, expanded) + 256 * 1024 * 1024
        free_bytes = shutil.disk_usage(raw_dir).free
        if free_bytes < reserve:
            raise ValueError(
                f"Ổ đĩa chỉ còn {_format_bytes(free_bytes)} nhưng cần khoảng "
                f"{_format_bytes(reserve)} cho dữ liệu giải nén."
            )
        for info in infos:
            member = safe_member_name(getattr(info, "filename", ""))
            if _archive_info_is_dir(info) or member is None:
                skipped.append({"name": str(getattr(info, "filename", "")), "reason": "unsafe_path_or_directory"})
                continue
            suffix = Path(member).suffix.lower()
            if suffix not in core.IMAGE_EXTS:
                skipped.append({"name": member, "reason": "unsupported_extension"})
                continue
            member_size = int(getattr(info, "file_size", 0) or 0)
            if member_size < 1 or member_size > MAX_ARCHIVE_MEMBER_BYTES:
                skipped.append({"name": member, "reason": "member_size_limit"})
                continue
            raw, read_error = _read_archive_member(archive, info)
            if read_error:
                skipped.append({"name": member, "reason": read_error})
                continue
            try:
                decoded = decode_archive_image(raw, MAX_ARCHIVE_IMAGE_PIXELS)
            except ValueError as exc:
                skipped.append({"name": member, "reason": str(exc)})
                continue
            destination = output_path(raw_dir, member, len(paths))
            destination.write_bytes(raw)
            paths.append(destination)
            if max_images is not None and len(paths) >= max_images:
                break
    return paths, {
        "source_type": "rar",
        "source": str(rar_path),
        "archive_entries": len(infos),
        "selected_images": len(paths),
        "skipped_entries": len(skipped),
        "skipped": skipped[:200],
        "source_groups": len({str(path.parent) for path in paths}),
        "backend": "rarfile + external RAR backend",
    }

def collect_directory(input_dir: Path, raw_dir: Path, max_images: Optional[int]) -> Tuple[List[Path], Dict[str, Any]]:
    """Index local originals without copying them or applying archive entry limits."""
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Thư mục ảnh không tồn tại: {input_dir}")
    paths, skipped, seen = [], [], {}
    source_paths = core.list_images(input_dir)
    for index, source in enumerate(source_paths, 1):
        try:
            if source.is_symlink() or not source.resolve().is_relative_to(input_dir):
                raise ValueError("unsafe_source_path")
            if source.stat().st_size > MAX_ARCHIVE_MEMBER_BYTES:
                raise ValueError("member_size_limit")
            digest = core.sha256_file(source)
            if digest in seen:
                skipped.append({"name": str(source), "reason": "exact_duplicate", "original": seen[digest]})
                continue
            seen[digest] = str(source)
            paths.append(source)
        except (OSError, ValueError) as exc:
            skipped.append({"name": str(source), "reason": str(exc)})
        finally:
            if index % 500 == 0 or index == len(source_paths):
                print(f"Kiểm kê {index}/{len(source_paths)} file; {len(paths)} ảnh duy nhất", flush=True)
        if max_images is not None and len(paths) >= max_images:
            break
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(raw_dir / "sources.json", {"source_root": str(input_dir), "paths": [str(p) for p in paths]})
    core.write_csv(raw_dir / "skipped.csv", skipped)
    return paths, {
        "source_type": "directory", "source": str(input_dir),
        "source_files": len(source_paths), "selected_images": len(paths),
        "skipped_entries": len(skipped), "skipped": skipped[:200],
        "duplicates": sum(r["reason"] == "exact_duplicate" for r in skipped),
        "storage_mode": "reference_originals", "source_groups": len({str(p.parent) for p in paths}),
    }


def prepare_manual_candidates(paths, data_dir, args):
    """Build only the requested human-review seed; keep remaining originals indexed."""
    import random
    shuffled = list(paths)
    random.Random(args.seed).shuffle(shuffled)
    rows, failures = [], []
    data_dir.mkdir(parents=True, exist_ok=True)
    core.write_class_map(data_dir)
    for source in shuffled:
        if len(rows) >= args.review_count:
            break
        try:
            if shutil.disk_usage(data_dir).free < 256 * 1024 * 1024:
                raise OSError("Ổ đĩa còn dưới 256 MB. Giải phóng dung lượng rồi thử lại với queue nhỏ hơn.")
            image = core.imread_rgb(source)
            if image is None:
                raise ValueError("Không đọc được ảnh; kiểm tra giới hạn 20 MiB/16 megapixel.")
            diag = core.analyze_quality(image, min_short_side=args.min_short_side)
            crop, mask, report = core.create_strict_pseudo_mask(image, out_size=args.out_size,
                                                             strict_line_threshold=args.strict_line_threshold)
            core.validate_mask(mask, crop.shape[:2])
            stem = core.sha256_file(source)
            image_rel, mask_rel, overlay_rel = f"images/{stem}.jpg", f"masks/{stem}.png", f"overlays/{stem}.jpg"
            if not core.imwrite_rgb(data_dir / image_rel, crop):
                raise OSError("Không ghi được crop ảnh.")
            if not core.imwrite_gray(data_dir / mask_rel, mask):
                raise OSError("Không ghi được mask.")
            if not core.imwrite_rgb(data_dir / overlay_rel, core.make_overlay(crop, mask)):
                raise OSError("Không ghi được overlay.")
            rows.append({
                "source_path": str(source), "source_group": str(source.parent), "sha256": stem,
                "phash": core.perceptual_hash(image), "image_path": image_rel, "mask_path": mask_rel,
                "overlay_path": overlay_rel, "mask_quality_score": report["mask_quality_score"],
                "main_line_count": report["main_line_count"], "line_pixel_ratio": report["line_pixel_ratio"],
                "quality_warning": diag.main_reason if diag.status != "ok" else "",
                "accepted": 0, "review_status": "pending", "split": "unassigned",
                "label_provenance": "unreviewed candidate; must be approved before training",
            })
            if len(rows) % 10 == 0:
                print(f"Đã chuẩn bị {len(rows)}/{args.review_count} mask chờ duyệt", flush=True)
        except OSError:
            raise
        except Exception as exc:
            failures.append({"source_path": str(source), "error": str(exc)})
    core.write_csv(data_dir / "labels_pseudo_strict.csv", rows)
    core.write_csv(data_dir / "review_failures.csv", failures)
    selected = {row["source_path"] for row in rows}
    write_json(data_dir / "remaining_sources.json", {
        "paths": [str(p) for p in paths if str(p) not in selected],
        "source_root": str(Path(args.input_dir).resolve()) if args.input_dir else str(Path(args.run_dir).resolve()),
        "out_size": args.out_size,
    })
    if not rows:
        raise ValueError("Không tạo được mask nào. Xem review_failures.csv; thử ảnh RGB/JPEG rõ hơn.")
    return {"total_input": len(paths), "review_candidates": len(rows), "accepted": 0,
            "remaining": len(paths) - len(rows), "failed_candidates": len(failures),
            "split_status": "deferred_until_human_review", "label_provenance": "human review required"}


def test_model(data_root: Path, labels_csv: Path, checkpoint: Path, out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    digest = core.sha256_file(checkpoint)
    model, checkpoint_input_size, _ = core.load_model_for_predict(checkpoint, None, digest)
    rows = [row for row in core.read_manifest(data_root, labels_csv) if row["split"] == "test"]
    if not rows:
        raise ValueError("Test split rỗng; cần ít nhất ba nhóm độc lập để chia train/val/test.")

    cm = np.zeros((len(core.CLASS_MAP), len(core.CLASS_MAP)), dtype=np.int64)
    prediction_dir = out_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        image = core.imread_rgb(data_root / str(row["image_path"]))
        target = cv2.imread(str(data_root / str(row["mask_path"])), cv2.IMREAD_UNCHANGED)
        if image is None or target is None:
            raise RuntimeError(f"Không đọc được dòng test: {row}")
        core.validate_mask(target, image.shape[:2])
        prediction, _ = model.predict_mask(image)
        prediction = cv2.resize(prediction, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST)
        cm += core.confusion_matrix(prediction, target)
        stem = safe_component(Path(str(row["image_path"])).stem, f"test_{index:04d}")
        core.imwrite_gray(prediction_dir / f"{stem}_mask.png", prediction)
        core.imwrite_rgb(prediction_dir / f"{stem}_overlay.jpg", core.make_overlay(image, prediction))

    result = {
        "split": "test",
        "sample_count": len(rows),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_input_size": checkpoint_input_size,
        "input_size": getattr(args, "input_size", checkpoint_input_size),
        "framework": core.MODEL_FRAMEWORK,
        "label_provenance": "classical CV pseudo-labels; not independent ground truth",
        **core.metrics_from_confusion(cm),
    }
    write_json(out_dir / "test_metrics.json", result)
    return result

def train_args(args: argparse.Namespace, data_dir: Path, model_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        data_root=str(data_dir),
        labels_csv=str(data_dir / "labels_pseudo_strict.csv"),
        out_dir=str(model_dir),
        input_size=args.input_size,
        model_size=args.model_size,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        max_train_hours=args.max_train_hours,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        dice_weight=args.dice_weight,
        grad_clip=args.grad_clip,
        patience=args.patience,
        resume="none",
        checkpoint_sha256="",
        device=args.device,
        num_workers=args.num_workers,
        amp=args.amp,
        cpu=args.device == "cpu",
    )


def prepare_review_queue(data_dir: Path, review_dir: Path, count: int) -> Dict[str, Any]:
    if count <= 0:
        raise ValueError("review_count phải dương.")
    labels_path = data_dir / "labels_pseudo_strict.csv"
    with labels_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = rows[:count]
    review_dir.mkdir(parents=True, exist_ok=True)
    review_rows = []
    for index, row in enumerate(selected, start=1):
        mask = cv2.imread(str(data_dir / row["mask_path"]), cv2.IMREAD_UNCHANGED)
        length_metrics = {}
        for class_id, class_name in ((2, "life_line"), (3, "head_line"), (4, "heart_line")):
            feature = core.line_feature_from_mask(mask, class_id) if mask is not None else {}
            length_metrics[f"length_{class_name}_norm"] = feature.get("length_norm", 0.0)
            length_metrics[f"length_{class_name}_px"] = feature.get("length_px", 0.0)
            length_metrics[f"component_{class_name}_count"] = feature.get("component_count", 0)
        review_rows.append({
            "review_id": f"review_{index:04d}",
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "overlay_path": row.get("overlay_path", ""),
            "mask_quality_score": row.get("mask_quality_score", ""),
            "main_line_count": row.get("main_line_count", ""),
            "line_pixel_ratio": row.get("line_pixel_ratio", ""),
            "score_life_line": row.get("score_life_line", ""),
            "score_head_line": row.get("score_head_line", ""),
            "score_heart_line": row.get("score_heart_line", ""),
            **length_metrics,
            "review_status": "pending",
            "review_notes": row.get("quality_warning", ""),
        })
    review_csv = review_dir / "review.csv"
    fields = [
        "review_id", "image_path", "mask_path", "overlay_path",
        "mask_quality_score", "main_line_count", "line_pixel_ratio",
        "score_life_line", "score_head_line", "score_heart_line",
        "length_life_line_norm", "length_life_line_px", "component_life_line_count",
        "length_head_line_norm", "length_head_line_px", "component_head_line_count",
        "length_heart_line_norm", "length_heart_line_px", "component_heart_line_count",
        "review_status", "review_notes",
    ]
    with review_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)

    html_rows = []
    for row in review_rows:
        image_src = "../01_preprocessed_dataset/" + row["image_path"].replace("\\", "/")
        mask_src = "../01_preprocessed_dataset/" + row["mask_path"].replace("\\", "/")
        overlay_src = "../01_preprocessed_dataset/" + row["overlay_path"].replace("\\", "/")
        details = (
            "mask_quality={}; main_line_count={}; line_pixel_ratio={}; "
            "life_score={}; head_score={}; heart_score={}; "
            "length_norm(life/head/heart)={}/{}/{}"
        ).format(
            row["mask_quality_score"], row["main_line_count"], row["line_pixel_ratio"],
            row["score_life_line"], row["score_head_line"], row["score_heart_line"],
            row["length_life_line_norm"], row["length_head_line_norm"], row["length_heart_line_norm"],
        )
        html_rows.append(
            "<tr><td>{}</td><td><img src='{}'></td><td><img src='{}'></td><td><img src='{}'></td>"
            "<td><code>{}</code><br>{}<br>status: pending</td></tr>".format(
                html.escape(row["review_id"]),
                html.escape(image_src, quote=True),
                html.escape(mask_src, quote=True),
                html.escape(overlay_src, quote=True),
                html.escape(row["image_path"]),
                html.escape(details),
            )
        )
    queue_html = review_dir / "queue.html"
    queue_html.write_text(
        "<!doctype html><meta charset='utf-8'><title>Palmistry mask review</title>"
        "<style>body{font:14px Arial;background:#111;color:#eee}table{border-collapse:collapse;width:100%}"
        "td{border:1px solid #444;padding:8px;vertical-align:top}img{width:240px;max-height:240px;object-fit:contain;background:#222}"
        "code{color:#9fe3ff}</style><h1>Review queue</h1>"
        "<p>Kiểm tra ảnh và overlay. Sửa mask nếu cần, sau đó mở review.csv và đặt "
        "review_status thành approved hoặc rejected. Chỉ approved được dùng làm seed.</p>"
        "<table><tr><th>ID</th><th>Ảnh</th><th>Mask</th><th>Overlay</th><th>Thông tin</th></tr>"
        + "".join(html_rows) + "</table>",
        encoding="utf-8",
    )
    (review_dir / "README.md").write_text(
        "# Review queue\n\n"
        "Mở queue.html để duyệt từng ảnh. Có thể sửa mask PNG bằng công cụ annotation "
        "của bạn; giữ class IDs 0..6; simian = 6. Sau đó sửa review.csv: dùng approved cho mask đã "
        "kiểm tra, rejected cho mask sai, pending cho mục chưa xem. Chạy bootstrap_review.py "
        "sau khi hoàn tất.\n",
        encoding="utf-8",
    )
    return {
        "queue_count": len(review_rows),
        "available_accepted": len(rows),
        "review_csv": str(review_csv),
        "queue_html": str(queue_html),
        "status_values": ["pending", "approved", "rejected"],
    }

def migrate_review_v2(source_run, target_run):
    """Copy v1 review assets, preserving originals; require fresh v2 review."""
    source_run, target_run = Path(source_run).resolve(), Path(target_run).resolve()
    if target_run.exists() or target_run.is_relative_to(source_run) or source_run.is_relative_to(target_run):
        raise ValueError("Đích nâng cấp phải là run mới, tách biệt run gốc.")
    source_data = source_run / "01_preprocessed_dataset"
    doc = json.loads((source_data / "class_map.json").read_text(encoding="utf-8"))
    legacy = json.loads((Path(__file__).parent / "class_map.v1.json").read_text(encoding="utf-8"))
    if doc != legacy:
        raise ValueError("Chỉ nâng cấp queue v1 hợp lệ. Queue v2 không cần nâng cấp lại.")
    for folder in (source_data, source_run / "review"):
        for item in folder.rglob("*"):
            if item.is_symlink() or not item.resolve().is_relative_to(source_run):
                raise ValueError("Không sao chép queue chứa đường dẫn liên kết ra ngoài run.")
    needed = sum(p.stat().st_size for p in source_data.rglob("*") if p.is_file())
    target_run.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(target_run.parent).free < needed + 256*1024*1024:
        raise OSError("Không đủ dung lượng tạo bản sao queue v2; queue gốc vẫn giữ nguyên.")
    with (source_run / "review/review.csv").open(encoding="utf-8", newline="") as handle:
        old_review = {r["image_path"]:r for r in csv.DictReader(handle)}
    target_data = target_run / "01_preprocessed_dataset"
    shutil.copytree(source_data, target_data)
    core.write_class_map(target_data)
    with (target_data / "labels_pseudo_strict.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(accepted=0, review_status="pending", split="unassigned", label_provenance="v1 mask copied; v2 simian review required")
    core.write_csv(target_data / "labels_pseudo_strict.csv", rows)
    review = prepare_review_queue(target_data, target_run / "review", len(old_review))
    review_path = target_run / "review/review.csv"
    with review_path.open(encoding="utf-8",newline="") as handle:
        new_rows = list(csv.DictReader(handle))
    for row in new_rows:
        old = old_review.get(row["image_path"], {})
        row["review_notes"] = ("Cần kiểm tra thêm simian (6). Trạng thái v1: " + old.get("review_status","pending") + ". " + old.get("review_notes","")).strip()
    core.write_csv(review_path,new_rows)
    summary = {"ok":True,"run_dir":str(target_run),"review":review,"migrated_from":str(source_run),
               "class_map_version":core.CLASS_VERSION,"review_status":"pending","originals_preserved":True}
    write_json(target_run / "pipeline_summary.json",summary)
    return summary


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory phải mới hoặc rỗng: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "00_raw"
    data_dir = run_dir / "01_preprocessed_dataset"
    model_dir = run_dir / "02_training"
    test_dir = run_dir / "03_test"
    summary: Dict[str, Any] = {
        "ok": False,
        "started_at": now_iso(),
        "run_dir": str(run_dir),
        "phases": [],
        "label_provenance": "classical CV pseudo-labels; not independent ground truth",
    }

    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        if args.input_zip:
            paths, ingestion = collect_zip(Path(args.input_zip), raw_dir, args.max_images)
        elif args.input_rar:
            paths, ingestion = collect_rar(Path(args.input_rar), raw_dir, args.max_images)
        else:
            paths, ingestion = collect_directory(Path(args.input_dir), raw_dir, args.max_images)
        if not paths:
            raise ValueError("Không có ảnh hợp lệ sau phase nhận ảnh.")
        summary["phases"].append({"name": "ingestion", "status": "completed", "finished_at": now_iso()})
        summary["ingestion"] = ingestion
        write_json(raw_dir / "ingestion.json", ingestion)

        build_summary = prepare_manual_candidates(paths, data_dir, args) if args.prepare_review else core.build_pseudomask_dataset(
            paths,
            data_dir,
            out_size=args.out_size,
            min_short_side=args.min_short_side,
            blur_threshold=args.blur_threshold,
            strict_line_threshold=args.strict_line_threshold,
            keep_threshold=args.keep_threshold,
            min_good_lines=args.min_good_lines,
            exact_dedupe=not args.no_exact_dedupe,
            max_images=None,
            seed=args.seed,
        )
        summary["phases"].append({"name": "preprocess_and_auto_label", "status": "completed", "finished_at": now_iso()})
        summary["preprocessing"] = build_summary
        if args.prepare_review:
            review = prepare_review_queue(data_dir, run_dir / "review", args.review_count)
            summary["review"] = review
            summary["phases"].append({"name": "human_review_queue", "status": "completed", "finished_at": now_iso()})
            summary["ok"] = True
            summary["finished_at"] = now_iso()
            write_json(run_dir / "pipeline_summary.json", summary)
            return summary


        if args.preprocess_only:
            summary["ok"] = True
            summary["finished_at"] = now_iso()
            write_json(run_dir / "pipeline_summary.json", summary)
            return summary


        train_result = core.train_model(train_args(args, data_dir, model_dir))
        summary["phases"].append({"name": "train", "status": "completed", "finished_at": now_iso()})
        summary["training"] = train_result
        best_checkpoint = model_dir / "checkpoints" / "best.npz"
        if not best_checkpoint.is_file():
            raise FileNotFoundError("Training không tạo được checkpoints/best.npz.")

        test_result = test_model(data_dir, data_dir / "labels_pseudo_strict.csv", best_checkpoint, test_dir, args)
        summary["phases"].append({"name": "test", "status": "completed", "finished_at": now_iso()})
        summary["test"] = test_result
        summary["checkpoint"] = str(best_checkpoint)
        summary["checkpoint_sha256"] = core.sha256_file(best_checkpoint)
        summary["ok"] = True
        summary["finished_at"] = now_iso()
        write_json(run_dir / "pipeline_summary.json", summary)
        return summary
    except Exception as exc:
        summary["error"] = str(exc)
        summary["fallback"] = "Ảnh gốc vẫn còn nguyên. Chọn queue nhỏ hơn hoặc run mới; xem báo cáo chi tiết trong run."
        summary["finished_at"] = now_iso()
        write_json(run_dir / "pipeline_summary.json", summary)
        raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end Palmistry ingestion, preprocessing, pseudo-label, split, train and test pipeline")
    parser.add_argument("--seed", type=int, default=42)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input_zip", help="ZIP ảnh đầu vào")
    source.add_argument("--input_rar", help="RAR ảnh đầu vào")
    source.add_argument("--input_dir", help="Thư mục ảnh đầu vào")
    parser.add_argument("--run_dir", required=True, help="Thư mục output mới hoặc rỗng")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--out_size", type=int, default=512)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--min_short_side", type=int, default=512)
    parser.add_argument("--blur_threshold", type=float, default=60.0)
    parser.add_argument("--strict_line_threshold", type=float, default=0.72)
    parser.add_argument("--keep_threshold", type=float, default=0.85)
    parser.add_argument("--min_good_lines", type=int, default=2)
    parser.add_argument("--no_exact_dedupe", action="store_true")
    parser.add_argument("--preprocess_only", action="store_true")
    parser.add_argument("--prepare_review", action="store_true", help="Tạo queue HTML/CSV để duyệt mask")
    parser.add_argument("--review_count", type=int, default=100)
    parser.add_argument("--model_size", choices=["tiny", "small", "medium"], default="small")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--max_train_hours", type=float, default=7.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.max_images <= 0:
        args.max_images = None
    core.validate_size(args.out_size)
    core.validate_size(args.input_size)
    result = run_pipeline(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
