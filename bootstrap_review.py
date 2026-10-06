#!/usr/bin/env python3
"""Human-review bootstrap for Palmistry pseudo-labels.

Workflow:
1. pipeline.py --prepare_review creates review/review.csv and review/queue.html.
2. A person marks rows approved/rejected and may replace mask_path with a corrected
   semantic PNG (class IDs 0..6) under 01_preprocessed_dataset.
3. This command trains a seed model only from approved masks, predicts the remaining
   accepted images, keeps high-confidence predictions, and optionally trains a final
   model from the combined manifest.

Generated labels are still pseudo-labels. The reviewed seed is the only human-reviewed
portion, and all generated rows carry their confidence and provenance in the CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

import palmistry_strict_auto_onefile as core
import pipeline as pipeline_lib


STATUS_VALUES = {"pending", "approved", "rejected"}
LINE_CLASS_IDS = (2, 3, 4, 6)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalise_rel(value: str) -> str:
    return str(value or "").replace("\\", "/").lstrip("./")


def inside(root: Path, rel_or_path: str) -> Tuple[Path, str]:
    raw = Path(str(rel_or_path).replace("\\", "/"))
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    root = root.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes dataset root: {rel_or_path}")
    return path, path.relative_to(root).as_posix()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "review_id", "image_path", "mask_path", "overlay_path", "split", "group_id",
        "split_basis", "accepted", "source_path", "source_group", "sha256", "phash",
        "label_provenance", "review_status", "review_notes", "bootstrap_confidence",
        "bootstrap_line_pixel_ratio", "bootstrap_foreground_confidence",
    ]
    keys = []
    for key in preferred + sorted({k for row in rows for k in row}):
        if key not in keys:
            keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_stem(value: str, fallback: str = "image") -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or ""))
    stem = stem.strip("._-")[:90]
    return stem or fallback


def load_base_manifest(data_root: Path, labels_csv: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows = read_csv(labels_csv)
    if not rows:
        raise ValueError(f"Manifest rỗng: {labels_csv}")
    mapping: Dict[str, Dict[str, Any]] = {}
    checked: List[Dict[str, Any]] = []
    for row in rows:
        if not row.get("image_path") or not row.get("mask_path"):
            raise ValueError("Manifest cần image_path và mask_path cho mọi dòng.")
        image_path, image_rel = inside(data_root, row["image_path"])
        mask_path, mask_rel = inside(data_root, row["mask_path"])
        image = core.imread_rgb(image_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}. Khôi phục file ảnh của queue trước khi train.")
        core.validate_mask(mask, image.shape[:2])
        out = dict(row)
        out["image_path"] = image_rel
        out["mask_path"] = mask_rel
        out["sha256"] = core.sha256_file(image_path)
        out["phash"] = core.perceptual_hash(image)
        out["source_group"] = str(row.get("source_group") or image_path.parent)
        out.setdefault("source_path", str(image_path))
        out.setdefault("accepted", "1")
        key = normalise_rel(image_rel)
        if key in mapping:
            raise ValueError(f"Trùng image_path trong manifest: {image_rel}")
        mapping[key] = out
        checked.append(out)
    return checked, mapping


def load_review(review_csv: Path, data_root: Path, base: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, int]]:
    rows = read_csv(review_csv)
    if not rows:
        raise ValueError(f"Review CSV rỗng: {review_csv}")
    by_image: Dict[str, Dict[str, str]] = {}
    counts = {"pending": 0, "approved": 0, "rejected": 0}
    for row in rows:
        status = str(row.get("review_status") or "pending").strip().lower()
        if status not in STATUS_VALUES:
            raise ValueError(f"review_status không hợp lệ: {status!r}; dùng pending/approved/rejected.")
        image_rel = normalise_rel(row.get("image_path", ""))
        if image_rel not in base:
            raise ValueError(f"Review image không có trong labels_pseudo_strict.csv: {image_rel}")
        if image_rel in by_image:
            raise ValueError(f"Review CSV có image_path trùng: {image_rel}")
        inside(data_root, row.get("image_path", ""))
        inside(data_root, row.get("mask_path", ""))
        row["review_status"] = status
        row["image_path"] = image_rel
        row["mask_path"] = normalise_rel(row.get("mask_path", ""))
        by_image[image_rel] = row
        counts[status] += 1
    return by_image, counts


def reviewed_rows(
    review_map: Dict[str, Dict[str, str]],
    base: Dict[str, Dict[str, Any]],
    data_root: Path,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for image_rel, review in review_map.items():
        if review["review_status"] != "approved":
            continue
        base_row = dict(base[image_rel])
        image_path, image_rel = inside(data_root, review["image_path"])
        mask_path, mask_rel = inside(data_root, review["mask_path"])
        image = core.imread_rgb(image_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}. Khôi phục file ảnh của queue trước khi train.")
        core.validate_mask(mask, image.shape[:2])
        base_row.update({
            "image_path": image_rel,
            "mask_path": mask_rel,
            "review_id": review.get("review_id", ""),
            "review_status": "approved",
            "review_notes": review.get("review_notes", ""),
            "label_provenance": "human-reviewed mask; approved by user",
            "accepted": "1",
            "sha256": core.sha256_file(image_path),
            "phash": core.perceptual_hash(image),
        })
        out.append(base_row)
    if not out:
        return []
    return core.group_rows(out, seed=42)


def confidence_prediction(
    model: Any,
    image: np.ndarray,
    input_size: int,
    device: Any = None,
) -> Tuple[np.ndarray, float, float, float]:
    """Run the NumPy checkpoint and return mask plus uncalibrated confidence."""
    probs = model.predict_proba(image)
    pred_small = np.argmax(probs, axis=0).astype(np.uint8)
    confidence_small = np.max(probs, axis=0).astype(np.float32)
    pred = cv2.resize(pred_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    confidence_map = cv2.resize(confidence_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    core.validate_mask(pred, image.shape[:2])
    foreground = np.isin(pred, LINE_CLASS_IDS)
    foreground_ratio = float(np.mean(foreground))
    if not np.any(foreground):
        return pred, 0.0, 0.0, foreground_ratio
    foreground_confidence = float(np.mean(confidence_map[foreground]))
    mean_confidence = float(np.mean(confidence_map))
    return pred, min(mean_confidence, foreground_confidence), foreground_confidence, foreground_ratio

def train_namespace(args: argparse.Namespace, data_root: Path, labels_csv: Path, out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        data_root=str(data_root),
        labels_csv=str(labels_csv),
        out_dir=str(out_dir),
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


def prepare_combined(
    base_rows: List[Dict[str, Any]],
    reviewed: List[Dict[str, Any]],
    data_root: Path,
    bootstrap_dir: Path,
    model: Any,
    model_input_size: int,
    device: Any,
    confidence_threshold: float,
    min_line_ratio: float,
    max_remaining: Optional[int],
    excluded_images=None,
    remaining_sources=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    approved_images = {normalise_rel(row["image_path"]) for row in reviewed}
    excluded_images = set(excluded_images or ())
    candidates = [row for row in base_rows if normalise_rel(row["image_path"]) not in approved_images
                  and normalise_rel(row["image_path"]) not in excluded_images]
    remaining_sources = remaining_sources or {}
    candidates += [{"source_path": path, "source_group": str(Path(path).parent), "raw_candidate": True}
                   for path in remaining_sources.get("paths", [])]
    holdout = [row for row in reviewed if row["split"] in {"val", "test"}]
    held_sources = {str(row["source_group"]) for row in holdout}
    held_subjects = {str(row.get("subject_id")) for row in holdout if row.get("subject_id")}
    held_hashes = [int(row["phash"], 16) for row in holdout]
    held_sha = {row["sha256"] for row in holdout}
    if max_remaining is not None:
        candidates = candidates[:max_remaining]
    mask_dir = data_root / "bootstrap_masks"
    prediction_dir = bootstrap_dir / "predictions"
    mask_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for index, original in enumerate(candidates, start=1):
        if index == 1 or index % 50 == 0 or index == len(candidates):
            print(f"Bootstrap {index}/{len(candidates)}; đã giữ {len(generated)}, bỏ qua {len(skipped)}", flush=True)
        if str(original.get("source_group")) in held_sources or str(original.get("subject_id")) in held_subjects:
            skipped.append({"image_path": original.get("image_path", original.get("source_path")), "reason": "held_out_source"})
            continue
        try:
            if original.get("raw_candidate"):
                source_root = Path(remaining_sources["source_root"])
                image_path, _ = inside(source_root, original["source_path"])
                raw = core.imread_rgb(image_path)
                if raw is None:
                    raise ValueError("unreadable_source")
                image, _, _ = core.crop_palm_auto(raw, out_size=int(remaining_sources.get("out_size", 512)))
                digest = core.sha256_file(image_path)
                image_rel = f"bootstrap_images/{digest}.jpg"
            else:
                image_path, image_rel = inside(data_root, original["image_path"])
                image = core.imread_rgb(image_path)
                digest = core.sha256_file(image_path)
            if image is None:
                raise ValueError("unreadable_image")
            phash = core.perceptual_hash(image)
            if digest in held_sha or any((int(phash, 16) ^ h).bit_count() <= 6 for h in held_hashes):
                skipped.append({"image_path": str(image_path), "reason": "held_out_duplicate"})
                continue
        except (OSError, ValueError, KeyError) as exc:
            skipped.append({"image_path": str(original.get("source_path", original.get("image_path"))), "reason": str(exc)})
            continue
        pred, confidence, foreground_confidence, line_ratio = confidence_prediction(
            model, image, model_input_size, device
        )
        reason = ""
        if confidence < confidence_threshold:
            reason = "below_confidence_threshold"
        elif line_ratio < min_line_ratio:
            reason = "below_min_line_pixel_ratio"
        if reason:
            skipped.append({
                "image_path": image_rel,
                "reason": reason,
                "bootstrap_confidence": confidence,
                "bootstrap_line_pixel_ratio": line_ratio,
            })
            continue
        stem = safe_stem(Path(image_rel).stem, f"image_{index:06d}")
        mask_path = mask_dir / f"{stem}_bootstrap_mask.png"
        if mask_path.exists():
            mask_path = mask_dir / f"{stem}_{index:06d}_bootstrap_mask.png"
        if original.get("raw_candidate") and not core.imwrite_rgb(data_root / image_rel, image):
            raise OSError("Không ghi được ảnh bootstrap; kiểm tra dung lượng ổ đĩa.")
        if not core.imwrite_gray(mask_path, pred):
            raise OSError("Không ghi được mask bootstrap; kiểm tra dung lượng ổ đĩa.")
        overlay_path = prediction_dir / f"{stem}_bootstrap_overlay.jpg"
        if not core.imwrite_rgb(overlay_path, core.make_overlay(image, pred)):
            raise OSError("Không ghi được overlay bootstrap; kiểm tra dung lượng ổ đĩa.")
        row = dict(original)
        row.update({
            "image_path": image_rel,
            "mask_path": mask_path.relative_to(data_root).as_posix(),
            "overlay_path": str(overlay_path),
            "accepted": "1",
            "sha256": core.sha256_file(data_root / image_rel),
            "phash": phash, "split": "train",
            "label_provenance": "bootstrap model pseudo-label; confidence thresholded",
            "bootstrap_confidence": confidence,
            "bootstrap_foreground_confidence": foreground_confidence,
            "bootstrap_line_pixel_ratio": line_ratio,
            "review_status": "",
            "review_notes": "",
        })
        generated.append(row)
    stats = {"candidate_count": len(candidates), "generated_count": len(generated), "skipped_count": len(skipped)}
    write_csv(bootstrap_dir / "bootstrap_skipped.csv", skipped)
    return generated, skipped, stats


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    data_root = run_dir / "01_preprocessed_dataset"
    labels_csv = data_root / "labels_pseudo_strict.csv"
    review_csv = Path(args.review_csv).resolve() if args.review_csv else run_dir / "review" / "review.csv"
    out_dir = Path(args.out_dir).resolve()
    if not data_root.is_dir() or not labels_csv.is_file():
        raise FileNotFoundError("Không tìm thấy 01_preprocessed_dataset/labels_pseudo_strict.csv trong run_dir.")
    if not review_csv.is_file():
        raise FileNotFoundError(f"Không tìm thấy review CSV: {review_csv}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"out_dir phải mới hoặc rỗng: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_rows, base_map = load_base_manifest(data_root, labels_csv)
    review_map, review_counts = load_review(review_csv, data_root, base_map)
    summary: Dict[str, Any] = {
        "ok": False,
        "started_at": now_iso(),
        "run_dir": str(run_dir),
        "data_root": str(data_root),
        "review_csv": str(review_csv),
        "review_counts": review_counts,
        "label_provenance": "human-reviewed seed + confidence-thresholded bootstrap pseudo-labels",
        "confidence_threshold": args.confidence,
        "min_main_line_ratio": args.min_line_ratio,
    }
    reviewed = reviewed_rows(review_map, base_map, data_root)
    summary["approved_seed_count"] = len(reviewed)
    if reviewed:
        write_csv(out_dir / "seed_labels.csv", reviewed)
    if args.validate_only:
        summary["ok"] = True
        summary["validation_only"] = True
        summary["message"] = (
            "Review đã được kiểm tra. Chưa train/sinh pseudo-label; chạy lại không có --validate_only "
            "sau khi có ít nhất 3 source/perceptual groups approved."
        )
        pipeline_lib.write_json(out_dir / "bootstrap_summary.json", summary)
        return summary

    if len(reviewed) < 3:
        raise ValueError("Cần duyệt approved ít nhất 3 ảnh thuộc 3 source/perceptual groups trước khi train seed.")
    seed_dir = out_dir / "seed_training"
    seed_args = train_namespace(args, data_root, out_dir / "seed_labels.csv", seed_dir)
    seed_result = core.train_model(seed_args)
    summary["seed_training"] = seed_result
    seed_checkpoint = seed_dir / "checkpoints" / "best.npz"
    if not seed_checkpoint.is_file():
        raise FileNotFoundError("Seed training không tạo được checkpoints/best.npz.")
    digest = core.sha256_file(seed_checkpoint)
    device = None
    model, checkpoint_input_size, validation_scores = core.load_model_for_predict(seed_checkpoint, device, digest)
    generated, skipped, generation_stats = prepare_combined(
        base_rows, reviewed, data_root, out_dir, model, checkpoint_input_size, device,
        args.confidence, args.min_line_ratio, args.max_remaining,
        excluded_images={key for key, row in review_map.items() if row["review_status"] != "approved"},
        remaining_sources=json.loads((data_root / "remaining_sources.json").read_text(encoding="utf-8"))
            if (data_root / "remaining_sources.json").is_file() else {},
    )
    combined = reviewed + generated
    if len(combined) < 3:
        raise ValueError("Không đủ reviewed/generated rows để tạo ba split độc lập.")
    # Freeze reviewed holdouts; generated masks are training data only.
    combined = core.group_rows(combined, seed=args.seed, assign=False)
    write_csv(out_dir / "labels_bootstrapped.csv", combined)
    summary.update({
        "seed_checkpoint": str(seed_checkpoint),
        "seed_checkpoint_sha256": digest,
        "seed_validation_scores": validation_scores,
        "generated": generation_stats,
        "combined_count": len(combined),
        "combined_split_counts": core.split_summary(combined),
        "labels_bootstrapped": str(out_dir / "labels_bootstrapped.csv"),
    })
    if args.train_final:
        final_dir = out_dir / "final_training"
        final_args = train_namespace(args, data_root, out_dir / "labels_bootstrapped.csv", final_dir)
        final_result = core.train_model(final_args)
        summary["final_training"] = final_result
        final_checkpoint = final_dir / "checkpoints" / "best.npz"
        if not final_checkpoint.is_file():
            raise FileNotFoundError("Final training không tạo được checkpoints/best.npz.")
        test_args = argparse.Namespace(input_size=args.input_size, device=args.device)
        test_result = pipeline_lib.test_model(
            data_root, out_dir / "labels_bootstrapped.csv", final_checkpoint, out_dir / "final_test", test_args,
        )
        test_result["label_provenance"] = "human-reviewed seed + confidence-thresholded bootstrap pseudo-labels"
        pipeline_lib.write_json(out_dir / "final_test" / "test_metrics.json", test_result)
        summary["final_checkpoint"] = str(final_checkpoint)
        summary["final_checkpoint_sha256"] = core.sha256_file(final_checkpoint)
        summary["final_test"] = test_result
    summary["ok"] = True
    summary["finished_at"] = now_iso()
    pipeline_lib.write_json(out_dir / "bootstrap_summary.json", summary)
    return summary


def preflight(run_dir, review_csv=None):
    data_root = Path(run_dir) / "01_preprocessed_dataset"
    review_csv = Path(review_csv).resolve() if review_csv else Path(run_dir) / "review" / "review.csv"
    rows = read_csv(review_csv)
    approved = [row for row in rows if row.get("review_status") == "approved"]
    if len(approved) < 3:
        raise ValueError(f"Mới có {len(approved)} mask approved. Hãy duyệt mask trước khi train; cần ít nhất 3 nhóm độc lập.")
    core.check_class_map(data_root / "class_map.json")
    _, base = load_base_manifest(data_root, data_root / "labels_pseudo_strict.csv")
    review_map, _ = load_review(review_csv, data_root, base)
    try:
        reviewed_rows(review_map, base, data_root)
    except ValueError as exc:
        raise ValueError("Chưa thể chia train/val/test độc lập. Cần metadata người/bàn tay và nhóm nguồn đúng; "
                         "không chia ngẫu nhiên ảnh cùng nguồn để ép train. Queue và kết quả duyệt vẫn được giữ. " + str(exc)) from exc


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"out_dir đã có dữ liệu: {out}; chọn tên mới.")
    try:
        if not args.validate_only:
            preflight(args.run_dir, args.review_csv)
        return _run(args)
    except Exception as exc:
        pipeline_lib.write_json(out / "bootstrap_summary.json", {
            "ok": False, "error": str(exc), "finished_at": now_iso(),
            "fallback": "Giữ nguyên mask đã duyệt. Kiểm tra nhóm nguồn, dung lượng và chọn tên bootstrap mới để thử lại.",
        })
        raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train from reviewed masks and bootstrap high-confidence labels for the remaining accepted images"
    )
    parser.add_argument("--run_dir", required=True, help="Output run_dir created by pipeline.py")
    parser.add_argument("--review_csv", default="", help="Default: run_dir/review/review.csv")
    parser.add_argument("--out_dir", required=True, help="New output directory for bootstrap artifacts")
    parser.add_argument("--validate_only", action="store_true", help="Validate review CSV and masks without model training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_size", type=int, default=256)
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
    parser.add_argument("--confidence", type=float, default=0.90, help="Minimum mean confidence for generated mask")
    parser.add_argument("--min_line_ratio", type=float, default=0.005, help="Minimum predicted line pixel ratio")
    parser.add_argument("--max_remaining", type=int, default=0, help="Optional cap for generated candidates; 0 means all")
    parser.add_argument("--train_final", action="store_true", help="Train final model and evaluate its held-out test split")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.max_remaining <= 0:
        args.max_remaining = None
    core.validate_size(args.input_size)
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence phải trong [0,1].")
    if args.min_line_ratio < 0.0 or args.min_line_ratio > 1.0:
        raise ValueError("--min_line_ratio phải trong [0,1].")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
