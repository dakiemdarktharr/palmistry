#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PALMISTRY STRICT AUTO PIPELINE - ONE FILE

Mục tiêu:
- Không cần người dùng tự tạo/sửa mask.
- Tự tạo pseudo-mask bằng Classical CV.
- Lọc cực gắt ảnh/mask yếu.
- Chỉ train 3 đường chính: Sinh đạo, Trí đạo, Tâm đạo + minor/unknown.
- Khi predict chỉ xuất phần đạt ngưỡng heuristic_score (chưa hiệu chuẩn).
- Nếu không đọc được, tự chẩn đoán lý do và yêu cầu chụp lại đúng vấn đề.

Commands:
  doctor
  build          Đọc folder ảnh local; remote download đã tắt.
  mask-folder    Tạo pseudo-mask từ folder ảnh local.
  single-mask    Tạo pseudo-mask cho 1 ảnh.
  train          Train segmentation model, checkpoint/resume/time limit.
  predict        Phân tích 1 ảnh, tự chẩn đoán lỗi nếu không đủ điều kiện.

Class map:
  0 background
  1 palm_area
  2 life_line / Sinh đạo
  3 head_line / Trí đạo
  4 heart_line / Tâm đạo
  5 minor_or_unknown_line / đường phụ hoặc chưa chắc tên
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
try:
    import requests
except Exception:
    requests = None
from prototype_common import (CLASS_MAP, CLASS_VERSION, MODEL_VERSION, validate_mask, validate_size,
    read_rgb, write_class_map, check_class_map, group_rows, split_summary, perceptual_hash, read_manifest,
    confusion_matrix, metrics_from_confusion, seed_everything, provenance, write_json, load_config)

try:
    import cv2
except Exception as e:
    print("ERROR: opencv-python chưa được cài. Chạy: pip install opencv-python", file=sys.stderr)
    raise

try:
    from PIL import Image, ImageOps
except Exception:
    print("ERROR: pillow chưa được cài. Chạy: pip install pillow", file=sys.stderr)
    raise

# -----------------------------
# Constants
# -----------------------------
from palm_geometry import LINE_IDS, infer_handedness, measure_line, resolve_transverse_lines

LINE_CLASSES = dict(LINE_IDS)
DEFAULT_COMMONS_QUERIES = [
    "palm hand lines",
    "open palm hand",
    "human palm",
    "palmistry hand",
    "hand palm close up",
    "left palm hand",
    "right palm hand",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# -----------------------------
# Utility
# -----------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in "._-":
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out[:180] if out else "file"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def list_images(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        return []
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def imread_rgb(path: Path) -> Optional[np.ndarray]:
    try:
        return read_rgb(path)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None

def imwrite_rgb(path: Path, img_rgb: np.ndarray) -> bool:
    try:
        ensure_dir(path.parent)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        ext = path.suffix.lower()
        if ext == ".jpg" or ext == ".jpeg":
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        else:
            ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def imwrite_gray(path: Path, img: np.ndarray) -> bool:
    try:
        ensure_dir(path.parent)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        ext = ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def resize_keep_aspect_pad(img: np.ndarray, out_size: int, pad_value: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    validate_size(out_size)
    h, w = img.shape[:2]
    scale = out_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    if img.ndim == 3:
        canvas = np.full((out_size, out_size, img.shape[2]), pad_value, dtype=img.dtype)
    else:
        canvas = np.full((out_size, out_size), pad_value, dtype=img.dtype)
    y0 = (out_size - nh) // 2
    x0 = (out_size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    meta = {"scale": scale, "x0": x0, "y0": y0, "new_w": nw, "new_h": nh, "orig_w": w, "orig_h": h}
    return canvas, meta


def normalize_score(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def percentile_clip_uint8(arr: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr.astype(np.float32) - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)

# -----------------------------
# Diagnostics and quality gate
# -----------------------------
@dataclass
class DiagnosticResult:
    status: str
    main_reason: str
    reason_vi: str
    fix_vi: str
    scores: Dict[str, float]
    measurements: Dict[str, float]


FAILURE_MESSAGES = {
    "NO_HAND_FOUND": (
        "Ảnh chưa thấy bàn tay rõ.",
        "Hãy chụp một bàn tay mở, lòng bàn tay hướng thẳng vào camera, nền đơn giản và đủ sáng."
    ),
    "NOT_PALM_SIDE": (
        "Ảnh có vẻ không phải mặt lòng bàn tay nên không thấy rõ các đường chỉ tay.",
        "Hãy lật tay lại để thấy lòng bàn tay, mở nhẹ các ngón và chụp đủ từ gốc ngón đến cổ tay."
    ),
    "PALM_CROPPED": (
        "Ảnh bị cắt mất một phần lòng bàn tay.",
        "Hãy chụp lại sao cho thấy đầy đủ từ gốc các ngón tay đến cổ tay, gồm cả rìa ngón cái và rìa ngoài bàn tay."
    ),
    "PALM_TOO_SMALL": (
        "Lòng bàn tay chiếm quá ít khung hình.",
        "Hãy đưa tay gần camera hơn để lòng bàn tay chiếm khoảng 60–80% ảnh."
    ),
    "BLURRY_IMAGE": (
        "Ảnh bị mờ/rung nên các đường nhỏ trên lòng bàn tay bị mất chi tiết.",
        "Hãy giữ điện thoại cố định, đặt tay cách camera khoảng 25–40 cm, chạm để lấy nét vào lòng bàn tay rồi chụp lại."
    ),
    "TOO_DARK": (
        "Ảnh quá tối nên các đường trong lòng bàn tay không đủ rõ để nhận diện.",
        "Hãy chụp lại ở nơi có ánh sáng đều hơn, gần cửa sổ ban ngày hoặc dưới đèn phòng sáng; tránh để bóng điện thoại che lòng bàn tay."
    ),
    "TOO_BRIGHT": (
        "Ảnh quá sáng hoặc bị cháy vùng lòng bàn tay.",
        "Hãy tránh ánh sáng chiếu thẳng hoặc đèn flash quá gần; chụp ở ánh sáng đều hơn."
    ),
    "STRONG_SHADOW": (
        "Lòng bàn tay bị bóng che nên đường chỉ tay không rõ.",
        "Hãy chụp nơi ánh sáng đều, tránh bóng của điện thoại hoặc ngón tay đổ lên lòng bàn tay."
    ),
    "GLARE_REFLECTION": (
        "Ảnh bị lóa trên lòng bàn tay.",
        "Hãy đổi góc chụp hoặc giảm ánh sáng trực tiếp để da không bị phản chiếu sáng mạnh."
    ),
    "HAND_TILTED_TOO_MUCH": (
        "Bàn tay đang nghiêng quá nhiều so với camera.",
        "Hãy đặt lòng bàn tay gần song song với camera, chụp thẳng hơn từ phía trước."
    ),
    "LOW_RESOLUTION": (
        "Ảnh có độ phân giải thấp, không đủ chi tiết để tách đường chỉ tay.",
        "Hãy gửi ảnh rõ hơn, tốt nhất cạnh ngắn tối thiểu khoảng 1024 px."
    ),
    "LOW_LINE_CONTRAST": (
        "Lòng bàn tay thấy được nhưng đường chỉ tay có độ tương phản thấp.",
        "Hãy chụp gần hơn một chút, dùng ánh sáng đều hoặc ánh sáng hơi nghiêng từ bên hông để các đường nổi rõ hơn."
    ),
    "NO_CONFIDENT_LINES": (
        "Hệ thống chưa nhận diện được đường chính nào đủ chắc từ ảnh này.",
        "Hãy chụp lại lòng bàn tay rõ hơn, đủ sáng, không mờ, không bị cắt mất cổ tay hoặc gốc ngón."
    ),
    "OK": ("Ảnh đạt điều kiện phân tích.", ""),
}


def basic_image_metrics(img_rgb: np.ndarray) -> Dict[str, float]:
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(np.mean(gray))
    std_contrast = float(np.std(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturated_hi = float(np.mean(gray > 245))
    saturated_lo = float(np.mean(gray < 10))
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))
    edge = cv2.Canny(gray, 60, 160)
    edge_density = float(np.mean(edge > 0))
    return {
        "width": float(w), "height": float(h), "short_side": float(min(h, w)),
        "mean_brightness": mean_brightness,
        "std_contrast": std_contrast,
        "laplacian_variance": lap_var,
        "saturated_hi_ratio": saturated_hi,
        "saturated_lo_ratio": saturated_lo,
        "saturation_mean": sat_mean,
        "edge_density": edge_density,
    }


def detect_skin_largest_contour(img_rgb: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], float]:
    """Return skin mask, bbox, score. Fallback hand/palm detector for fully automatic pipeline."""
    h, w = img_rgb.shape[:2]
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    # Broad skin range; intentionally permissive then morph clean.
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    mask1 = cv2.inRange(ycrcb, lower, upper)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # Additional permissive HSV skin-like zone.
    mask2 = cv2.inRange(hsv, np.array([0, 15, 40]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask, None, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < 0.02 * h * w:
        return mask, None, normalize_score(area / (h*w), 0.0, 0.1)
    x, y, bw, bh = cv2.boundingRect(cnt)
    area_ratio = area / float(h*w)
    bbox_ratio = (bw * bh) / float(h*w)
    score = 0.0
    score += 0.45 * normalize_score(area_ratio, 0.05, 0.45)
    score += 0.25 * normalize_score(bbox_ratio, 0.12, 0.75)
    score += 0.15 * (1.0 - min(abs((bw / max(bh, 1)) - 0.75), 0.75) / 0.75)
    score += 0.15 * (1.0 if x > 0 and y > 0 and x + bw < w and y + bh < h else 0.5)
    return mask, (x, y, bw, bh), float(np.clip(score, 0, 1))


def crop_palm_auto(img_rgb: np.ndarray, out_size: int = 1024) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Auto crop palm-like ROI. Uses skin contour; fallback center square."""
    h, w = img_rgb.shape[:2]
    skin_mask, bbox, hand_score = detect_skin_largest_contour(img_rgb)
    method = "skin_bbox"
    cropped_from = [0, 0, w, h]
    if bbox is not None:
        x, y, bw, bh = bbox
        # Expand bbox, but try to keep palm large. Since fingers may inflate box, we still crop whole hand.
        pad_x = int(0.12 * bw)
        pad_y = int(0.12 * bh)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + bw + pad_x)
        y1 = min(h, y + bh + pad_y)
        crop = img_rgb[y0:y1, x0:x1]
        crop_mask = skin_mask[y0:y1, x0:x1]
        cropped_from = [int(x0), int(y0), int(x1), int(y1)]
    else:
        method = "center_square_fallback"
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = img_rgb[y0:y0+side, x0:x0+side]
        crop_mask = np.ones((side, side), dtype=np.uint8) * 255
        cropped_from = [int(x0), int(y0), int(x0+side), int(y0+side)]
        hand_score = 0.35
    crop_sq, meta = resize_keep_aspect_pad(crop, out_size, pad_value=0)
    mask_sq, _ = resize_keep_aspect_pad(crop_mask, out_size, pad_value=0)
    mask_sq = (mask_sq > 0).astype(np.uint8) * 255
    # Refine palm area: large closing, fill holes.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask_sq = cv2.morphologyEx(mask_sq, cv2.MORPH_CLOSE, kernel, iterations=2)
    meta.update({"crop_method": method, "hand_score": float(hand_score), "crop_xyxy": cropped_from})
    return crop_sq, mask_sq, meta


def analyze_quality(img_rgb: np.ndarray, min_short_side: int = 1024) -> DiagnosticResult:
    m = basic_image_metrics(img_rgb)
    _, bbox, hand_score = detect_skin_largest_contour(img_rgb)
    brightness = m["mean_brightness"]
    contrast = m["std_contrast"]
    lap = m["laplacian_variance"]
    short_side = m["short_side"]
    sat_hi = m["saturated_hi_ratio"]
    edge_density = m["edge_density"]

    brightness_score = 1.0 - min(abs(brightness - 135.0) / 110.0, 1.0)
    blur_score = normalize_score(lap, 45.0, 260.0)
    contrast_score = normalize_score(contrast, 20.0, 65.0)
    resolution_score = normalize_score(short_side, min_short_side * 0.65, min_short_side)
    glare_score = 1.0 - normalize_score(sat_hi, 0.03, 0.25)
    edge_score = normalize_score(edge_density, 0.015, 0.12)
    palm_detection_score = float(hand_score)
    image_quality_score = float(np.clip(
        0.20 * brightness_score + 0.25 * blur_score + 0.20 * contrast_score +
        0.15 * resolution_score + 0.10 * glare_score + 0.10 * palm_detection_score,
        0, 1
    ))
    line_visibility_score = float(np.clip(0.45 * contrast_score + 0.35 * edge_score + 0.20 * blur_score, 0, 1))

    scores = {
        "image_quality_score": image_quality_score,
        "brightness_score": float(np.clip(brightness_score, 0, 1)),
        "blur_score": float(np.clip(blur_score, 0, 1)),
        "contrast_score": float(np.clip(contrast_score, 0, 1)),
        "resolution_score": float(np.clip(resolution_score, 0, 1)),
        "glare_score": float(np.clip(glare_score, 0, 1)),
        "palm_detection_score": palm_detection_score,
        "line_visibility_score": line_visibility_score,
    }
    # Main reason priority.
    reason = "OK"
    if palm_detection_score < 0.25:
        reason = "NO_HAND_FOUND"
    elif short_side < min_short_side * 0.65:
        reason = "LOW_RESOLUTION"
    elif blur_score < 0.35:
        reason = "BLURRY_IMAGE"
    elif brightness < 65:
        reason = "TOO_DARK"
    elif brightness > 215 or sat_hi > 0.22:
        reason = "TOO_BRIGHT"
    elif contrast_score < 0.35 or line_visibility_score < 0.35:
        reason = "LOW_LINE_CONTRAST"
    elif image_quality_score < 0.55:
        # choose weakest score
        weakest = min(scores.items(), key=lambda kv: kv[1])[0]
        reason = {
            "brightness_score": "TOO_DARK" if brightness < 135 else "TOO_BRIGHT",
            "blur_score": "BLURRY_IMAGE",
            "contrast_score": "LOW_LINE_CONTRAST",
            "resolution_score": "LOW_RESOLUTION",
            "glare_score": "GLARE_REFLECTION",
            "palm_detection_score": "NO_HAND_FOUND",
            "line_visibility_score": "LOW_LINE_CONTRAST",
        }.get(weakest, "LOW_LINE_CONTRAST")

    status = "ok" if reason == "OK" else "need_retake"
    reason_vi, fix_vi = FAILURE_MESSAGES[reason]
    return DiagnosticResult(status, reason, reason_vi, fix_vi, scores, m)

# -----------------------------
# Classical CV pseudo-mask
# -----------------------------
def clahe_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def gabor_max_response(gray: np.ndarray, ksize: int = 17) -> np.ndarray:
    responses = []
    for theta in np.linspace(0, np.pi, 12, endpoint=False):
        kernel = cv2.getGaborKernel((ksize, ksize), sigma=4.0, theta=theta, lambd=8.0, gamma=0.45, psi=0)
        kernel -= kernel.mean()
        resp = cv2.filter2D(gray, cv2.CV_32F, kernel)
        responses.append(np.abs(resp))
    mx = np.max(np.stack(responses, axis=0), axis=0)
    return percentile_clip_uint8(mx, 2, 99)


def directional_blackhat(gray: np.ndarray) -> np.ndarray:
    outs = []
    for length in (15, 23, 31):
        for angle in (0, 30, 60, 90, 120, 150):
            kernel = np.zeros((length, length), dtype=np.uint8)
            center = length // 2
            rad = math.radians(angle)
            dx = int(math.cos(rad) * center)
            dy = int(math.sin(rad) * center)
            p1 = (center - dx, center - dy)
            p2 = (center + dx, center + dy)
            cv2.line(kernel, p1, p2, 1, 1)
            bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            outs.append(bh)
    out = np.max(np.stack(outs, axis=0), axis=0)
    return percentile_clip_uint8(out, 1, 99)


def remove_small_components(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1
    return out.astype(np.uint8) * 255


def zhang_suen_thinning(binary: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Pure numpy fallback thinning. Input 0/255, output 0/255."""
    img = (binary > 0).astype(np.uint8)
    prev = np.zeros_like(img)
    count = 0
    while not np.array_equal(img, prev) and count < max_iter:
        prev = img.copy()
        for step in (0, 1):
            to_remove = []
            rows, cols = img.shape
            # iterate non-border foreground only
            ys, xs = np.where(img[1:-1, 1:-1] > 0)
            ys = ys + 1; xs = xs + 1
            for y, x in zip(ys, xs):
                p2 = img[y-1, x]
                p3 = img[y-1, x+1]
                p4 = img[y, x+1]
                p5 = img[y+1, x+1]
                p6 = img[y+1, x]
                p7 = img[y+1, x-1]
                p8 = img[y, x-1]
                p9 = img[y-1, x-1]
                ps = [p2,p3,p4,p5,p6,p7,p8,p9]
                B = sum(ps)
                if B < 2 or B > 6:
                    continue
                A = 0
                seq = ps + [p2]
                for k in range(8):
                    if seq[k] == 0 and seq[k+1] == 1:
                        A += 1
                if A != 1:
                    continue
                if step == 0:
                    if p2 * p4 * p6 != 0:
                        continue
                    if p4 * p6 * p8 != 0:
                        continue
                else:
                    if p2 * p4 * p8 != 0:
                        continue
                    if p2 * p6 * p8 != 0:
                        continue
                to_remove.append((y, x))
            for y, x in to_remove:
                img[y, x] = 0
        count += 1
    return (img * 255).astype(np.uint8)


def skeletonize_mask(binary: np.ndarray) -> np.ndarray:
    binary = (binary > 0).astype(np.uint8) * 255
    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            return cv2.ximgproc.thinning(binary)
    except Exception:
        pass
    try:
        from skimage.morphology import skeletonize
        sk = skeletonize(binary > 0)
        return (sk.astype(np.uint8) * 255)
    except Exception:
        # For speed, thin at half resolution if very large, then upsample.
        h, w = binary.shape
        if max(h, w) > 768:
            small = cv2.resize(binary, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
            sk = zhang_suen_thinning(small, max_iter=80)
            return cv2.resize(sk, (w, h), interpolation=cv2.INTER_NEAREST)
        return zhang_suen_thinning(binary, max_iter=100)


def estimate_thumb_side(palm_mask: np.ndarray) -> str:
    return infer_handedness(palm_mask, mirrored=None)["thumb_side_image"]


def component_features(comp_mask: np.ndarray) -> Dict[str, float]:
    ys, xs = np.where(comp_mask > 0)
    if len(xs) == 0:
        return {}
    h, w = comp_mask.shape
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    length = len(xs)
    cx, cy = float(np.mean(xs) / w), float(np.mean(ys) / h)
    # PCA orientation: angle 0 horizontal, pi/2 vertical.
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    if len(pts) > 2:
        pts -= pts.mean(axis=0, keepdims=True)
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, np.argmax(vals)]
        angle = float(abs(math.atan2(v[1], v[0])))
        if angle > math.pi/2:
            angle = math.pi - angle
    else:
        angle = 0.0
    return {
        "x0": x0/w, "x1": x1/w, "y0": y0/h, "y1": y1/h,
        "bw": bw/w, "bh": bh/h, "length_px": float(length),
        "cx": cx, "cy": cy, "angle": angle,
        "horizontalness": float(1.0 - min(abs(angle - 0.0) / (math.pi/2), 1.0)),
        "verticalness": float(1.0 - min(abs(angle - math.pi/2) / (math.pi/2), 1.0)),
        "diagonalness": float(1.0 - min(abs(angle - math.pi/4) / (math.pi/4), 1.0)),
    }


def score_component_as_line(feat: Dict[str, float], thumb_side: str) -> Dict[str, float]:
    if not feat:
        return {"life_line": 0.0, "head_line": 0.0, "heart_line": 0.0, "minor": 0.0}
    cx, cy = feat["cx"], feat["cy"]
    bw, bh = feat["bw"], feat["bh"]
    length_norm = normalize_score(feat["length_px"], 30, 600)
    horiz = feat["horizontalness"]
    diag = feat["diagonalness"]
    vert = feat["verticalness"]
    # Heart: upper zone, long horizontal/curved line under fingers.
    heart_zone = 1.0 - min(abs(cy - 0.30) / 0.22, 1.0)
    heart_score = 0.36*heart_zone + 0.24*horiz + 0.16*normalize_score(bw, 0.16, 0.55) + 0.24*length_norm
    # Head: middle zone, horizontal/diagonal, long enough.
    head_zone = 1.0 - min(abs(cy - 0.50) / 0.24, 1.0)
    head_score = 0.34*head_zone + 0.18*horiz + 0.16*diag + 0.14*normalize_score(bw, 0.14, 0.60) + 0.18*length_norm
    # Life: thumb side, vertical range/curve near Venus mount.
    if thumb_side == "left":
        side_score = 1.0 - min(abs(cx - 0.28) / 0.32, 1.0)
    else:
        side_score = 1.0 - min(abs(cx - 0.72) / 0.32, 1.0)
    if thumb_side == "unknown":
        side_score = 0.0
    lower_span = normalize_score(feat["y1"] - feat["y0"], 0.12, 0.50)
    life_score = 0.30*side_score + 0.22*lower_span + 0.14*diag + 0.10*vert + 0.24*length_norm
    return {
        "life_line": float(np.clip(life_score, 0, 1)),
        "head_line": float(np.clip(head_score, 0, 1)),
        "heart_line": float(np.clip(heart_score, 0, 1)),
        "minor": float(np.clip(0.45*length_norm + 0.20*max(horiz, diag, vert), 0, 1)),
    }


def create_strict_pseudo_mask(img_rgb: np.ndarray, out_size: int = 1024, strict_line_threshold: float = 0.72) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return crop_rgb, semantic_mask, report."""
    crop, palm_mask, crop_meta = crop_palm_auto(img_rgb, out_size=out_size)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = clahe_gray(gray_blur)
    blackhat = directional_blackhat(clahe)
    gabor = gabor_max_response(clahe)
    edges = cv2.Canny(clahe, 40, 130)
    # Combine maps. Lines in palm are darker; blackhat strong. Gabor/edges assist.
    combined = cv2.addWeighted(blackhat, 0.52, gabor, 0.33, 0)
    combined = cv2.addWeighted(combined, 0.85, edges, 0.15, 0)
    # Restrict to palm/hand area, erode boundary to avoid contour edges.
    pm = (palm_mask > 0).astype(np.uint8) * 255
    pm_erode = cv2.erode(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
    combined = cv2.bitwise_and(combined, pm_erode)
    # Adaptive threshold on nonzero palm pixels.
    palm_vals = combined[pm_erode > 0]
    if palm_vals.size == 0:
        line_binary = np.zeros_like(gray)
    else:
        thr = max(18, int(np.percentile(palm_vals, 88)))
        _, line_binary = cv2.threshold(combined, thr, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    line_binary = cv2.morphologyEx(line_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    line_binary = remove_small_components(line_binary, min_area=max(12, int(out_size*out_size*0.000015)))
    skeleton = skeletonize_mask(line_binary)
    skeleton = remove_small_components(skeleton, min_area=max(8, int(out_size*out_size*0.000006)))
    thumb_side = estimate_thumb_side(pm)

    # Semantic mask initially palm area.
    sem = np.zeros((out_size, out_size), dtype=np.uint8)
    sem[pm > 0] = 1
    # Connected components on skeleton.
    sk_bin = (skeleton > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sk_bin, 8)
    comps = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < max(10, out_size * 0.012):
            continue
        comp = (labels == i).astype(np.uint8)
        feat = component_features(comp)
        if not feat:
            continue
        scores = score_component_as_line(feat, thumb_side)
        comps.append({"idx": i, "area": int(area), "feat": feat, "scores": scores, "mask": comp})
    # Greedy assign best component per main line with minimum threshold.
    assigned = set()
    line_scores = {"life_line": 0.0, "head_line": 0.0, "heart_line": 0.0}
    line_lengths = {"life_line": 0.0, "head_line": 0.0, "heart_line": 0.0}
    label_to_class = {"life_line": 2, "head_line": 3, "heart_line": 4}
    for line_name in ["life_line", "head_line", "heart_line"]:
        best = None
        best_score = 0.0
        for c in comps:
            if c["idx"] in assigned:
                continue
            s = c["scores"][line_name]
            if s > best_score:
                best_score = s
                best = c
        if best is not None and best_score >= strict_line_threshold:
            sem[best["mask"] > 0] = label_to_class[line_name]
            assigned.add(best["idx"])
            line_scores[line_name] = float(best_score)
            line_lengths[line_name] = float(best["feat"]["length_px"])
    # Simian requires supervised annotations; a missing heart/head is insufficient evidence.
    line_scores["simian_line"] = 0.0
    line_lengths["simian_line"] = 0.0
    # Remaining line-like components = minor/unknown, but only if sufficiently plausible.
    minor_pixels = 0
    for c in comps:
        if c["idx"] in assigned:
            continue
        if c["scores"]["minor"] >= 0.25:
            sem[c["mask"] > 0] = 5
            minor_pixels += int(np.sum(c["mask"] > 0))
    # Dilate line classes slightly for train masks.
    sem2 = sem.copy()
    for cls in (2, 3, 4, 5):
        cm = (sem == cls).astype(np.uint8) * 255
        if np.sum(cm) > 0:
            cm = cv2.dilate(cm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            sem2[cm > 0] = cls
    sem = sem2
    line_pixel_ratio = float(np.mean(np.isin(sem, [2,3,4,5])))
    main_count = int(sum(1 for v in line_scores.values() if v >= strict_line_threshold))
    mask_quality_score = float(np.clip(
        0.25 * normalize_score(line_pixel_ratio, 0.002, 0.035) +
        0.25 * (main_count / 3.0) +
        0.25 * np.mean(list(line_scores.values())) +
        0.25 * crop_meta.get("hand_score", 0.0),
        0, 1
    ))
    report = {
        "crop_meta": crop_meta,
        "thumb_side_estimate": thumb_side,
        "line_scores": line_scores,
        "line_lengths": line_lengths,
        "main_line_count": main_count,
        "minor_pixels": minor_pixels,
        "line_pixel_ratio": line_pixel_ratio,
        "mask_quality_score": mask_quality_score,
        "component_count": len(comps),
        "strict_line_threshold": strict_line_threshold,
    }
    validate_mask(sem, crop.shape[:2])
    report.update({"inference_source": "classical CV fallback", "model_used": False, "class_map_version": CLASS_VERSION})
    return crop, sem, report


def make_overlay(img_rgb: np.ndarray, sem: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    colors = {
        0: (0, 0, 0),
        1: (50, 180, 50),
        2: (255, 0, 0),      # life
        3: (0, 80, 255),      # head
        4: (255, 0, 255),     # heart
        5: (255, 200, 0),     # minor
        6: (0, 230, 230),     # simian
    }
    overlay = img_rgb.copy()
    color_img = np.zeros_like(img_rgb)
    for cls, col in colors.items():
        color_img[sem == cls] = col
    mask = sem > 0
    overlay[mask] = (img_rgb[mask] * (1 - alpha) + color_img[mask] * alpha).astype(np.uint8)
    return overlay

# -----------------------------
# Dataset build / download
# -----------------------------
def download_commons_images(out_dir: Path, target_count: int = 1000, max_downloads: int = 5000,
                            queries: Optional[List[str]] = None, sleep: float = 0.05) -> List[Path]:
    raise RuntimeError("Remote dataset download is disabled. Supply a documented local --input_dir.")


def split_rows(rows: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    return group_rows(rows, seed=seed)

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def build_pseudomask_dataset(input_paths: List[Path], output_dir: Path, out_size: int = 1024,
                             min_short_side: int = 1024, blur_threshold: float = 60.0,
                             strict_line_threshold: float = 0.72, keep_threshold: float = 0.85,
                             min_good_lines: int = 2, exact_dedupe: bool = True,
                             max_images: Optional[int] = None, seed: int = 42) -> Dict[str, Any]:
    validate_size(out_size)
    if not input_paths:
        raise ValueError("Empty input dataset: supply local images.")
    ensure_dir(output_dir)
    img_dir = ensure_dir(output_dir / "images")
    mask_dir = ensure_dir(output_dir / "masks")
    overlay_dir = ensure_dir(output_dir / "overlays")
    rejected_dir = ensure_dir(output_dir / "rejected")
    report_dir = ensure_dir(output_dir / "reports")
    seen_hashes = set()
    rows_all: List[Dict[str, Any]] = []
    rows_keep: List[Dict[str, Any]] = []
    rejected_counts: Dict[str, int] = {}
    if max_images:
        input_paths = input_paths[:max_images]
    for p in input_paths:
        rec: Dict[str, Any] = {"source_path": str(p)}
        try:
            if exact_dedupe:
                h = sha256_file(p)
                if h in seen_hashes:
                    rec["accepted"] = 0; rec["reject_reason"] = "exact_duplicate"
                    rejected_counts["exact_duplicate"] = rejected_counts.get("exact_duplicate", 0) + 1
                    rows_all.append(rec)
                    continue
                seen_hashes.add(h)
                rec["sha256"] = h
            img = imread_rgb(p)
            if img is None:
                rec["accepted"] = 0; rec["reject_reason"] = "bad_read"
                rejected_counts["bad_read"] = rejected_counts.get("bad_read", 0) + 1
                rows_all.append(rec); continue
            diag = analyze_quality(img, min_short_side=min_short_side)
            rec.update({f"diag_{k}": v for k, v in diag.scores.items()})
            rec.update({f"measure_{k}": v for k, v in diag.measurements.items()})
            if diag.measurements["short_side"] < min_short_side:
                rec["accepted"] = 0; rec["reject_reason"] = "too_small"
                rejected_counts["too_small"] = rejected_counts.get("too_small", 0) + 1
                rows_all.append(rec); continue
            if diag.measurements["laplacian_variance"] < blur_threshold:
                rec["accepted"] = 0; rec["reject_reason"] = "too_blurry"
                rejected_counts["too_blurry"] = rejected_counts.get("too_blurry", 0) + 1
                rows_all.append(rec); continue
            if diag.scores["image_quality_score"] < 0.65 or diag.scores["palm_detection_score"] < 0.35:
                rec["accepted"] = 0; rec["reject_reason"] = diag.main_reason.lower()
                rejected_counts[rec["reject_reason"]] = rejected_counts.get(rec["reject_reason"], 0) + 1
                rows_all.append(rec); continue
            rec.update({"source_group": str(p.parent), "phash": perceptual_hash(img), "sha256": sha256_file(p), "class_map_version": CLASS_VERSION, "label_provenance": "classical pseudo-label; not independent ground truth"})
            crop, sem, rep = create_strict_pseudo_mask(img, out_size=out_size, strict_line_threshold=strict_line_threshold)
            rec.update({"mask_quality_score": rep["mask_quality_score"], "main_line_count": rep["main_line_count"], "line_pixel_ratio": rep["line_pixel_ratio"], "component_count": rep["component_count"], "thumb_side": rep["thumb_side_estimate"]})
            for ln, s in rep["line_scores"].items():
                rec[f"score_{ln}"] = s
            keep = (rep["mask_quality_score"] >= keep_threshold and rep["main_line_count"] >= min_good_lines)
            stem = safe_name(p.stem)[:100] + "_" + hashlib.md5(str(p).encode("utf-8")).hexdigest()[:8]
            if keep:
                img_path = img_dir / f"{stem}.jpg"
                mask_path = mask_dir / f"{stem}_mask.png"
                overlay_path = overlay_dir / f"{stem}_overlay.jpg"
                if any(x.exists() for x in (img_path, mask_path, overlay_path)):
                    raise FileExistsError(f"Output collision: {img_path}; use a new directory.")
                imwrite_rgb(img_path, crop)
                imwrite_gray(mask_path, sem)
                imwrite_rgb(overlay_path, make_overlay(crop, sem))
                rec.update({
                    "accepted": 1,
                    "reject_reason": "",
                    "image_path": str(img_path.relative_to(output_dir)),
                    "mask_path": str(mask_path.relative_to(output_dir)),
                    "overlay_path": str(overlay_path.relative_to(output_dir)),
                })
                rows_keep.append(rec.copy())
            else:
                rec["accepted"] = 0
                rec["reject_reason"] = "low_mask_quality_or_few_lines"
                rejected_counts["low_mask_quality_or_few_lines"] = rejected_counts.get("low_mask_quality_or_few_lines", 0) + 1
                # Save a few rejected overlays for debugging, but not all.
                if rejected_counts["low_mask_quality_or_few_lines"] <= 80:
                    rej_sub = ensure_dir(rejected_dir / "low_mask_quality_or_few_lines")
                    imwrite_rgb(rej_sub / f"{stem}_overlay.jpg", make_overlay(crop, sem))
            rows_all.append(rec)
        except Exception as e:
            rec["accepted"] = 0; rec["reject_reason"] = "exception"; rec["error"] = repr(e)
            rejected_counts["exception"] = rejected_counts.get("exception", 0) + 1
            rows_all.append(rec)
    rows_keep = split_rows(rows_keep, seed=seed)
    write_csv(output_dir / "labels_pseudo_strict.csv", rows_keep)
    write_csv(output_dir / "labels_pseudo_strict_all.csv", rows_all)
    write_class_map(output_dir)
    summary = {
        "split_counts": split_summary(rows_keep),
        "label_provenance": "classical CV pseudo-labels; pipeline mechanics only",
        "total_input": len(input_paths),
        "accepted": len(rows_keep),
        "rejected": len(rows_all) - len(rows_keep),
        "rejected_counts": rejected_counts,
        "out_size": out_size,
        "strict_line_threshold": strict_line_threshold,
        "keep_threshold": keep_threshold,
        "min_good_lines": min_good_lines,
    }
    with open(report_dir / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    create_eda(output_dir, rows_keep, rows_all)
    return summary


def create_eda(output_dir: Path, rows_keep: List[Dict[str, Any]], rows_all: List[Dict[str, Any]]) -> None:
    """Write EDA tables using the standard csv module; plotting is optional UI work."""
    report_dir = ensure_dir(output_dir / "reports" / "eda")
    if not rows_all:
        return
    counts: Dict[str, int] = {}
    for row in rows_all:
        reason = row.get("reject_reason") or "accepted"
        counts[reason] = counts.get(reason, 0) + 1
    write_json(report_dir / "rejected_counts.json", counts)
    write_csv(report_dir / "image_metrics_all.csv", rows_all)
    write_csv(report_dir / "image_metrics_accepted.csv", rows_keep)
    sample_paths = [output_dir / row["overlay_path"] for row in rows_keep[:25] if row.get("overlay_path")]
    images = []
    for sample_path in sample_paths:
        image = imread_rgb(sample_path)
        if image is not None:
            images.append(cv2.resize(image, (220, 220)))
    if images:
        cols = 5
        rows = math.ceil(len(images) / cols)
        canvas = np.zeros((rows * 220, cols * 220, 3), dtype=np.uint8)
        for index, image in enumerate(images):
            y = (index // cols) * 220
            x = (index % cols) * 220
            canvas[y:y + 220, x:x + 220] = image
        imwrite_rgb(report_dir / "sample_overlay_grid.jpg", canvas)

# NumPy model/training
# -------------------
# Project policy: model training and inference use NumPy only.  No PyTorch,
# scikit-learn, TensorFlow or other ML framework is imported here.
TORCH_AVAILABLE = False  # retained only so old callers fail closed
NUMPY_MODEL_AVAILABLE = True
MODEL_FEATURE_NAMES = (
    "red", "green", "blue", "gray", "saturation", "blackhat", "gradient", "x", "y",
)
MODEL_FRAMEWORK = "NumPy diagonal-Gaussian pixel classifier"


def _model_features(image: np.ndarray) -> np.ndarray:
    """Build deterministic per-pixel colour, line-response and position features."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Model features require an RGB image.")
    h, w = image.shape[:2]
    rgb = image.astype(np.float32) / 255.0
    gray_u8 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1] / 255.0
    blackhat_u8 = cv2.morphologyEx(
        gray_u8,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    blackhat = blackhat_u8.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy).clip(0.0, 1.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = xx / max(w - 1, 1)
    y = yy / max(h - 1, 1)
    return np.stack(
        [rgb[..., 0], rgb[..., 1], rgb[..., 2], gray, saturation, blackhat, gradient, x, y],
        axis=-1,
    ).astype(np.float32)


class NumpyPixelModel:
    """Small, inspectable segmentation model fitted from pixel statistics.

    Each class stores a diagonal Gaussian in the engineered feature space.  The
    model is trained and evaluated with NumPy and saved as a compressed .npz;
    this keeps the checkpoint portable and avoids executable/pickle formats.
    """

    def __init__(self, means, stds, priors, input_size, metadata=None):
        self.means = np.asarray(means, dtype=np.float32)
        self.stds = np.maximum(np.asarray(stds, dtype=np.float32), 1e-3)
        self.priors = np.maximum(np.asarray(priors, dtype=np.float32), 1e-8)
        self.priors = self.priors / float(self.priors.sum())
        self.input_size = validate_size(int(input_size))
        self.metadata = dict(metadata or {})
        self.prototype_training_provenance = self.metadata.get("run_info", {})
        counts = self.metadata.get("class_sample_counts")
        self.learned_classes = np.asarray(counts, dtype=np.float64) > 0 if counts is not None else np.ones(len(CLASS_MAP), dtype=bool)
        if self.learned_classes.shape != (len(CLASS_MAP),) or not self.learned_classes.any():
            raise ValueError("Invalid/missing learned-class support in checkpoint.")
        if self.means.shape != (len(CLASS_MAP), len(MODEL_FEATURE_NAMES)):
            raise ValueError("Invalid NumPy model means shape.")
        if self.stds.shape != self.means.shape or self.priors.shape != (len(CLASS_MAP),):
            raise ValueError("Invalid NumPy model checkpoint shapes.")

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        features = _model_features(resized)
        delta = (features[..., None, :] - self.means[None, None, :, :]) / self.stds[None, None, :, :]
        logits = -0.5 * np.sum(delta * delta, axis=-1)
        logits += np.log(self.priors)[None, None, :]
        logits -= np.sum(np.log(self.stds), axis=1)[None, None, :]
        logits[..., ~self.learned_classes] = -np.inf
        logits -= np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(np.clip(logits, -80.0, 40.0))
        probs[..., ~self.learned_classes] = 0.0
        probs /= np.maximum(np.sum(probs, axis=-1, keepdims=True), 1e-8)
        return probs.transpose(2, 0, 1).astype(np.float32)

    def predict_mask(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.predict_proba(image)
        return np.argmax(probs, axis=0).astype(np.uint8), probs


def _save_numpy_checkpoint(path: Path, model: NumpyPixelModel, metadata: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = dict(metadata)
    if "class_sample_counts" in model.metadata:
        payload["class_sample_counts"] = model.metadata["class_sample_counts"]
    payload.setdefault("class_map", {str(k): v for k, v in CLASS_MAP.items()})
    payload.setdefault("class_map_version", CLASS_VERSION)
    payload.setdefault("model_version", MODEL_VERSION)
    payload.setdefault("framework", MODEL_FRAMEWORK)
    np.savez_compressed(
        str(tmp),
        means=model.means.astype(np.float32),
        stds=model.stds.astype(np.float32),
        priors=model.priors.astype(np.float32),
        metadata=np.asarray(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    )
    generated = Path(str(tmp) + ".npz")
    if generated != path:
        os.replace(generated, path)
    else:
        os.replace(tmp, path)


def _load_resized_pair(data_root: Path, row: Dict[str, Any], input_size: int) -> Tuple[np.ndarray, np.ndarray]:
    image = imread_rgb(data_root / str(row["image_path"]))
    mask = cv2.imread(str(data_root / str(row["mask_path"])), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        raise RuntimeError(f"Cannot read training row: {row}")
    validate_mask(mask, image.shape[:2])
    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    validate_mask(mask, image.shape[:2])
    return image, mask.astype(np.uint8)


def _fit_numpy_model(data_root: Path, rows: List[Dict[str, Any]], input_size: int, seed: int, max_pixels_per_class: int = 6000) -> NumpyPixelModel:
    feature_count = len(MODEL_FEATURE_NAMES)
    sums = np.zeros((len(CLASS_MAP), feature_count), dtype=np.float64)
    squares = np.zeros_like(sums)
    counts = np.zeros(len(CLASS_MAP), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for row in rows:
        image, mask = _load_resized_pair(data_root, row, input_size)
        features = _model_features(image).reshape(-1, feature_count)
        labels = mask.reshape(-1)
        for class_id in range(len(CLASS_MAP)):
            indexes = np.flatnonzero(labels == class_id)
            if indexes.size == 0:
                continue
            if indexes.size > max_pixels_per_class:
                indexes = rng.choice(indexes, size=max_pixels_per_class, replace=False)
            values = features[indexes].astype(np.float64)
            sums[class_id] += values.sum(axis=0)
            squares[class_id] += np.square(values).sum(axis=0)
            counts[class_id] += values.shape[0]
    if not np.any(counts):
        raise ValueError("No valid pixels found for NumPy model training.")
    global_mean = sums.sum(axis=0) / max(float(counts.sum()), 1.0)
    global_var = np.maximum(squares.sum(axis=0) / max(float(counts.sum()), 1.0) - global_mean * global_mean, 0.0)
    means = np.zeros_like(sums, dtype=np.float32)
    stds = np.zeros_like(sums, dtype=np.float32)
    for class_id in range(len(CLASS_MAP)):
        if counts[class_id] > 0:
            means[class_id] = sums[class_id] / counts[class_id]
            variance = np.maximum(squares[class_id] / counts[class_id] - means[class_id] ** 2, 0.0)
            stds[class_id] = np.sqrt(variance).astype(np.float32)
        else:
            means[class_id] = global_mean
            stds[class_id] = np.sqrt(global_var).astype(np.float32)
    # A square-root prior prevents background pixels from overwhelming rare lines.
    priors = np.sqrt(counts.astype(np.float32) + 1.0)
    priors /= float(priors.sum())
    stds = np.maximum(stds, 0.025)
    return NumpyPixelModel(
        means,
        stds,
        priors,
        input_size,
        {"class_sample_counts": counts.tolist(), "feature_names": list(MODEL_FEATURE_NAMES)},
    )


class PalmMaskDataset:
    """Compatibility reader exposing manifest rows without a tensor framework."""

    def __init__(self, data_root: Path, labels_csv: Path, split: str, input_size: int = 512, augment: bool = False):
        if augment and split != "train":
            raise ValueError("Validation/test augmentation is forbidden.")
        self.data_root = Path(data_root)
        self.input_size = validate_size(input_size)
        self.rows = [row for row in read_manifest(self.data_root, labels_csv) if row["split"] == split]

    def __len__(self):
        return len(self.rows)

    def _load(self, row):
        return _load_resized_pair(self.data_root, row, self.input_size)

    def __getitem__(self, index):
        return self._load(self.rows[index])


def compute_dice_metrics(prediction, target, num_classes=len(CLASS_MAP)) -> Dict[str, float]:
    pred = np.asarray(prediction)
    target = np.asarray(target)
    if pred.ndim == 4:
        pred = np.argmax(pred, axis=1)
    if target.ndim == 4:
        target = np.argmax(target, axis=1)
    if pred.ndim == 2:
        pred = pred[None, ...]
    if target.ndim == 2:
        target = target[None, ...]
    out = {}
    for class_id in range(1, num_classes):
        p = pred == class_id
        t = target == class_id
        denom = int(p.sum() + t.sum())
        out[f"dice_{CLASS_MAP[class_id]}"] = float(2 * np.logical_and(p, t).sum() / denom) if denom else 0.0
    out["dice_main_lines_mean"] = float(np.mean([out.get("dice_life_line", 0.0), out.get("dice_head_line", 0.0), out.get("dice_heart_line", 0.0)]))
    return out


def _evaluate_numpy_model(model: NumpyPixelModel, data_root: Path, rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], np.ndarray]:
    cm = np.zeros((len(CLASS_MAP), len(CLASS_MAP)), dtype=np.int64)
    for row in rows:
        image, target = _load_resized_pair(data_root, row, model.input_size)
        prediction, _ = model.predict_mask(image)
        cm += confusion_matrix(prediction, target)
    return metrics_from_confusion(cm), cm


def _write_history_csv(path: Path, history: List[Dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def train_model(args: argparse.Namespace) -> Dict[str, Any]:
    validate_parameters(args)
    seed_everything(args.seed)
    data_root = Path(args.data_root)
    labels_csv = Path(args.labels_csv)
    out_dir = ensure_dir(Path(args.out_dir))
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    log_dir = ensure_dir(out_dir / "logs")
    input_size = validate_size(int(args.input_size))
    audited_rows = read_manifest(data_root, labels_csv)
    train_rows = [row for row in audited_rows if row["split"] == "train"]
    val_rows = [row for row in audited_rows if row["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError("Non-empty independent train and val splits required.")
    run_info = provenance(args, labels_csv)
    run_info["split_counts"] = split_summary(audited_rows)
    run_info["label_provenance"] = sorted({str(row.get("label_provenance") or row.get("cls_quality") or "unverified annotations") for row in audited_rows})
    run_info["evaluation_scope"] = "fixture-only" if all("deterministic test fixture" in value for value in run_info["label_provenance"]) else "annotation/pseudo-label mechanics; no independent accuracy claim"
    run_info["framework"] = MODEL_FRAMEWORK
    run_info["model_version"] = MODEL_VERSION
    run_info["device"] = "cpu"
    write_json(out_dir / "run_config.json", run_info)
    model = _fit_numpy_model(data_root, train_rows, input_size, int(args.seed), int(getattr(args, "max_pixels_per_class", 6000)))
    model.metadata.update({"run_info": run_info, "input_size": input_size})
    detailed, _ = _evaluate_numpy_model(model, data_root, val_rows)
    metrics = {f"dice_{name}": values["dice"] for name, values in detailed["per_class"].items()}
    metric_values = [float(metrics.get(f"dice_{name}")) for name in LINE_CLASSES.values() if metrics.get(f"dice_{name}") is not None]
    metric = float(np.mean(metric_values)) if metric_values else 0.0
    metrics["dice_main_lines_mean"] = metric
    write_json(log_dir / "validation_metrics.json", {"split": "val", "provenance": run_info, **detailed})
    history = [{"epoch": 0, "train_loss": 0.0, "val_loss": 0.0, **metrics}]
    _write_history_csv(log_dir / "history.csv", history)
    write_json(log_dir / "history.json", history)
    checkpoint_meta = {
        "class_map": {str(k): v for k, v in CLASS_MAP.items()},
        "class_map_version": CLASS_VERSION,
        "model_version": MODEL_VERSION,
        "framework": MODEL_FRAMEWORK,
        "input_size": input_size,
        "validation_scores": metrics,
        "run_info": run_info,
        "epochs_requested": int(args.epochs),
        "training_method": "deterministic diagonal Gaussian class statistics over engineered pixels",
    }
    best_checkpoint = ckpt_dir / "best.npz"
    last_checkpoint = ckpt_dir / "last.npz"
    _save_numpy_checkpoint(last_checkpoint, model, checkpoint_meta)
    _save_numpy_checkpoint(best_checkpoint, model, checkpoint_meta)
    (ckpt_dir / "last.npz.sha256").write_text(sha256_file(last_checkpoint), encoding="ascii")
    (ckpt_dir / "best.npz.sha256").write_text(sha256_file(best_checkpoint), encoding="ascii")
    write_json(out_dir / "class_validation_scores.json", metrics)
    return {"best_metric": metric, "history": history, "framework": MODEL_FRAMEWORK, "checkpoint": str(best_checkpoint)}


def finish_partial_accumulation(*_args, **_kwargs):
    # Kept as a no-op compatibility hook for callers of the old trainer.
    return None


def safe_checkpoint(checkpoint, device=None, expected_sha256=""):
    from prototype_common import validate_npz_size
    checkpoint = Path(checkpoint)
    if not expected_sha256 or len(str(expected_sha256)) != 64:
        raise ValueError("Checkpoint requires explicit --checkpoint-sha256 from a trusted source.")
    if checkpoint.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("Checkpoint exceeds prototype 512 MiB limit.")
    if sha256_file(checkpoint).lower() != str(expected_sha256).lower():
        raise ValueError("Checkpoint SHA-256 mismatch.")
    if checkpoint.suffix.lower() != ".npz":
        raise ValueError("Only NumPy .npz checkpoints are supported by this project.")
    validate_npz_size(checkpoint, 512 * 1024 * 1024)
    try:
        with np.load(checkpoint, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            means = np.asarray(payload["means"], dtype=np.float32)
            stds = np.asarray(payload["stds"], dtype=np.float32)
            priors = np.asarray(payload["priors"], dtype=np.float32)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid NumPy checkpoint: {exc}") from exc
    expected_map = {str(k): v for k, v in CLASS_MAP.items()}
    if not isinstance(metadata, dict) or metadata.get("class_map") != expected_map or metadata.get("class_map_version") != CLASS_VERSION or metadata.get("model_version") != MODEL_VERSION:
        raise ValueError("Incompatible checkpoint schema, class map or model version.")
    validate_size(int(metadata.get("input_size")))
    return {"means": means, "stds": stds, "priors": priors, "metadata": metadata}


def load_model_for_predict(checkpoint: Path, device=None, expected_sha256=""):
    ck = safe_checkpoint(checkpoint, device, expected_sha256)
    metadata = ck["metadata"]
    model = NumpyPixelModel(ck["means"], ck["stds"], ck["priors"], metadata["input_size"], metadata)
    model.prototype_training_provenance = metadata.get("run_info", {})
    return model, int(metadata["input_size"]), metadata.get("validation_scores", {})

# Prediction and Vietnamese output
# -----------------------------
def line_feature_from_mask(sem: np.ndarray, cls: int) -> Dict[str, Any]:
    return measure_line(sem, cls, skeletonize_mask)


def vietnamese_reading(accepted: Dict[str, Dict[str, Any]]) -> str:
    return "Geometry only; no personality, health or future inference:\n" + "\n".join(
        f"{name}: tỷ lệ so với lòng bàn tay {item.get('length_norm', 0):.3f}" for name, item in accepted.items())

def describe_length(x: float) -> str:
    if x >= 0.55: return "dài"
    if x >= 0.30: return "vừa"
    return "ngắn"


def _predict_image(args: argparse.Namespace) -> Dict[str, Any]:
    img_path = Path(args.image)
    img = imread_rgb(img_path)
    if img is None:
        res = DiagnosticResult("need_retake", "BAD_READ", "Không đọc được file ảnh.", "Hãy gửi lại ảnh định dạng JPG hoặc PNG rõ ràng.", {}, {})
        return asdict(res)
    diag = analyze_quality(img, min_short_side=args.min_short_side)
    if diag.status != "ok" and diag.scores.get("image_quality_score",0) < args.min_image_quality:
        out = asdict(diag); out["status"] = "need_retake"; return out
    crop, pseudo_sem, pseudo_report = create_strict_pseudo_mask(img, out_size=args.out_size, strict_line_threshold=args.strict_line_threshold)
    sem = pseudo_sem
    model_scores: Dict[str, Any] = {}
    model = None
    if args.checkpoint and Path(args.checkpoint).exists():
        model, input_size, val_scores = load_model_for_predict(Path(args.checkpoint), None, args.checkpoint_sha256)
        probs = model.predict_proba(crop)
        pred = np.argmax(probs, axis=0).astype(np.uint8)
        pred = cv2.resize(pred, (args.out_size, args.out_size), interpolation=cv2.INTER_NEAREST)
        sem = pred
        model_scores["validation_scores"] = val_scores
        for cls, name in LINE_CLASSES.items():
            cls_mask = pred == cls
            if np.any(cls_mask):
                pmap = cv2.resize(probs[cls], (args.out_size, args.out_size), interpolation=cv2.INTER_LINEAR)
                model_scores[name] = float(np.mean(pmap[cls_mask]))
            else:
                model_scores[name] = 0.0
    else:
        # Use deterministic classical pseudo scores as the explicit fallback.
        model_scores = dict(pseudo_report.get("line_scores", {}))
        model_scores["validation_scores"] = {}
    mirrored = {"yes": True, "no": False}.get(getattr(args, "mirrored", "unknown"))
    skin_mask, _, _ = detect_skin_largest_contour(img)
    handedness = infer_handedness(skin_mask, mirrored=mirrored)
    features = resolve_transverse_lines({name: line_feature_from_mask(sem, cls) for cls, name in LINE_CLASSES.items()})
    accepted: Dict[str, Dict[str, Any]] = {}
    rejected: Dict[str, Any] = {}
    for cls, name in LINE_CLASSES.items():
        feat = features[name]
        pseudo_s = float(pseudo_report.get("line_scores", {}).get(name, 0.0))
        pred_s = float(model_scores.get(name, pseudo_s))
        val_scores = model_scores.get("validation_scores", {}) if isinstance(model_scores.get("validation_scores", {}), dict) else {}
        val_key = f"dice_{name}"
        raw_val = val_scores.get(val_key)
        val_s = float(raw_val if raw_val is not None else (0.0 if args.require_validation_scores else 1.0))
        template_s = pred_s if model is not None else pseudo_s
        length_s = normalize_score(feat.get("length_norm", 0.0), 0.10, 0.55) if feat.get("detected") else 0.0
        final = min(diag.scores.get("image_quality_score",0), diag.scores.get("line_visibility_score",0), pred_s, template_s, max(length_s, 0.35), val_s)
        item = {"heuristic_score": float(final), "model_ranking_score": pred_s, "template_score": template_s, "validation_score": val_s, **feat}
        if feat.get("detected") and final >= args.read_threshold and handedness["side"] != "unknown":
            accepted[name] = item
        else:
            rejected[name] = item
    if not accepted:
        # Determine best user-actionable reason.
        reason = diag.main_reason if diag.main_reason != "OK" else "NO_CONFIDENT_LINES"
        if diag.scores.get("line_visibility_score", 0) < 0.55:
            reason = "LOW_LINE_CONTRAST"
        reason_vi, fix_vi = FAILURE_MESSAGES.get(reason, FAILURE_MESSAGES["NO_CONFIDENT_LINES"])
        out = {
            "status": "need_retake",
            "main_reason": reason,
            "reason_vi": reason_vi,
            "fix_vi": fix_vi,
            "scores": diag.scores,
            "line_scores": rejected,
            "pseudo_report": pseudo_report,
        }
    else:
        reading = vietnamese_reading(accepted)
        out = {
            "status": "success",
            "accepted_lines": accepted,
            "rejected_lines": rejected,
            "scores": diag.scores,
            "reading_vi": reading,
            "pseudo_report": pseudo_report,
        }
    validate_mask(sem, crop.shape[:2])
    out["handedness"] = handedness
    out["measurement_version"] = "palm-geodesic-v2"
    if handedness["side"] == "unknown":
        out.update(main_reason="HAND_SIDE_UNCERTAIN", reason_vi="Chưa xác định chắc chắn tay trái/phải.", fix_vi="Chụp một lòng bàn tay mở, ngón hướng lên và khai báo ảnh có lật gương không.")
    out["segmentation_available"] = True
    out["crop_meta"] = pseudo_report["crop_meta"]
    out["training_provenance"] = getattr(model, "prototype_training_provenance", {}) if args.checkpoint else {"labels": "classical pseudo-labels; not independent ground truth"}
    if args.out_mask and not imwrite_gray(Path(args.out_mask), sem):
        raise OSError("Could not write mask.")
    if args.out_overlay and not imwrite_rgb(Path(args.out_overlay), make_overlay(crop, sem)):
        raise OSError("Could not write overlay.")
    if args.print_reading and out.get("status") == "success":
        print(out["reading_vi"])
    elif args.print_reading:
        print(out.get("reason_vi", "Không thể phân tích ảnh này."))
        print(out.get("fix_vi", ""))
    return out


def predict_image(args: argparse.Namespace) -> Dict[str, Any]:
    import uuid
    request_id = getattr(args, "request_id", None) or uuid.uuid4().hex
    started = time.perf_counter()
    outputs = []
    safe_json = False
    try:
        outputs = [Path(x).resolve() for x in (args.out_json, args.out_mask, args.out_overlay) if x]
        protected = {Path(args.image).resolve()}
        if args.checkpoint:
            protected.add(Path(args.checkpoint).resolve())
        if len(set(outputs)) != len(outputs) or protected.intersection(outputs):
            outputs = []
            raise ValueError("Output paths must be distinct and cannot overwrite input/checkpoint.")
        safe_json = True
        for path in outputs:
            path.unlink(missing_ok=True)
        validate_parameters(args)
        if args.checkpoint and not Path(args.checkpoint).is_file():
            raise FileNotFoundError("Explicit checkpoint does not exist; classical fallback was not substituted.")
        if args.checkpoint and not args.checkpoint_sha256:
            raise ValueError("An explicit trusted --checkpoint-sha256 is required.")
        if not args.checkpoint and not args.allow_classical_fallback:
            raise ValueError("Supply --checkpoint and --checkpoint-sha256, or explicitly --allow-classical-fallback.")
        seed_everything(getattr(args, "seed", 42))
        out = _predict_image(args)
    except Exception as e:
        for path in outputs:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        out = {"status": "error", "main_reason": type(e).__name__, "error": str(e)}
    used = bool(args.checkpoint) and out.get("segmentation_available", False)
    out.update({"schema_version": 1, "request_id": request_id, "model_used": used,
                "inference_source": ("trained model" if used else "classical CV fallback") if out.get("segmentation_available") else "none",
                "class_map_version": CLASS_VERSION, "elapsed_seconds": time.perf_counter() - started,
                "score_semantics": "uncalibrated heuristic ranking; not probability",
                "provenance": provenance(args)})
    out["success_marker"] = request_id if out.get("segmentation_available") else None
    try:
        if safe_json:
            save_json_if_needed(args, out)
    except OSError as e:
        out.update(status="error", error=f"Cannot write output JSON: {e}", success_marker=None)
    print(json.dumps(out, ensure_ascii=False, allow_nan=False))
    return out


def save_json_if_needed(args: argparse.Namespace, obj: Dict[str, Any]) -> None:
    out_json = getattr(args, "out_json", None)
    if out_json:
        ensure_dir(Path(out_json).parent)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------------
# CLI
# -----------------------------
def cmd_doctor(args: argparse.Namespace) -> None:
    print("Python:", sys.version)
    print("OpenCV:", cv2.__version__)
    print("NumPy:", np.__version__)
    print("Matplotlib: not required by the core pipeline")
    print("Requests: not required; remote downloads are disabled")
    print("Model framework:", MODEL_FRAMEWORK)
    print("Torch: disabled by project policy (NumPy model only)")
    print("scikit-learn: disabled by project policy")
    print("Class map:", CLASS_VERSION)
    print("Model inference: optional trusted NumPy .npz checkpoint plus SHA-256")
    print("Core CV and NumPy model: OK")

def cmd_build(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root)
    dataset_dir = ensure_dir(project_root / args.output_dir)
    raw_dir = ensure_dir(dataset_dir / "00_raw_downloads")
    if args.input_dir:
        paths = list_images(Path(args.input_dir))
    else:
        print("Remote downloads are disabled; supply --input_dir.")
        paths = download_commons_images(raw_dir, target_count=args.target_count, max_downloads=args.max_downloads)
    print(f"Tổng ảnh đầu vào: {len(paths)}")
    summary = build_pseudomask_dataset(
        paths, dataset_dir,
        out_size=args.out_size,
        min_short_side=args.min_short_side,
        blur_threshold=args.blur_threshold,
        strict_line_threshold=args.strict_line_threshold,
        keep_threshold=args.keep_threshold,
        min_good_lines=args.min_good_lines,
        exact_dedupe=not args.no_exact_dedupe,
        max_images=args.max_images, seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Dataset CSV:", dataset_dir / "labels_pseudo_strict.csv")


def cmd_mask_folder(args: argparse.Namespace) -> None:
    paths = list_images(Path(args.input_dir))
    summary = build_pseudomask_dataset(
        paths, Path(args.output_dir), out_size=args.out_size,
        min_short_side=args.min_short_side,
        blur_threshold=args.blur_threshold,
        strict_line_threshold=args.strict_line_threshold,
        keep_threshold=args.keep_threshold,
        min_good_lines=args.min_good_lines,
        exact_dedupe=not args.no_exact_dedupe,
        max_images=args.max_images, seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_single_mask(args: argparse.Namespace) -> None:
    img = imread_rgb(Path(args.image))
    if img is None:
        raise RuntimeError("Không đọc được ảnh.")
    out = ensure_dir(Path(args.output_dir))
    diag = analyze_quality(img, min_short_side=args.min_short_side)
    crop, sem, rep = create_strict_pseudo_mask(img, out_size=args.out_size, strict_line_threshold=args.strict_line_threshold)
    imwrite_rgb(out / "crop.jpg", crop)
    imwrite_gray(out / "pseudo_mask.png", sem)
    imwrite_rgb(out / "overlay.jpg", make_overlay(crop, sem))
    report = {"diagnostic": asdict(diag), "pseudo_report": rep, "class_map": CLASS_MAP, "class_map_version": CLASS_VERSION, "inference_source": "classical CV fallback", "model_used": False}
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Palmistry strict fully automatic pipeline")
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=42)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")

    b = sub.add_parser("build", help="Read local images and build pseudo-masks; remote download disabled")
    b.add_argument("--project_root", default=".")
    b.add_argument("--output_dir", default="dataset_strict")
    b.add_argument("--input_dir", default="")
    b.add_argument("--target_count", type=int, default=10000)
    b.add_argument("--max_downloads", type=int, default=30000)
    b.add_argument("--max_images", type=int, default=0)
    b.add_argument("--out_size", type=int, default=1024)
    b.add_argument("--min_short_side", type=int, default=1024)
    b.add_argument("--blur_threshold", type=float, default=60.0)
    b.add_argument("--strict_line_threshold", type=float, default=0.72)
    b.add_argument("--keep_threshold", type=float, default=0.85)
    b.add_argument("--min_good_lines", type=int, default=2)
    b.add_argument("--no_exact_dedupe", action="store_true")

    mf = sub.add_parser("mask-folder", help="Tạo pseudo-mask lọc gắt từ folder ảnh")
    mf.add_argument("--input_dir", required=True)
    mf.add_argument("--output_dir", required=True)
    mf.add_argument("--max_images", type=int, default=0)
    mf.add_argument("--out_size", type=int, default=1024)
    mf.add_argument("--min_short_side", type=int, default=1024)
    mf.add_argument("--blur_threshold", type=float, default=60.0)
    mf.add_argument("--strict_line_threshold", type=float, default=0.72)
    mf.add_argument("--keep_threshold", type=float, default=0.85)
    mf.add_argument("--min_good_lines", type=int, default=2)
    mf.add_argument("--no_exact_dedupe", action="store_true")

    sm = sub.add_parser("single-mask", help="Tạo pseudo-mask cho 1 ảnh")
    sm.add_argument("--image", required=True)
    sm.add_argument("--output_dir", default="single_mask_output")
    sm.add_argument("--out_size", type=int, default=1024)
    sm.add_argument("--min_short_side", type=int, default=512)
    sm.add_argument("--strict_line_threshold", type=float, default=0.72)

    tr = sub.add_parser("train", help="Train segmentation model với dataset strict pseudo-mask")
    tr.add_argument("--data_root", required=True)
    tr.add_argument("--labels_csv", required=True)
    tr.add_argument("--out_dir", required=True)
    tr.add_argument("--input_size", type=int, default=512)
    tr.add_argument("--model_size", choices=["tiny","small","medium"], default="small")
    tr.add_argument("--batch_size", type=int, default=6)
    tr.add_argument("--grad_accum", type=int, default=2)
    tr.add_argument("--epochs", type=int, default=180)
    tr.add_argument("--max_train_hours", type=float, default=7.0)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-3)
    tr.add_argument("--dropout", type=float, default=0.18)
    tr.add_argument("--dice_weight", type=float, default=1.0)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--patience", type=int, default=20)
    tr.add_argument("--resume", choices=["auto","none"], default="none")
    tr.add_argument("--checkpoint-sha256", default="")
    tr.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    tr.add_argument("--num_workers", type=int, default=0)
    tr.add_argument("--amp", action="store_true", default=True)
    tr.add_argument("--cpu", action="store_true")

    pr = sub.add_parser("predict", help="Phân tích 1 ảnh, tự chẩn đoán lỗi, chỉ đọc line > threshold")
    pr.add_argument("--image", required=True)
    pr.add_argument("--mirrored", choices=["yes", "no", "unknown"], default="unknown")
    pr.add_argument("--checkpoint", default="")
    pr.add_argument("--checkpoint-sha256", default="")
    pr.add_argument("--allow-classical-fallback", action="store_true")
    pr.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    pr.add_argument("--request-id", default="")
    pr.add_argument("--out_json", default="prediction_result.json")
    pr.add_argument("--out_mask", default="predicted_mask.png")
    pr.add_argument("--out_overlay", default="prediction_overlay.jpg")
    pr.add_argument("--out_size", type=int, default=1024)
    pr.add_argument("--min_short_side", type=int, default=512)
    pr.add_argument("--min_image_quality", type=float, default=0.50)
    pr.add_argument("--strict_line_threshold", type=float, default=0.72)
    pr.add_argument("--read_threshold", type=float, default=0.85)
    pr.add_argument("--require_validation_scores", action="store_true")
    pr.add_argument("--print_reading", action="store_true", default=True)
    pr.add_argument("--cpu", action="store_true")
    fixture = sub.add_parser("fixture", help="Generate deterministic synthetic test-only images and masks")
    fixture.add_argument("--output_dir", default="artifacts/fixture")
    evaluation = sub.add_parser("evaluate", help="Compare aligned semantic masks; never used for model selection")
    evaluation.add_argument("--prediction", required=True)
    evaluation.add_argument("--target", required=True)
    evaluation.add_argument("--dataset-name", required=True)
    evaluation.add_argument("--split", choices=["train", "val", "test", "fixture-only"], required=True)
    evaluation.add_argument("--inference-source", choices=["trained model", "classical CV fallback", "deterministic test fixture"], required=True)
    evaluation.add_argument("--output", default="artifacts/evaluation.json")
    return p


def validate_parameters(args):
    for key in ("out_size", "input_size"):
        if hasattr(args, key):
            validate_size(getattr(args, key))
    for key in ("batch_size", "grad_accum", "epochs", "patience", "min_short_side"):
        if hasattr(args, key) and getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("strict_line_threshold", "keep_threshold", "read_threshold", "min_image_quality", "dropout"):
        if hasattr(args, key) and not 0 <= getattr(args, key) <= 1:
            raise ValueError(f"{key} must be finite in [0, 1]")
    for key in ("lr", "grad_clip", "weight_decay", "dice_weight", "blur_threshold", "max_train_hours"):
        if hasattr(args, key) and (not math.isfinite(getattr(args, key)) or getattr(args, key) < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
    if getattr(args, "lr", 1) == 0 or getattr(args, "num_workers", 0) < 0:
        raise ValueError("lr must be positive and num_workers nonnegative")


def main():
    parser = make_parser()
    # Config supplies defaults; explicit CLI flags always win.
    pre, _ = parser.parse_known_args()
    config = load_config(pre.config)
    parser.set_defaults(seed=config["seed"])
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                destinations = {a.dest for a in child._actions}
                child.set_defaults(**{k: v for k, v in config.items() if k in destinations})
    args = parser.parse_args()
    if args.cmd != "predict":
        validate_parameters(args)
    if getattr(args, "max_images", 0) == 0:
        args.max_images = None
    if args.cmd == "doctor": cmd_doctor(args)
    elif args.cmd == "build": cmd_build(args)
    elif args.cmd == "mask-folder": cmd_mask_folder(args)
    elif args.cmd == "single-mask": cmd_single_mask(args)
    elif args.cmd == "train": train_model(args)
    elif args.cmd == "predict":
        result = predict_image(args)
        if result.get("status") == "error" or not result.get("segmentation_available"):
            raise SystemExit(2)
    elif args.cmd == "fixture":
        from prototype_fixture import generate_fixture
        print(json.dumps(generate_fixture(Path(args.output_dir)), indent=2))
    elif args.cmd == "evaluate":
        pred = cv2.imread(args.prediction, cv2.IMREAD_UNCHANGED)
        target = cv2.imread(args.target, cv2.IMREAD_UNCHANGED)
        result = {"dataset": args.dataset_name, "split": args.split, "inference_source": args.inference_source,
                  "provenance": provenance(args), **metrics_from_confusion(confusion_matrix(pred, target))}
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
    else: parser.print_help()

if __name__ == "__main__":
    main()
