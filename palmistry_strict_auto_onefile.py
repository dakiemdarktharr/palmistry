#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PALMISTRY STRICT AUTO PIPELINE - ONE FILE

Mục tiêu:
- Không cần người dùng tự tạo/sửa mask.
- Tự tạo pseudo-mask bằng Classical CV.
- Lọc cực gắt ảnh/mask yếu.
- Chỉ train 3 đường chính: Sinh đạo, Trí đạo, Tâm đạo + minor/unknown.
- Khi predict chỉ xuất phần đạt ngưỡng confidence cao.
- Nếu không đọc được, tự chẩn đoán lý do và yêu cầu chụp lại đúng vấn đề.

Commands:
  doctor
  build          Tự tải ảnh hoặc đọc folder ảnh, preprocess, tạo pseudo-mask, EDA.
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

import numpy as np

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

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x

try:
    import requests
except Exception:
    requests = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# -----------------------------
# Constants
# -----------------------------
CLASS_MAP = {
    0: "background",
    1: "palm_area",
    2: "life_line",
    3: "head_line",
    4: "heart_line",
    5: "minor_or_unknown_line",
}
LINE_CLASSES = {2: "life_line", 3: "head_line", 4: "heart_line"}
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
    return [p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]


def imread_rgb(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        try:
            img = Image.open(path).convert("RGB")
            return np.array(img)
        except Exception:
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
    """Very rough thumb side: side with more palm mass in lower/middle zones."""
    m = (palm_mask > 0).astype(np.uint8)
    h, w = m.shape
    zone = m[int(0.30*h):int(0.85*h), :]
    left = float(np.sum(zone[:, :w//2]))
    right = float(np.sum(zone[:, w//2:]))
    # thumb side tends to have larger Venus mount mass; if uncertain default left
    if right > left * 1.08:
        return "right"
    return "left"


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
    return crop, sem, report


def make_overlay(img_rgb: np.ndarray, sem: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    colors = {
        0: (0, 0, 0),
        1: (50, 180, 50),
        2: (255, 0, 0),      # life
        3: (0, 80, 255),      # head
        4: (255, 0, 255),     # heart
        5: (255, 200, 0),     # minor
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
    if requests is None:
        raise RuntimeError("requests chưa được cài. Chạy pip install requests hoặc dùng --input_dir.")
    ensure_dir(out_dir)
    queries = queries or DEFAULT_COMMONS_QUERIES
    session = requests.Session()
    session.headers.update({"User-Agent": "PalmistryStrictAutoDatasetBuilder/1.1 (local research; contact: local)"})
    downloaded: List[Path] = []
    seen_urls = set()
    seen_titles = set()
    meta_path = out_dir.parent / "metadata" / "commons_metadata.jsonl"
    ensure_dir(meta_path.parent)
    search_queries = []
    for q in queries:
        search_queries.append(q)
    search_queries += [
        "palm hand", "open palm", "human palm", "palmar crease", "hand lines",
        "left palm", "right palm", "palmistry", "palm close up", "open hand"
    ]
    categories = [
        "Category:Palms of hands",
        "Category:Human hands",
        "Category:Hands",
        "Category:Palmistry",
        "Category:Palmar creases"
    ]
    def save_page(page: Dict[str, Any], source: str, meta_f) -> bool:
        nonlocal downloaded
        title = page.get("title", "")
        if title in seen_titles:
            return False
        seen_titles.add(title)
        info_list = page.get("imageinfo") or []
        if not info_list:
            return False
        info = info_list[0]
        url = info.get("url")
        mime = info.get("mime", "")
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        if not url or url in seen_urls:
            return False
        seen_urls.add(url)
        if not mime.startswith("image/"):
            return False
        if width < 512 or height < 512:
            return False
        ext = Path(url.split("?")[0]).suffix.lower()
        if ext not in IMAGE_EXTS:
            ext = ".jpg"
        fname = safe_name(title if title else url) + ext
        dst = out_dir / fname
        if dst.exists():
            return False
        try:
            img_resp = session.get(url, timeout=30, stream=True)
            if img_resp.status_code != 200:
                return False
            with open(dst, "wb") as f:
                for chunk in img_resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
            if imread_rgb(dst) is None:
                dst.unlink(missing_ok=True)
                return False
            meta_f.write(json.dumps({"source": source, "title": title, "url": url, "width": width, "height": height, "mime": mime}, ensure_ascii=False) + "\n")
            downloaded.append(dst)
            time.sleep(sleep)
            return True
        except Exception:
            try:
                dst.unlink(missing_ok=True)
            except Exception:
                pass
            return False
    def run_query(params: Dict[str, Any], source: str, meta_f) -> None:
        cont = None
        attempts = 0
        while len(downloaded) < target_count and len(seen_urls) < max_downloads and attempts < 200:
            attempts += 1
            req = dict(params)
            if cont:
                req.update(cont)
            try:
                r = session.get("https://commons.wikimedia.org/w/api.php", params=req, timeout=25)
                if r.status_code != 200:
                    break
                data = r.json()
            except Exception:
                break
            pages = data.get("query", {}).get("pages", {})
            if pages:
                for _, page in pages.items():
                    if len(downloaded) >= target_count or len(seen_urls) >= max_downloads:
                        break
                    save_page(page, source, meta_f)
            cont = data.get("continue")
            if not cont:
                break
    with open(meta_path, "a", encoding="utf-8") as meta_f:
        for q in search_queries:
            if len(downloaded) >= target_count or len(seen_urls) >= max_downloads:
                break
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": q,
                "gsrnamespace": 6,
                "gsrlimit": 50,
                "prop": "imageinfo",
                "iiprop": "url|mime|size|extmetadata",
                "format": "json",
            }
            print(f"Commons search: {q}")
            run_query(params, f"search:{q}", meta_f)
        for cat in categories:
            if len(downloaded) >= target_count or len(seen_urls) >= max_downloads:
                break
            params = {
                "action": "query",
                "generator": "categorymembers",
                "gcmtitle": cat,
                "gcmnamespace": 6,
                "gcmlimit": 50,
                "prop": "imageinfo",
                "iiprop": "url|mime|size|extmetadata",
                "format": "json",
            }
            print(f"Commons category: {cat}")
            run_query(params, f"category:{cat}", meta_f)
    return downloaded

def split_rows(rows: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)
    for i, row in enumerate(rows):
        if i < n_train:
            row["split"] = "train"
        elif i < n_train + n_val:
            row["split"] = "val"
        else:
            row["split"] = "test"
    return rows


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
                             max_images: Optional[int] = None) -> Dict[str, Any]:
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
    for p in tqdm(input_paths, desc="Auto pseudo-mask strict"):
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
    rows_keep = split_rows(rows_keep)
    write_csv(output_dir / "labels_pseudo_strict.csv", rows_keep)
    write_csv(output_dir / "labels_pseudo_strict_all.csv", rows_all)
    with open(output_dir / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_MAP, f, ensure_ascii=False, indent=2)
    summary = {
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
    report_dir = ensure_dir(output_dir / "reports" / "eda")
    if not rows_all:
        return
    # Simple CSV count by reason.
    counts: Dict[str, int] = {}
    for r in rows_all:
        reason = r.get("reject_reason") or "accepted"
        counts[reason] = counts.get(reason, 0) + 1
    with open(report_dir / "rejected_counts.json", "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)
    if pd is not None:
        pd.DataFrame(rows_all).to_csv(report_dir / "image_metrics_all.csv", index=False)
        pd.DataFrame(rows_keep).to_csv(report_dir / "image_metrics_accepted.csv", index=False)
    if plt is None:
        return
    try:
        def hist(key: str, fname: str, title: str):
            vals = [float(r[key]) for r in rows_all if key in r and str(r[key]) not in ("", "nan")]
            if not vals: return
            plt.figure(figsize=(7,4))
            plt.hist(vals, bins=40)
            plt.title(title)
            plt.xlabel(key); plt.ylabel("count")
            plt.tight_layout(); plt.savefig(report_dir / fname); plt.close()
        hist("measure_laplacian_variance", "blur_laplacian_hist.png", "Blur/Laplacian variance")
        hist("diag_image_quality_score", "image_quality_hist.png", "Image quality score")
        hist("mask_quality_score", "mask_quality_hist.png", "Mask quality score")
        hist("line_pixel_ratio", "line_pixel_ratio_hist.png", "Line pixel ratio")
        plt.figure(figsize=(8,4))
        plt.bar(list(counts.keys()), list(counts.values()))
        plt.xticks(rotation=45, ha="right")
        plt.title("Accepted/rejected counts")
        plt.tight_layout(); plt.savefig(report_dir / "accepted_rejected_counts.png"); plt.close()
        # Sample grid
        sample_paths = [output_dir / r["overlay_path"] for r in rows_keep[:25] if r.get("overlay_path")]
        imgs = []
        for sp in sample_paths:
            im = imread_rgb(sp)
            if im is not None:
                imgs.append(cv2.resize(im, (220,220)))
        if imgs:
            cols = 5; rows = math.ceil(len(imgs)/cols)
            canvas = np.zeros((rows*220, cols*220, 3), dtype=np.uint8)
            for i, im in enumerate(imgs):
                y = (i//cols)*220; x=(i%cols)*220
                canvas[y:y+220, x:x+220] = im
            imwrite_rgb(report_dir / "sample_overlay_grid.jpg", canvas)
    except Exception:
        pass

# -----------------------------
# PyTorch model/training
# -----------------------------
if TORCH_AVAILABLE:
    class PalmMaskDataset(Dataset):
        def __init__(self, data_root: Path, labels_csv: Path, split: str, input_size: int = 512, augment: bool = False):
            self.data_root = Path(data_root)
            self.input_size = input_size
            self.augment = augment
            rows = []
            if pd is not None:
                df = pd.read_csv(labels_csv)
                rows = df.to_dict("records")
            else:
                with open(labels_csv, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
            self.rows = [r for r in rows if str(r.get("split", "train")) == split]
            if not self.rows and split == "train":
                self.rows = rows
        def __len__(self): return len(self.rows)
        def _load(self, r):
            img = imread_rgb(self.data_root / str(r["image_path"]))
            mask = cv2.imread(str(self.data_root / str(r["mask_path"])), cv2.IMREAD_UNCHANGED)
            if img is None or mask is None:
                raise RuntimeError(f"Cannot read {r}")
            img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
            return img, mask.astype(np.int64)
        def __getitem__(self, idx):
            img, mask = self._load(self.rows[idx])
            if self.augment:
                img, mask = augment_image_mask(img, mask)
            x = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
            mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
            std = torch.tensor([0.229,0.224,0.225])[:,None,None]
            x = (x - mean) / std
            y = torch.from_numpy(mask).long()
            return x, y

    def augment_image_mask(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        # flip horizontal with class unchanged, because canonical direction is not guaranteed in auto dataset.
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        angle = random.uniform(-10, 10)
        scale = random.uniform(0.92, 1.08)
        tx = random.uniform(-0.04, 0.04) * w
        ty = random.uniform(-0.04, 0.04) * h
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        M[:,2] += [tx, ty]
        img = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        mask = cv2.warpAffine(mask, M, (w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # photometric jitter
        if random.random() < 0.75:
            alpha = random.uniform(0.85, 1.18)
            beta = random.uniform(-18, 18)
            img = np.clip(img.astype(np.float32)*alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.35:
            noise = np.random.normal(0, random.uniform(1, 5), img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32)+noise, 0, 255).astype(np.uint8)
        if random.random() < 0.15:
            k = random.choice([3,5])
            img = cv2.GaussianBlur(img, (k,k), 0)
        return img, mask

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, dropout=0.0):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            )
        def forward(self, x): return self.block(x)

    class SmallUNet(nn.Module):
        def __init__(self, num_classes=6, base=32, dropout=0.15):
            super().__init__()
            self.enc1 = ConvBlock(3, base, dropout=0.0)
            self.enc2 = ConvBlock(base, base*2, dropout=dropout/2)
            self.enc3 = ConvBlock(base*2, base*4, dropout=dropout)
            self.enc4 = ConvBlock(base*4, base*8, dropout=dropout)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(base*8, base*16, dropout=dropout)
            self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
            self.dec4 = ConvBlock(base*16, base*8, dropout=dropout)
            self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
            self.dec3 = ConvBlock(base*8, base*4, dropout=dropout)
            self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
            self.dec2 = ConvBlock(base*4, base*2, dropout=dropout/2)
            self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
            self.dec1 = ConvBlock(base*2, base, dropout=0.0)
            self.out = nn.Conv2d(base, num_classes, 1)
        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)

    def dice_loss(logits, target, num_classes=6, ignore_background=True, eps=1e-6):
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target.clamp(0, num_classes-1), num_classes).permute(0,3,1,2).float()
        start = 1 if ignore_background else 0
        dices = []
        for c in range(start, num_classes):
            p = probs[:, c]
            t = onehot[:, c]
            inter = (p*t).sum(dim=(1,2))
            denom = p.sum(dim=(1,2)) + t.sum(dim=(1,2)) + eps
            dices.append(1 - ((2*inter + eps) / denom))
        return torch.stack(dices, dim=1).mean()

    @torch.no_grad()
    def compute_dice_metrics(logits, target, num_classes=6) -> Dict[str, float]:
        pred = torch.argmax(logits, dim=1)
        out = {}
        for c in range(1, num_classes):
            p = (pred == c).float()
            t = (target == c).float()
            inter = (p*t).sum().item()
            denom = p.sum().item() + t.sum().item()
            dice = (2*inter + 1e-6) / (denom + 1e-6)
            out[f"dice_{CLASS_MAP[c]}"] = float(dice)
        line_dices = [out.get("dice_life_line",0), out.get("dice_head_line",0), out.get("dice_heart_line",0)]
        out["dice_main_lines_mean"] = float(np.mean(line_dices))
        return out


def train_model(args: argparse.Namespace) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch chưa được cài. Chạy pip install torch torchvision")
    data_root = Path(args.data_root)
    labels_csv = Path(args.labels_csv)
    out_dir = ensure_dir(Path(args.out_dir))
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    log_dir = ensure_dir(out_dir / "logs")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_ds = PalmMaskDataset(data_root, labels_csv, "train", args.input_size, augment=True)
    val_ds = PalmMaskDataset(data_root, labels_csv, "val", args.input_size, augment=False)
    if len(val_ds) == 0:
        print("Không có split val; dùng 10% train làm val tạm.")
        val_ds = train_ds
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    base = 24 if args.model_size == "tiny" else 32 if args.model_size == "small" else 48
    model = SmallUNet(num_classes=6, base=base, dropout=args.dropout).to(device)
    class_weights = torch.tensor([0.10, 0.35, 2.4, 2.4, 2.4, 1.2], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))
    start_epoch = 0; best_metric = -1.0; history = []
    last_ckpt = ckpt_dir / "last.pt"
    best_ckpt = ckpt_dir / "best.pt"
    if args.resume == "auto" and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if "scaler" in ck and ck["scaler"] is not None:
            try: scaler.load_state_dict(ck["scaler"])
            except Exception: pass
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_metric = float(ck.get("best_metric", -1.0))
        history = ck.get("history", [])
        print(f"Resume từ epoch {start_epoch}, best={best_metric:.4f}")
    start_time = time.time()
    max_seconds = args.max_train_hours * 3600 if args.max_train_hours else None
    no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        train_loss = 0.0; steps = 0
        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}")):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                logits = model(x)
                ce = F.cross_entropy(logits, y, weight=class_weights)
                dl = dice_loss(logits, y, num_classes=6)
                loss = (ce + args.dice_weight * dl) / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            train_loss += float(loss.item()) * args.grad_accum; steps += 1
            if max_seconds and (time.time() - start_time) > max_seconds * 0.98:
                print("Gần hết thời gian train, lưu checkpoint và dừng an toàn.")
                break
        # Validation
        model.eval(); metrics_accum: Dict[str, List[float]] = {}; val_loss = 0.0; nval = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="val"):
                x = x.to(device); y = y.to(device)
                logits = model(x)
                ce = F.cross_entropy(logits, y, weight=class_weights)
                dl = dice_loss(logits, y, num_classes=6)
                loss = ce + args.dice_weight * dl
                val_loss += float(loss.item()); nval += 1
                md = compute_dice_metrics(logits, y)
                for k, v in md.items(): metrics_accum.setdefault(k, []).append(v)
        metrics = {k: float(np.mean(v)) for k, v in metrics_accum.items()}
        metric = metrics.get("dice_main_lines_mean", 0.0)
        row = {"epoch": epoch, "train_loss": train_loss/max(steps,1), "val_loss": val_loss/max(nval,1), **metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        if pd is not None:
            pd.DataFrame(history).to_csv(log_dir / "history.csv", index=False)
        with open(log_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        improved = metric > best_metric
        if improved:
            best_metric = metric; no_improve = 0
        else:
            no_improve += 1
        ck = {
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch, "best_metric": best_metric, "history": history,
            "class_map": CLASS_MAP, "input_size": args.input_size,
            "model_size": args.model_size, "base": base,
            "validation_scores": metrics,
        }
        torch.save(ck, last_ckpt)
        if improved:
            torch.save(ck, best_ckpt)
            with open(out_dir / "class_validation_scores.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        if no_improve >= args.patience:
            print("Early stopping.")
            break
        if max_seconds and (time.time() - start_time) > max_seconds * 0.98:
            break
    return {"best_metric": best_metric, "history": history}

# -----------------------------
# Prediction and Vietnamese output
# -----------------------------
def line_feature_from_mask(sem: np.ndarray, cls: int) -> Dict[str, Any]:
    m = (sem == cls).astype(np.uint8) * 255
    if np.sum(m) == 0:
        return {"detected": False}
    sk = skeletonize_mask(m)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((sk > 0).astype(np.uint8), 8)
    comps = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 5:
            comp = (labels == i).astype(np.uint8)
            feat = component_features(comp)
            if feat:
                comps.append(feat)
    if not comps:
        return {"detected": False}
    # merge summary
    length = sum(f["length_px"] for f in comps)
    largest = max(comps, key=lambda f: f["length_px"])
    h, w = sem.shape
    density = float(np.mean(m > 0))
    # endpoints/intersections proxy from skeleton neighbors
    sk_bin = (sk > 0).astype(np.uint8)
    kernel = np.ones((3,3), dtype=np.uint8)
    neigh = cv2.filter2D(sk_bin, -1, kernel)
    endpoints = int(np.sum((sk_bin == 1) & (neigh == 2)))
    intersections = int(np.sum((sk_bin == 1) & (neigh >= 4)))
    return {
        "detected": True,
        "length_px": float(length),
        "length_norm": float(length / max(h, w)),
        "density": density,
        "component_count": len(comps),
        "largest_component_length_px": float(largest["length_px"]),
        "cx": largest["cx"], "cy": largest["cy"],
        "angle": largest["angle"],
        "horizontalness": largest["horizontalness"],
        "verticalness": largest["verticalness"],
        "diagonalness": largest["diagonalness"],
        "endpoint_proxy_count": endpoints,
        "intersection_proxy_count": intersections,
    }


def vietnamese_reading(accepted: Dict[str, Dict[str, Any]]) -> str:
    parts = ["KẾT QUẢ PHÂN TÍCH CHỈ TAY", ""]
    if "life_line" in accepted:
        f = accepted["life_line"]
        cont = "tương đối liền mạch" if f.get("component_count", 1) <= 2 else "có vài đoạn tách nhỏ"
        parts += [
            "Sinh đạo:",
            f"Sinh đạo được nhận diện rõ, độ dài tương đối {describe_length(f.get('length_norm',0))} và {cont}. Dạng này nghiêng về nền tảng sức bền ổn, khả năng hồi phục tốt và có xu hướng bám lâu với mục tiêu khi đã chọn đúng môi trường.",
            ""
        ]
    if "head_line" in accepted:
        f = accepted["head_line"]
        direction = "thiên ngang/thực tế" if f.get("horizontalness",0) >= f.get("diagonalness",0) else "hơi xuôi chéo, thiên về trực giác/sáng tạo"
        parts += [
            "Trí đạo:",
            f"Trí đạo được nhận diện rõ và {direction}. Điều này nghiêng về kiểu tư duy có khả năng phân tích, học sâu và tự xây hệ thống suy nghĩ riêng; nếu đường hơi xuôi chéo, yếu tố sáng tạo và trực giác sẽ nổi bật hơn.",
            ""
        ]
    if "heart_line" in accepted:
        f = accepted["heart_line"]
        parts += [
            "Tâm đạo:",
            f"Tâm đạo được nhận diện rõ ở vùng trên lòng bàn tay, độ dài tương đối {describe_length(f.get('length_norm',0))}. Hệ đọc nghiêng về kiểu cảm xúc rõ, biết quan tâm và cần sự ổn định trong các mối quan hệ thân thiết.",
            ""
        ]
    # Summary based on accepted lines
    traits = []
    if "head_line" in accepted: traits.append("tư duy sâu")
    if "heart_line" in accepted: traits.append("cảm xúc rõ")
    if "life_line" in accepted: traits.append("khả năng bền bỉ")
    if not traits: traits = ["một số đặc điểm đủ điều kiện phân tích"]
    parts += [
        "Tổng hợp:",
        "Các đường đủ điều kiện phân tích cho thấy người này nghiêng về nhóm có " + ", ".join(traits) + ". Khi có mục tiêu rõ và môi trường phù hợp, người này có khả năng phát triển ổn định hơn.",
    ]
    return "\n".join(parts)


def describe_length(x: float) -> str:
    if x >= 0.55: return "dài"
    if x >= 0.30: return "vừa"
    return "ngắn"


def load_model_for_predict(checkpoint: Path, device: torch.device):
    ck = torch.load(checkpoint, map_location=device)
    base = int(ck.get("base", 32))
    model = SmallUNet(num_classes=6, base=base, dropout=0.0).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    input_size = int(ck.get("input_size", 512))
    val_scores = ck.get("validation_scores", {})
    return model, input_size, val_scores


def predict_image(args: argparse.Namespace) -> Dict[str, Any]:
    img_path = Path(args.image)
    img = imread_rgb(img_path)
    if img is None:
        res = DiagnosticResult("need_retake", "BAD_READ", "Không đọc được file ảnh.", "Hãy gửi lại ảnh định dạng JPG hoặc PNG rõ ràng.", {}, {})
        return asdict(res)
    diag = analyze_quality(img, min_short_side=args.min_short_side)
    if diag.status != "ok" and diag.scores.get("image_quality_score",0) < args.min_image_quality:
        out = asdict(diag); out["status"] = "need_retake"; save_json_if_needed(args, out); return out
    crop, pseudo_sem, pseudo_report = create_strict_pseudo_mask(img, out_size=args.out_size, strict_line_threshold=args.strict_line_threshold)
    sem = pseudo_sem
    model_scores: Dict[str, Any] = {}
    if args.checkpoint and Path(args.checkpoint).exists():
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch chưa được cài, không thể dùng checkpoint.")
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model, input_size, val_scores = load_model_for_predict(Path(args.checkpoint), device)
        inp = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(inp.transpose(2,0,1)).float()/255.0
        mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
        std = torch.tensor([0.229,0.224,0.225])[:,None,None]
        x = ((x-mean)/std).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = np.argmax(probs, axis=0).astype(np.uint8)
            pred = cv2.resize(pred, (args.out_size,args.out_size), interpolation=cv2.INTER_NEAREST)
            sem = pred
            # Mean confidence per line class.
            for cls, name in LINE_CLASSES.items():
                cls_mask = pred == cls
                if np.any(cls_mask):
                    # resize prob cls to out_size
                    pmap = cv2.resize(probs[cls], (args.out_size,args.out_size), interpolation=cv2.INTER_LINEAR)
                    model_scores[name] = float(np.mean(pmap[cls_mask]))
                else:
                    model_scores[name] = 0.0
            model_scores["validation_scores"] = val_scores
    else:
        # Use pseudo scores as fallback.
        model_scores = dict(pseudo_report.get("line_scores", {}))
        model_scores["validation_scores"] = {}

    accepted: Dict[str, Dict[str, Any]] = {}
    rejected: Dict[str, Any] = {}
    for cls, name in LINE_CLASSES.items():
        feat = line_feature_from_mask(sem, cls)
        pseudo_s = float(pseudo_report.get("line_scores", {}).get(name, 0.0))
        pred_s = float(model_scores.get(name, pseudo_s))
        val_scores = model_scores.get("validation_scores", {}) if isinstance(model_scores.get("validation_scores", {}), dict) else {}
        val_key = f"dice_{name}"
        val_s = float(val_scores.get(val_key, 1.0 if not args.require_validation_scores else 0.0))
        template_s = pseudo_s
        length_s = normalize_score(feat.get("length_norm", 0.0), 0.10, 0.55) if feat.get("detected") else 0.0
        final = min(diag.scores.get("image_quality_score",0), diag.scores.get("line_visibility_score",0), pred_s, template_s, max(length_s, 0.35), val_s)
        item = {"final_score": float(final), "model_score": pred_s, "template_score": template_s, "validation_score": val_s, **feat}
        if feat.get("detected") and final >= args.read_threshold:
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
    if args.out_mask:
        imwrite_gray(Path(args.out_mask), sem)
    if args.out_overlay:
        imwrite_rgb(Path(args.out_overlay), make_overlay(crop, sem))
    save_json_if_needed(args, out)
    if args.print_reading and out.get("status") == "success":
        print(out["reading_vi"])
    elif args.print_reading:
        print(out.get("reason_vi", "Không thể phân tích ảnh này."))
        print(out.get("fix_vi", ""))
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
    print("Pandas:", getattr(pd, "__version__", "not installed"))
    print("Matplotlib:", "installed" if plt is not None else "not installed")
    print("Requests:", "installed" if requests is not None else "not installed")
    print("Torch:", torch.__version__ if TORCH_AVAILABLE else "not installed")
    if TORCH_AVAILABLE:
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    print("OK")


def cmd_build(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root)
    dataset_dir = ensure_dir(project_root / args.output_dir)
    raw_dir = ensure_dir(dataset_dir / "00_raw_downloads")
    if args.input_dir:
        paths = list_images(Path(args.input_dir))
    else:
        print("Không có --input_dir, bắt đầu tải ảnh từ Wikimedia Commons.")
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
        max_images=args.max_images,
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
        max_images=args.max_images,
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
    report = {"diagnostic": asdict(diag), "pseudo_report": rep, "class_map": CLASS_MAP}
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Palmistry strict fully automatic pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")

    b = sub.add_parser("build", help="Tự tải hoặc đọc ảnh local, tạo dataset pseudo-mask lọc gắt")
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
    tr.add_argument("--resume", choices=["auto","none"], default="auto")
    tr.add_argument("--num_workers", type=int, default=0)
    tr.add_argument("--amp", action="store_true", default=True)
    tr.add_argument("--cpu", action="store_true")

    pr = sub.add_parser("predict", help="Phân tích 1 ảnh, tự chẩn đoán lỗi, chỉ đọc line > threshold")
    pr.add_argument("--image", required=True)
    pr.add_argument("--checkpoint", default="")
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
    return p


def main():
    parser = make_parser()
    args = parser.parse_args()
    if getattr(args, "max_images", 0) == 0:
        args.max_images = None
    if args.cmd == "doctor": cmd_doctor(args)
    elif args.cmd == "build": cmd_build(args)
    elif args.cmd == "mask-folder": cmd_mask_folder(args)
    elif args.cmd == "single-mask": cmd_single_mask(args)
    elif args.cmd == "train": train_model(args)
    elif args.cmd == "predict": predict_image(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()
