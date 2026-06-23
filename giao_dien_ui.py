import os
import sys
import json
import time
import math
import shutil
import zipfile
import threading
import subprocess
from pathlib import Path
from collections import deque

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, Response, jsonify, send_from_directory


PROJECT_DIR = Path(__file__).resolve().parent

PORT = 8501

NGROK_AUTHTOKEN = ""
USE_NGROK = False

WEBCAM_INDEX = 0
USB_CAMERA_INDEX = 1
CAMERA_INDEX = WEBCAM_INDEX

DISPLAY_FPS = 25
JPEG_QUALITY = 80

PREDICT_EVERY_SECONDS = 0.80
READ_THRESHOLD = 0.85

LOCK_ON_FIRST_GOOD_OVERLAY = True
GOOD_OVERLAY_MIN_LINES = 2

STABLE_SECONDS = 3.0
STABILITY_TOLERANCE = 0.10

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FLIP_CAMERA = True

CHECKPOINT_PATH = ""
CODE_PATH = PROJECT_DIR / "palmistry_strict_auto_onefile.py"

BUNDLE_ZIP_EDA = PROJECT_DIR / "palmistry_colab_bundle_eda.zip"
BUNDLE_ZIP_BASIC = PROJECT_DIR / "palmistry_colab_bundle.zip"
BUNDLE_DIR = PROJECT_DIR / ".web_bundle"
TMP_DIR = PROJECT_DIR / ".web_live_tmp"
STATIC_DIR = PROJECT_DIR / ".web_static"

TMP_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

LINE_CLASSES = {
    2: {"name": "Sinh đạo", "key": "life_line", "color": (0, 0, 255)},
    3: {"name": "Trí đạo", "key": "head_line", "color": (0, 255, 0)},
    4: {"name": "Tâm đạo", "key": "heart_line", "color": (255, 0, 0)},
}

app = Flask(__name__)


def load_json(path):
    path = Path(path)

    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def find_checkpoint():
    if CHECKPOINT_PATH:
        p = Path(CHECKPOINT_PATH)

        if not p.is_absolute():
            p = PROJECT_DIR / p

        if p.exists():
            return p

    candidates = [
        PROJECT_DIR / "runs" / "palmistry_roboflow_v1" / "checkpoints" / "best.pt",
        PROJECT_DIR / "runs" / "palmistry_roboflow_v1" / "checkpoints" / "last.pt",
        PROJECT_DIR / "runs" / "palmistry_roboflow_768_v1" / "checkpoints" / "best.pt",
        PROJECT_DIR / "runs" / "palmistry_roboflow_768_v1" / "checkpoints" / "last.pt",
    ]

    for p in candidates:
        if p.exists():
            return p

    found = list(PROJECT_DIR.rglob("best.pt"))

    if found:
        return found[0]

    found = list(PROJECT_DIR.rglob("last.pt"))

    if found:
        return found[0]

    return None


def find_bundle_zip():
    if BUNDLE_ZIP_EDA.exists():
        return BUNDLE_ZIP_EDA

    if BUNDLE_ZIP_BASIC.exists():
        return BUNDLE_ZIP_BASIC

    zips = list(PROJECT_DIR.glob("*bundle*.zip"))

    if zips:
        return zips[0]

    return None


def extract_bundle():
    bundle_zip = find_bundle_zip()

    if bundle_zip is None:
        return None

    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)

    BUNDLE_DIR.mkdir(exist_ok=True)

    with zipfile.ZipFile(bundle_zip, "r") as z:
        z.extractall(BUNDLE_DIR)

    return bundle_zip


def class_map_from_bundle():
    path = BUNDLE_DIR / "dataset_meta" / "class_map.json"
    raw = load_json(path)

    if raw:
        out = {}

        for k, v in raw.items():
            try:
                out[int(k)] = v
            except Exception:
                pass

        if out:
            return out

    return {
        0: "background",
        1: "palm_area",
        2: "life_line",
        3: "head_line",
        4: "heart_line",
    }


def read_mask(path):
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if arr is None:
        return None

    if arr.ndim == 3:
        if np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2]):
            arr = arr[:, :, 0]
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    return arr.astype(np.int64)


def estimate_line_metrics(mask_path, frame_shape=None):
    mask = read_mask(mask_path)

    if mask is None:
        return {}

    if frame_shape is not None:
        h, w = frame_shape[:2]

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)

    h, w = mask.shape[:2]
    diag = math.sqrt(w * w + h * h)
    metrics = {}

    for cls, info in LINE_CLASSES.items():
        binary = (mask == cls).astype(np.uint8)
        area = int(binary.sum())

        if area < 8:
            metrics[info["key"]] = {
                "name": info["name"],
                "present": False,
                "length_norm": 0.0,
                "continuity": 0.0,
                "angle_deg": 0.0,
                "center_x": 0.0,
                "center_y": 0.0,
                "components": 0,
                "area": 0,
            }
            continue

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
        largest_area = max(areas) if areas else area
        continuity = largest_area / max(1, area)
        components = max(0, n_labels - 1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(c, False) for c in contours)
        length_px = perimeter / 2.0
        length_norm = length_px / max(1.0, diag)

        ys, xs = np.where(binary > 0)
        center_x = float(xs.mean() / max(1, w))
        center_y = float(ys.mean() / max(1, h))

        angle_deg = 0.0

        if len(xs) >= 2:
            pts = np.column_stack([xs, ys]).astype(np.float32)
            mean, eigenvectors = cv2.PCACompute(pts, mean=None)
            vx, vy = eigenvectors[0]
            angle_deg = math.degrees(math.atan2(vy, vx))

        metrics[info["key"]] = {
            "name": info["name"],
            "present": True,
            "length_norm": round(float(length_norm), 4),
            "continuity": round(float(continuity), 4),
            "angle_deg": round(float(angle_deg), 2),
            "center_x": round(float(center_x), 4),
            "center_y": round(float(center_y), 4),
            "components": int(components),
            "area": int(area),
        }

    return metrics


def metrics_vector(metrics):
    vals = []

    for key in ["life_line", "head_line", "heart_line"]:
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
    life = metrics.get("life_line", {})
    head = metrics.get("head_line", {})
    heart = metrics.get("heart_line", {})

    present_count = sum(1 for m in [life, head, heart] if m.get("present"))

    if present_count == 0:
        return "Chưa đọc rõ đường chỉ tay. Hãy đưa lòng bàn tay vào giữa khung hình, mở tay thẳng hơn và để ánh sáng đều hơn."

    result_parts = []

    if life.get("present"):
        length = _safe_float(life, "length_norm")
        continuity = _safe_float(life, "continuity")
        center_x = _safe_float(life, "center_x")
        components = int(_safe_float(life, "components", 1))

        if length >= 0.24:
            length_text = "dài và nổi bật"
            life_meaning = "cho thấy xu hướng bền bỉ, có khả năng theo đuổi mục tiêu trong thời gian dài"
        elif length >= 0.18:
            length_text = "khá rõ và có độ dài tốt"
            life_meaning = "thể hiện sự ổn định, biết duy trì nhịp sống và kế hoạch cá nhân"
        elif length >= 0.12:
            length_text = "ở mức vừa phải"
            life_meaning = "cho thấy năng lượng cân bằng, nhưng dễ thay đổi theo môi trường hoặc thói quen sinh hoạt"
        else:
            length_text = "ngắn hoặc chưa thật rõ"
            life_meaning = "phần sinh đạo chưa đủ mạnh để đọc sâu, cần khung hình rõ hơn"

        if continuity >= 0.78:
            continuity_text = "đường đi liền mạch"
            life_style = "cách hành động khá chắc chắn và ít bị ngắt quãng"
        elif continuity >= 0.58:
            continuity_text = "có vài đoạn chuyển nhẹ"
            life_style = "có xu hướng linh hoạt, đôi lúc đổi nhịp nhưng vẫn giữ hướng chính"
        else:
            continuity_text = "còn đứt hoặc bị nhiễu"
            life_style = "kết quả sinh đạo phụ thuộc nhiều vào độ rõ của ảnh hiện tại"

        if center_x < 0.42:
            position_text = "nằm sát vùng gò ngón cái"
        elif center_x > 0.58:
            position_text = "mở rộng nhiều vào giữa lòng bàn tay"
        else:
            position_text = "nằm khá cân bằng trong lòng bàn tay"

        if components <= 2:
            component_text = "cấu trúc đường khá gọn"
        elif components <= 5:
            component_text = "có một số đoạn nhỏ tách rời"
        else:
            component_text = "bị chia thành nhiều đoạn nhỏ, có thể do nhiễu ảnh hoặc đường tay mảnh"

        result_parts.append(
            f"Sinh đạo {length_text}, {continuity_text}, {position_text}; {component_text}. "
            f"Nhận xét chính: {life_meaning}; {life_style}."
        )

    if head.get("present"):
        length = _safe_float(head, "length_norm")
        continuity = _safe_float(head, "continuity")
        angle = _safe_float(head, "angle_deg")
        center_y = _safe_float(head, "center_y")
        components = int(_safe_float(head, "components", 1))

        if abs(angle) <= 12:
            angle_text = "thiên ngang"
            thinking_text = "nghiêng về tư duy logic, thực tế và thích phân tích vấn đề theo dữ kiện rõ ràng"
        elif angle > 12:
            angle_text = "nghiêng xuống rõ"
            thinking_text = "nghiêng về trực giác, trí tưởng tượng và khả năng liên tưởng linh hoạt"
        else:
            angle_text = "hướng lên nhẹ"
            thinking_text = "có xu hướng chủ động, quyết đoán và phản ứng nhanh khi xử lý vấn đề"

        if length >= 0.22:
            depth_text = "khá dài nên khả năng tập trung và đào sâu vấn đề nổi bật"
        elif length >= 0.15:
            depth_text = "có độ dài vừa phải nên cách suy nghĩ cân bằng giữa nhanh và chắc"
        else:
            depth_text = "chưa thật dài hoặc chưa rõ trong ảnh nên phần trí đạo cần đọc thận trọng hơn"

        if continuity >= 0.72:
            flow_text = "mạch suy nghĩ ổn định"
        elif continuity >= 0.52:
            flow_text = "mạch suy nghĩ linh hoạt, dễ đổi hướng khi có thông tin mới"
        else:
            flow_text = "đường còn đứt hoặc nhiễu, có thể do ánh sáng hoặc tay chưa nằm đúng khung hình"

        if center_y < 0.45:
            position_text = "nằm hơi cao trong lòng bàn tay"
        elif center_y > 0.58:
            position_text = "nằm thấp hơn trung bình"
        else:
            position_text = "nằm ở vùng giữa lòng bàn tay"

        if components <= 2:
            component_text = "đường tương đối gọn"
        elif components <= 5:
            component_text = "đường có một vài đoạn phụ"
        else:
            component_text = "đường bị tách thành nhiều đoạn nhỏ nên độ tin cậy giảm"

        result_parts.append(
            f"Trí đạo {angle_text}, {position_text}; {component_text}. "
            f"Nhận xét chính: {thinking_text}; {depth_text}; {flow_text}."
        )

    if heart.get("present"):
        length = _safe_float(heart, "length_norm")
        continuity = _safe_float(heart, "continuity")
        center_y = _safe_float(heart, "center_y")
        components = int(_safe_float(heart, "components", 1))

        if length >= 0.22:
            emotion_length_text = "dài và dễ quan sát"
            emotion_text = "cách biểu lộ cảm xúc khá rõ, có xu hướng coi trọng sự gắn kết trong các mối quan hệ"
        elif length >= 0.15:
            emotion_length_text = "ở mức vừa"
            emotion_text = "cảm xúc có xu hướng cân bằng, không quá bộc lộ nhưng vẫn có chiều sâu"
        else:
            emotion_length_text = "ngắn hoặc chưa rõ"
            emotion_text = "chưa đủ dữ liệu để đọc mạnh về cảm xúc, nên cần khung hình rõ hơn"

        if continuity >= 0.75:
            emotion_flow_text = "đường khá liền nên cảm xúc tương đối ổn định"
        elif continuity >= 0.55:
            emotion_flow_text = "đường có vài đoạn thay đổi nên cảm xúc linh hoạt, dễ chịu ảnh hưởng bởi bối cảnh"
        else:
            emotion_flow_text = "đường chưa liền, có thể do ảnh mờ hoặc lòng bàn tay chưa mở đủ"

        if center_y < 0.42:
            position_text = "nằm cao trong lòng bàn tay"
        elif center_y > 0.58:
            position_text = "nằm thấp hơn trung bình"
        else:
            position_text = "nằm ở vùng giữa phía trên lòng bàn tay"

        if components <= 2:
            component_text = "đường có cấu trúc gọn"
        elif components <= 5:
            component_text = "đường có một số đoạn tách nhẹ"
        else:
            component_text = "đường bị chia thành nhiều đoạn nhỏ nên cần ảnh rõ hơn để đọc chắc"

        result_parts.append(
            f"Tâm đạo {emotion_length_text}, {position_text}; {component_text}. "
            f"Nhận xét chính: {emotion_text}; {emotion_flow_text}."
        )

    life_score = 0.0
    head_score = 0.0
    heart_score = 0.0

    if life.get("present"):
        life_score = (
            0.55 * _safe_float(life, "length_norm")
            + 0.45 * _safe_float(life, "continuity")
        )

    if head.get("present"):
        head_score = (
            0.45 * _safe_float(head, "length_norm")
            + 0.35 * _safe_float(head, "continuity")
            + 0.20 * min(abs(_safe_float(head, "angle_deg")) / 45.0, 1.0)
        )

    if heart.get("present"):
        heart_score = (
            0.50 * _safe_float(heart, "length_norm")
            + 0.50 * _safe_float(heart, "continuity")
        )

    trait_scores = [
        (life_score, "sức bền và khả năng duy trì mục tiêu"),
        (head_score, "tư duy và cách xử lý vấn đề"),
        (heart_score, "cảm xúc và cách kết nối với người khác"),
    ]

    trait_scores = sorted(trait_scores, reverse=True, key=lambda x: x[0])

    if present_count >= 2:
        result_parts.append(
            f"Tổng hợp cá nhân: dấu tay này nổi bật nhất ở {trait_scores[0][1]}, tiếp theo là {trait_scores[1][1]}."
        )
    else:
        only_line = ""

        if life.get("present"):
            only_line = "sinh đạo"
        elif head.get("present"):
            only_line = "trí đạo"
        elif heart.get("present"):
            only_line = "tâm đạo"

        result_parts.append(
            f"Tổng hợp cá nhân: hiện hệ thống đọc rõ nhất {only_line}, nên nhận xét tập trung chủ yếu vào đường này."
        )

    result_parts.append("Bấm R hoặc nút Đo lại để quét bàn tay mới.")

    return " ".join(result_parts)


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


class LiveEngine:
    def __init__(self):
        self.checkpoint = find_checkpoint()
        self.cap = None

        self.camera_index = CAMERA_INDEX
        self.requested_camera_index = CAMERA_INDEX
        self.camera_name = "Webcam"

        self.frame_lock = threading.Lock()
        self.latest_frame = None

        self.latest_mask = None
        self.latest_overlay_exact = None
        self.latest_metrics = {}
        self.latest_json = {}
        self.latest_reading = ""

        self.status = "Camera đang chuẩn bị"
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
                self.cap = cap
                self.camera_index = index

                if index == WEBCAM_INDEX:
                    self.camera_name = "Webcam"
                elif index == USB_CAMERA_INDEX:
                    self.camera_name = "Camera USB"
                else:
                    self.camera_name = f"Camera {index}"

                self.status = f"{self.camera_name} sẵn sàng"
                print(self.status)
                return True

            cap.release()

        self.status = f"Không nhận được hình từ camera {index}"
        print(self.status)
        return False

    def set_camera(self, index):
        self.requested_camera_index = int(index)
        self.reset()
        self.status = f"Đang chuyển sang camera {index}"

    def camera_loop(self):
        self.open_camera(self.requested_camera_index)

        frame_delay = 1.0 / max(1, DISPLAY_FPS)

        while not self.stop:
            start = time.time()

            if self.camera_index != self.requested_camera_index or self.cap is None:
                self.open_camera(self.requested_camera_index)
                time.sleep(0.1)
                continue

            if self.locked:
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.status = "Camera chưa gửi khung hình"
                time.sleep(0.03)
                continue

            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            with self.frame_lock:
                self.latest_frame = frame.copy()

            elapsed = time.time() - start
            sleep_time = max(0.001, frame_delay - elapsed)
            time.sleep(sleep_time)

    def predict_loop(self):
        while not self.stop:
            if self.locked:
                time.sleep(0.05)
                continue

            now = time.time()

            if now - self.last_predict_time < PREDICT_EVERY_SECONDS:
                time.sleep(0.02)
                continue

            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is None:
                time.sleep(0.05)
                continue

            self.predicting = True
            self.status = "Model đang tìm bàn tay rõ trong khung hình"

            try:
                mask, exact_overlay, data, metrics, reading = self.run_predict(frame)

                self.latest_mask = mask
                self.latest_overlay_exact = exact_overlay
                self.latest_json = data
                self.latest_metrics = metrics
                self.latest_reading = reading

                self.last_predict_time = time.time()
                self.status = "Model sẵn sàng cho khung hình tiếp theo"

                self.update_lock(exact_overlay, metrics, reading)

            except Exception as e:
                self.status = f"Model tạm thời chưa đọc được khung hình này: {e}"
                print(self.status)
                self.last_predict_time = time.time()

            self.predicting = False

    def run_predict(self, frame):
        if self.checkpoint is None:
            raise FileNotFoundError("Không tìm thấy checkpoint best.pt hoặc last.pt.")

        if not CODE_PATH.exists():
            raise FileNotFoundError("Không tìm thấy palmistry_strict_auto_onefile.py.")

        frame_path = TMP_DIR / "live_frame.jpg"
        out_json = TMP_DIR / "prediction_result.json"
        out_mask = TMP_DIR / "predicted_mask.png"
        out_overlay = TMP_DIR / "prediction_overlay.jpg"

        cv2.imwrite(str(frame_path), frame)

        cmd = [
            sys.executable,
            str(CODE_PATH),
            "predict",
            "--checkpoint",
            str(self.checkpoint),
            "--image",
            str(frame_path),
            "--out_json",
            str(out_json),
            "--out_mask",
            str(out_mask),
            "--out_overlay",
            str(out_overlay),
            "--read_threshold",
            str(READ_THRESHOLD),
        ]

        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=child_env,
        )

        if result.returncode != 0:
            print("Predict stderr:")
            print(result.stderr)

        data = load_json(out_json)
        metrics = estimate_line_metrics(out_mask, frame.shape)

        reading = fallback_reading(metrics)

        mask = read_mask(out_mask)

        if mask is not None and mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)

        exact_overlay = cv2.imread(str(out_overlay)) if out_overlay.exists() else None

        if exact_overlay is None:
            exact_overlay = overlay_mask_on_frame(frame, mask)

        if exact_overlay is None:
            exact_overlay = frame.copy()

        if exact_overlay.shape[:2] != frame.shape[:2]:
            exact_overlay = cv2.resize(exact_overlay, (frame.shape[1], frame.shape[0]))

        return mask, exact_overlay, data, metrics, reading

    def update_lock(self, exact_overlay, metrics, reading):
        if self.locked:
            return

        present_count = sum(1 for m in metrics.values() if m.get("present"))

        if present_count < GOOD_OVERLAY_MIN_LINES:
            self.lock_start = None
            self.history.clear()
            self.status = f"Đưa lòng bàn tay rõ hơn vào khung hình ({present_count}/{GOOD_OVERLAY_MIN_LINES} đường)"
            return

        if LOCK_ON_FIRST_GOOD_OVERLAY:
            self.locked = True
            self.locked_frame = draw_hud(
                exact_overlay.copy(),
                "KET QUA KHOA - Bam R hoac nut Do lai",
                (0, 255, 255),
            )
            self.locked_metrics = json.loads(json.dumps(metrics, ensure_ascii=False))
            self.locked_reading = str(reading)

            self.latest_metrics = self.locked_metrics
            self.latest_reading = self.locked_reading

            self.status = "Khung hình đang khóa. Bấm R hoặc Đo lại để quét bàn tay mới."
            print(self.status)
            return

        self.history.append(metrics)

        if not is_stable(self.history):
            self.lock_start = None
            self.status = "Giữ tay yên thêm một chút"
            return

        if self.lock_start is None:
            self.lock_start = time.time()
            self.status = "Tay ổn định, đang đếm thời gian"
            return

        stable_elapsed = time.time() - self.lock_start

        if stable_elapsed >= STABLE_SECONDS:
            self.locked = True
            self.locked_frame = draw_hud(
                exact_overlay.copy(),
                "KET QUA KHOA - Bam R hoac nut Do lai",
                (0, 255, 255),
            )
            self.locked_metrics = json.loads(json.dumps(metrics, ensure_ascii=False))
            self.locked_reading = str(reading)

            self.latest_metrics = self.locked_metrics
            self.latest_reading = self.locked_reading

            self.status = "Khung hình đang khóa. Bấm R hoặc Đo lại để quét bàn tay mới."
            print(self.status)

    def get_display_frame(self):
        if self.locked and self.locked_frame is not None:
            return self.locked_frame.copy()

        with self.frame_lock:
            raw = None if self.latest_frame is None else self.latest_frame.copy()

        if raw is not None:
            return draw_hud(
                raw,
                "LIVE CAMERA",
                (255, 255, 255),
            )

        blank = np.zeros((720, 1280, 3), dtype=np.uint8)

        cv2.putText(
            blank,
            "NO CAMERA FRAME",
            (60, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255),
            3,
        )

        cv2.putText(
            blank,
            str(self.status),
            (60, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            blank,
            f"CAMERA_INDEX = {self.requested_camera_index}",
            (60, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        return blank

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

    def reset(self):
        self.locked = False
        self.locked_frame = None
        self.locked_metrics = {}
        self.locked_reading = ""

        self.latest_mask = None
        self.latest_overlay_exact = None
        self.latest_metrics = {}
        self.latest_reading = ""
        self.latest_json = {}

        self.lock_start = None
        self.history.clear()

        self.status = "Đưa lòng bàn tay vào khung hình"

    def lock_now(self):
        if self.locked:
            return

        frame = self.get_display_frame()
        self.locked = True
        self.locked_frame = draw_hud(
            frame.copy(),
            "KHOA THU CONG - Bam R hoac nut Do lai",
            (0, 255, 255),
        )
        self.locked_metrics = self.latest_metrics
        self.locked_reading = self.latest_reading or fallback_reading(self.latest_metrics)
        self.status = "Khung hình đang khóa thủ công."

    def payload(self):
        metrics = self.locked_metrics if self.locked else self.latest_metrics
        reading = self.locked_reading if self.locked else self.latest_reading

        stable_time = 0.0

        if self.lock_start is not None and not self.locked:
            stable_time = time.time() - self.lock_start

        return {
            "status": self.status,
            "camera": self.camera_index,
            "camera_name": self.camera_name,
            "requested_camera": self.requested_camera_index,
            "checkpoint": str(self.checkpoint) if self.checkpoint else "Không tìm thấy checkpoint",
            "predicting": self.predicting,
            "locked": self.locked,
            "stable_time": round(stable_time, 2),
            "stable_seconds": STABLE_SECONDS,
            "display_fps": DISPLAY_FPS,
            "predict_every": PREDICT_EVERY_SECONDS,
            "metrics": metrics,
            "reading": reading,
        }


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


def make_metric_charts():
    history_path = BUNDLE_DIR / "logs" / "history.csv"

    if not history_path.exists():
        return []

    hist = pd.read_csv(history_path)

    if "epoch" not in hist.columns:
        return []

    hist["epoch"] = pd.to_numeric(hist["epoch"], errors="coerce")
    hist = hist.dropna(subset=["epoch"])

    charts = []

    loss_cols = [c for c in ["train_loss", "val_loss"] if c in hist.columns]
    dice_cols = [
        c
        for c in [
            "dice_life_line",
            "dice_head_line",
            "dice_heart_line",
            "dice_main_lines_mean",
            "dice_palm_area",
        ]
        if c in hist.columns
    ]

    switch_epoch = 79

    if len(hist) > 0 and hist["epoch"].max() < switch_epoch:
        switch_epoch = int(hist["epoch"].max())

    if loss_cols:
        plt.figure(figsize=(9, 4))

        for c in loss_cols:
            plt.plot(hist["epoch"], hist[c], label=c)

        plt.axvline(switch_epoch, linestyle="--", linewidth=2)

        plt.text(
            switch_epoch + 1,
            float(hist[loss_cols].max().max()) * 0.92,
            "Chuyển từ dữ liệu tự preprocess\nsang Roboflow",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.15),
        )

        plt.title("Loss theo epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.25)

        out = STATIC_DIR / "loss_chart.png"
        plt.tight_layout()
        plt.savefig(out, dpi=140)
        plt.close()

        charts.append(out.name)

    if dice_cols:
        plt.figure(figsize=(9, 4))

        for c in dice_cols:
            plt.plot(hist["epoch"], hist[c], label=c)

        plt.axvline(switch_epoch, linestyle="--", linewidth=2)

        plt.text(
            switch_epoch + 1,
            float(hist[dice_cols].max().max()) * 0.92,
            "Chuyển sang Roboflow",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.15),
        )

        plt.title("Dice theo epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.legend()
        plt.grid(alpha=0.25)

        out = STATIC_DIR / "dice_chart.png"
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
    eval_df = pd.read_csv(eval_csv)

    if "eval_mask_path" not in eval_df.columns:
        return []

    masks = [BUNDLE_DIR / p for p in eval_df["eval_mask_path"].tolist()]
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
    eval_df = pd.read_csv(eval_csv)

    if "eval_mask_path" not in eval_df.columns:
        return None

    masks = [BUNDLE_DIR / p for p in eval_df["eval_mask_path"].tolist()]
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

    df = pd.DataFrame(rows)

    plt.figure(figsize=(8, 4))
    plt.bar(df["class"], df["pixels"])
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
    shutil.rmtree(STATIC_DIR, ignore_errors=True)
    STATIC_DIR.mkdir(exist_ok=True)

    bundle_zip = extract_bundle()

    charts = []
    heatmaps = []
    pixel_chart = None
    split_chart = None
    sample_overlays = []

    report = load_json(BUNDLE_DIR / "reports" / "convert_report.json")

    if report:
        split_chart = make_split_chart(report.get("split_counts", {}))

    charts = make_metric_charts()
    heatmaps = make_heatmaps()
    pixel_chart = make_pixel_distribution()

    overlay_dir = BUNDLE_DIR / "sample_overlays"

    if overlay_dir.exists():
        files = sorted([
            p
            for p in overlay_dir.glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ])[:6]

        for i, p in enumerate(files):
            out = STATIC_DIR / f"sample_overlay_{i}{p.suffix.lower()}"
            shutil.copy(p, out)
            sample_overlays.append(out.name)

    return {
        "bundle_zip": str(bundle_zip) if bundle_zip else "",
        "charts": charts,
        "heatmaps": heatmaps,
        "pixel_chart": pixel_chart,
        "split_chart": split_chart,
        "sample_overlays": sample_overlays,
    }


EDA_ASSETS = prepare_eda_assets()


def file_size_mb(path):
    if path is None:
        return "Không có"

    path = Path(path)

    if not path.exists():
        return "Không có"

    return f"{path.stat().st_size / 1024 / 1024:.2f} MB"


def model_info_html():
    checkpoint = find_checkpoint()
    report = load_json(BUNDLE_DIR / "reports" / "convert_report.json")
    class_map = class_map_from_bundle()

    history_path = BUNDLE_DIR / "logs" / "history.csv"
    history_table = ""

    if history_path.exists():
        hist = pd.read_csv(history_path)
        last = hist.tail(1)
        history_table = last.to_html(index=False, classes="table")
    else:
        history_table = "<p>Không tìm thấy history.csv trong bundle.</p>"

    class_rows = ""

    for k, v in class_map.items():
        class_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

    chart_html = ""

    if EDA_ASSETS.get("split_chart"):
        chart_html += f'<div class="card"><h3>Train / Valid / Test</h3><img src="/static_generated/{EDA_ASSETS["split_chart"]}"></div>'

    for name in EDA_ASSETS.get("charts", []):
        chart_html += f'<div class="card"><h3>{name}</h3><img src="/static_generated/{name}"></div>'

    if EDA_ASSETS.get("pixel_chart"):
        chart_html += f'<div class="card"><h3>Pixel distribution</h3><img src="/static_generated/{EDA_ASSETS["pixel_chart"]}"></div>'

    if EDA_ASSETS.get("heatmaps"):
        heat_html = ""

        for name in EDA_ASSETS["heatmaps"]:
            heat_html += f'<img src="/static_generated/{name}">'

        chart_html += f'<div class="card"><h3>Heatmap vị trí xuất hiện đường chỉ tay</h3><div class="image-grid">{heat_html}</div></div>'

    if EDA_ASSETS.get("sample_overlays"):
        overlay_html = ""

        for name in EDA_ASSETS["sample_overlays"]:
            overlay_html += f'<img src="/static_generated/{name}">'

        chart_html += f'<div class="card"><h3>Sample overlays từ dataset</h3><div class="image-grid">{overlay_html}</div></div>'

    if not chart_html:
        chart_html = '<div class="card"><p>Chưa có EDA assets. Hãy đặt palmistry_colab_bundle_eda.zip vào folder project.</p></div>'

    return f"""
    <div class="grid2">
        <div class="card">
            <h2>Model Info</h2>
            <table class="table">
                <tr><td>Project dir</td><td>{PROJECT_DIR}</td></tr>
                <tr><td>Code predict</td><td>{CODE_PATH}</td></tr>
                <tr><td>Checkpoint</td><td>{checkpoint}</td></tr>
                <tr><td>Checkpoint size</td><td>{file_size_mb(checkpoint)}</td></tr>
                <tr><td>Bundle zip</td><td>{EDA_ASSETS.get("bundle_zip", "")}</td></tr>
                <tr><td>Webcam index</td><td>{WEBCAM_INDEX}</td></tr>
                <tr><td>USB camera index</td><td>{USB_CAMERA_INDEX}</td></tr>
                <tr><td>Display FPS</td><td>{DISPLAY_FPS}</td></tr>
                <tr><td>Predict interval</td><td>{PREDICT_EVERY_SECONDS} giây</td></tr>
            </table>
        </div>

        <div class="card">
            <h2>Dataset Report</h2>
            <table class="table">
                <tr><td>Tổng ảnh</td><td>{report.get("total_images_found", "Không có")}</td></tr>
                <tr><td>Ảnh dùng được</td><td>{report.get("accepted", "Không có")}</td></tr>
                <tr><td>Ảnh bị loại</td><td>{report.get("rejected", "Không có")}</td></tr>
                <tr><td>Split counts</td><td>{report.get("split_counts", "Không có")}</td></tr>
            </table>
        </div>
    </div>

    <div class="card">
        <h2>Class Map</h2>
        <table class="table">
            <tr><th>ID</th><th>Class</th></tr>
            {class_rows}
        </table>
    </div>

    <div class="card">
        <h2>Metric cuối trong history.csv</h2>
        {history_table}
    </div>

    {chart_html}
    """


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
        <div class="title">Palmistry Live Camera</div>
    </div>

    <div class="tabs">
        <div class="tab-btn active" onclick="showTab('live', this)">Live Camera + Prediction</div>
        <div class="tab-btn" onclick="showTab('info', this)">Model Info + EDA</div>
    </div>

    <div id="live" class="tab-content active">
        <div class="live-layout">
            <div class="card video-card">
                <img id="video" src="/video_feed">
            </div>

            <div>
                <div class="card">
                    <h2>Camera</h2>
                    <button class="green" onclick="setCamera(0)">Webcam</button>
                    <button class="purple" onclick="setCamera(1)">Camera USB</button>
                    <br>
                    <input id="customCamera" type="number" min="0" value="0">
                    <button class="secondary" onclick="setCustomCamera()">Chọn camera index</button>
                    <p class="small">Webcam thường là 0. Camera USB thường là 1, 2 hoặc 3.</p>
                </div>

                <div class="card">
                    <h2>Điều khiển</h2>
                    <button onclick="resetResult()">Đo lại</button>
                    <button class="secondary" onclick="lockResult()">Khóa khung hình</button>
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

    <script>
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

            let badges = "";
            badges += `<span class="badge">${data.camera_name}</span>`;
            badges += `<span class="badge">Index: ${data.camera}</span>`;
            badges += `<span class="badge">Web FPS: ${data.display_fps}</span>`;
            badges += `<span class="badge">Predict mỗi: ${data.predict_every}s</span>`;
            badges += `<span class="badge">${data.locked ? "Khung hình đang khóa" : "Đang tìm bàn tay"}</span>`;
            badges += `<span class="badge">${data.predicting ? "Model đang chạy" : "Model chờ frame tiếp theo"}</span>`;
            badges += `<p class="small">${data.status}</p>`;

            document.getElementById('badges').innerHTML = badges;

            const m = data.metrics || {};

            document.getElementById('metrics').innerHTML =
                metricBox("Sinh đạo", m.life_line) +
                metricBox("Trí đạo", m.head_line) +
                metricBox("Tâm đạo", m.heart_line);

            document.getElementById('reading').innerText =
                data.reading || "Chưa có kết quả. Hãy đưa lòng bàn tay rõ hơn vào camera.";
        }

        async function resetResult() {
            await fetch('/api/reset', {method: 'POST'});
            await updateStatus();
        }

        async function lockResult() {
            await fetch('/api/lock', {method: 'POST'});
            await updateStatus();
        }

        async function setCamera(index) {
            await fetch('/api/camera/' + index, {method: 'POST'});
            await updateStatus();
        }

        async function setCustomCamera() {
            const index = document.getElementById("customCamera").value;
            await setCamera(index);
        }

        document.addEventListener("keydown", function(e) {
            if (e.key === "r" || e.key === "R") {
                resetResult();
            }
        });

        setInterval(updateStatus, 500);
        updateStatus();
    </script>
</body>
</html>
"""


def html_page():
    return HTML_TEMPLATE.replace("{INFO_HTML}", model_info_html())


@app.route("/")
def index():
    return html_page()


@app.route("/video_feed")
def video_feed():
    def generate():
        frame_delay = 1.0 / max(1, DISPLAY_FPS)

        while True:
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


@app.route("/api/status")
def api_status():
    return jsonify(engine.payload())


@app.route("/api/reset", methods=["POST"])
def api_reset():
    engine.reset()
    return jsonify({"ok": True})


@app.route("/api/lock", methods=["POST"])
def api_lock():
    engine.lock_now()
    return jsonify({"ok": True})


@app.route("/api/camera/<int:index>", methods=["POST"])
def api_camera(index):
    engine.set_camera(index)
    return jsonify({"ok": True, "camera": index})


@app.route("/static_generated/<path:name>")
def static_generated(name):
    return send_from_directory(STATIC_DIR, name)


def start_ngrok():
    if not USE_NGROK:
        return

    if not NGROK_AUTHTOKEN.strip():
        print("USE_NGROK=True nhưng NGROK_AUTHTOKEN đang trống.")
        return

    try:
        from pyngrok import ngrok

        ngrok.set_auth_token(NGROK_AUTHTOKEN.strip())
        public_url = ngrok.connect(PORT, "http")

        print()
        print("NGROK PUBLIC URL:")
        print(public_url)
        print()

    except Exception as e:
        print("Không mở được ngrok:", e)


def main():
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

    cam_thread = threading.Thread(target=engine.camera_loop, daemon=True)
    pred_thread = threading.Thread(target=engine.predict_loop, daemon=True)

    cam_thread.start()
    pred_thread.start()

    start_ngrok()

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=True,
        use_reloader=False,
    )

    engine.stop = True


if __name__ == "__main__":
    main()