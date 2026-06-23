import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

CLASS_ALIASES = {
    "life": 2,
    "life_line": 2,
    "life line": 2,
    "lifeline": 2,
    "head": 3,
    "head_line": 3,
    "head line": 3,
    "headline": 3,
    "heart": 4,
    "heart_line": 4,
    "heart line": 4,
    "heartline": 4,
    "fate": 5,
    "fate_line": 5,
    "fate line": 5,
    "fateline": 5
}

MASK_CLASS_NAMES = {
    0: "background",
    1: "palm_area",
    2: "life_line",
    3: "head_line",
    4: "heart_line",
    5: "fate_line"
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def norm_name(x):
    x = str(x).strip().lower().replace("-", "_")
    x = re.sub(r"\s+", " ", x)
    return x

def parse_names_from_yaml(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    names = {}
    m = re.search(r"names\s*:\s*\[([^\]]+)\]", text, re.S)
    if m:
        parts = [p.strip().strip("'\"") for p in m.group(1).split(",")]
        return {i: p for i, p in enumerate(parts)}
    lines = text.splitlines()
    in_names = False
    list_index = 0
    for line in lines:
        if re.match(r"^\s*names\s*:\s*$", line):
            in_names = True
            continue
        if in_names:
            if re.match(r"^\s*-\s*['\"]?([^'\"]+)['\"]?\s*$", line):
                value = re.match(r"^\s*-\s*['\"]?([^'\"]+)['\"]?\s*$", line).group(1).strip()
                names[list_index] = value
                list_index += 1
                continue
            m2 = re.match(r"^\s*(\d+)\s*:\s*['\"]?([^'\"]+)['\"]?\s*$", line)
            if m2:
                names[int(m2.group(1))] = m2.group(2).strip()
                continue
            if line.strip() == "":
                continue
            if re.match(r"^\S", line):
                break
    return names

def find_yaml(root):
    candidates = list(Path(root).rglob("data.yaml")) + list(Path(root).rglob("data.yml"))
    return candidates[0] if candidates else None

def find_images(root):
    return [p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def possible_label_paths(img_path, root):
    p = Path(img_path)
    out = [p.with_suffix(".txt")]
    parts = list(p.parts)
    for key in ["images", "image", "imgs"]:
        if key in parts:
            idx = parts.index(key)
            new_parts = parts[:]
            new_parts[idx] = "labels"
            out.append(Path(*new_parts).with_suffix(".txt"))
    rel = p.relative_to(root)
    out.append(Path(root) / "labels" / rel.with_suffix(".txt").name)
    if len(rel.parts) >= 2:
        out.append(Path(root) / rel.parts[0] / "labels" / rel.with_suffix(".txt").name)
    seen = []
    for x in out:
        if x not in seen:
            seen.append(x)
    return seen

def read_yolo_label(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) >= 5:
                try:
                    rows.append([float(v) for v in vals])
                except Exception:
                    pass
    return rows

def class_id_to_mask_class(cls_id, names):
    name = names.get(int(cls_id), str(int(cls_id)))
    key = norm_name(name)
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    return CLASS_ALIASES.get(key.replace("_", " "), 0)

def denorm_xy(x, y, w, h):
    return int(round(float(x) * w)), int(round(float(y) * h))

def get_keypoints_from_row(row, w, h):
    vals = row[5:]
    pts = []
    if len(vals) < 4:
        return pts
    if len(vals) % 3 == 0:
        for i in range(0, len(vals), 3):
            x, y, v = vals[i], vals[i + 1], vals[i + 2]
            if v > 0 and 0 <= x <= 1 and 0 <= y <= 1:
                pts.append(denorm_xy(x, y, w, h))
    elif len(vals) % 2 == 0:
        for i in range(0, len(vals), 2):
            x, y = vals[i], vals[i + 1]
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append(denorm_xy(x, y, w, h))
    return pts

def bbox_to_line(row, w, h):
    _, xc, yc, bw, bh = row[:5]
    cx, cy = denorm_xy(xc, yc, w, h)
    bw_px = max(2, int(round(bw * w)))
    bh_px = max(2, int(round(bh * h)))
    if bw_px >= bh_px:
        return [(max(0, cx - bw_px // 2), cy), (min(w - 1, cx + bw_px // 2), cy)]
    return [(cx, max(0, cy - bh_px // 2)), (cx, min(h - 1, cy + bh_px // 2))]

def draw_line_mask(mask, pts, cls, thickness):
    if len(pts) < 2:
        return
    pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(mask, [pts_np], False, int(cls), int(thickness), lineType=cv2.LINE_AA)

def create_palm_area(mask, pts, w, h):
    if len(pts) < 3:
        mask[mask == 0] = 1
        return
    arr = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(arr)
    area = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(area, hull, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))
    area = cv2.dilate(area, kernel, iterations=2)
    mask[(area > 0) & (mask == 0)] = 1

def make_overlay(img, mask):
    overlay = img.copy()
    colors = {
        1: (70, 70, 70),
        2: (255, 60, 60),
        3: (60, 255, 60),
        4: (60, 120, 255),
        5: (255, 220, 60)
    }
    for cls, color in colors.items():
        m = mask == cls
        if np.any(m):
            col = np.array(color, dtype=np.uint8)
            overlay[m] = (0.55 * overlay[m] + 0.45 * col).astype(np.uint8)
    return overlay

def safe_stem(path):
    s = Path(path).stem
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:120]

def infer_split(path):
    parts = [p.lower() for p in Path(path).parts]
    if "valid" in parts or "val" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return "train"

def convert_dataset(input_dir, output_dir, out_size, line_thickness, min_lines, ignore_fate):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    reports_dir = output_dir / "reports"
    for d in [images_dir, masks_dir, overlays_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
    yaml_path = find_yaml(input_dir)
    names = parse_names_from_yaml(yaml_path) if yaml_path else {}
    images = find_images(input_dir)
    rows_out = []
    rejected = []
    split_counts = {}
    for img_path in tqdm(images, desc="Convert Roboflow"):
        label_path = None
        for cand in possible_label_paths(img_path, input_dir):
            if cand.exists():
                label_path = cand
                break
        if label_path is None:
            rejected.append({"image": str(img_path), "reason": "no_label"})
            continue
        labels = read_yolo_label(label_path)
        if not labels:
            rejected.append({"image": str(img_path), "reason": "empty_label"})
            continue
        try:
            with Image.open(img_path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                orig_w, orig_h = im.size
                im_resized = im.resize((out_size, out_size), Image.BILINEAR)
                img_np = np.array(im_resized)
        except Exception as e:
            rejected.append({"image": str(img_path), "reason": f"bad_image:{e}"})
            continue
        mask = np.zeros((out_size, out_size), dtype=np.uint8)
        all_pts = []
        detected = {"life": 0, "head": 0, "heart": 0, "fate": 0}
        for row in labels:
            cls = class_id_to_mask_class(int(row[0]), names)
            if ignore_fate and cls == 5:
                continue
            if cls == 0:
                continue
            pts_orig = get_keypoints_from_row(row, orig_w, orig_h)
            if len(pts_orig) < 2:
                pts_orig = bbox_to_line(row, orig_w, orig_h)
            pts = []
            for x, y in pts_orig:
                nx = max(0, min(out_size - 1, int(round(x / max(1, orig_w) * out_size))))
                ny = max(0, min(out_size - 1, int(round(y / max(1, orig_h) * out_size))))
                pts.append((nx, ny))
            if len(pts) >= 2:
                draw_line_mask(mask, pts, cls, line_thickness)
                all_pts.extend(pts)
                if cls == 2:
                    detected["life"] += 1
                elif cls == 3:
                    detected["head"] += 1
                elif cls == 4:
                    detected["heart"] += 1
                elif cls == 5:
                    detected["fate"] += 1
        good_main = sum(1 for k in ["life", "head", "heart"] if detected[k] > 0)
        if good_main < min_lines:
            rejected.append({"image": str(img_path), "reason": "not_enough_main_lines", "detected": detected})
            continue
        create_palm_area(mask, all_pts, out_size, out_size)
        stem = safe_stem(img_path)
        out_img = images_dir / f"{stem}.jpg"
        out_mask = masks_dir / f"{stem}_mask.png"
        out_overlay = overlays_dir / f"{stem}_overlay.jpg"
        Image.fromarray(img_np).save(out_img, quality=95)
        Image.fromarray(mask).save(out_mask)
        Image.fromarray(make_overlay(img_np, mask)).save(out_overlay, quality=90)
        split = infer_split(img_path)
        split_counts[split] = split_counts.get(split, 0) + 1
        rows_out.append({
            "image_path": str(out_img.relative_to(output_dir)).replace("\\", "/"),
            "mask_path": str(out_mask.relative_to(output_dir)).replace("\\", "/"),
            "split": split,
            "cls_quality": "roboflow_label",
            "reg_line_density": float(np.mean(mask >= 2)),
            "life_present": int(detected["life"] > 0),
            "head_present": int(detected["head"] > 0),
            "heart_present": int(detected["heart"] > 0),
            "fate_present": int(detected["fate"] > 0)
        })
    csv_path = output_dir / "labels_pseudo_strict.csv"
    fieldnames = ["image_path", "mask_path", "split", "cls_quality", "reg_line_density", "life_present", "head_present", "heart_present", "fate_present"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    with open(output_dir / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(MASK_CLASS_NAMES, f, ensure_ascii=False, indent=2)
    with open(reports_dir / "convert_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "yaml": str(yaml_path) if yaml_path else None,
            "names": names,
            "total_images_found": len(images),
            "accepted": len(rows_out),
            "rejected": len(rejected),
            "split_counts": split_counts,
            "rejected_sample": rejected[:100]
        }, f, ensure_ascii=False, indent=2)
    print(json.dumps({"accepted": len(rows_out), "rejected": len(rejected), "csv": str(csv_path), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("convert", nargs="?")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="dataset_roboflow_converted")
    parser.add_argument("--out_size", type=int, default=1024)
    parser.add_argument("--line_thickness", type=int, default=11)
    parser.add_argument("--min_lines", type=int, default=1)
    parser.add_argument("--ignore_fate", action="store_true")
    args = parser.parse_args()
    convert_dataset(args.input_dir, args.output_dir, args.out_size, args.line_thickness, args.min_lines, args.ignore_fate)

if __name__ == "__main__":
    main()
