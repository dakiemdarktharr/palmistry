"""Deterministic geometric test fixture. This is not a human palm dataset."""
from pathlib import Path
import csv
import hashlib

import cv2
import numpy as np
from PIL import Image

from prototype_common import (validate_mask, write_class_map, write_json, sha256,
                              perceptual_hash, group_rows, split_summary)


def generate_fixture(root):
    from palmistry_strict_auto_onefile import crop_palm_auto
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    expected = {}
    for index in range(6):
        rng = np.random.default_rng(400 + index)
        # Independent geometric masks, never the output of the segmenter under test.
        mask = np.zeros((128, 128), dtype=np.uint8)
        center = (int(rng.integers(45, 83)), int(rng.integers(49, 79)))
        axes = (int(rng.integers(25, 39)), int(rng.integers(35, 48)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1, cv2.LINE_8)
        cx, cy = center
        for cls, points in (
            (2, [(cx-16, cy-19), (cx-22, cy), (cx-9, cy+26)]),
            (3, [(cx-12, cy-4), (cx+5, cy+2), (cx+21, cy+5)]),
            (4, [(cx-13, cy-18), (cx+5, cy-13), (cx+22, cy-19)]),
            (5, [(cx+8, cy+11), (cx+6, cy+25)]),
        ):
            cv2.polylines(mask, [np.array(points, dtype=np.int32)], False, cls, 2, cv2.LINE_8)
        image = rng.integers(15, 50, (128, 128, 3), dtype=np.uint8)
        image[mask == 1] = np.clip(np.array([205, 148, 108]) + rng.integers(-14, 15, (int((mask == 1).sum()), 3)), 0, 255)
        image[mask >= 2] = [65, 40, 28]
        validate_mask(mask, image.shape[:2])
        folder = root / "images" / f"case_{index}"
        folder.mkdir(parents=True, exist_ok=True)
        image_path = folder / "input.png"
        mask_path = folder / "mask.png"
        # Re-running the fixture may overwrite its own known generated files only.
        Image.fromarray(image).save(image_path)
        Image.fromarray(mask).save(mask_path)
        rows.append({"image_path": image_path.relative_to(root).as_posix(),
                     "mask_path": mask_path.relative_to(root).as_posix(),
                     "source_group": f"synthetic_case_{index}", "source_path": f"case_{index}/input.png",
                     "subject_id": f"test-only-geometric-{index}", "hand_id": "not-a-human-hand",
                     "sha256": sha256(image_path), "phash": perceptual_hash(image),
                     "label_provenance": "deterministic test fixture; geometric labels; no real performance evidence"})
        expected[str(index)] = {"rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                                "class_ids": np.unique(mask).tolist()}
    rows = group_rows(rows, seed=42)
    with (root / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_class_map(root)
    test = next(row for row in rows if row["split"] == "test")
    rgb = np.array(Image.open(root / test["image_path"]).convert("RGB"))
    target = np.array(Image.open(root / test["mask_path"]))
    crop, _, meta = crop_palm_auto(rgb, 128)
    x0, y0, x1, y1 = meta["crop_xyxy"]
    aligned = np.zeros((128, 128), np.uint8)
    resized = cv2.resize(target[y0:y1, x0:x1], (meta["new_w"], meta["new_h"]), interpolation=cv2.INTER_NEAREST)
    aligned[meta["y0"]:meta["y0"]+meta["new_h"], meta["x0"]:meta["x0"]+meta["new_w"]] = resized
    Image.fromarray(rgb).save(root / "input.png")
    Image.fromarray(aligned).save(root / "expected_mask.png")
    Image.fromarray(crop).save(root / "expected_crop.png")
    report = {"fixture": "geometric-v1", "inference_source": "deterministic test fixture", "test_only": True,
              "split_counts": split_summary(rows), "expected": expected, "demo_split": "test",
              "demo_source": test["image_path"], "crop_meta": meta,
              "limitation": "Synthetic shape mechanics only. Subject IDs describe generated cases, not people."}
    write_json(root / "expected.json", report)
    return report
