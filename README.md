## Pipeline keypoint mới (Heart / Head / Life / Fate)

Mở **http://127.0.0.1:8501/keypoints** để kiểm duyệt 6 điểm mỗi đường, đo B-spline, nhập metadata, train CNN NumPy từ scratch và tạo pseudo-label. Không dùng PyTorch/scikit-learn. Hướng dẫn và giới hạn: [docs/KEYPOINT_PIPELINE.md](docs/KEYPOINT_PIPELINE.md).

Project `palm_keypoints_main` có 400 ảnh chờ duyệt và 11.600 nguồn còn lại. Cần nhãn keypoint thật trước khi train; checkpoint segmentation cũ không dùng trực tiếp cho mô hình mới. Các phần mask/camera bên dưới là legacy và được giữ để đối chiếu.

# Palm-line segmentation · reproducible computer-vision prototype

A small Python prototype for image quality assessment, palm-line segmentation
and visualization. It makes an existing classical-CV → mask → NumPy pixel model → Flask
workflow reproducible and testable without requiring a real dataset to run
the mechanics. It is **only a prototype**.

Thử nghiệm thị giác máy tính có thể tái lập: đánh giá chất lượng ảnh, phân đoạn
đường lòng bàn tay và hiển thị kết quả. Các hình học này không chứng minh khả
năng dự đoán tính cách, sức khỏe, hành vi hay tương lai.

There is **no real dataset or real-data trained checkpoint included**. A
deterministic geometric fixture and a one-epoch fixture checkpoint can be
generated locally. Their scores are test-only and establish no real-world
accuracy or scientific validity. Obtain consent before capturing hands.

## Architecture and scope

```text
local RGB image → bounded decode → quality assessment → palm crop
  → explicit classical CV OR explicit compatible trained checkpoint
  → canonical semantic mask + crop overlay + structured JSON
  → evaluation against an independently supplied, aligned target mask

camera preview → latest-frame slot → one subprocess inference worker
  → unique temporary request directory → request-ID success check
  → synchronized result snapshot → live overlay / manual lock
```

The UI's subprocess worker is a deliberate compatibility fallback. It reloads
the model each request; frames arriving during inference replace the single
pending frame. A result's mask, overlay, reading, metrics and frame ID remain
together in crop coordinates. Raw camera preview is labeled as having no valid
prediction. No checkpoint/hash means preview-only mode, without a
repeated inference failure loop.

`class_map.v2.json` is the active shared ontology; v1 is retained for explicit migration:

| ID | Meaning |
|---|---|
| 0 | background |
| 1 | palm_area |
| 2 | life_line |
| 3 | head_line |
| 4 | heart_line |
| 5 | minor_or_unknown_line |
| 6 | simian_line |

The traditional line names identify geometric classes only. Legacy converter
`fate_line` annotations require explicit `--class-map-version fate-v0`, which
merges them into class 5 and records that conversion. Unversioned datasets and
incompatible checkpoints are rejected rather than silently mixed.

## Repository

| File | Purpose |
|---|---|
| `palmistry_strict_auto_onefile.py` | Existing doctor, build, mask-folder, single-mask, train and predict CLI; added fixture/evaluate commands |
| `tien_xu_ly.py` | Annotation conversion, integer rasterization, polygon rejection and collision checks |
| `giao_dien_ui.py` | Local Flask camera preview and request-isolated inference |
| `pipeline.py` | Bounded ZIP/RAR/folder ingestion, preprocessing, auto pseudo-labeling, split audit and review queue |
| `bootstrap_review.py` | Trains from approved masks, confidence-filters model pseudo-labels and optionally trains the final model |
| `prototype_common.py` | Ontology, bounded image decoding, mask validation, group split audit, metrics and provenance |
| `prototype_fixture.py` | Deterministic RGB images, geometric masks and expected hashes; test-only |
| `prototype_config.json`, `.env.example` | Seed, CPU device, paths, thresholds, interval, timeout and zero retention |
| `requirements.txt`, `requirements-model.txt` | Tested pinned runtime; NumPy model has no extra ML framework |
| `tests/` | Data, model, CLI, UI, security and cleanup regressions |
| `scripts/verify.py` | Runs existing commands and records evidence; contains no duplicate model logic |
| `docs/verification.md`, `docs/verification.json` | Verification coverage, exact recorded commands, measured results and limitations |
| `docs/dataset_sources.md` | Candidate hand/palm datasets, counts, access constraints and import guidance |
| `artifacts/` | Ignored generated images, checkpoints, logs and reports |

## Environment setup — PowerShell, Python 3.11

Run from the repository root. No shell activation is necessary:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe palmistry_strict_auto_onefile.py doctor
```

The core environment already contains everything needed for preprocessing, NumPy model training and inference:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

This setup was separately installed and tested in `.venv-clean` without global
site packages: Python 3.11.9, OpenCV 5.0.0, NumPy 2.4.6, Flask 3.1.3 and
the pinned plotting/runtime packages. The model is a deterministic NumPy
diagonal-Gaussian pixel classifier; PyTorch, scikit-learn and pandas are not
installed or imported. See the saved [package freeze](docs/environment-freeze.txt).
These are exact version pins, not a cryptographic wheel lock or a tested
cross-platform environment.

## Fixture and regression commands

Run the complete evidence workflow, including preprocessing, NumPy training, prediction, evaluation, offline UI IPC and loopback HTTP:

```powershell
.\.venv\Scripts\python.exe scripts/verify.py --output artifacts/verification
```

Or run individual fixture/tests commands:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe palmistry_strict_auto_onefile.py fixture --output_dir artifacts/fixture
```

`geometric-v1` creates six 128×128 RGB images with integer masks containing
IDs 0–5, a manifest, canonical class map and deterministic expected hashes.
Conservative perceptual grouping yields 1 train image, 3 validation images,
and 2 test images, in one group per split. These are generated cases, not
human subjects. `input.png` selects one synthetic test image;
`expected_mask.png` is its geometric target transformed into inference crop
coordinates. This target is not generated by the segmenter being evaluated.

## Preprocessing local images

```powershell
.\.venv\Scripts\python.exe palmistry_strict_auto_onefile.py single-mask --image artifacts/fixture/input.png --output_dir artifacts/preprocessed --out_size 128 --min_short_side 128
```

For local photos, keep normal quality thresholds and use a new output folder:

```powershell
.\.venv\Scripts\python.exe palmistry_strict_auto_onefile.py --seed 42 mask-folder --input_dir C:/data/palms --output_dir artifacts/local-dataset --out_size 256
.\.venv\Scripts\python.exe tien_xu_ly.py convert --input_dir C:/data/annotations --output_dir artifacts/converted --out_size 256 --line_thickness 3 --min_lines 2
```

These two local-data examples require data you supply; they were not run on
a real dataset. `build --input_dir C:/data/palms` is also available, but remote
downloads are disabled. Retain source, consent, author, license, attribution
and checksums for local datasets. No license provenance is invented here.

The converter uses LINE_8, rejects fewer-than-three or collinear palm points,
hashes source-relative filenames and refuses existing output collisions. Its
palm hull and bounding-box center-lines remain **pseudo-labels**, not anatomical
ground truth. Conversion records rejected samples. If split auditing fails,
it saves the conversion report and exits with an error; those files are not
training-ready.

Training manifests require `image_path`, `mask_path`, `split` and source/group
metadata. Optional `subject_id` and `hand_id` strengthen grouping. Actual image
hashes are recomputed before training. Samples sharing a subject, source,
identical bytes or a pHash distance ≤6 are grouped transitively. All three
splits must be nonempty; source-folder fallback cannot establish subject
independence. A single-folder input may therefore be rejected. Do not duplicate
images or invent subject IDs to force a split. The pHash audit uses seven exact candidate bands and has no 10,000-sample prototype cap.

## End-to-end pipeline

Run the complete ingestion, preprocessing, automatic pseudo-labeling,
leakage-safe train/validation/test split, model training and test evaluation
with one command:

    .\.venv\Scripts\python.exe pipeline.py --input_zip C:\data\palms.zip --run_dir artifacts\pipeline_001 --out_size 512 --input_size 256 --epochs 60 --device cpu

    .\.venv\Scripts\python.exe pipeline.py --input_rar C:\data\palms.rar --run_dir artifacts\pipeline_002 --out_size 512 --input_size 256 --epochs 60 --device cpu

A local image folder can be used instead:

    .\.venv\Scripts\python.exe pipeline.py --input_dir C:\data\palms --run_dir artifacts\pipeline_001

The run directory contains 00_raw, 01_preprocessed_dataset, 02_training,
03_test and pipeline_summary.json. ZIP/RAR paths are checked for traversal,
member-count, compressed-size and expanded-size limits. Source subfolders are
preserved as independent groups for the split audit; a flat folder/archive with
fewer than three independent groups stops with an error instead of leaking
near-duplicate images across splits. Add --preprocess_only to stop after
preprocessing, auto-labeling and split generation.

Automatic labels are generated by the classical CV line extractor and are
marked as pseudo-labels in every manifest and report. They are useful for
testing pipeline mechanics, but they are not human ground truth and should
not be treated as evidence of real-world accuracy. The train phase requires
the NumPy model from the core environment; no optional ML environment is required.

### Human review 400 mask đầu và bootstrap pseudo-label

Trên giao diện, chọn **Chuẩn bị queue review** sau khi nhập dataset. Thư mục ảnh
được kiểm kê và lọc trùng SHA-256, không chép lại TIFF và không áp giới hạn 20.000
entry của archive cho tổng dataset. Chỉ 400 ảnh được tạo crop, mask và overlay;
các ảnh còn lại được lập chỉ mục để xử lý sau khi duyệt. Mọi mask mới là `pending`,
chưa chia train/val/test. Trang duyệt có 40 dòng/trang và lưu trạng thái từng dòng.

Bấm lại với tên run cũ tạo run mới; dữ liệu cũ vẫn được giữ. App nhớ queue sau khi
khởi động lại. Nếu job bị ngắt, giao diện báo rõ để chạy lại. Ảnh gốc phải được giữ
nguyên đường dẫn vì pipeline tham chiếu trực tiếp. Train chỉ bắt đầu sau khi kiểm
tra mask approved và đủ nhóm nguồn độc lập. Các tập val/test đã chia cho seed được
giữ cố định, pseudo-label mới chỉ được vào train; ảnh rejected/pending không thành
pseudo-label. Xem [các lỗi, cách sửa và fallback](docs/REVIEW_PIPELINE_FIXES.md).


Để dùng kiểm duyệt thủ công làm seed đáng tin cậy, chạy phase tiền xử lý và tạo queue:

    .\.venv\Scripts\python.exe pipeline.py --input_zip C:\data\palms.zip --run_dir artifacts\pipeline_review_001 --prepare_review --review_count 400 --out_size 512 --input_size 256

Mở artifacts/pipeline_review_001/review/queue.html để xem ảnh và overlay. Nếu cần,
sửa file mask PNG bằng công cụ annotation, giữ class IDs 0..6 (simian = 6) và đúng kích thước ảnh.
Sau đó mở review/review.csv, đặt review_status thành approved cho mask đã kiểm tra,
rejected cho mask sai, hoặc giữ pending nếu chưa xem. Có thể ghi lý do vào
review_notes.

Khi đã duyệt xong, chạy bootstrap:

    .\.venv\Scripts\python.exe bootstrap_review.py --run_dir artifacts\pipeline_review_001 --out_dir artifacts\bootstrap_001 --confidence 0.90 --min_line_ratio 0.005 --train_final --device cpu

Script kiểm tra lại mọi path và class ID, train seed model chỉ từ các mask approved,
dùng seed model dự đoán các ảnh gốc còn lại trong remaining_sources.json, chỉ giữ dự đoán đạt ngưỡng confidence
và tỷ lệ pixel đường tối thiểu, rồi ghi labels_bootstrapped.csv. Các dòng được sinh tự
động mang provenance và confidence riêng; ảnh không đạt ngưỡng nằm trong
bootstrap_skipped.csv để không bị ép thành nhãn. --train_final train model cuối trên
manifest kết hợp và chạy test split độc lập. Cần ít nhất ba source/perceptual groups độc lập;
không tách bản sao để ép đủ split.

## Training

One real CPU training epoch on the generated fixture:

```powershell
$env:OMP_NUM_THREADS = '1'
.\.venv\Scripts\python.exe palmistry_strict_auto_onefile.py --seed 42 train --data_root artifacts/fixture --labels_csv artifacts/fixture/labels.csv --out_dir artifacts/model-smoke --input_size 32 --epochs 1 --batch_size 1 --grad_accum 2 --model_size tiny --device cpu --resume none
```

This produces a **fixture-only checkpoint**, not a useful palm model. Training
records the seed, device, model version, label provenance, split counts,
manifest hash, application source hashes, Git revision and full configuration.
Python and NumPy are seeded. A final partial
accumulation window is stepped. Input sizes must be 32–2048 and divisible by
16. Validation is nonaugmented and independent; missing validation stops
training. Test samples are never used for model selection.

For your own verified masks, replace `--data_root` and `--labels_csv` with the
versioned local dataset paths. Real-data training has not been measured.
Resume is not needed for this statistical model; rerunning with the same manifest
and seed is deterministic. The `device` flag is retained for CLI compatibility and
always runs on CPU.

## Inference

Explicit classical fallback on the synthetic fixture:

```powershell
.\.venv\Scripts\python.exe palmistry_strict_auto_onefile.py predict --image artifacts/fixture/input.png --allow-classical-fallback --out_size 128 --min_short_side 128 --min_image_quality 0 --out_json artifacts/classical/result.json --out_mask artifacts/classical/mask.png --out_overlay artifacts/classical/overlay.png
```

The zero quality threshold is for this synthetic test only. For a real photo,
omit it. Results use `heuristic_score` / ranking semantics, not calibrated
probabilities. Low quality can cause abstention; there is no learned OOD detector.

Use the fixture checkpoint generated by the training command:

```powershell
$modelHash = (Get-FileHash artifacts/model-smoke/checkpoints/best.npz -Algorithm SHA256).Hash.ToLower()
.\.venv\Scripts\python.exe palmistry_strict_auto_onefile.py predict --image artifacts/fixture/input.png --checkpoint artifacts/model-smoke/checkpoints/best.npz --checkpoint-sha256 $modelHash --device cpu --out_size 128 --min_short_side 128 --min_image_quality 0 --out_json artifacts/model/result.json --out_mask artifacts/model/mask.png --out_overlay artifacts/model/overlay.png
```

For an external checkpoint, obtain its hash from a trusted source; computing
a hash yourself does not establish trust. Checkpoints use a strict `.npz` schema,
strict state-dict shapes, matching class-map/model versions and bounded file
size. There is no recursive checkpoint discovery and no unsafe loading fallback.

Every prediction response identifies its request, `model_used` and
`inference_source`: `trained model`, `classical CV fallback`, or `none` when
no segmentation was produced. The fixture generator separately identifies
`deterministic test fixture`. A missing explicitly requested checkpoint always
fails, even with the fallback flag. A bad image writes structured JSON and
returns exit 2. A successful segmentation can still abstain from accepting
individual line labels. JSON `success_marker` confirms only that this
request produced segmentation artifacts.

## Evaluation and visualization

```powershell
.\.venv\Scripts\python.exe palmistry_strict_auto_onefile.py evaluate --prediction artifacts/classical/mask.png --target artifacts/fixture/expected_mask.png --dataset-name geometric-v1 --split fixture-only --inference-source 'classical CV fallback' --output artifacts/classical/evaluation.json
.\.venv\Scripts\python.exe palmistry_strict_auto_onefile.py evaluate --prediction artifacts/model/mask.png --target artifacts/fixture/expected_mask.png --dataset-name geometric-v1 --split fixture-only --inference-source 'trained model' --output artifacts/model/evaluation.json
```

Inspect `artifacts/classical/overlay.png` and `artifacts/model/overlay.png`.
Masks/overlays use crop coordinates; evaluation targets must be aligned at
the same shape. Reports include per-class IoU, Dice, precision, recall, F1,
support pixels and a confusion matrix. Undefined metrics are null; macro IoU
averages nonempty unions. Evaluation labels supplied by a user describe that
user's dataset; they do not certify its annotation quality.

Measured on **2026-09-11**, with the commands above executed through
`scripts/verify.py` in the clean CPU environment:

| Metric | Value | Dataset / split | Command / limitation |
|---|---:|---|---|
| Classical mean IoU | 0.286904 | geometric-v1, fixture-only; one synthetic test image, 16,384 pixels | `evaluate` classical command above; no real accuracy claim |
| One-epoch model mean IoU | 0.050990 | Same aligned synthetic test image | `evaluate` model command above; random-initialized tiny model trained for one epoch |
| Model mean main-line Dice | 0.000000 | geometric-v1 validation, 3 images at 32×32 | `train` above; validation-only selection metric |
| NumPy-only regressions | 53 passed, 0 failed, 5 legacy ML tests skipped | Generated fixtures/mocks; no real people | `python -m unittest discover -s tests -v` |
| Legacy framework tests | 5 skipped by project policy | PyTorch/scikit-learn are not installed | `python -m unittest discover -s tests -v`; skipped model checks |

Full measured values, per-class metrics, exact command arguments, process
startup samples and offline UI latency are in [verification.json](docs/verification.json).
These scores do not constitute a benchmark. The model score is poor; this
work verifies reproducibility and failure handling, not model quality.

## Flask demo, camera and privacy

For the latest confirmed defects, fixes and validation, see
[the 2026-10-07 project audit](docs/PROJECT_AUDIT_2026-10-07.md).
GitHub Actions runs `scripts/verify.py` on Windows and Linux from a clean
checkout; `scripts/verify_desktop.py` checks the embedded desktop template and
download flow without using a camera or the personal dataset.

```powershell
.\.venv\Scripts\python.exe giao_dien_ui.py
```

Clicking the Desktop shortcut `Palmistry Live Camera.lnk` runs `run_palmistry.bat`, starts the local server in a hidden process, waits for `/keypoints` to return HTTP 200, and opens `palmistry_desktop.py` in a native PySide6 application window. The interface stays inside the app with no browser tabs or address bar. Clicking the shortcut again brings the existing app window to the front. File selection, CSV downloads, review pages and full-size images are supported inside the app. The server remains available after closing the window so background training can finish; reopening the shortcut reconnects to it. Launcher errors appear in a dialog; server and desktop logs are in `artifacts/keypoints-ui.*.log` and `artifacts/desktop.*.log`. Desktop dependencies are included in `requirements.txt`; no Opera installation is required.

Open [the local demo](http://127.0.0.1:8501). A working OS camera and driver
are needed for camera preview. Camera capture was **not tested on real
hardware**; tests use a fake camera and actual offline fixture IPC. With no
checkpoint, the UI stays in preview-only mode. Manual lock requires a valid
prediction. A prediction failure clears the old result and disables the
worker until restart, preventing repeated failures.

For model mode, configure an explicit compatible checkpoint and its trusted hash:

```powershell
$env:PALM_CHECKPOINT = 'artifacts/model-smoke/checkpoints/best.npz'
$env:PALM_CHECKPOINT_SHA256 = (Get-FileHash $env:PALM_CHECKPOINT -Algorithm SHA256).Hash.ToLower()
$env:PALM_CONFIG = 'prototype_config.json'
.\.venv\Scripts\python.exe giao_dien_ui.py
```

The example checkpoint remains fixture-only. `.env.example` lists shell
settings; `.env` files are not loaded automatically. CLI `--config` and UI
`PALM_CONFIG` select JSON settings; explicit CLI options override defaults.
The default worker interval is 2 seconds **after** each prediction and its
timeout is 30 seconds. Model reload overhead makes this unsuitable for
high-frame-rate inference. One worker retains only the newest waiting frame.

The app binds to **127.0.0.1 only**. LAN access and ngrok remain unavailable. The UI provides an explicit ZIP/RAR upload button; automatic startup extraction remains disabled. Local endpoints have Host/peer checks,
CSRF protection on POST, Origin checks, a process-local rate limit, CSP and
no-store headers. Do not expose the Flask development server through a proxy.
Network authentication, TLS and deployment hardening are deferred.

Frames live in memory except for one unique, account-local OS temporary
directory per prediction. It is cleaned on normal success, error and timeout;
retention is fixed to zero. No camera frames are saved in the repository.
Ctrl+C signals shutdown, releases the camera and joins workers. Abrupt crashes
may leave OS temporary remnants; the files are neither encrypted nor securely
erased, and actual driver shutdown behavior remains unverified. CLI outputs
persist at the paths you explicitly request.

## Implemented, verified, deferred

Implemented and tested: canonical IDs, integer masks, degenerate polygon
rejection, collision refusal, conservative split audit, independent validation,
safe checkpoint loading, seed/config/source provenance, final accumulation
step, structured inference errors, request-isolated IPC, consistent live/locked
snapshots, local-only guards, cleanup and deterministic fixture evaluation.

Deferred or blocked: real annotated data and useful checkpoints; independent
accuracy evaluation; confidence calibration; persistent in-process model
loading; real camera and hardware-accelerator checks; safe remote download/attribution; authenticated network deployment.
Heuristic cropping, line naming, pseudo-label circularity and missed identity
leakage remain limitations. Read the [full verification and limitation report](docs/verification.md).

## Truthful two-line resume bullet

- Built a reproducible computer-vision prototype for palm-image preprocessing, segmentation and visualization with explicit inference provenance.
- Added deterministic fixture evaluation and 53 passing CPU regression tests covering data integrity, checkpoint safety and local demo failure handling.

## Work together / Cùng nghiên cứu

Interested in image segmentation, datasets or evaluation?
[Email Khôi](mailto:daethphate@gmail.com) or [say hi on Instagram](https://www.instagram.com/koiluuuuv/).
Based in HCMC, open to meeting offline.

Bạn muốn cùng tìm hiểu phân đoạn ảnh, dữ liệu hoặc đánh giá mô hình?
Mình ở TP.HCM và muốn tìm bạn có thể gặp trực tiếp.

## Windows installer and ZIP/RAR dataset import

The repository includes a PowerShell installer for Windows. From a PowerShell terminal in the repository:

    .\installer\install.bat
    .\installer\install.bat -WithModel

The installer copies the app to %LOCALAPPDATA%\Palmistry, creates an isolated
.venv, installs requirements.txt, creates Start Menu and Desktop shortcuts,
and launches the local app. `-WithModel` is retained for compatibility and does
not install an extra ML framework because the model uses NumPy. Python 3.11+ and
an internet connection for pip are required.

Inside the app, choose a ZIP or RAR in the Nhập ảnh từ ZIP hoặc RAR card and press Nhận ZIP/RAR và nhập ảnh. Large archives are uploaded in 16 MB chunks with retry and progress reporting, then imported in a background job so a 4GB RAR does not hit a single-request timeout. Real RAR archives are extracted in one batch backend process instead of spawning a process per image, and the UI reports extraction and image-validation progress. Valid image files are decoded and written to dataset/images; each import creates a report under dataset/imports. The default limits are 8 GB compressed, 32 GB expanded, 100 MB per member and 20,000 members. Set PALM_MAX_ARCHIVE_BYTES, PALM_MAX_ARCHIVE_EXPANDED_BYTES, PALM_MAX_ARCHIVE_MEMBER_BYTES, PALM_MAX_ARCHIVE_MEMBERS, PALM_ARCHIVE_UPLOAD_CHUNK_BYTES, PALM_ARCHIVE_UPLOAD_DIR and PALM_DATASET_DIR to tune them for available disk and bandwidth. Non-image, corrupt, directory, unsafe and over-sized entries are reported as skipped. Password-protected and multi-volume RAR archives are rejected with an actionable message; decrypt or combine them first. No archive is imported automatically at startup.

The same app now exposes the end-to-end pipeline in the **Pipeline dataset → review → train** card. Enter a run name, choose the number of masks to review (400 by default), and press **Chuẩn bị queue review**. When preprocessing finishes, **Mở trang duyệt mask** opens the review table; each row can be marked pending, approved or rejected and saved from the browser. Set the pseudo-label confidence and output name, then press **Bootstrap + train model cuối**. Jobs run in the background and their logs/results remain under artifacts/<run-name>; no pipeline CLI command is required for normal use.

## Bốn đường và phép đo v2

Mở `/analyze` từ app để nhận một ảnh bàn tay, chọn trạng thái lật gương và chạy CV thử
nghiệm hoặc model đã cấu hình. Kết quả chỉ gồm các đường được chấp nhận trong head,
heart, life và simian. Đây là **reproducible computer-vision prototype**, chưa được xác
nhận độ chính xác trên người thật; không tạo kết luận tính cách từ hình học chỉ tay.

`palm_geometry.py` đo đường chính trên đồ thị skeleton với cạnh chéo có trọng số, chia
cho đường kính nội tiếp lớn nhất của vùng bàn tay (đại lượng thay thế bề rộng lòng bàn
tay trong ảnh 2D). Không cộng mọi nhánh hay chia theo kích thước canvas. Nhận diện bên
tay yêu cầu ảnh lòng bàn tay mở, ngón hướng lên và biết ảnh có lật gương hay không;
trường hợp thiếu dấu hiệu trả `unknown`. Camera chuyển cấu hình flip cho inference.

Simian dùng class 6, giữ nguyên class 5 là đường phụ/chưa rõ. Model v1 phải được train
lại với schema v2. Model không được dự đoán lớp không có mẫu học. CV cổ điển chưa tự
nhận diện simian: cần annotation đã duyệt hoặc model học từ annotation simian. Nếu
head/heart và simian xung đột, hệ thống không báo ba đường như những phép đo độc lập.

Nút **Tạo bản sao queue v2 (simian)** giữ nguyên queue v1 và tạo bản sao để kiểm tra
thêm simian; trạng thái bản sao là pending. Không xóa ảnh gốc, không tự approved, không
dùng đường phụ làm nhãn simian. Điều kiện nhóm nguồn/người độc lập khi train vẫn giữ.

[Nghiên cứu palmistry và đặc tả phép đo](docs/PALMISTRY_RESEARCH.md) lưu nguồn sơ cấp,
các khác biệt giữa trường phái, giới hạn bằng chứng và điều kiện kiểm chứng CV tiếp
theo. Không có bảng quy đổi chiều dài đường sang tính cách cá nhân.
