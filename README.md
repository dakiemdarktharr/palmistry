# Palmistry · palm-line segmentation

An experimental Python computer vision project for detecting and segmenting palm lines, with preprocessing, training and a camera UI.

Thử nghiệm computer vision bằng Python để phát hiện và phân đoạn đường chỉ tay, có tiền xử lý, huấn luyện và giao diện camera.

**Status / Trạng thái:** experimental source code. This repository currently contains three Python scripts; datasets, trained checkpoints and a packaged demo are not included. / Repo hiện có ba script Python, chưa kèm dataset, checkpoint đã huấn luyện hoặc bản demo đóng gói.

## What's inside / Nội dung

- `palmistry_strict_auto_onefile.py`: classical CV pseudo-masks, dataset preparation, segmentation training and prediction tools. / Tạo pseudo-mask, chuẩn bị dữ liệu, huấn luyện và dự đoán.
- `giao_dien_ui.py`: Flask camera interface using OpenCV. Prediction requires a compatible trained checkpoint. / Giao diện camera Flask; dự đoán cần checkpoint phù hợp.
- `tien_xu_ly.py`: image and label preprocessing utilities. / Công cụ tiền xử lý ảnh và nhãn.

The pipeline distinguishes palm area and three major palm-line classes, plus minor/unknown lines. The preprocessing script uses a different label for class 5; review class mappings before combining datasets.

Pipeline phân biệt vùng bàn tay và ba nhóm đường chính, cùng đường phụ/chưa xác định. Script tiền xử lý đặt tên lớp 5 khác pipeline; cần đối chiếu mapping trước khi ghép dữ liệu.

## Explore / Bắt đầu tìm hiểu

Read the command-line options in `palmistry_strict_auto_onefile.py`: `doctor`, `build`, `mask-folder`, `single-mask`, `train`, `predict`. Imports include NumPy, OpenCV and Pillow; other components use PyTorch, pandas, Matplotlib, tqdm, requests and Flask. A tested environment and dependency lockfile are not yet documented.

Xem các lệnh trên trong script chính. Repo chưa có môi trường chạy được kiểm chứng hoặc file khóa phiên bản thư viện; cần chuẩn bị dữ liệu và checkpoint trước khi thử dự đoán.

## Scope / Phạm vi

This is an image-processing experiment. Palm-line labels do not establish scientific predictions about personality, health or the future. No accuracy benchmark is claimed here.

Đây là thử nghiệm xử lý ảnh. Nhãn đường chỉ tay không chứng minh khả năng dự đoán tính cách, sức khỏe hay tương lai. Repo chưa công bố benchmark độ chính xác.

## Work together / Cùng nghiên cứu

Interested in image segmentation, datasets or evaluation? [Email Khôi](mailto:daethphate@gmail.com) or [say hi on Instagram](https://www.instagram.com/koiluuuuv/). Based in HCMC, open to meeting offline.

Bạn muốn cùng tìm hiểu phân đoạn ảnh, dữ liệu hoặc đánh giá mô hình? Mình ở TP.HCM và muốn tìm bạn có thể gặp trực tiếp.
