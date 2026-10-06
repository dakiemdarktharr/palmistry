# Pipeline keypoint Heart / Head / Life / Fate

Bấm shortcut **Palmistry Live Camera** trên Desktop để mở cửa sổ ứng dụng. `http://127.0.0.1:8501/keypoints` vẫn dùng được khi cần kiểm tra qua browser. Đây là pipeline mới; màn hình mask/camera cũ được giữ dưới nhãn legacy. Không chuyển nhãn Simian hoặc minor_unknown sang Fate.

## Các bước trên giao diện

1. Chọn queue nguồn và tạo project keypoint. Bản cài cục bộ hiện có queue `palm_keypoints_main` với 400 ảnh chờ duyệt; bản clone GitHub không chứa dữ liệu cá nhân này. Chỉ thêm JPEG xem trước tối đa 1024 px; không sao chép TIFF 16+ GB lần nữa.
2. Đặt 6 điểm theo thứ tự trên từng đường, kéo để sửa, xem B-spline và độ dài. Đường không nhìn thấy phải đánh dấu **Không nhìn thấy**; bỏ trống không có nghĩa absent. Dùng Lưu nháp khi chưa hoàn thành. Duyệt ảnh kiểm tra đầy đủ hình học, bên tay, gương và hai điểm chuẩn.
3. Gán 2 điểm chuẩn bề rộng lòng bàn tay ở cùng mức ngang dưới gốc các ngón, không tính ngón cái. Sử dụng cùng quy ước trên mọi ảnh. Độ dài chuẩn hóa phụ thuộc độ nhất quán của quy ước này.
4. Nhập metadata thật bằng CSV hoặc nhập trên từng ảnh. Cùng người phải có cùng `subject_id` ở mọi session. Các cột CSV: `source_path,subject_id,mirrored,source_group,handedness`. `source_path` là đường dẫn tuyệt đối hoặc tương đối từ `source_root` trong remaining_sources.json. `mirrored` = yes/no/unknown, `handedness` = left/right/unknown; nhóm nguồn tùy chọn. Thay đổi bên tay/gương của ảnh đã duyệt đưa ảnh về pending.
5. Bấm Chia tập và train. Chỉ annotation approved được dùng. Gộp người/nguồn/trùng hash/pHash trước chia train/val/test. Nếu không có ít nhất 3 nhóm độc lập thì dừng với thông báo, không tách ảnh trùng để ép train. Trường hợp không biết người, nguồn chỉ là fallback bảo thủ, không chứng minh độc lập người.
6. Bấm Tạo pseudo-label. Chọn 100 ảnh để chạy kiểm tra trước hoặc 0 để xử lý tất cả. Kết quả nằm trong `auto_labels.jsonl` và `manual_review.jsonl`. Thiếu metadata người trước hết được đưa vào manual review, tránh đưa người trong test vào train. Bổ sung CSV rồi chạy lại bằng run mới.
7. Xem báo cáo → chọn run pseudo → Mở 400 đề xuất trong queue mới. Offset 0, 400, 800... chọn các batch tiếp theo. Queue đề xuất luôn pending. Nguồn không đọc được được ghi vào `unreadable_sources.json`. Queue này chỉ dùng để duyệt/xuất JSON, không tự chia lại tập để làm rò rỉ held-out. Huấn luyện vòng đầu chỉ dùng nhãn người duyệt; nhãn pseudo chưa được tự hợp nhất vào train.
8. Nhận diện ảnh dùng checkpoint keypoint gần nhất của project. Đường vượt kiểm định có độ dài; các dự đoán cần kiểm tra hiển thị nét đứt và lý do. Ảnh upload không được ghi thành dataset tự động.

## JSON annotation

Schema `palm-keypoints-v1`, tọa độ `[x,y]` chuẩn hóa trong [0,1] theo ảnh xem trước đã xử lý EXIF. Mỗi ảnh gồm:

- `image_id`, `revision`, `status`: pending/approved/rejected.
- `handedness`, `mirrored`, `subject_id`, `source_group` và ghi chú.
- `palm_width_points`: 2 điểm chuẩn bề rộng.
- `lines`: chính xác heart_line, head_line, life_line, fate_line. Mỗi đường có `status` unreviewed/present/absent và `points` gồm 6 điểm nếu present, rỗng nếu absent.

Thứ tự: Heart/Head từ phía ngón út về ngón cái; Life từ gần ngón trỏ về cổ tay; Fate từ cổ tay về ngón giữa. Đây là quy ước annotation, không dùng vị trí trái/phải của ảnh để gán tên đường. Không tự đảo nhãn khi lật gương. Nhãn train cho bên tay học phía tay biểu kiến; inference chỉ đổi thành bên tay thật khi biết trạng thái gương và qua kiểm định.

Lưu atomic, tăng revision và từ chối HTTP 409 nếu tab khác đã sửa. Trong khi train/export/pseudo đang đọc project, giao diện khóa ghi annotation. Manifest train lưu snapshot nhãn, nhóm, split và SHA-256 để đối chiếu. Xuất YOLO gồm một đối tượng palm, 26 điểm (24 điểm đường + 2 chuẩn), visibility 0/2, cùng annotations.json và split.json. Không cài Ultralytics.

## Kiến trúc

- `palm_keypoints/data.py`: JSON, validation, chuyển queue bằng cách nối review.csv với labels_pseudo_strict.csv khi cần, chống trùng/chia tập, export và queue đề xuất.
- `palm_keypoints/model.py`: CNN nhỏ NumPy từ scratch. RGB letterbox 64×64 → conv3×3 stride2 8 kênh → average pool2 → conv3×3 stride2 16 kênh → dense96 → 52 tọa độ + 4 presence + 1 phía tay. Backprop conv/dense, masked Huber, BCE, dropout và Adam đều NumPy. Không PyTorch/scikit-learn/backbone tải sẵn.
- `palm_keypoints/spline.py`: spline mở cubic bằng SciPy splprep/splev, mặc định s=0 đi qua 6 mốc. Một dãy tham số duy nhất tạo 257 điểm hiển thị; tích phân chuẩn đạo hàm spline để tính arc length. Không dùng skeleton hay morphology để tạo nhãn/đường cuối.
- `palm_keypoints/workflow.py`: train, early stopping theo val, đo test sau chọn model, inference và pseudo QC. Checkpoint NPZ allow_pickle=False, shape/schema/SHA-256 được kiểm tra.
- `palm_keypoints/web.py`, `templates/keypoints.html`: canvas annotation, upload CSV/ảnh, tác vụ nền và báo cáo. Tác vụ chạy ngoài request HTTP; log UTF-8 rõ lỗi trên Windows.
- `keypoint_pipeline.py`: CLI tương đương để tái lập: from-review, train, predict, pseudo, export-yolo, review-pseudo.

## Độ tin cậy và fallback

B-spline cho một đường liên tục theo cấu trúc; không chứng minh keypoint nằm đúng nếp chỉ tay. Kiểm tra điểm trùng/thiếu, nằm ngoài ảnh, đảo thứ tự/góc gấp, tự cắt, spline vượt biên, độ cong chuẩn hóa, tỷ lệ độ dài và sai số tích phân. Nếu fit không hợp lệ chỉ trả polyline xem trước, không tự nhận là đường đạt chuẩn. Sai số tích phân số không bao gồm sai số nhãn, phối cảnh hay độ phân giải ảnh; đây không phải chiều dài vật lý cm/mm.

Confidence là thứ hạng dựa trên presence × độ ổn định MC-dropout, **không phải xác suất đã hiệu chuẩn**. Auto-label còn yêu cầu bằng chứng validation: ít nhất 3 ảnh present + 3 absent mỗi đường, precision ≥ .95, recall ≥ .90, PCK@.05 bề rộng ≥ .90; chuẩn bề rộng lỗi trung bình ≤ .05; bên tay có đủ mẫu và accuracy ≥ .95. Các ngưỡng là chính sách ban đầu cần được kiểm chứng trên dữ liệu thật, không bảo đảm độ chính xác trên ảnh mới. Nếu thiếu ví dụ absent hoặc chưa đạt ngưỡng, hệ thống chuyển sang review thay vì giả vờ đã có model tốt.

CNN 64×64 là baseline nhẹ có thể huấn luyện từ scratch; khả năng tổng quát hóa và độ chính xác chỉ tay thật chưa được đo khi 400 ảnh chưa có keypoint người duyệt. Các checkpoint pixel segmentation cũ không chuyển trực tiếp sang CNN này. Không tuyên bố model đã train trên 400 ảnh thật.

Lỗi tạo queue: giữ nguyên ảnh gốc và ghi creation_error.json, sửa nguồn/định dạng rồi dùng tên project mới. Lỗi train: giữ best.npz đã lưu nếu có, xem history.json/job.log và chạy lại vào run mới. Pseudo giữ tách biệt file đạt/không đạt; model thiếu validation hoặc metadata không tự vượt cổng kiểm tra. Các nhãn cũ và checkpoint cũ được giữ riêng.

## Kiểm chứng

`tests/test_keypoints.py` kiểm tra cung tròn/đường thẳng, chuẩn hóa theo kích thước, keypoint thiếu/trùng/cắt, gradient sai phân cho hai conv và hai dense, loss giảm, NPZ round-trip, mask loss cho absent, mirror, schema Fate khác Simian, mapping queue legacy, split chống trùng, train→evaluate→pseudo, API revision/metadata/export và queue pending.

Bằng chứng chạy tổng thể: `artifacts/verification-keypoints-final/`. Kiểm thử browser: `artifacts/keypoints-browser-evidence.json`. Fixture tổng hợp nằm ở project có tiền tố `zz_fixture_`; không phải dữ liệu/độ chính xác thực tế của 400 ảnh. Không push GitHub.

Tài liệu API spline chính thức: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html và https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splev.html. SciPy đánh dấu các hàm này legacy nhưng vẫn hỗ trợ; chúng được cô lập trong một module theo yêu cầu, có thể thay bằng BSpline sau khi đối chiếu hồi quy.

Thiếu metadata người được phát hiện trước khi hash/decode ảnh nguồn, tránh quét lại hàng chục GB chỉ để trả lỗi metadata. Kiểm thử xác nhận nhánh này không gọi bộ giải mã ảnh.
