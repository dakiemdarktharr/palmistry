# Palm Studio: 3 đường và quẹt để lưu

Bấm **Palmistry Live Camera** trên Desktop. App mở cửa sổ riêng; URL kiểm tra là `http://127.0.0.1:8501/keypoints`. Mục tiêu hiện tại là **100 ảnh**. Ảnh gốc, các ảnh ngoài chỉ tiêu và nhãn cũ không bị xóa.

## Dán nhãn

1. Chọn bộ ảnh. App bắt đầu ở ảnh pending đầu tiên.
2. Chọn **Tâm đạo**, **Trí đạo**, **Sinh đạo**. Với chế độ **Vẽ đường**, kéo dọc nếp chỉ tay từ đầu đến cuối; app lấy 6 mốc cách đều theo chiều dài nét vẽ rồi fit B-spline. Có thể dùng **Đặt điểm** để bấm 6 mốc thủ công. Kéo từng mốc để sửa. Kính phóng đại giúp xem chi tiết.
3. Chọn **Bề rộng**, bấm hai mép lòng bàn tay ở cùng mức ngang dưới gốc các ngón, không tính ngón cái.
4. Kéo thanh **Quẹt ở đây để lưu** sang trái nếu tay trái, sang phải nếu tay phải. Nút Tay trái/Tay phải làm cùng thao tác. Chỉ chuyển ảnh sau khi server đã ghi thành công. Quẹt ngắn, dọc hoặc bị hủy không lưu.
5. Có thể để trống đường khó thấy, bề rộng hoặc metadata. Quẹt vẫn lưu nhãn một phần. Trống là **unreviewed**, không được tự coi là **absent**. Điểm chưa hoàn tất được giữ lại nhưng không dùng làm target huấn luyện.
6. **Hoàn tác quẹt** khôi phục bản nhãn trước lần quẹt và mở lại ảnh. **Hoàn tác** hoặc Ctrl+Z hoàn tác sửa điểm. Phím 1–4 chọn công cụ; T/P lưu tay trái/phải (ngoài ô nhập liệu). Phím trái/phải khi focus thanh quẹt cũng lưu bên tay.
7. Nút ảnh trước/tiếp và đổi bộ ảnh lưu nháp trước khi chuyển. Đóng hoặc tải lại khi còn thay đổi có cảnh báo. Lỗi ghi/HTTP 409 giữ bản nháp trên màn hình; dùng Tải bản nháp JSON trước khi khôi phục hoặc tải lại.

Tâm đạo/Trí đạo: phía ngón út về ngón cái. Sinh đạo: gần ngón trỏ về cổ tay. Đây là quy ước thứ tự điểm, không gán tên đường dựa trên trái/phải của ảnh. Quẹt gán bên tay do người dùng xác nhận; không cần thông tin lật gương.

## Công cụ và dữ liệu

Nút **Công cụ** mở drawer gồm tạo bộ 100 ảnh, CSV metadata, huấn luyện, export, pseudo-label và thử nhận diện. Chỉ phần dán nhãn xuất hiện trên màn hình chính.

CSV tối thiểu: `source_path,subject_id`; `source_group` tùy chọn. Ảnh cùng người phải có cùng subject_id ở mọi lần chụp. Đường dẫn source_path tuyệt đối hoặc tương đối từ source_root trong remaining_sources.json. Cột handedness/mirrored cũ vẫn đọc được để tương thích; mirrored không tham gia target bên tay v2.

Huấn luyện chỉ đọc ảnh approved và gộp nhóm người/nguồn/trùng SHA/pHash trước chia train/val/test. Cần ít nhất 3 nhóm độc lập. Không tự đặt subject_id hoặc tách ảnh cùng session để ép train. Metadata trống được lưu nhưng không chứng minh độc lập người. Mẫu hướng dẫn và queue pseudo không được tự chia lại làm tập train gốc.

## Schema v2 và nhãn thiếu

`palm-keypoints-v2` có heart_line, head_line, life_line, mỗi đường tối đa 6 điểm, cùng 2 điểm bề rộng: tổng **20 điểm**, **40 tọa độ**. Annotation giữ revision, status, handedness, subject_id, source_group, notes. handedness left/right được ghi khi quẹt; unknown khi chưa chọn. mirrored là trường tương thích cũ, không cần nhập.

- Đường present có đúng 6 điểm; unreviewed có 0–6 điểm chưa hoàn tất; absent chỉ có khi được xác nhận rõ trong dữ liệu cũ/API, không sinh từ phần bỏ trống.
- Trạng thái approved nghĩa người dùng đã lưu ảnh; không có nghĩa mọi trường đều đủ. Mỗi target có mask riêng. Điểm của đường unreviewed, presence của đường unreviewed, bề rộng chưa đủ 2 điểm và bên tay unknown không đóng góp loss/gradient.
- Evaluate không tính đường chưa có nhãn thành false negative, không đưa ảnh thiếu bề rộng vào sai số chuẩn hóa. Metric không có mẫu trả null kèm số mẫu, không ghi NaN.
- B-spline được xem trước khi chưa có bề rộng; chưa có tỷ lệ độ dài. Khi có bề rộng, tỷ lệ là arc length / bề rộng ảnh, không phải đơn vị cm/mm.
- JSON giữ mọi trạng thái và điểm chưa hoàn tất. YOLO-Pose có shape `[20, 3]`; mốc chưa được gán có visibility 0, các tọa độ hợp lệ có visibility 2. Dùng annotations.json để phân biệt unknown và absent.

Lưu JSON atomic và tăng revision. HTTP 409 ngăn ghi đè sửa đổi từ cửa sổ khác. Quẹt có endpoint riêng `/api/keypoints/swipe/<project>/<image_id>`; bên tay và annotation được ghi trong cùng lần thay thế file.

## Chuyển dữ liệu 4 đường cũ

`python keypoint_pipeline.py migrate-v2 --project artifacts/keypoints/palm_keypoints_main`

Lệnh sao lưu project.json và toàn bộ annotation v1 vào `backups/schema_v1_*/` trước khi ghi v2. Fate được giữ trong legacy_lines và bản sao lưu, không đổi thành một trong 3 đường mới. Tương thích đọc v1 cũng được hỗ trợ trong API. Mô hình v1 có 26 điểm bị từ chối bởi schema/model-version/shape; cần train mô hình v2 từ nhãn người dùng, không sửa số chiều checkpoint cũ để nạp nhầm.

`python keypoint_pipeline.py limit-review --project artifacts/keypoints/palm_keypoints_main --count 100` giảm queue, giữ mọi ảnh approved/rejected, chuyển nguồn dư sang remaining_sources.json và giữ file ảnh/nhãn trên đĩa. Nếu ảnh đã duyệt vượt chỉ tiêu thì dừng. Chỉ migrate/resize sau khi lưu nháp, đóng cửa sổ dán nhãn và không có tác vụ đang chạy.

## Kiến trúc và build giao diện

React + Mantine + React Aria useMove nằm trong frontend/. Giao diện được build sẵn tại static/labeler/, chạy cục bộ trong PySide6, không dùng CDN, không cần Node khi mở app. Token giao diện và bố cục: [LABELER_DESIGN.md](LABELER_DESIGN.md).

Để sửa UI: cài Node 24, chạy `npm ci`, `npm test`, `npm run build` trong frontend/. Commit source, lockfile và static/labeler/. License thư viện được ghi vào THIRD_PARTY_LICENSES.txt. CI build và kiểm tra lại trên Windows/Ubuntu. Installer chỉ cần static đã build, không chép node_modules/cache.

CNN NumPy: RGB letterbox 64×64 → conv3×3 stride2 8 kênh → average pool2 → conv3×3 stride2 16 kênh → dense96 → 40 tọa độ + 3 presence + 1 bên tay. Backprop, masked Huber/BCE, dropout, Adam đều NumPy. Checkpoint allow_pickle=False, SHA/schema/shape/expanded size được kiểm tra.

## Độ tin cậy và kiểm chứng

Fit B-spline không chứng minh điểm nằm đúng nếp tay. Kiểm tra điểm trùng, bounds, hướng/góc, tự cắt, overshoot, độ cong và tích phân; fit không đạt chỉ dùng xem trước. Auto-label vẫn yêu cầu ít nhất 3 mẫu present + 3 absent mỗi đường, precision ≥ .95, recall ≥ .90, PCK@.05 ≥ .90 và lỗi bề rộng ≤ .05. Bên tay cần ít nhất 6 nhãn đã biết, accuracy ≥ .95. Thiếu nhãn/validation không được tự vượt cổng này. Với dữ liệu toàn unknown ở một đường, đường đó chưa được auto-label. Không tuyên bố độ chính xác trên dữ liệu thật khi chưa đo.

`tests/test_keypoints.py` kiểm tra spline, gradient, mask dữ liệu thiếu, migrate giữ bản gốc, export, train với nhãn một phần, split chống trùng và API swipe/revision. `scripts/verify_desktop.py` chạy app thật với fixture riêng: vẽ freehand, bề rộng, lỗi ghi rồi retry, quẹt trái/phải, cancel, hoàn tác, lưu ảnh trống, download và bảo vệ đóng cửa sổ. Fixture không vào dataset của người dùng; ảnh kiểm chứng tại artifacts/desktop-swipe/.
