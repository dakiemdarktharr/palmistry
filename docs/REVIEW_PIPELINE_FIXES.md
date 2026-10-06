# Rà soát và khắc phục pipeline review — 2026-09-12

## Kết quả trên dataset thật

- Run: `pipeline_ui_001_ca643763`, hoàn tất từ 19:33:34 đến 19:40:58 (giờ Việt Nam).
- 24.000 file TIFF → 12.000 ảnh duy nhất; 12.000 bản trùng được bỏ qua khi xử lý, không xóa ảnh gốc.
- Đủ 400 crop + mask + overlay, tất cả `pending`, 0 ảnh lỗi trong đợt tạo queue. 11.600 ảnh còn lại được lập chỉ mục.
- Toàn bộ run mới khoảng 34,8 MiB; không chép thêm hàng chục GiB TIFF.
- Giao diện: http://127.0.0.1:8501/pipeline_review/pipeline_ui_001_ca643763

## Nguyên nhân, cách sửa và fallback

| Vấn đề | Cách sửa đã thực hiện | Fallback / cách phục hồi |
|---|---|---|
| Queue báo quá 20.000 ảnh | Tách giới hạn entry archive khỏi tổng thư mục dataset; lọc trùng theo SHA-256 trước preprocessing | Giảm số mask review trên UI để chuẩn bị đợt nhỏ; log kiểm kê cho biết ảnh bị bỏ qua |
| Dataset bị nhập trùng, ổ D gần đầy | Pipeline tham chiếu ảnh gốc và chỉ tạo 400 crop/mask/overlay cần duyệt | Giữ nguyên đường dẫn ảnh gốc; nếu thiếu chỗ thì giải phóng dung lượng hoặc chuyển cấu hình dataset trước khi chạy run mới. Chưa xóa bản trùng trên đĩa và chưa ngăn việc người dùng nhập lại archive |
| Chờ lâu không biết tiến độ | Bật log subprocess không đệm; tiến độ kiểm kê mỗi 500 file, mask mỗi 10 ảnh, bootstrap mỗi 50 ứng viên | Xem log và thông báo lỗi trong UI; nếu job bị ngắt, chạy lại bằng tên run mới. Lần đầu vẫn phải đọc toàn bộ file để xác định trùng chính xác |
| Lọc chất lượng/chia tập làm queue thất bại trước khi được duyệt | Nhánh review tạo ứng viên không phụ thuộc ngưỡng auto-accept; giữ cảnh báo chất lượng và `unassigned`, không tự approved | Sửa mask bằng công cụ annotation hoặc đánh rejected; ảnh hỏng được ghi vào `review_failures.csv` |
| Chỉ hai nhóm nguồn, chưa đủ train/val/test | Kiểm tra trước khi train và báo thiếu metadata/nhóm độc lập; giữ nguyên queue | Bổ sung metadata người/bàn tay/nhóm nguồn có bằng chứng vào manifest trước bootstrap. Không đoán ID theo tên file, không ép chia ảnh cùng nguồn. Nếu chưa có metadata, tiếp tục review và hoãn train |
| Audit giới hạn 10.000, so sánh pHash chậm | Bỏ giới hạn prototype; dùng 7 dải hash để tìm ứng viên, vẫn kiểm khoảng cách Hamming ≤6 và hợp nhóm bắc cầu | Nếu nhiều ảnh quá giống khiến nhóm chỉ còn 1–2, bổ sung dữ liệu độc lập; không bỏ kiểm tra leakage |
| Retry trùng tên run; mất queue khi mở lại | UI cấp tên run mới khi tên đã có dữ liệu; lưu trạng thái job nguyên tử và khôi phục sau restart | Run cũ được giữ trong artifacts; job bị ngắt được báo rõ, chọn chạy lại. Đây là restart từ đầu, chưa phải resume giữa chừng |
| Nhiều request ảnh khiến trang duyệt chậm/429 | Phân trang 40 mask, lazy-load; request ảnh artifact không tiêu thụ quota API thông thường | Đi từng trang và mở từng ảnh lớn; các API ghi vẫn giữ giới hạn và CSRF |
| Mask trông đen dù mã lớp đúng | Endpoint xem màu với palette và chú giải, bấm ảnh để xem lớn | PNG semantic 0..5 không bị thay đổi; có thể tải file gốc để sửa bằng công cụ annotation |
| Lưu review đồng thời có thể mất dòng thay đổi | Khóa cả chuỗi đọc–sửa–ghi và thay file nguyên tử | Mỗi dòng có kết quả lưu; nếu lỗi, thử lưu lại sau khi kiểm tra dung lượng/quyền. Cùng một dòng thì lần lưu cuối thắng |
| Rejected/pending lọt vào pseudo-label; chia lại held-out sau train seed | Loại các dòng chưa approved khỏi ứng viên, chỉ thêm pseudo-label vào train; giữ cố định val/test và chặn nguồn/người/hash trùng | Mẫu không đạt confidence/line-ratio nằm trong `bootstrap_skipped.csv`; nếu audit phát hiện leakage thì dừng, sửa nhóm dữ liệu trước khi thử run mới |
| Bootstrap lỗi thiếu ảnh/custom review CSV/ghi file | Thông báo ảnh không đọc được, tôn trọng đường dẫn review tùy chọn, kiểm tra ghi mask/crop/overlay; ghi summary lỗi và fallback | Khôi phục file bị thiếu, sửa đường dẫn/quyền/dung lượng rồi dùng tên bootstrap mới; không thay mask đã duyệt |
| RAR giải nén theo từng entry chậm, đường dẫn bất thường, backend treo | Batch backend; kiểm tra tất cả entry trước giải nén; temp cùng ổ dataset; rollback file của lần nhập thất bại; timeout 30 phút | Với RAR không được hỗ trợ/hỏng/mã hóa hoặc quá timeout, giải nén có kiểm soát rồi nén các ảnh thành ZIP và nhập lại; xem report/log lỗi |
| Hết chỗ/không có quyền ghi | Kiểm tra dung lượng khi tạo queue; HTTP JSON lỗi storage rõ ràng; installer dừng khi pip thất bại | Giải phóng dung lượng/cấp quyền thư mục, kiểm tra Python và mạng cài dependency rồi chạy lại |

## Kiểm chứng

`scripts/verify.py --output artifacts/verification-review-recovery-final` kiểm tra compile,
imports, HTTP loopback, dependency, toàn bộ unittest, preprocessing, train NumPy,
predict/evaluate, UI subprocess và đường dẫn ảnh lỗi. Bài bootstrap dùng fixture tổng
hợp thực sự train seed → giữ split → train final → test; không approved hay train trên
400 mask thật. Kết quả cuối: 69 bài, 64 pass và 5 skip do thuộc nhánh PyTorch cũ; dự án không dùng PyTorch.

Bằng chứng lưu trong `artifacts/verification-review-recovery-final/`: `evidence.json`,
`tests.log`, `real-queue.json`, `browser.json`, `review-browser.png`. Kiểm thử trình duyệt
kiểm tra khôi phục tên run, link review, trang 1 và 10, ảnh tải thành công, không có lỗi
JavaScript/HTTP. Kiểm thử HTTP riêng đọc hơn 300 ảnh liên tục và kiểm tra API vẫn đáp ứng.

## Giới hạn còn phải xử lý bằng dữ liệu thật

Mask tự trích hiện vẫn là gợi ý; quan sát trực tiếp có mask không bám đúng các đường.
Chưa chứng minh độ chính xác trace/độ dài hoặc chất lượng nhận diện thực tế. Người dùng
cần sửa hoặc từ chối mask sai. UI hiện lưu trạng thái và ghi chú; sửa hình học mask cần
công cụ annotation bên ngoài. Dataset chỉ có session1/session2, chưa đủ metadata để
chứng minh chia tập độc lập. Không bảo đảm phần mềm không bao giờ còn lỗi; các kết quả
ở đây là phạm vi đã chạy và có bằng chứng, không phải xác nhận mọi thiết bị/archive.
