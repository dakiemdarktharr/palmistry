# Thẻ dán nhãn 3 đường

Một thẻ ảnh ở giữa màn hình. Ba công cụ đặt điểm và công cụ bề rộng ở cạnh ảnh; thanh quẹt ở chân thẻ. Quẹt sang trái nghĩa là tay trái, sang phải nghĩa là tay phải; cả hai đều lưu dữ liệu. Vùng ảnh dành riêng cho vẽ và kéo điểm, vùng chân thẻ dành riêng cho quẹt. Nút tay trái/phải cung cấp thao tác tương đương bằng bàn phím.

Màu nền #eef1f7, giấy #ffffff, chữ #202835, tâm đạo #d54c75, trí đạo #15869c, sinh đạo #a57612, hành động #6650ab. Chữ Segoe UI dùng sẵn trên Windows, không tải font ngoài. Màu đường gắn với công cụ và đường cong tương ứng; chuyển động chỉ phản hồi thao tác quẹt/lưu, tôn trọng reduced-motion.

Mantine cung cấp nút, select, progress, drawer, modal, tooltip và trạng thái tải. React Aria useMove chuẩn hóa chuột/cảm ứng cho thanh quẹt. Các thư viện được đóng gói cục bộ, app không cần truy cập CDN hoặc chạy Node khi mở. Tài liệu: https://mantine.dev/getting-started/ và https://react-aria.adobe.com/useMove.

Dữ kiện chưa nhập giữ trạng thái unreviewed hoặc mảng rỗng, không chuyển thành absent. Nhãn một phần vẫn được lưu khi quẹt. Mô hình chỉ tính loss trên những trường có nhãn; nhận diện và đo độ dài giữ kiểm định hình học/validation. Schema v2 chỉ có tâm đạo, trí đạo, sinh đạo (20 điểm kể cả bề rộng); mô hình v1 bốn đường không được nạp nhầm.
