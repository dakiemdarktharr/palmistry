# Palmistry và phép đo bốn đường bàn tay

## Kết luận nghiên cứu

Có thể xây dựng một hệ thống CV đo các đặc điểm nhìn thấy được: vị trí đường, trace,
chiều dài tương đối, kiểu nếp ngang và bên tay khi điều kiện chụp cho phép. Các phép đo
này không tự cung cấp một thang đo tính cách. Trong những nguồn đã đối chiếu, chưa xác
lập được một công thức đã kiểm chứng dùng riêng bốn tỷ lệ head, heart, life, simian để
kết luận tính cách của một cá nhân. Vì vậy bản triển khai chỉ xuất kết quả hình học;
không có bảng quy đổi hình dạng cơ thể thành phẩm chất cá nhân.

## Ý nghĩa và giới hạn của các tên đường

Head, heart và life là các tên truyền thống của palmistry. Để annotation nhất quán,
cần gắn mỗi tên với một mô tả hình học và ví dụ mask đã duyệt, thay vì để người gán
nhãn tự suy diễn theo các trường phái khác nhau. Một tên nghe giống cơ quan hay khả
năng tâm lý không biến đường đó thành phép đo cơ quan hoặc khả năng ấy.

NCBI/HPO định nghĩa single transverse palmar crease là trường hợp các nếp ngang xa và
gần hợp thành một nếp ngang. Simian là tên đồng nghĩa được dùng trong một số tài liệu.
Đây là lý do về hình thái để không ép mọi bàn tay phải có đồng thời bốn đường độc lập.
Bản triển khai chỉ dùng định nghĩa hình học; không diễn giải y khoa từ sự hiện diện
của nếp này.[1]

| Nhãn mục tiêu | Quy ước annotation của ứng dụng |
|---|---|
| Life | Trace quanh vùng gốc ngón cái; phải xác định được phía ngón cái hoặc kiểm duyệt thủ công |
| Head | Nếp ngang giữa lòng bàn tay theo bộ ví dụ được thống nhất khi gán nhãn |
| Heart | Nếp ngang phía dưới các ngón theo cùng bộ ví dụ |
| Simian | Một nếp ngang đơn hợp nhất; không tự suy ra chỉ vì head hoặc heart bị bỏ sót |

Các mô tả trong bảng là hợp đồng annotation đề xuất cho ứng dụng, không phải chứng
nhận rằng bộ lọc ảnh hiện tại luôn phân biệt đúng từng cấu trúc. Đường đứt, nhánh nối,
ảnh nghiêng hoặc không thấy hết lòng bàn tay phải được đưa vào nhóm chưa chắc chắn.

## Đối chiếu quan niệm truyền thống

Cheiro trình bày head như một chỉ dấu về hoạt động tinh thần, heart về tình cảm và
life gắn với sức sống hoặc tuổi thọ. Các diễn giải còn phụ thuộc vị trí xuất phát,
độ cong, sự nối giữa đường và nhiều dấu phụ. Độ dài đơn thuần không bao quát ngay cả
hệ thống quan niệm của tác giả. Sách còn có phần head và heart nhập lại với nhau.
Đây là nguồn lịch sử sơ cấp để biết palmistry từng tuyên bố gì, không phải nghiên
cứu kiểm chứng các tuyên bố đó.[2]

Louis Williams, trong *Key to Palmistry* (1902), dùng cách gọi và định vị heart/thought
khác với một số tác giả khác; tác giả cũng kết hợp bề rộng, độ sâu, hướng và dấu cắt.
Điểm khác biệt này làm cho việc trộn các bảng diễn giải từ nhiều sách dễ sinh mâu
thuẫn. Các quan niệm về vai trò tay trái/phải trong sách là tuyên bố của trường phái,
không được dùng làm thuật toán xác định bên tay hoặc tính cách trong ứng dụng.[3]

Hai hệ thống trên không cung cấp, trong các phần đã khảo sát, một ngưỡng thống nhất
kiểu “chiều dài / đường kính lòng bàn tay = x” với một thang tính cách và sai số đo
được kiểm chứng. Tự chọn ngưỡng rồi viết kết luận chắc chắn sẽ bổ sung một quy tắc
không được nguồn hỗ trợ. Bản nghiên cứu không tạo các ngưỡng kiểu đó.

## Các nghiên cứu có vẻ liên quan nhưng cần đọc đúng phạm vi

Prasad và Chai công bố một hệ thống ảnh bàn tay trong *The Computer Journal*. Abstract
nêu bốn giai đoạn xử lý ảnh và báo các con số độ chính xác cho phát hiện đường cùng
phần phân tích hành vi. Chỉ phần abstract và thông tin xuất bản được truy cập ở đây;
không coi đó là đã kiểm tra toàn bộ thiết kế nghiên cứu, nguồn nhãn tâm lý hay phép
kiểm định độc lập. Các con số do tác giả báo cáo không được chuyển thành độ chính xác
của Palmistry hoặc thành bằng chứng cho công thức bốn tỷ lệ đang xét.[4]

Venurkar và cộng sự nghiên cứu 200 sinh viên, dùng mẫu vân tay và bảng hỏi NERIS; thiết
kế là cắt ngang. Dữ liệu này không phải độ dài bốn đường lòng bàn tay. Việc hơn 90%
người tham gia hài lòng với kết quả bảng hỏi không đồng nghĩa một máy đọc chỉ tay dự
đoán đúng tính cách ở mức đó. Do khác đặc trưng đầu vào, nghiên cứu không xác lập
quy tắc chuyển bốn chiều dài thành kết luận cá nhân cho ứng dụng.[5]

Lucas, Dhugga và Henneberg đo đường life và bàn tay trên 60 thi thể, không tìm thấy
tương quan có ý nghĩa giữa độ dài đường và tuổi thọ. Đây là kết quả về tuổi thọ,
không phải phép kiểm tra trực tiếp tính cách. Nghiên cứu minh họa vì sao một tên đường
và một tuyên bố trong sách không đủ để tạo chức năng dự báo.[6]

Pham Van và cộng sự nghiên cứu segmentation đường bàn tay bằng U-Net và mô-đun ngữ
cảnh, có bộ dữ liệu được gán nhãn cho bài toán đó. Bài này hỗ trợ việc tách đánh giá
segmentation khỏi diễn giải palmistry. Không đưa kiến trúc hay điểm số của bài báo
vào mô hình NumPy hiện tại; phép đo trên một dataset không phải chất lượng đã chứng
minh trên ảnh của ứng dụng.[7]

Các nguồn được chọn gồm định nghĩa hình thái, sách palmistry sơ cấp, bài nghiên cứu
CV và các công bố có vẻ ủng hộ liên hệ tâm lý. Việc đối chiếu không chỉ chọn các nguồn
cùng quan điểm. Giới hạn truy cập full text được nêu riêng thay vì điền suy đoán về
phương pháp hoặc số liệu chưa đọc được.

## Định nghĩa phép đo để có thể kiểm chứng

Bản v2 đo trong hệ tọa độ ảnh đã crop. Semantic mask được làm mảnh thành skeleton;
sau đó dựng đồ thị pixel có trọng số: cạnh ngang/dọc dài 1 pixel, cạnh chéo dài căn 2.
Không cộng tất cả pixel ở mọi nhánh. Độ dài được lấy trên đường chính giữa các đầu
mút của thành phần liên thông lớn nhất; các đường quá phân nhánh, quá phân mảnh hoặc
vòng kín không rõ đầu mút được trả chưa xác định.

Mẫu số chuẩn hóa là đường kính vòng tròn nội tiếp lớn nhất trong vùng bàn tay đã
segment. Đây là một đại lượng thay thế bề rộng lòng bàn tay trong ảnh 2D, không phải
bề rộng giải phẫu đo bằng thước. Công thức triển khai là:

`length_to_palm_ratio = principal_trace_length_px / largest_inscribed_palm_diameter_px`

Việc chia cho đại lượng thuộc bàn tay giúp hạn chế phụ thuộc padding và khoảng cách
chụp, khi ảnh được scale đồng đều. Nó không khắc phục phối cảnh nghiêng, bàn tay co,
mặt lưng, che ngón cái hoặc vùng segment sai. Những trường hợp đó cần chụp lại hoặc
annotation tốt hơn. Tỷ lệ có thể lớn hơn 1 nếu đường cong dài hơn mẫu số; không được
ép nó thành xác suất hay phần trăm tính cách.

Một đường cần có trạng thái phân biệt: đo được, chưa rõ, không thấy tín hiệu hoặc
xung đột nhãn. “Không nhận diện được” không phải bằng chứng đường thực sự không tồn
tại. Giao diện kết quả bỏ các đường chưa được chấp nhận và không điền chiều dài 0
để rồi dùng nó như dữ liệu đã đo.

## Một bàn tay, bên tay và ảnh lật gương

Nhận diện được thực hiện theo hình dáng bàn tay mở, ngón hướng lên và phía ngón cái.
Nếu không nhận rõ ngón cái, không đủ dấu hiệu tư thế hoặc có nhiều vùng bàn tay,
thuật toán trả `unknown`, thay cho mặc định tay trái. Kết quả này là ước lượng CV,
chưa được hiệu chỉnh thành xác suất trên dữ liệu bàn tay thật.

Với một ảnh đơn không có thông tin nguồn, ảnh lật gương có thể đảo hình dáng hai bên.
Giao diện cho chọn “không lật”, “có lật” hoặc “chưa biết”; không đoán trạng thái này.
Camera nội bộ chuyển cấu hình flip hiện có thành metadata cho tiến trình inference.
Điều đó vẫn chưa xác minh liệu driver hoặc thiết bị đã lật ảnh trước khi giao cho app.

Trái/phải không đồng nghĩa thuận/không thuận. Ứng dụng không tự suy tay thuận, giới
tính hay bất kỳ thuộc tính tính cách nào từ bên tay hoặc hình học đường.

## Schema, dữ liệu và model

Schema `palm-lines-v2` giữ nguyên 0..5 và thêm `6 = simian_line`. Các lớp nền, vùng bàn
tay và đường phụ vẫn cần cho segmentation; kết quả đường mục tiêu chỉ gồm bốn tên.
Class 5 vẫn là đường phụ/chưa rõ, tuyệt đối không được đổi tên thành simian.

Checkpoint sáu lớp v1 không tự nhận biết lớp thứ bảy. Loader từ chối checkpoint không
khớp schema. Model NumPy mới lưu số pixel đã học của từng lớp và không dự đoán lớp
không có mẫu học; điều này tránh tạo simian giả chỉ vì đã thêm một tên vào class map.

Queue v1 có thể nâng cấp bằng nút tạo bản sao v2. Bản gốc và ghi chú cũ được giữ, bản
sao cần duyệt lại trạng thái simian. Không tự thêm pixel simian và không tự approved.
400 mask mới hoặc nâng cấp vẫn cần các nhãn thật phù hợp trước khi train. Chỉ có hai
nhóm session không đủ chứng minh chia train/val/test độc lập; metadata người/bàn tay
và nhóm nguồn vẫn phải được bổ sung có bằng chứng.

CV cổ điển hiện không tự xác nhận simian từ sự thiếu head/heart. Chỉ mask có annotation
simian hoặc model thực sự đã học lớp này mới cung cấp ứng viên simian. Khi kết quả
đồng thời báo simian và head/heart riêng, hệ thống trả xung đột để kiểm tra thay vì
cộng trùng cùng một đường. Đây là cơ chế thận trọng; không bao quát mọi biến thể của
nếp ngang đơn/phụ ngoài bộ dữ liệu đã duyệt.

## Cách kiểm chứng tiếp theo

Bộ test hình học kiểm tra đường ngang, chéo, nhánh, đoạn rời, padding, scale và lật
gương trên hình tổng hợp. Test dữ liệu kiểm tra schema, class chưa học và bảo toàn
queue khi nâng cấp. Đây là kiểm chứng cơ học phần mềm, không phải accuracy thực tế.

Để đánh giá CV trên người thật cần tập test có nhãn bên tay, trạng thái lật gương,
mask bốn lớp và trace độc lập. Cần báo riêng tỷ lệ nhận đúng bên tay, tỷ lệ abstain,
precision/recall của từng đường, lỗi chiều dài tương đối và lỗi trùng simian/head/
heart. Tập test phải tách người với tập train; các ảnh cùng một người hoặc gần trùng
không được phân tán qua các tập.

Nếu chưa có đủ nhãn simian thì chức năng phải tiếp tục thể hiện “chưa hỗ trợ bằng
model đã học” thay vì dùng đường phụ làm mẫu thay thế. Nếu silhouette sai, không thể
sửa phép đo chỉ bằng đổi hệ số tỷ lệ. Nếu không biết ảnh lật gương, xác nhận nguồn
ảnh hoặc chụp lại là fallback đúng. Không chuyển các lỗi CV này thành nhận định về
người trong ảnh.

## Nguồn

1. NCBI MedGen / Human Phenotype Ontology. [Single transverse palmar crease](https://ncbi.nlm.nih.gov/medgen/96108). Định nghĩa hình thái; truy cập 12-09-2026.
2. Cheiro. [Palmistry for All](https://www.gutenberg.org/cache/epub/20480/pg20480-images.html). Các chương II, III, VII; bản số hóa Project Gutenberg. Nguồn lịch sử về quan niệm truyền thống.
3. Louis Williams. [Key to Palmistry](https://www.gutenberg.org/cache/epub/79203/pg79203-images.html). International Institute of Science, 1902; bản số hóa 2026. Các phần Life, Heart, Thought và Symbols.
4. Shitala Prasad, Tingting Chai. [Palmprint for Individual’s Personality Behavior Analysis](https://academic.oup.com/comjnl/article-abstract/65/2/355/5860494). Online 2020; The Computer Journal 65(2), 2022, 355–370. Chỉ truy cập abstract/thông tin xuất bản.
5. Shreya Venurkar và cộng sự. [Decoding Human Personality Through Dermatoglyphics](https://pmc.ncbi.nlm.nih.gov/articles/PMC9678115/). Cureus, 2022, 14(10):e30445. Nghiên cứu vân tay và bảng hỏi, không phải bốn chiều dài chỉ tay.
6. Teghan Lucas, Amrita Dhugga, Maciej Henneberg. [Predicting longevity from the line of life: Is it accurate?](https://researchnow.flinders.edu.au/en/publications/predicting-longevity-from-the-line-of-life-is-it-accurate/). Bản ghi nghiên cứu tại Flinders; dùng abstract, không suy rộng kết quả sang đánh giá tính cách.
7. Toan Pham Van và cộng sự. [Efficient Palm-Line Segmentation with U-Net Context Fusion Module](https://arxiv.org/abs/2102.12127). ACOMP 2020; arXiv 2021. Nghiên cứu segmentation.


## Bằng chứng chạy phiên bản hiện tại

Bộ verify cuối có 83 bài: 78 pass, 5 skip thuộc nhánh PyTorch cũ không áp dụng.
Train, inference, evaluate NumPy và HTTP đều chạy thành công trên fixture; test lớp
simian xác nhận có thể học class 6 khi có pixel đã gán nhãn và không dự đoán class
này khi chưa có mẫu. Đây không phải mô hình simian đã được chứng minh trên ảnh thật.

Chrome kiểm tra nhận ảnh TIFF, trạng thái chụp lại và sao chép queue v2 không đổi CSV
gốc. Queue v2 có 400 pending, 0 approved. Không có lỗi JavaScript trong bước đã thử.
Bằng chứng: `verification-four-lines-v2.json` và `artifacts/verification-four-lines-v2/`.
