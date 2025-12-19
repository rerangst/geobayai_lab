# Chương 3: Đặc điểm Bài toán Ship Detection

## 4.1. Giới thiệu về Ship Detection

Ship Detection từ ảnh vệ tinh là bài toán phát hiện và định vị các tàu biển trên đại dương, vùng ven biển, và trong các cảng từ dữ liệu viễn thám. Đây là một trong những ứng dụng quan trọng nhất của deep learning trong lĩnh vực giám sát hàng hải, với nhiều ứng dụng thực tiễn từ an ninh quốc phòng đến bảo vệ môi trường và quản lý nghề cá.

Bài toán ship detection có thể được formulate theo nhiều cách tùy thuộc vào yêu cầu ứng dụng. Dạng cơ bản nhất là binary detection: xác định có hay không có tàu trong một vùng ảnh, thường được sử dụng trong các hệ thống cảnh báo sớm. Dạng phổ biến hơn là object detection với bounding box: định vị vị trí từng tàu bằng bounding box (horizontal hoặc oriented) kèm confidence score. Dạng nâng cao hơn bao gồm cả classification: không chỉ phát hiện mà còn phân loại loại tàu (fishing vessel, cargo ship, tanker, etc.). Dạng chi tiết nhất là instance segmentation: tạo mask pixel-level cho từng tàu, cho phép phân tích kích thước và hình dạng chính xác.

Việt Nam với đường bờ biển dài hơn 3,260 km và vùng đặc quyền kinh tế rộng lớn có nhu cầu cấp thiết về hệ thống giám sát hàng hải. Ship detection từ vệ tinh cung cấp khả năng giám sát diện rộng, liên tục, và độc lập - bổ sung quan trọng cho các phương tiện giám sát truyền thống như radar ven biển, tàu tuần tra, và máy bay.

## 4.2. Tầm quan trọng và Ứng dụng

### 4.2.1. Giám sát An ninh Hàng hải

Ship detection đóng vai trò then chốt trong việc đảm bảo an ninh vùng biển quốc gia. Hệ thống có thể phát hiện các tàu lạ xâm nhập vùng biển, theo dõi hoạt động của tàu nước ngoài, và hỗ trợ lực lượng chức năng trong việc tuần tra và kiểm soát. Khả năng giám sát 24/7 của ảnh SAR đặc biệt quan trọng cho các hoạt động đêm và trong điều kiện thời tiết xấu.

Việc phát hiện "dark vessels" - các tàu cố tình tắt hệ thống AIS (Automatic Identification System) để tránh bị theo dõi - là một ứng dụng quan trọng. Bằng cách so sánh detections từ vệ tinh với dữ liệu AIS, có thể xác định các tàu không tuân thủ quy định khai báo, thường liên quan đến hoạt động bất hợp pháp như buôn lậu, đánh bắt trái phép, hoặc vi phạm lãnh hải.

### 4.2.2. Chống Đánh bắt Bất hợp pháp (IUU Fishing)

IUU (Illegal, Unreported, and Unregulated) fishing là vấn đề nghiêm trọng đe dọa nguồn lợi thủy sản và sinh kế của ngư dân. Ước tính IUU fishing chiếm 20-30% tổng sản lượng đánh bắt toàn cầu, với thiệt hại hàng tỷ USD mỗi năm. Ship detection từ vệ tinh, kết hợp với phân loại tàu đánh cá và phân tích hành vi, là công cụ quan trọng để phát hiện và ngăn chặn IUU fishing.

Cuộc thi xView3 được tổ chức đặc biệt để thúc đẩy nghiên cứu về phát hiện dark fishing vessels. Dataset xView3-SAR với hơn 243,000 maritime objects đã trở thành benchmark quan trọng cho bài toán này. Các giải pháp top sử dụng deep learning đã đạt được hiệu suất đáng kể trong việc phát hiện tàu đánh cá từ ảnh SAR.

### 4.2.3. Quản lý Giao thông Hàng hải

Ship detection hỗ trợ quản lý và giám sát giao thông hàng hải, đặc biệt tại các tuyến đường biển đông đúc và các cảng lớn. Thông tin về vị trí và mật độ tàu giúp tối ưu hóa điều phối giao thông, giảm thiểu nguy cơ va chạm, và cải thiện hiệu quả hoạt động cảng.

Đối với các eo biển quan trọng như Malacca Strait, ship detection cung cấp bức tranh toàn cảnh về lưu lượng tàu thuyền, hỗ trợ dự báo tắc nghẽn và lập kế hoạch điều hướng.

### 4.2.4. Ứng phó Tìm kiếm Cứu nạn (SAR Operations)

Trong các tình huống tìm kiếm cứu nạn trên biển, ship detection có thể hỗ trợ xác định vị trí tàu gặp nạn hoặc tìm kiếm tàu cứu hộ gần nhất. Khả năng quét diện rộng của vệ tinh bổ sung cho các phương tiện tìm kiếm truyền thống như trực thăng và tàu cứu hộ.

### 4.2.5. Phát hiện Tràn dầu từ Tàu

Ship detection thường được kết hợp với oil spill detection để truy nguyên nguồn gốc tràn dầu. Khi phát hiện vết dầu loang, việc xác định tàu gần đó (đặc biệt là tankers) giúp điều tra nguyên nhân và quy trách nhiệm. Trong một số trường hợp, có thể phát hiện dầu đang được xả từ tàu đang di chuyển.

## 4.3. Các Thách thức trong Ship Detection

### 4.3.1. Kích thước Đối tượng Đa dạng

Một trong những thách thức lớn nhất của ship detection là sự đa dạng cực lớn về kích thước tàu. Một tàu container lớn như Ever Given (400m dài) có thể chiếm hàng nghìn pixels trong ảnh độ phân giải cao, trong khi một tàu đánh cá nhỏ chỉ vài mét có thể chỉ chiếm vài pixels trong ảnh Sentinel-1 với GSD 10m.

Sự chênh lệch tỷ lệ này đặt ra yêu cầu về kiến trúc multi-scale. Các detector phải có khả năng phát hiện cả tàu lớn (cần receptive field rộng và semantic context) và tàu nhỏ (cần feature map high-resolution để giữ spatial detail). Feature Pyramid Network (FPN) và các biến thể là giải pháp phổ biến, cho phép detection ở nhiều mức resolution khác nhau.

Small object detection trong viễn thám còn khó khăn hơn trong ảnh tự nhiên do: (1) đối tượng nhỏ hơn nhiều so với kích thước ảnh (một ảnh Sentinel-1 scene có thể 25,000×25,000 pixels), (2) ít pixels cung cấp ít thông tin đặc trưng, và (3) dễ bị nhầm với noise đặc biệt trong ảnh SAR.

### 4.3.2. Môi trường Biển Phức tạp

Môi trường biển tạo ra nhiều nhiễu và false positives trong ship detection:

**Sóng và bọt sóng (Sea Clutter):** Sóng biển, đặc biệt whitecaps khi gió mạnh, có thể tạo ra các điểm sáng trên ảnh SAR giống với tín hiệu từ tàu nhỏ. Trong ảnh quang học, ánh sáng phản xạ từ mặt biển (sun glint) gây nhiễu tương tự.

**Vật thể trôi nổi:** Rác thải biển, container trôi, phao đánh dấu, và các vật thể khác có thể bị phát hiện nhầm là tàu, đặc biệt trong ảnh SAR không có thông tin màu sắc.

**Điều kiện thời tiết:** Mưa, sương mù ảnh hưởng đến ảnh quang học. Gió mạnh làm tăng độ nhám bề mặt biển, giảm contrast giữa tàu và background trong ảnh SAR.

**Đảo nhỏ và đá ngầm:** Trong vùng ven biển có nhiều đảo nhỏ, bãi đá có thể bị nhầm với tàu neo đậu.

### 4.3.3. Tàu gần Bờ và trong Cảng

Phát hiện tàu gần bờ biển và trong cảng đặc biệt khó khăn do:

**Background phức tạp:** Bờ biển, cầu cảng, container yards, nhà xưởng tạo ra background đa dạng và phức tạp, khác biệt với biển mở uniform.

**Occlusion:** Tàu có thể bị che khuất một phần bởi cầu cảng, cẩu, hoặc các tàu khác.

**Mật độ cao:** Trong các bến cảng lớn, hàng trăm tàu có thể neo đậu sát nhau, tạo ra thách thức cho việc phân tách từng tàu riêng lẻ.

**Land-water boundary:** Ranh giới đất-nước không rõ ràng trong một số ảnh, dẫn đến false positives từ các cấu trúc trên bờ.

### 4.3.4. Oriented Objects và Aspect Ratio

Tàu biển có đặc điểm hình học đặc trưng: hình dạng dài và hẹp với aspect ratio cao (tàu container có thể dài 400m nhưng chỉ rộng 60m, aspect ratio 6:1 hoặc cao hơn). Hơn nữa, tàu có thể xuất hiện theo bất kỳ hướng nào khi nhìn từ trên xuống.

Horizontal Bounding Box (HBB) truyền thống không phù hợp với các đối tượng dài và nghiêng:
- Box lớn hơn nhiều so với đối tượng thực, chứa nhiều background
- Khó phân tách các tàu xếp nghiêng gần nhau
- IoU threshold có thể không phản ánh đúng chất lượng detection

Oriented Bounding Box (OBB) hay Rotated Bounding Box được sử dụng để giải quyết vấn đề này, biểu diễn bounding box với thêm tham số góc xoay θ. Các detector như Rotated Faster R-CNN, RoI Transformer, và S²A-Net được thiết kế đặc biệt cho oriented detection.

### 4.3.5. Dense and Occluded Ships

Trong các khu vực đông đúc như bến cảng, tàu có thể:
- Neo đậu sát nhau, boxes overlap nhiều
- Một tàu bị che bởi tàu khác (occlusion)
- Các thành phần tàu (thân, cabin, cẩu) bị tách rời trong detection

Non-Maximum Suppression (NMS) chuẩn có thể loại bỏ nhầm các detections hợp lệ khi IoU cao. Soft-NMS và các phương pháp NMS cải tiến được sử dụng để giảm thiểu vấn đề này.

### 4.3.6. Đặc thù của Ảnh SAR

Synthetic Aperture Radar (SAR) là nguồn dữ liệu chính cho ship detection do khả năng hoạt động 24/7 bất kể thời tiết. Tuy nhiên, ảnh SAR có những đặc thù riêng:

**Speckle Noise:** Ảnh SAR chứa speckle - một dạng noise multiplicative đặc trưng của radar coherent. Speckle làm giảm chất lượng ảnh và có thể gây nhầm với đối tượng nhỏ.

**Side-looking geometry:** SAR chụp nghiêng, gây ra các hiệu ứng hình học như foreshortening, layover, và shadow. Tàu lớn có thể có shadow dài, cần được xử lý để không bị detect nhầm thành nhiều tàu.

**Không có thông tin màu sắc:** SAR cung cấp thông tin backscatter intensity, không có màu sắc như ảnh quang học. Việc phân loại loại tàu khó khăn hơn.

**Dual polarization:** Sentinel-1 cung cấp VV và VH polarization, mỗi kênh mang thông tin khác nhau. Việc fusion hiệu quả hai kênh là quan trọng.

## 4.4. SAR vs Optical Imagery cho Ship Detection

### 4.4.1. Ưu điểm của SAR

**All-weather capability:** SAR có thể hoạt động trong mọi điều kiện thời tiết - mây, mưa, sương mù không ảnh hưởng đến tín hiệu radar. Điều này đặc biệt quan trọng cho giám sát liên tục ở các vùng biển nhiệt đới thường xuyên có mây.

**Day-night operation:** SAR sử dụng nguồn năng lượng chủ động (active sensor), không phụ thuộc vào ánh sáng mặt trời. Có thể chụp ảnh 24/7, quan trọng cho surveillance.

**High contrast metal targets:** Tàu biển với cấu trúc kim loại phản xạ radar mạnh, tạo ra contrast cao so với mặt biển. Ngay cả tàu nhỏ cũng có thể tạo ra tín hiệu bright spot dễ phát hiện.

**Sentinel-1 free data:** Sentinel-1 của ESA cung cấp dữ liệu SAR miễn phí với coverage toàn cầu, độ phân giải 10m, và revisit time 6 ngày (với constellation 2 vệ tinh).

### 4.4.2. Hạn chế của SAR

**Speckle noise:** Cần preprocessing để giảm speckle, có thể mất một số detail.

**Lower resolution:** SAR miễn phí (Sentinel-1) có GSD 10m, thấp hơn nhiều so với ảnh quang học thương mại (0.3-1m). Tàu nhỏ khó phát hiện.

**Limited classification:** Không có thông tin màu sắc khiến việc phân loại loại tàu và phân biệt với các đối tượng khác khó khăn hơn.

**Complex interpretation:** Ảnh SAR đòi hỏi expertise để giải đoán, không trực quan như ảnh quang học.

### 4.4.3. Ưu điểm của Optical Imagery

**High resolution:** Ảnh quang học thương mại như WorldView-3 có GSD 0.3m, cho phép nhận dạng chi tiết cấu trúc tàu.

**Rich information:** 3 kênh RGB (hoặc multispectral với nhiều kênh hơn) cung cấp thông tin màu sắc hữu ích cho classification.

**Intuitive interpretation:** Ảnh trực quan, dễ kiểm tra và validate kết quả.

**Ship type classification:** Có thể nhận dạng chi tiết loại tàu dựa trên hình dạng, màu sắc, cấu trúc superstructure.

### 4.4.4. Hạn chế của Optical Imagery

**Cloud obstruction:** Mây che phủ hoàn toàn ảnh bên dưới. Ở vùng nhiệt đới, cloud cover có thể trên 70% thời gian.

**Night limitation:** Chỉ hoạt động ban ngày với đủ ánh sáng.

**Sun glint:** Ánh sáng phản xạ từ mặt biển có thể gây nhiễu.

**Cost:** Ảnh độ phân giải cao từ commercial providers có chi phí đáng kể.

### 4.4.5. Fusion Approach

Cách tiếp cận tối ưu là kết hợp cả SAR và optical:
- SAR cho continuous surveillance và initial detection
- Optical cho verification và detailed classification khi điều kiện cho phép
- Multi-modal fusion trong deep learning models

Một số nghiên cứu đã phát triển models có thể xử lý cả SAR và optical input, hoặc transfer learning giữa hai modalities.
