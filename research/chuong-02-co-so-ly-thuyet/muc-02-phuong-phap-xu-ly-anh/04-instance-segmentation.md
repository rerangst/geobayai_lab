# Chương 2: Instance Segmentation trong Viễn thám

## 3.17. Định nghĩa và Sự Khác biệt với Semantic Segmentation

Instance Segmentation là bài toán kết hợp giữa object detection và semantic segmentation, yêu cầu mô hình không chỉ phân loại từng pixel mà còn phân biệt từng instance riêng biệt của cùng một lớp. Trong khi semantic segmentation gán tất cả pixels của các con tàu vào cùng một lớp "ship", instance segmentation gán mỗi con tàu một ID riêng biệt, cho phép phân biệt và đếm từng tàu.

Output của instance segmentation là tập hợp các mask riêng lẻ, mỗi mask tương ứng với một instance kèm theo class label và confidence score: I = {(m₁, c₁, s₁), (m₂, c₂, s₂), ...}, trong đó mᵢ là binary mask cho instance i, cᵢ là class label, và sᵢ là confidence score. Các masks có thể overlap trong trường hợp occlusion, khác với semantic segmentation nơi mỗi pixel chỉ thuộc một lớp.

Về mặt ứng dụng, instance segmentation cung cấp thông tin chi tiết hơn semantic segmentation, cho phép đếm chính xác số lượng đối tượng, phân tích đặc điểm từng đối tượng riêng lẻ, và tracking đối tượng qua các khung hình trong video hoặc chuỗi ảnh thời gian.

## 3.18. Ứng dụng Instance Segmentation trong Viễn thám

### 3.18.1. Đếm và Phân tích Từng Tàu Biển

Trong giám sát hàng hải, instance segmentation cho phép đếm chính xác số lượng tàu trong một vùng biển hoặc tại một cảng. Hơn nữa, mask của từng tàu cung cấp thông tin về kích thước và hình dạng, hỗ trợ việc phân loại loại tàu và ước tính tonnage. Khi kết hợp với dữ liệu AIS, mỗi instance có thể được cross-reference để xác định danh tính tàu.

Trong các bến cảng đông đúc với nhiều tàu neo đậu sát nhau, instance segmentation vượt trội so với detection (bounding box chồng lấp) và semantic segmentation (không phân biệt được các tàu dính nhau). Mask chính xác cho từng tàu cho phép tách riêng các tàu ngay cả khi chúng chạm hoặc gần nhau.

### 3.18.2. Đếm Từng Tòa nhà và Đánh giá Thiệt hại

Trong đánh giá thiệt hại sau thiên tai (như cuộc thi xView2), instance segmentation cho phép đếm số lượng tòa nhà và phân loại mức độ thiệt hại cho từng tòa. Thông tin này quan trọng cho việc ước tính thiệt hại và lập kế hoạch cứu trợ.

Mỗi instance building có thể được gán một trong các mức độ thiệt hại: no damage, minor damage, major damage, destroyed. Kết quả là thống kê chi tiết về số lượng công trình ở mỗi mức độ thiệt hại trong vùng ảnh hưởng.

### 3.18.3. Đếm Phương tiện Giao thông

Instance segmentation được sử dụng để đếm xe trong bãi đỗ xe, trên đường phố, và tại các nút giao thông. So với detection, segmentation mask cung cấp thông tin chính xác hơn về vị trí và hướng của từng xe, đặc biệt trong các tình huống đông đúc với nhiều xe đỗ sát nhau.

### 3.18.4. Phân tích Từng Vết Dầu Loang

Trong trường hợp có nhiều vết dầu loang riêng biệt trong cùng một ảnh (ví dụ từ nhiều nguồn khác nhau hoặc bị chia tách bởi dòng chảy), instance segmentation cho phép phân tích từng vết riêng lẻ: tính diện tích, xác định hướng lan tỏa, và truy nguyên nguồn gốc riêng biệt cho mỗi vết.

## 3.19. Kiến trúc Instance Segmentation

### 3.19.1. Mask R-CNN

Mask R-CNN là kiến trúc instance segmentation phổ biến nhất, mở rộng Faster R-CNN bằng cách thêm một branch dự đoán segmentation mask cho mỗi detected object.

**Kiến trúc:** Mask R-CNN có bốn thành phần chính: (1) backbone + FPN để trích xuất multi-scale features, (2) Region Proposal Network (RPN) để tạo object proposals, (3) RoI Align để extract fixed-size features từ mỗi proposal, và (4) Head networks gồm classification branch, box regression branch, và mask branch.

**RoI Align:** Cải tiến quan trọng so với RoI Pooling trong Faster R-CNN. RoI Pooling sử dụng quantization (làm tròn tọa độ) có thể gây misalignment giữa RoI và feature map, đặc biệt có hại cho mask prediction yêu cầu độ chính xác pixel-level. RoI Align sử dụng bilinear interpolation để sample features tại các vị trí chính xác, cải thiện đáng kể chất lượng mask.

**Mask Branch:** Một fully convolutional network nhỏ (thường 4 conv layers theo sau bởi deconv) dự đoán binary mask cho mỗi class. Output có kích thước m×m×K, với m thường là 28 hoặc 14, và K là số lớp. Mask được dự đoán riêng cho mỗi class (class-specific masks), nhưng trong inference chỉ mask của class được dự đoán bởi classification branch được sử dụng.

**Loss Function:** Total loss là tổng của classification loss, box regression loss, và mask loss:
L = Lcls + Lbox + Lmask

Mask loss là average binary cross-entropy, chỉ tính trên mask của ground truth class để tránh competition giữa các class.

**Ưu điểm:** Mask R-CNN đạt kết quả state-of-the-art và là baseline cho hầu hết các nghiên cứu instance segmentation. Kiến trúc modular cho phép dễ dàng thay đổi backbone hoặc thêm các branch khác.

**Trong TorchGeo:** Thư viện TorchGeo (**Chương 5**) cung cấp Mask R-CNN với các backbones pre-trained trên ảnh vệ tinh, hữu ích cho các bài toán như building instance segmentation trong xView2 challenge.

### 3.19.2. YOLACT và YOLACT++

YOLACT (You Only Look At CoefficienTs) là real-time instance segmentation model, đạt tốc độ 30+ FPS với độ chính xác cạnh tranh.

**Ý tưởng chính:** Thay vì dự đoán mask trực tiếp cho mỗi instance, YOLACT tách bài toán thành hai phần song song: (1) Prototype generation branch tạo ra k prototype masks (thường k=32) có kích thước full image, và (2) Prediction head dự đoán k mask coefficients cho mỗi detected instance. Mask cuối cùng được tạo bằng linear combination của prototypes với coefficients, sau đó crop theo bounding box.

**Ưu điểm:** Tốc độ nhanh do prototype được chia sẻ giữa tất cả instances và mask assembly chỉ là matrix multiplication. Phù hợp cho các ứng dụng real-time hoặc xử lý video.

**YOLACT++:** Cải tiến với deformable convolution, improved backbone (ResNet-101-DCN), và mask rescoring branch để cải thiện chất lượng mask.

### 3.19.3. SOLOv2

SOLO (Segmenting Objects by Locations) và SOLOv2 là các phương pháp instance segmentation không dựa trên detection (box-free).

**Ý tưởng:** Chia ảnh thành S×S grid, mỗi cell chịu trách nhiệm dự đoán instance có center rơi vào cell đó. Mỗi cell dự đoán: (1) category probability, và (2) instance mask. Mask được dự đoán ở một branch riêng với category.

**SOLOv2 improvements:** Sử dụng dynamic convolution - kernel weights được predicted để generate mask thay vì predict mask trực tiếp, cải thiện chất lượng và giảm memory. Matrix NMS nhanh hơn traditional NMS.

### 3.19.4. PointRend

PointRend cải thiện chất lượng mask boundaries bằng cách render mask như một rendering problem.

**Ý tưởng:** Thay vì predict mask ở fixed resolution rồi upsample, PointRend iteratively refine predictions tại uncertain points (thường ở boundaries). Bắt đầu từ coarse mask, chọn N points có uncertainty cao nhất, predict refined values cho những points đó, rồi lặp lại.

**Kết hợp với Mask R-CNN:** PointRend có thể được thêm vào Mask R-CNN như một module bổ sung, cải thiện đáng kể chất lượng mask ở object boundaries với chi phí tính toán thấp.

## 3.20. Panoptic Segmentation

### 3.20.1. Định nghĩa

Panoptic Segmentation thống nhất semantic segmentation và instance segmentation thành một task duy nhất. Mỗi pixel trong ảnh được gán một class label (như semantic segmentation) và một instance ID (như instance segmentation). Các lớp được chia thành "things" (countable objects như xe, tàu, tòa nhà - có instance IDs riêng biệt) và "stuff" (uncountable regions như bầu trời, biển, rừng - không có instance IDs).

**Output format:** Mỗi pixel (i,j) được gán (cᵢⱼ, idᵢⱼ) với c là class và id là instance ID. Đối với stuff classes, tất cả pixels có cùng class có cùng id. Đối với things classes, mỗi instance có id riêng.

### 3.20.2. Ứng dụng trong Viễn thám

Panoptic segmentation phù hợp cho các ứng dụng cần cả semantic understanding và instance-level analysis. Ví dụ, trong giám sát cảng biển: background (sea, land) được segment như stuff classes, còn các tàu và container được segment như things classes với instance IDs riêng biệt.

### 3.20.3. Kiến trúc Panoptic FPN

Panoptic FPN mở rộng Mask R-CNN với semantic segmentation branch để xử lý stuff classes. Stuff branch là một simple FPN-based semantic segmentation network. Output từ instance branch (things) và stuff branch (stuff) được merge theo rules: nếu pixel được cover bởi instance mask với confidence cao, gán cho instance đó; ngược lại gán theo stuff prediction.

## 3.21. Metrics Đánh giá Instance Segmentation

### 3.21.1. Mask AP (Average Precision)

Tương tự AP trong object detection, nhưng IoU được tính giữa predicted masks và ground truth masks thay vì bounding boxes:

Mask IoU = |Predicted Mask ∩ GT Mask| / |Predicted Mask ∪ GT Mask|

AP được tính ở các ngưỡng IoU khác nhau: AP@0.5, AP@0.75, AP@[0.5:0.95]. COCO metric (AP@[0.5:0.95]) là tiêu chuẩn phổ biến nhất.

### 3.21.2. AP cho Small/Medium/Large Objects

COCO chia instances theo diện tích mask:
- Small: area < 32² pixels
- Medium: 32² ≤ area ≤ 96² pixels
- Large: area > 96² pixels

Report AP riêng cho mỗi category size để đánh giá chi tiết performance theo kích thước đối tượng.

### 3.21.3. Panoptic Quality (PQ)

Đối với panoptic segmentation, Panoptic Quality được định nghĩa:

PQ = (Σ IoU(p,g)) / (|TP| + ½|FP| + ½|FN|)

trong đó (p,g) là matched pairs giữa predicted và ground truth segments. PQ có thể phân tích thành:

PQ = SQ × RQ

với SQ (Segmentation Quality) = Σ IoU / |TP| đo chất lượng segmentation của matched segments, và RQ (Recognition Quality) = |TP| / (|TP| + ½|FP| + ½|FN|) đo khả năng detection.

## 3.22. So sánh và Lựa chọn Phương pháp

### 3.22.1. Khi nào dùng Semantic Segmentation

Sử dụng semantic segmentation khi:
- Chỉ cần biết vùng nào thuộc lớp nào, không cần phân biệt từng instance
- Các lớp không overlap (mỗi pixel thuộc một lớp duy nhất)
- Đối tượng có thể merge thành vùng lớn (như rừng, mặt nước, đô thị)
- Cần inference nhanh hơn instance segmentation

Ví dụ: Land cover mapping, flood mapping, oil spill extent estimation.

### 3.22.2. Khi nào dùng Instance Segmentation

Sử dụng instance segmentation khi:
- Cần đếm số lượng đối tượng
- Cần phân tích đặc điểm từng đối tượng riêng lẻ
- Cần tracking đối tượng qua thời gian
- Các đối tượng có thể chạm hoặc overlap nhau

Ví dụ: Ship counting, building damage assessment per building, vehicle counting.

### 3.22.3. Trade-offs

| Aspect | Semantic Seg | Instance Seg |
|--------|--------------|--------------|
| **Output** | Dense label map | Set of masks + labels |
| **Instance info** | Không | Có |
| **Tốc độ** | Nhanh hơn | Chậm hơn |
| **Complexity** | Thấp hơn | Cao hơn |
| **Training data** | Pixel-level labels | Per-instance masks |

### 3.22.4. Hybrid Approaches

Trong nhiều ứng dụng viễn thám, hybrid approach có thể được sử dụng:
- Semantic segmentation cho background/stuff classes (sea, land)
- Instance segmentation cho countable objects (ships, buildings)
- Post-processing để extract instances từ semantic masks (connected component analysis)

Ví dụ, cho oil spill: semantic segmentation phân vùng toàn bộ oil spill region, sau đó connected component analysis tách thành các vết dầu riêng biệt nếu cần.

---

## Kết chương

Chương này đã trình bày các kiến thức nền tảng về CNN và các bài toán xử lý ảnh trong viễn thám. Từ các thành phần cơ bản của CNN (convolution, pooling, activation) đến các kiến trúc backbone hiện đại (ResNet, EfficientNet, ViT, Swin Transformer), và các phương pháp xử lý ảnh từ classification, object detection, đến semantic và instance segmentation.

Các kiến thức này tạo nền tảng để hiểu các implementation cụ thể trong **Chương 5** về thư viện TorchGeo - công cụ chuyên biệt cho deep learning trong viễn thám. TorchGeo cung cấp các mô hình pre-trained, datasets, và transforms được tối ưu cho ảnh vệ tinh, cho phép áp dụng các phương pháp đã học vào thực tế.

Tiếp theo, **Chương 6** sẽ phân tích ba cuộc thi xView Challenges (object detection, building damage assessment, maritime detection) và 15 giải pháp hàng đầu, minh họa cách các kiến trúc và kỹ thuật đã học được áp dụng để giải quyết các bài toán viễn thám thách thức trong thực tế.
