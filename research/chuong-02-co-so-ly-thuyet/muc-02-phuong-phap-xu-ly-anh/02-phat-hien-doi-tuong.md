# Chương 2: Phát hiện Đối tượng (Object Detection) trong Viễn thám

## 3.7. Định nghĩa Bài toán Object Detection

Object Detection là bài toán kết hợp giữa localization (xác định vị trí) và classification (phân loại), yêu cầu mô hình không chỉ nhận dạng đối tượng trong ảnh mà còn xác định chính xác vị trí của chúng thông qua bounding box. Khác với image classification gán một nhãn cho toàn bộ ảnh, object detection có thể phát hiện nhiều đối tượng thuộc nhiều lớp khác nhau trong cùng một ảnh, mỗi đối tượng được biểu diễn bởi một bounding box và nhãn lớp kèm theo confidence score.

Về mặt hình thức, output của object detection là tập hợp các detection D = {(b₁, c₁, s₁), (b₂, c₂, s₂), ...}, trong đó bᵢ là bounding box (thường được biểu diễn bởi 4 giá trị: tọa độ góc trên trái x, y, chiều rộng w, và chiều cao h, hoặc tọa độ hai góc đối diện), cᵢ là nhãn lớp, và sᵢ là confidence score thể hiện độ tin cậy của detection. Quá trình post-processing như Non-Maximum Suppression (NMS) được áp dụng để loại bỏ các detection trùng lặp.

Trong viễn thám, object detection được sử dụng rộng rãi cho nhiều ứng dụng: phát hiện tàu biển để giám sát hàng hải, phát hiện máy bay tại sân bay, đếm xe trong bãi đỗ xe và trên đường giao thông, phát hiện các công trình xây dựng, và nhiều ứng dụng khác. Mỗi ứng dụng đặt ra những thách thức riêng về kích thước đối tượng, mật độ, và điều kiện chụp ảnh.

## 3.8. Các Thách thức Đặc thù trong Viễn thám

### 3.8.1. Đối tượng Nhỏ (Small Object Detection)

Một trong những thách thức lớn nhất của object detection trong viễn thám là kích thước nhỏ của đối tượng so với kích thước ảnh. Trong ảnh vệ tinh độ phân giải trung bình (như Sentinel-1 với 10m GSD), một tàu đánh cá nhỏ có thể chỉ chiếm vài pixel. Ngay cả trong ảnh độ phân giải cao như WorldView-3 (0.3m GSD), các phương tiện vẫn nhỏ hơn đáng kể so với đối tượng trong các dataset tự nhiên như COCO hay Pascal VOC.

Đối tượng nhỏ gặp nhiều khó khăn trong các kiến trúc CNN tiêu chuẩn do: thông tin đặc trưng bị mất qua các lớp pooling liên tiếp, feature map ở các layer sâu có độ phân giải thấp không đủ để biểu diễn đối tượng nhỏ, và tỷ lệ foreground/background trong anchor-based method cực kỳ không cân bằng. Các kỹ thuật như Feature Pyramid Network (FPN), high-resolution prediction head, và multi-scale training được sử dụng để giải quyết vấn đề này.

### 3.8.2. Mật độ Đối tượng Cao (Dense Object Detection)

Trong nhiều scenario viễn thám, đối tượng xuất hiện với mật độ rất cao. Một bến cảng có thể chứa hàng trăm container xếp chồng, một bãi đỗ xe có hàng nghìn xe, và một đoàn tàu biển có thể gồm nhiều tàu xếp sát nhau. Mật độ cao gây ra vấn đề occlusion (đối tượng che khuất lẫn nhau) và khó khăn cho NMS trong việc phân biệt các detection gần nhau.

Các phương pháp giải quyết bao gồm: sử dụng IoU threshold thấp hơn trong NMS, áp dụng Soft-NMS thay vì hard suppression, và sử dụng các kiến trúc được thiết kế cho dense detection như FCOS và CenterNet.

### 3.8.3. Đối tượng Nghiêng (Oriented Object Detection)

Không giống như ảnh tự nhiên với đối tượng thường có hướng "đứng" cố định, đối tượng trong ảnh viễn thám (chụp từ trên xuống) có thể xuất hiện theo bất kỳ hướng nào. Một con tàu có thể nằm ngang, dọc, hoặc nghiêng góc 45°. Bounding box ngang (horizontal bounding box - HBB) truyền thống không fit tốt với các đối tượng dài và hẹp như tàu, máy bay, gây ra nhiều background trong box và khó khăn trong việc phân tách các đối tượng xếp nghiêng gần nhau.

Oriented Bounding Box (OBB) hay Rotated Bounding Box được sử dụng để giải quyết vấn đề này. OBB được biểu diễn bởi 5 tham số: tọa độ tâm (cx, cy), chiều rộng w, chiều cao h, và góc xoay θ. Các kiến trúc như Rotated Faster R-CNN, RoI Transformer, và Oriented R-CNN được phát triển đặc biệt cho bài toán oriented detection.

### 3.8.4. Sự Đa dạng về Tỷ lệ (Multi-scale Objects)

Trong cùng một ảnh vệ tinh, đối tượng có thể xuất hiện ở nhiều tỷ lệ rất khác nhau. Một tàu container lớn có thể dài hàng trăm pixel trong khi một tàu đánh cá nhỏ chỉ vài pixel. Sự khác biệt tỷ lệ cực đoan này đòi hỏi detector phải có khả năng xử lý đồng thời cả đối tượng lớn và nhỏ.

Feature Pyramid Network (FPN) và các biến thể như PANet, BiFPN là các giải pháp phổ biến, xây dựng multi-scale feature maps và thực hiện detection ở nhiều mức độ phân giải khác nhau. Các đối tượng lớn được detect từ feature map có độ phân giải thấp (nhiều semantic information), trong khi đối tượng nhỏ được detect từ feature map có độ phân giải cao (nhiều spatial detail).

## 3.9. Kiến trúc Two-stage Detector

### 3.9.1. Nguyên lý Hoạt động

Two-stage detector chia quá trình detection thành hai giai đoạn riêng biệt. Giai đoạn đầu tiên (Region Proposal Network - RPN) tạo ra các vùng ứng viên (region proposals) có khả năng chứa đối tượng, không phân biệt lớp cụ thể. Giai đoạn thứ hai thực hiện classification và bounding box regression trên các proposals này, quyết định lớp của đối tượng và tinh chỉnh vị trí bounding box.

Cách tiếp cận hai giai đoạn cho phép mỗi giai đoạn tập trung vào một nhiệm vụ cụ thể, thường dẫn đến accuracy cao hơn so với one-stage detector. Tuy nhiên, tốc độ inference chậm hơn do cần xử lý tuần tự hai giai đoạn.

### 3.9.2. Faster R-CNN

Faster R-CNN là kiến trúc two-stage detector phổ biến nhất, được đề xuất bởi Ren và cộng sự năm 2015. Kiến trúc gồm ba thành phần chính: backbone network (như ResNet hoặc VGG) trích xuất feature map từ ảnh đầu vào, Region Proposal Network (RPN) tạo các object proposals từ feature map, và Fast R-CNN head thực hiện classification và bounding box regression trên các proposals.

**Region Proposal Network (RPN):** RPN là một fully convolutional network trượt trên feature map, tại mỗi vị trí dự đoán k anchor boxes với các tỷ lệ và kích thước khác nhau. Với mỗi anchor, RPN output hai giá trị: objectness score (xác suất anchor chứa đối tượng) và 4 giá trị regression để điều chỉnh vị trí anchor thành proposal. Các proposals có objectness score cao được chọn cho giai đoạn sau.

**RoI Pooling/Align:** Do các proposals có kích thước khác nhau, cần cơ chế để extract feature vector có kích thước cố định từ mỗi proposal. RoI Pooling chia proposal thành lưới cố định (ví dụ 7×7) và max pool trong mỗi ô. RoI Align cải tiến bằng cách sử dụng bilinear interpolation, tránh quantization error và cải thiện accuracy cho các đối tượng nhỏ.

**Classification và Box Regression Head:** Feature vector từ mỗi proposal được đưa qua các lớp fully connected để dự đoán: (1) xác suất thuộc mỗi lớp (K+1 lớp, bao gồm background), và (2) 4×K giá trị regression để tinh chỉnh bounding box cho mỗi lớp.

### 3.9.3. Feature Pyramid Network (FPN)

FPN được đề xuất để cải thiện khả năng detect multi-scale objects. Ý tưởng cốt lõi là xây dựng một pyramid of feature maps bằng cách kết hợp bottom-up pathway (forward pass thông thường) với top-down pathway và lateral connections.

Trong bottom-up pathway, feature map giảm dần về spatial resolution và tăng semantic richness qua các stage của backbone. Trong top-down pathway, feature map được upsample (thường 2×) và cộng với feature map từ bottom-up pathway cùng resolution thông qua lateral connection (convolution 1×1 để match số kênh). Kết quả là các feature map ở mọi scale đều có cả low-level detail và high-level semantics.

FPN có thể được tích hợp với Faster R-CNN bằng cách: (1) chạy RPN trên tất cả các mức của pyramid, (2) assign proposals cho các mức dựa trên kích thước (đối tượng nhỏ → mức cao resolution, đối tượng lớn → mức thấp resolution), và (3) thực hiện RoI pooling từ mức tương ứng.

### 3.9.4. Cascade R-CNN

Cascade R-CNN giải quyết vấn đề mismatch giữa IoU threshold trong training và quality của proposals. Trong Faster R-CNN tiêu chuẩn, một IoU threshold cố định (thường 0.5) được sử dụng để định nghĩa positive/negative proposals. Proposals có IoU thấp hơn ngưỡng này không được huấn luyện, dẫn đến detector không tối ưu cho các detection có IoU cao.

Cascade R-CNN sử dụng nhiều detection heads xếp chồng, mỗi head được huấn luyện với IoU threshold tăng dần (ví dụ 0.5 → 0.6 → 0.7). Output proposals từ head trước được sử dụng làm input cho head sau, cho phép progressive refinement. Trong inference, output của head cuối cùng được sử dụng, cho kết quả detection chất lượng cao hơn.

## 3.10. Kiến trúc One-stage Detector

### 3.10.1. Nguyên lý Hoạt động

One-stage detector (còn gọi là single-shot detector) thực hiện detection trong một lần forward pass duy nhất, không có giai đoạn tạo proposals riêng biệt. Mạng trực tiếp dự đoán bounding boxes và class probabilities từ feature map, thường sử dụng dense prediction trên một lưới các vị trí.

Ưu điểm chính của one-stage detector là tốc độ inference nhanh, phù hợp cho các ứng dụng real-time. Nhược điểm truyền thống là accuracy thấp hơn two-stage do phải xử lý extreme class imbalance (rất nhiều background so với foreground). Tuy nhiên, các kỹ thuật hiện đại như Focal Loss đã thu hẹp đáng kể khoảng cách này.

### 3.10.2. YOLO Family

YOLO (You Only Look Once) là họ one-stage detector phổ biến nhất, được phát triển qua nhiều phiên bản với cải tiến liên tục về accuracy và speed.

**YOLOv1-v3:** Các phiên bản ban đầu thiết lập kiến trúc cơ bản: chia ảnh thành grid S×S, mỗi cell dự đoán B bounding boxes và C class probabilities. YOLOv2 giới thiệu anchor boxes và batch normalization. YOLOv3 sử dụng multi-scale prediction tương tự FPN và Darknet-53 backbone.

**YOLOv4-v5:** Tích hợp nhiều "bag of freebies" và "bag of specials" - các kỹ thuật cải thiện training và inference. CSPDarknet backbone, SPP (Spatial Pyramid Pooling), và PANet neck được sử dụng. YOLOv5 (không phải từ tác giả gốc) được viết bằng PyTorch với nhiều cải tiến về engineering.

**YOLOv7-v8:** YOLOv7 giới thiệu E-ELAN (Extended Efficient Layer Aggregation Network) và nhiều cải tiến training. YOLOv8 là phiên bản mới nhất từ Ultralytics, với kiến trúc anchor-free, decoupled head, và nhiều cải tiến về training và inference.

**YOLOv9-v10:** Các phiên bản gần đây nhất (2024) với GELAN (Generalized Efficient Layer Aggregation Network) trong YOLOv9 và NMS-free design trong YOLOv10, hướng đến tối ưu cho edge deployment.

Đối với ship detection trong viễn thám, các biến thể YOLO được fine-tune trên các dataset như SSDD và HRSID đã cho kết quả state-of-the-art với mAP trên 90%.

### 3.10.3. RetinaNet và Focal Loss

RetinaNet được đề xuất bởi Lin và cộng sự (2017) cùng với Focal Loss, giải quyết vấn đề class imbalance trong one-stage detector và đạt accuracy ngang với two-stage detector.

Kiến trúc RetinaNet gồm: ResNet + FPN backbone, classification subnet (4 lớp conv 256 channel + conv output KA classes), và box regression subnet (4 lớp conv 256 channel + conv output 4A values), trong đó A là số anchors mỗi vị trí và K là số lớp.

Focal Loss là đóng góp quan trọng nhất, đã được trình bày trong phần Classification. Trong context detection, background locations chiếm đa số (hàng nghìn đến hàng triệu) so với foreground (vài chục đến vài trăm). Focal Loss down-weight easy negatives, cho phép mạng tập trung vào hard examples.

### 3.10.4. Anchor-free Detector

Anchor-free detector loại bỏ khái niệm predefined anchor boxes, thay vào đó dự đoán trực tiếp các thuộc tính của bounding box. Điều này đơn giản hóa design, giảm hyperparameters (không cần tune anchor sizes và aspect ratios), và có thể cải thiện performance cho các đối tượng có hình dạng bất thường.

**CenterNet:** Dự đoán object center như heatmap peak, sau đó regress width, height, và offset từ center. Không cần NMS do mỗi object chỉ tạo một center point.

**FCOS (Fully Convolutional One-Stage Object Detection):** Dự đoán 4 distances từ mỗi foreground pixel đến 4 cạnh của bounding box, kết hợp với centerness score để suppress low-quality predictions.

## 3.11. Metrics Đánh giá Object Detection

### 3.11.1. Intersection over Union (IoU)

IoU (còn gọi là Jaccard Index) đo lường mức độ overlap giữa predicted bounding box và ground truth bounding box:

IoU = Area(Prediction ∩ Ground Truth) / Area(Prediction ∪ Ground Truth)

IoU có giá trị từ 0 (không overlap) đến 1 (overlap hoàn toàn). Một detection được coi là True Positive nếu IoU với một ground truth box vượt qua ngưỡng nhất định (thường 0.5 hoặc 0.75). Mỗi ground truth chỉ được match với một prediction, predictions còn lại là False Positive.

Đối với oriented bounding box (OBB), IoU được tính tương tự nhưng phức tạp hơn về mặt tính toán do cần xử lý polygon intersection thay vì rectangle intersection.

### 3.11.2. Precision-Recall và Average Precision (AP)

Tương tự classification, Precision và Recall cho detection được định nghĩa:

Precision = TP / (TP + FP) = số detection đúng / tổng số detection
Recall = TP / (TP + FN) = số detection đúng / tổng số ground truth

Bằng cách thay đổi confidence threshold, ta thu được Precision-Recall curve. Average Precision (AP) là diện tích dưới đường P-R curve, tính bằng cách interpolate hoặc lấy tổng Riemann. AP tổng hợp cả Precision và Recall thành một metric duy nhất.

### 3.11.3. Mean Average Precision (mAP)

mAP là trung bình của AP trên tất cả các lớp:

mAP = (1/K) × Σ APₖ

mAP thường được report ở các ngưỡng IoU khác nhau:
- mAP@0.5: IoU threshold 0.5 (tiêu chuẩn Pascal VOC)
- mAP@0.75: IoU threshold 0.75 (yêu cầu localization chính xác hơn)
- mAP@[0.5:0.95]: Trung bình mAP từ IoU 0.5 đến 0.95 với step 0.05 (tiêu chuẩn COCO)

Đối với các dataset viễn thám như DOTA, SSDD, và HRSID, mAP@0.5 là metric phổ biến nhất được báo cáo.

### 3.11.4. Các Metrics Bổ sung

**AP cho Small/Medium/Large Objects:** COCO chia đối tượng theo diện tích (small < 32², medium 32²-96², large > 96²) và report AP riêng cho mỗi loại. Điều này quan trọng cho viễn thám với nhiều đối tượng nhỏ.

**AR (Average Recall):** Tương tự AP nhưng đo Recall, thường report AR@1, AR@10, AR@100 (maximum detections per image).

**F1-Score tại ngưỡng cố định:** Một số paper report F1 tại confidence threshold cụ thể, cung cấp single-point metric thay vì curve.
