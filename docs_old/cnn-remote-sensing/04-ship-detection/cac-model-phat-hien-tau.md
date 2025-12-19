# Các Model Deep Learning cho Ship Detection

## 4.5. Tổng quan về Các Hướng Tiếp cận

Các phương pháp deep learning cho ship detection có thể được phân loại theo nhiều tiêu chí. Theo số stage, có two-stage detectors (region proposal + classification) và one-stage detectors (direct prediction). Theo loại bounding box, có horizontal bounding box (HBB) detectors và oriented bounding box (OBB) detectors. Theo backbone, có CNN-based và Transformer-based approaches.

Lựa chọn phương pháp phụ thuộc vào yêu cầu cụ thể: accuracy vs speed, HBB vs OBB, loại ảnh (SAR/optical), và tài nguyên deployment. Trong phần này, chúng ta sẽ phân tích chi tiết các kiến trúc phổ biến nhất cho ship detection.

## 4.6. Two-stage Detectors

### 4.6.1. Faster R-CNN với FPN

Faster R-CNN kết hợp với Feature Pyramid Network là baseline mạnh cho ship detection, đặc biệt khi yêu cầu accuracy cao.

**Kiến trúc cho Ship Detection:**
- **Backbone:** ResNet-50 hoặc ResNet-101 pre-trained trên ImageNet, hoặc pre-trained trên satellite imagery từ TorchGeo
- **Neck:** FPN để xử lý multi-scale ships
- **RPN:** Region Proposal Network với anchors được tune cho ship aspect ratios
- **Head:** Classification + Box Regression

**Tuning Anchors cho Ships:**
Ships có aspect ratios đặc trưng (thường 3:1 đến 8:1), khác với đối tượng thông thường (1:1 đến 2:1). Anchor boxes cần được điều chỉnh:
- Aspect ratios: [0.2, 0.33, 0.5, 1.0, 2.0, 3.0, 5.0]
- Scales phụ thuộc vào GSD của ảnh và kích thước tàu target

**Kết quả điển hình:**
Trên dataset SSDD (SAR Ship Detection Dataset), Faster R-CNN + FPN đạt mAP@0.5 khoảng 85-90% tùy cấu hình.

### 4.6.2. Cascade R-CNN

Cascade R-CNN cải thiện Faster R-CNN bằng multi-stage refinement, đặc biệt hiệu quả cho high-quality detections với IoU cao.

**Cải tiến cho Ship Detection:**
- Nhiều stage với IoU threshold tăng dần (0.5 → 0.6 → 0.7)
- Mỗi stage refine proposals từ stage trước
- Cuối cùng output high-quality boxes

Cascade R-CNN phù hợp khi cần localization chính xác, ví dụ để ước tính kích thước tàu hoặc kết hợp với segmentation.

### 4.6.3. Rotated Faster R-CNN

Để xử lý oriented ships, Rotated Faster R-CNN mở rộng Faster R-CNN với oriented bounding boxes.

**Thay đổi chính:**
- **Rotated Anchors:** Mỗi anchor có thêm góc rotation, thường discretize thành 6-12 góc từ 0° đến 180°
- **Rotated RoI Pooling/Align:** Extract features từ rotated regions, đòi hỏi xử lý hình học phức tạp hơn
- **Box Regression:** Predict 5 values (cx, cy, w, h, θ) thay vì 4

**Loss cho Angle Regression:**
Angle regression có vấn đề periodicity (0° và 180° tương đương cho nhiều shapes). Các giải pháp:
- Smooth L1 loss với angle normalization
- Circular Smooth Label (CSL): treat angle như classification problem
- Gaussian Wasserstein Distance (GWD): model boxes như Gaussian distributions

### 4.6.4. RoI Transformer

RoI Transformer là kiến trúc chuyên biệt cho oriented detection, sử dụng Spatial Transformer để learn rotation.

**Ý tưởng chính:**
- Stage 1: Detect horizontal RoIs như Faster R-CNN thông thường
- Stage 2: RoI Transformer học transformation từ horizontal RoI sang oriented RoI
- Rotated Position Sensitive (RPS) RoI Align để extract aligned features

**Ưu điểm:**
- Không cần rotated anchors (giảm hyperparameters)
- Learn rotation end-to-end
- State-of-the-art trên DOTA và các dataset viễn thám

### 4.6.5. Oriented R-CNN

Oriented R-CNN là kiến trúc efficient cho oriented detection, cải tiến Faster R-CNN với oriented proposals.

**Đặc điểm:**
- **Oriented RPN:** Trực tiếp propose oriented boxes từ FPN features
- **Rotated RoI Align:** Extract features từ oriented proposals
- **Midpoint Offset Representation:** Biểu diễn oriented box hiệu quả hơn

Oriented R-CNN đạt kết quả tốt với tốc độ nhanh hơn RoI Transformer.

## 4.7. One-stage Detectors (YOLO Family)

### 4.7.1. YOLOv5 cho Ship Detection

YOLOv5 là một trong những detectors phổ biến nhất cho ship detection do cân bằng tốt giữa accuracy và speed.

**Customizations cho SAR Ship Detection:**
- **Input size:** Tăng lên 1024×1024 hoặc cao hơn để detect small ships
- **Anchors:** Clustering trên ship dataset để optimize anchor sizes
- **Data augmentation:** Mosaic, MixUp đặc biệt hiệu quả
- **Model size:** YOLOv5m hoặc YOLOv5l cho accuracy tốt hơn

**Kết quả trên SSDD:**
YOLOv5l đạt mAP@0.5 92-95% với inference speed ~50 FPS trên GPU modern.

### 4.7.2. YOLOv7 và Biến thể

YOLOv7 giới thiệu nhiều cải tiến về architecture và training strategies.

**YOLOv7-LDS (Lightweight Detection for SAR):**
Biến thể tối ưu cho SAR ship detection:
- Giảm 26.7% parameters so với baseline
- Duy trì accuracy cao
- Phù hợp cho edge deployment

**Innovations:**
- E-ELAN (Extended Efficient Layer Aggregation Network)
- Compound scaling method
- Planned re-parameterized convolution

### 4.7.3. YOLOv8

YOLOv8 từ Ultralytics với nhiều cải tiến về architecture và usability.

**Đặc điểm mới:**
- **Anchor-free:** Không cần predefined anchors, giảm hyperparameters
- **Decoupled head:** Tách biệt classification và regression heads
- **C2f module:** Efficient feature extraction
- **Task-specific heads:** Hỗ trợ detection, segmentation, pose

**Cho Ship Detection:**
YOLOv8 với custom training trên ship datasets đạt state-of-the-art performance với inference nhanh.

### 4.7.4. YOLOv9 và YOLOv10

Các phiên bản mới nhất (2024) với cải tiến tiếp tục.

**YOLOv9:**
- GELAN (Generalized Efficient Layer Aggregation Network)
- PGI (Programmable Gradient Information)
- First application cho ship detection trong một số papers gần đây

**YOLOv10:**
- NMS-free design: Loại bỏ NMS post-processing
- Optimized cho TinyML và edge deployment
- Inference ~10ms trên hardware phù hợp

### 4.7.5. AC-YOLO và Các Biến thể SAR-specific

Nhiều biến thể YOLO được thiết kế đặc biệt cho SAR ship detection:

**AC-YOLO (2025):**
- Dựa trên YOLO11
- Attention mechanisms cho SAR features
- Coordinate attention blocks

**YOLO-SD:**
- Multi-scale convolution
- Feature Transformer module
- Optimized cho small ship detection

**GDB-YOLOv5s:**
- Ghost convolution cho lightweight
- BiFPN cho multi-scale fusion
- Designed cho SAR imagery

## 4.8. Oriented One-stage Detectors

### 4.8.1. Rotated YOLO Variants

Các biến thể YOLO hỗ trợ oriented bounding boxes:

**YOLOv5-OBB:**
- Extension của YOLOv5 với oriented output
- Predict (cx, cy, w, h, θ) thay vì (cx, cy, w, h)
- Angle loss với circular smoothing

**Rotated-RetinaNet:**
- RetinaNet với rotated anchors
- Focal loss cho class imbalance
- Hỗ trợ nhiều datasets: DOTA, HRSC2016

### 4.8.2. S²A-Net (Single Shot Alignment Network)

S²A-Net là oriented detector hiệu quả với feature alignment mechanism.

**Components:**
- Feature Alignment Module (FAM): Align features theo orientation
- Oriented Detection Module (ODM): Predict oriented boxes
- Active Rotating Filters: Learn rotation-invariant features

### 4.8.3. R³Det

R³Det sử dụng feature refinement cho oriented detection.

**Ý tưởng:**
- Initial horizontal detection
- Feature Refinement Module (FRM) để refine features
- Progressive regression từ horizontal sang oriented

## 4.9. Specialized Architectures cho Ship Detection

### 4.9.1. HO-ShipNet

HO-ShipNet (Hardware-Oriented Ship Network) được thiết kế cho on-board processing.

**Đặc điểm:**
- Lightweight CNN architecture
- Optimized cho FPGA/edge deployment
- 95% accuracy trên satellite imagery datasets
- Real-time processing capability

### 4.9.2. MSS-Net

MSS-Net (Multi-Scale Ship Network) focus vào multi-scale detection.

**Cải tiến:**
- +4.8% mAP so với Faster R-CNN
- +4.4% so với RetinaNet
- 26.7% giảm parameters cho lightweight version

### 4.9.3. CircleNet (xView3 Winner)

CircleNet là giải pháp đạt giải nhất xView3 challenge cho maritime detection.

**Innovations:**
- CenterNet-inspired với U-Net-like structure
- High-resolution output (stride-2)
- Custom SAR normalization (sigmoid activation)
- Reduced Focal Loss
- Label noise handling với entropy regularization

**Ensemble Strategy:**
12 models (3 backbones × 4 configurations):
- EfficientNet-B4, B5, V2S backbones
- Different image sizes và augmentations

## 4.10. Attention Mechanisms cho Ship Detection

### 4.10.1. CBAM (Convolutional Block Attention Module)

CBAM thêm channel và spatial attention vào CNN:
- Channel Attention: "what" to attend
- Spatial Attention: "where" to attend

Tích hợp CBAM vào backbone (ResNet, EfficientNet) cải thiện ship detection, đặc biệt cho small ships.

### 4.10.2. SE-Net (Squeeze-and-Excitation)

SE blocks recalibrate channel-wise features:
- Squeeze: Global average pooling
- Excitation: FC layers + sigmoid
- Scale: Multiply features với learned weights

SE-ResNet và SE-EfficientNet là backbones hiệu quả cho ship detection.

### 4.10.3. Coordinate Attention

Coordinate Attention encode spatial information vào channel attention:
- Capture long-range dependencies
- Preserve positional information
- Hiệu quả cho oriented objects như ships

## 4.11. So sánh và Lựa chọn Model

### 4.11.1. Bảng So sánh Performance

| Model | Dataset | mAP@0.5 | Speed (FPS) | Đặc điểm |
|-------|---------|---------|-------------|----------|
| Faster R-CNN + FPN | SSDD | 87% | ~15 | Baseline mạnh |
| YOLOv5l | SSDD | 94% | ~50 | Cân bằng tốt |
| YOLOv8m | SSDD | 95% | ~60 | State-of-the-art |
| Rotated FRCNN | HRSC2016 | 89% | ~10 | Oriented detection |
| S²A-Net | DOTA-Ship | 90% | ~12 | Oriented efficient |

### 4.11.2. Recommendations theo Use Case

**Surveillance (Real-time):**
- YOLOv8-nano hoặc YOLOv10-nano
- Sacrifice some accuracy for speed
- Suitable cho continuous monitoring

**High Accuracy (Offline Analysis):**
- Cascade R-CNN hoặc ensemble
- Rotated variants cho oriented ships
- Accept slower inference

**Oriented Ships (Port Monitoring):**
- RoI Transformer hoặc Oriented R-CNN
- S²A-Net cho speed-accuracy balance

**Edge Deployment:**
- YOLOv5-nano hoặc custom lightweight
- HO-ShipNet variants
- Consider FPGA/embedded optimization

**SAR-specific:**
- Models pre-trained trên SAR datasets
- AC-YOLO, GDB-YOLOv5s
- Attention mechanisms for speckle handling
