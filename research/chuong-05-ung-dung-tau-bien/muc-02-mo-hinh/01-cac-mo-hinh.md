# Chương 5: Các Model Áp dụng cho Ship Detection

## 5.5. Tổng quan Lựa chọn Model

Các mô hình object detection đã được trình bày chi tiết trong **Chương 3** có thể được áp dụng trực tiếp cho bài toán ship detection. Lựa chọn mô hình phù hợp phụ thuộc vào:

- **Accuracy vs Speed trade-off:** Real-time surveillance cần one-stage detectors (YOLO, RetinaNet), offline analysis cho phép two-stage (Faster R-CNN, Cascade R-CNN)
- **HBB vs OBB:** Tàu trong cảng/ven biển cần oriented detectors (Rotated Faster R-CNN, Oriented R-CNN)
- **Loại dữ liệu:** SAR vs optical imagery
- **Tài nguyên deployment:** Edge devices cần lightweight models
- **Dataset size:** Transfer learning từ TorchGeo weights (**Chương 3.5**)

## 5.6. Kiến trúc Thích hợp cho Ship Detection

### 5.6.1. Two-stage Detectors cho High Accuracy

**Faster R-CNN + FPN** (xem **Chương 3.2.1**)
- Baseline mạnh cho ship detection
- FPN xử lý tốt multi-scale ships (từ small fishing boats đến large cargo vessels)
- **Tuning quan trọng:** Anchor aspect ratios phải phù hợp với ships (3:1 đến 8:1 thay vì 1:1-2:1 như objects tự nhiên)
- **Transfer learning:** Sử dụng pretrained weights từ TorchGeo (**Chương 3.5**) cho satellite imagery
- **Kết quả:** mAP@0.5 85-90% trên SSDD dataset

**Cascade R-CNN** (xem **Chương 3.2.2**)
- Multi-stage refinement cho high-quality bounding boxes
- Phù hợp khi cần localization chính xác (ví dụ: ước tính kích thước tàu, kết hợp segmentation)
- IoU thresholds tăng dần (0.5 → 0.6 → 0.7) cải thiện precision

### 5.6.2. Oriented Detectors cho Dense Ports

**Rotated Faster R-CNN** (xem **Chương 3.2.3**)
- Predict oriented bounding boxes (cx, cy, w, h, θ)
- Giải quyết vấn đề HBB chứa nhiều background cho đối tượng dài và hẹp
- Sử dụng Rotated RoI Align để extract features
- Quan trọng cho scenarios: tàu trong cảng, tàu neo gần nhau

**Oriented R-CNN**
- Efficient oriented detection
- Midpoint offset representation cho stable angle regression
- Nhanh hơn RoI Transformer với accuracy tương đương

### 5.6.3. One-stage Detectors cho Real-time

**YOLO Family** (xem **Chương 3.3**)
- **YOLOv5:** Cân bằng tốt accuracy-speed, phổ biến nhất
  - YOLOv5l: mAP@0.5 92-95% trên SSDD, ~50 FPS
  - Anchor-based, cần clustering anchors trên ship dataset
- **YOLOv8:** Anchor-free, decoupled head
  - State-of-the-art cho ship detection
  - Custom training heads cho detection + classification
- **YOLOv10:** NMS-free design cho ultra-fast inference

**SAR-specific YOLO variants:**
- **AC-YOLO (2025):** Attention mechanisms cho SAR features
- **GDB-YOLOv5s:** Ghost convolution + BiFPN cho lightweight
- **YOLO-SD:** Multi-scale convolution cho small ships

### 5.6.4. Specialized Architectures

**CircleNet (xView3 Winner)** - xem **Chương 4.3.2**
- CenterNet-inspired cho SAR ship detection
- High-resolution output (stride-2)
- Custom SAR normalization
- Ensemble 12 models: 3 backbones × 4 configurations
- Top solution trong xView3 maritime challenge

**HO-ShipNet**
- Hardware-Oriented Network cho on-board satellite processing
- Lightweight CNN optimized cho FPGA/edge
- 95% accuracy với real-time capability

## 5.7. Transfer Learning và Pretrained Weights

### 5.7.1. From TorchGeo Models

**TorchGeo** (xem **Chương 3**) cung cấp pretrained weights đặc biệt phù hợp cho ship detection:

**Classification backbones** (Chương 3.2):
- ResNet50 pretrained on Sentinel-2 imagery
- Vision Transformers (ViT, Swin) pretrained on SSL4EO
- EfficientNet variants cho lightweight deployment

**Segmentation models** (Chương 3.3):
- U-Net variants có thể fine-tune cho ship instance segmentation
- DeepLabV3+ cho semantic segmentation maritime areas

**Change detection models** (Chương 3.4):
- Track ship movements across time-series imagery
- Detect new vessels appearing in monitored regions

### 5.7.2. Training Strategy

1. **Load pretrained backbone** từ TorchGeo (ResNet, ViT, Swin)
2. **Modify input layer** nếu cần (SAR: 2 channels VV+VH)
3. **Add detection head** (FPN, YOLO head, etc.)
4. **Fine-tune** trên ship detection dataset với:
   - Lower LR cho backbone layers
   - Higher LR cho detection head
   - Custom anchors cho ship aspect ratios
5. **Augmentation:** Rotation, flip, scale, mosaic, mixup

## 5.8. Multi-modal Fusion SAR + Optical

Một số approaches kết hợp cả SAR và optical imagery:

**Early fusion:** Concatenate SAR và optical channels trước backbone
**Late fusion:** Separate branches cho SAR và optical, merge features
**Cross-modal learning:** Transfer knowledge từ optical models sang SAR

## 5.9. Benchmarks và Kết quả

### 5.9.1. SAR Ship Detection (SSDD)

| Model | mAP@0.5 | Speed (FPS) | Notes |
|-------|---------|-------------|-------|
| Faster R-CNN + FPN | 87% | ~15 | Baseline |
| YOLOv5l | 94% | ~50 | Best balance |
| YOLOv8m | 95% | ~60 | SOTA |
| AC-YOLO | 96% | ~45 | SAR-specific |

### 5.9.2. Oriented Detection (HRSC2016)

| Model | mAP | Notes |
|-------|-----|-------|
| Rotated Faster R-CNN | 89% | Baseline OBB |
| S²A-Net | 95% | Feature alignment |
| Oriented R-CNN | 96% | Efficient |

### 5.9.3. xView3-SAR Challenge

| Solution | Aggregate Score | Key Components |
|----------|----------------|----------------|
| 1st (CircleNet) | 0.603 | Ensemble, stride-2, custom norm |
| 2nd | 0.589 | Multi-scale, attention |
| 3rd | 0.576 | Cascade, TTA |
| Baseline | 0.200 | Single model |

## 5.10. Recommendations

**Cho Maritime Surveillance (Real-time):**
- YOLOv8-nano hoặc YOLOv10 với custom ship anchors
- Pretrained weights từ TorchGeo Sentinel-1 models
- Edge optimization (TensorRT, ONNX)

**Cho Research và High Accuracy:**
- Cascade R-CNN với ResNet101-FPN backbone
- Pretrained on TorchGeo satellite datasets
- Ensemble multiple models

**Cho Port Monitoring (Oriented Ships):**
- Oriented R-CNN hoặc S²A-Net
- Custom OBB annotations
- Post-processing: Soft-NMS cho dense ships

**Cho IUU Fishing Detection:**
- Học từ xView3 top solutions (**Chương 4.3**)
- CircleNet-like architecture
- Multi-modal: SAR + AIS + bathymetry

---

**Tóm tắt:** Phần này đã khảo sát các mô hình object detection từ Chương 3 được áp dụng cho ship detection, với focus vào transfer learning từ TorchGeo và specialized adaptations cho maritime domain. Phần tiếp theo sẽ trình bày pipeline hoàn chỉnh từ data đến deployment.
