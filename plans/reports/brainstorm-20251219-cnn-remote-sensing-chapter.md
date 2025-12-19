# Brainstorm: CNN/Deep Learning trong Viễn thám - Chương Báo cáo Chi tiết

## Tóm tắt

Báo cáo này phân tích và đề xuất cấu trúc cho một chương chi tiết về ứng dụng CNN/Deep Learning trong viễn thám, tập trung vào hai bài toán: **phát hiện tàu (ship detection)** và **nhận dạng vết dầu loang (oil spill detection)**. Nội dung bao gồm giới thiệu CNN, phương pháp áp dụng, các model nổi tiếng từ TorchGeo, và datasets chuẩn quốc tế.

---

## 1. Cấu trúc Chương Đề xuất

```
1. Giới thiệu Mạng CNN (Convolutional Neural Network)
   1.1. Kiến trúc cơ bản của CNN
   1.2. Các thành phần chính: Convolution, Pooling, Activation
   1.3. Backbone networks: ResNet, VGG, EfficientNet

2. Phương pháp sử dụng CNN với ảnh vệ tinh
   2.1. Classification (Phân loại) → Mục đích và ứng dụng
   2.2. Object Detection (Phát hiện đối tượng) → Mục đích và ứng dụng
   2.3. Semantic Segmentation (Phân đoạn ngữ nghĩa) → Mục đích và ứng dụng
   2.4. Instance Segmentation → Mục đích và ứng dụng

3. Bài toán Phát hiện Tàu (Ship Detection)
   3.1. Đặc điểm bài toán
   3.2. Các model CNN phổ biến
   3.3. Quy trình phát hiện/nhận dạng
   3.4. Datasets nổi tiếng

4. Bài toán Nhận dạng Vết dầu loang (Oil Spill Detection)
   4.1. Đặc điểm bài toán
   4.2. Các model CNN phổ biến
   4.3. Quy trình phát hiện/phân đoạn
   4.4. Datasets nổi tiếng

5. TorchGeo - Thư viện Deep Learning cho Viễn thám
   5.1. Tổng quan TorchGeo
   5.2. Các model hỗ trợ chi tiết
   5.3. Pre-trained weights theo loại sensor
```

---

## 2. Nội dung Chi tiết

### 2.1. Giới thiệu về Mạng CNN

#### 2.1.1. Kiến trúc CNN Cơ bản

**CNN (Convolutional Neural Network)** là kiến trúc mạng nơ-ron sâu được thiết kế đặc biệt để xử lý dữ liệu có cấu trúc lưới (grid-like topology), điển hình là ảnh.

| Thành phần | Mô tả | Vai trò |
|------------|-------|---------|
| **Convolutional Layer** | Áp dụng bộ lọc (filter/kernel) trượt qua ảnh | Trích xuất đặc trưng cục bộ (edges, textures, patterns) |
| **Pooling Layer** | Giảm kích thước không gian (Max/Average Pooling) | Giảm tính toán, tăng tính bất biến với dịch chuyển |
| **Activation Function** | ReLU, Sigmoid, Softmax | Thêm tính phi tuyến cho model |
| **Fully Connected Layer** | Kết nối tất cả neurons | Tổng hợp đặc trưng cho classification |

#### 2.1.2. Backbone Networks phổ biến

| Backbone | Năm | Đặc điểm | Ứng dụng trong Viễn thám |
|----------|-----|----------|--------------------------|
| **VGG16/19** | 2014 | 16-19 layers, modular convolution | Baseline, feature extraction |
| **ResNet-50/101/152** | 2015 | Residual connections, giải quyết vanishing gradient | Backbone chính cho object detection, segmentation |
| **EfficientNet B0-B7** | 2019 | Compound scaling, cân bằng depth/width/resolution | State-of-the-art accuracy với ít parameters |
| **Swin Transformer** | 2021 | Hierarchical vision transformer | Kết hợp local + global context |

### 2.2. Phương pháp sử dụng CNN với ảnh vệ tinh

#### 2.2.1. Classification (Phân loại)

**Output:** Nhãn lớp cho toàn bộ ảnh hoặc tile

**Mục đích trong viễn thám:**
- **Land Cover Classification**: Phân loại lớp phủ mặt đất (rừng, nước, đô thị)
- **Scene Classification**: Phân loại cảnh (cảng biển, sân bay, khu dân cư)
- **Vessel Type Classification**: Phân loại loại tàu (tàu đánh cá, tàu hàng, tàu dầu)
- **Oil vs Look-alike Classification**: Phân biệt dầu loang với hiện tượng tương tự (sóng, tảo)

**Kiến trúc điển hình:** ResNet, EfficientNet, ViT (Vision Transformer)

**Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

#### 2.2.2. Object Detection (Phát hiện đối tượng)

**Output:** Bounding boxes + confidence scores + class labels

**Mục đích trong viễn thám:**
- **Ship Detection**: Phát hiện vị trí tàu trên biển
- **Aircraft Detection**: Phát hiện máy bay tại sân bay
- **Vehicle Counting**: Đếm phương tiện giao thông
- **Oil Platform Detection**: Phát hiện giàn khoan dầu

**Hai hướng tiếp cận chính:**

| Loại | Models | Đặc điểm |
|------|--------|----------|
| **Two-stage** | Faster R-CNN, Mask R-CNN, FPN | Accuracy cao, tốc độ chậm hơn |
| **One-stage** | YOLO series, SSD, RetinaNet | Real-time, trade-off accuracy |

**Metrics:** mAP (mean Average Precision), IoU (Intersection over Union), Precision-Recall curve

#### 2.2.3. Semantic Segmentation (Phân đoạn ngữ nghĩa)

**Output:** Mask pixel-level với nhãn lớp cho từng pixel

**Mục đích trong viễn thám:**
- **Oil Spill Segmentation**: Phân vùng chính xác vết dầu loang
- **Water Body Mapping**: Lập bản đồ sông, hồ, biển
- **Urban Footprint Extraction**: Trích xuất vùng đô thị
- **Flood Mapping**: Lập bản đồ ngập lụt

**Kiến trúc Encoder-Decoder:**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Encoder   │───▶│   Bottleneck │───▶│   Decoder   │
│  (Feature   │    │   (ASPP,    │    │  (Upsample, │
│  extraction)│    │   FPN, etc) │    │   recover)  │
└─────────────┘    └─────────────┘    └─────────────┘
      │                                       │
      └─────── Skip Connections ──────────────┘
```

**Models phổ biến:**

| Model | Đặc điểm | Ưu điểm |
|-------|----------|---------|
| **U-Net** | Symmetric encoder-decoder với skip connections | Bảo toàn chi tiết biên, ít dữ liệu train |
| **DeepLabV3+** | ASPP (Atrous Spatial Pyramid Pooling) | Multi-scale context, sharp boundaries |
| **FPN** | Feature Pyramid Network | Multi-scale detection |
| **PSPNet** | Pyramid Pooling Module | Global context aggregation |
| **HRNet** | High-Resolution representations | Duy trì resolution cao |

**Metrics:** IoU, mIoU, Dice Score, Pixel Accuracy

#### 2.2.4. Instance Segmentation

**Output:** Mask riêng biệt cho từng instance + bounding box + class

**Mục đích:** Phân biệt từng tàu riêng lẻ, từng tòa nhà riêng lẻ

**Models:** Mask R-CNN, YOLACT, SOLOv2

---

### 2.3. Bài toán Phát hiện Tàu (Ship Detection)

#### 2.3.1. Đặc điểm bài toán

| Thách thức | Mô tả |
|------------|-------|
| **Kích thước đa dạng** | Tàu nhỏ (vài pixel) đến tàu lớn (hàng trăm pixel) |
| **Môi trường phức tạp** | Sóng biển, bọt sóng, nhiễu SAR (speckle noise) |
| **Tàu gần bờ** | Khó phân biệt với cầu cảng, bến tàu |
| **Tàu dày đặc** | Nhiều tàu xếp chồng, occlusion |
| **Góc nghiêng** | Oriented bounding box cần thiết |

#### 2.3.2. Các Model CNN phổ biến

**A. Two-stage Detectors:**

| Model | Framework | Đặc điểm | Performance |
|-------|-----------|----------|-------------|
| **Faster R-CNN + FPN** | Detectron2, MMDetection | ResNet backbone + Feature Pyramid | mAP 85-90% trên SSDD |
| **Rotated Faster R-CNN** | MMRotate | Oriented bounding boxes | Phù hợp tàu có góc nghiêng |
| **Cascade R-CNN** | MMDetection | Multi-stage refinement | Accuracy cao hơn |

**B. One-stage Detectors (YOLO family):**

| Model | Năm | Đặc điểm | Hiệu năng SAR |
|-------|-----|----------|---------------|
| **YOLOv5** | 2020 | Anchor-based, CSPDarknet | mAP 92-95% SSDD |
| **YOLOv7-LDS** | 2023 | Lightweight, SAR-optimized | 26.7% ít parameters |
| **YOLOv8** | 2023 | Anchor-free, decoupled head | SOTA accuracy |
| **YOLOv9** | 2024 | GELAN + PGI | First application for ships |
| **YOLOv10/11** | 2024-25 | NMS-free, TinyML ready | 10.34ms inference |
| **AC-YOLO** | 2025 | YOLO11-based, SAR specific | Latest architecture |

**C. Specialized Architectures:**

| Model | Nguồn | Đặc điểm |
|-------|-------|----------|
| **HO-ShipNet** | Literature | Hardware-oriented CNN, 95% accuracy |
| **MSS-Net** | Literature | +4.8% mAP vs Faster R-CNN |
| **YOLO-SD** | MDPI | Multi-scale convolution + Feature Transformer |

#### 2.3.3. Quy trình Phát hiện Tàu

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        SHIP DETECTION PIPELINE                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA ACQUISITION                                                          │
│    ├─ Optical: WorldView-3 (0.3m), Planet (3-5m), Sentinel-2 (10m)          │
│    └─ SAR: Sentinel-1 (5-40m), ICEYE (0.25m), TerraSAR-X (1m)               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING                                                             │
│    ├─ SAR: Speckle filtering, Sigma0 calibration, VV/VH polarization        │
│    ├─ Optical: Atmospheric correction, pan-sharpening                        │
│    ├─ Tiling: 512×512 hoặc 800×800 với overlap (10-20%)                     │
│    └─ Augmentation: Mosaic, RandomAffine, MixUp, RandomFlip, Color jitter   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. MODEL SELECTION                                                           │
│    ├─ High accuracy needed → Faster R-CNN / Cascade R-CNN                    │
│    ├─ Real-time detection → YOLOv8 / YOLOv10                                 │
│    ├─ Oriented boxes needed → Rotated RCNN / Oriented YOLO                   │
│    └─ Edge deployment → YOLOv10-nano / AC-YOLO lightweight                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. TRAINING                                                                  │
│    ├─ Backbone: ResNet50/EfficientNet pretrained on ImageNet                │
│    ├─ Loss: Focal Loss (class imbalance), CIoU/DIoU (bbox regression)       │
│    ├─ Optimizer: SGD with momentum / AdamW                                   │
│    └─ Learning rate: Cosine annealing, warmup                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. INFERENCE                                                                 │
│    ├─ Sliding window: 2048×2048 tiles với 1536px stride overlap             │
│    ├─ TTA (Test Time Augmentation): Flip, rotation                          │
│    ├─ NMS (Non-Maximum Suppression): IoU threshold 0.5-0.7                   │
│    └─ Confidence threshold: 0.3-0.5                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. POST-PROCESSING                                                           │
│    ├─ AIS correlation (nếu có)                                               │
│    ├─ Vessel classification: Fishing / Non-fishing / Infrastructure         │
│    └─ Tracking across temporal images                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.3.4. Datasets Phát hiện Tàu Nổi tiếng

| Dataset | Nguồn | Số ảnh | Số tàu | Loại | Đặc điểm |
|---------|-------|--------|--------|------|----------|
| **SSDD** | Sentinel-1, TerraSAR-X, RadarSat-2 | 1,160 | 2,456 | SAR | 1-15m resolution, đa phân cực |
| **HRSID** | Sentinel-1, TerraSAR-X | 5,604 | 16,951 | SAR | 800×800 pixels, high quality |
| **xView3-SAR** | Sentinel-1 | 991 scenes | 243,018 | SAR | 1,400 gigapixels, IUU fishing focus |
| **HRSC2016** | Optical satellite | 1,070 | 2,976 | Optical | Rotated bbox, 3-level class hierarchy |
| **HRSC2016-MS** | Extended HRSC | 1,680 | 7,655 | Optical | Multi-scale ships |
| **ShipRSImageNet** | Multiple sources | 3,435 | 17,573 | Optical | 50 ship categories, 4-level hierarchy |
| **DOTA** | Aerial/Satellite | 2,806 | ~10K ships | Optical | 15 object classes including ships |
| **FGSD** | 17 major ports | 2,612 | 5,634 | Optical | 43 fine-grained ship classes |
| **xView1** | WorldView-3 | 847 | ~100K | Optical | 60 classes, ships included |

---

### 2.4. Bài toán Nhận dạng Vết dầu loang (Oil Spill Detection)

#### 2.4.1. Đặc điểm bài toán

| Thách thức | Mô tả |
|------------|-------|
| **Look-alikes** | Dầu loang vs sóng thấp, tảo biển, film sinh học |
| **Shape không đều** | Vết dầu có hình dạng bất định, thay đổi theo thời gian |
| **Kích thước đa dạng** | Từ vài m² đến hàng km² |
| **Điều kiện môi trường** | Gió, sóng ảnh hưởng đến độ tương phản SAR |
| **Dữ liệu hiếm** | Sự kiện tràn dầu không thường xuyên |

#### 2.4.2. Tại sao dùng SAR?

SAR (Synthetic Aperture Radar) là lựa chọn hàng đầu cho oil spill detection:

| Ưu điểm SAR | Giải thích |
|-------------|------------|
| **All-weather** | Hoạt động trong mây, mưa, ban đêm |
| **Contrast rõ** | Dầu làm giảm độ nhám bề mặt → dark spot trên ảnh SAR |
| **Sentinel-1 miễn phí** | Global coverage, 10m resolution, near-real-time |
| **VV polarization** | Tối ưu cho phát hiện dầu |

#### 2.4.3. Các Model CNN phổ biến

**A. Semantic Segmentation Models:**

| Model | Đặc điểm | Hiệu năng |
|-------|----------|-----------|
| **U-Net** | Encoder-decoder với skip connections | IoU 96% (optimized) |
| **DeepLabV3+** | ASPP + decoder, EfficientNet backbone | 98.14% accuracy, MIoU 0.7872 |
| **Dual Attention U-Net** | Channel + Position Attention Maps | Improved boundary detection |
| **FPN** | Multi-scale feature fusion | Robust với kích thước khác nhau |

**B. Two-stage Approach (Detection + Segmentation):**

| Stage | Model | Mô tả |
|-------|-------|-------|
| **Stage 1: Classification** | 23-layer CNN | Phân loại có/không có dầu |
| **Stage 2: Segmentation** | 5-stage U-Net | Phân vùng chính xác vết dầu |

**C. Hybrid CNN-Transformer:**

| Model | Đặc điểm |
|-------|----------|
| **CNN + ViT** | CNN cho local features, ViT cho global context |
| **DaNet** | Dual attention network cho UAV/SAR |

**D. Detection Models (Bounding Box):**

| Model | Ứng dụng |
|-------|----------|
| **YOLOv4** | Object detection cho oil spill |
| **Faster R-CNN** | Two-stage detection |

#### 2.4.4. Quy trình Nhận dạng Vết dầu loang

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      OIL SPILL DETECTION PIPELINE                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. SAR DATA ACQUISITION                                                      │
│    ├─ Sentinel-1: VV + VH polarization, 10m resolution                       │
│    ├─ Near-real-time (1-3 hours from acquisition)                            │
│    └─ GRD (Ground Range Detected) product                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. SAR PREPROCESSING                                                         │
│    ├─ Orbit correction                                                       │
│    ├─ Thermal noise removal                                                  │
│    ├─ Radiometric calibration → Sigma0 (dB)                                  │
│    ├─ Speckle filtering: Lee, Frost, Gamma-MAP                              │
│    ├─ Geometric terrain correction                                           │
│    └─ Land masking                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. TRADITIONAL DETECTION (Optional baseline)                                 │
│    ├─ Thresholding: Adaptive threshold trên Sigma0                          │
│    ├─ CFAR: Constant False Alarm Rate                                        │
│    └─ Edge detection: Sobel, Canny                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. DEEP LEARNING DETECTION                                                   │
│    ├─ Model: DeepLabV3+ / U-Net / Dual Attention U-Net                      │
│    ├─ Backbone: EfficientNet / ResNet50                                      │
│    ├─ Input: 2048×2048 tiles (Sigma0 dB)                                     │
│    ├─ Output: Binary mask (oil / background) hoặc                            │
│    │          Multi-class (oil / look-alike / sea / land / ship)            │
│    └─ Loss: Dice + Focal + CrossEntropy                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. POST-PROCESSING                                                           │
│    ├─ Morphological operations: erosion, dilation, opening, closing         │
│    ├─ Connected component analysis                                           │
│    ├─ Small region filtering (min area threshold)                            │
│    └─ Confidence thresholding                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. LOOK-ALIKE DISCRIMINATION                                                 │
│    ├─ Feature extraction: shape, texture, contrast                           │
│    ├─ Ancillary data: wind speed, sea state                                  │
│    ├─ Classifier: CNN / Random Forest                                        │
│    └─ Expert validation                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. OUTPUT & REPORTING                                                        │
│    ├─ GeoTIFF mask với coordinate                                            │
│    ├─ Area estimation (km²)                                                  │
│    ├─ Alert generation                                                        │
│    └─ Integration với CleanSeaNet / response systems                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.4.5. Datasets Vết dầu loang Nổi tiếng

| Dataset | Nguồn | Số ảnh | Định dạng | Đặc điểm |
|---------|-------|--------|-----------|----------|
| **EMSA CleanSeaNet** | European Maritime Safety Agency | 1,112 scenes | Sentinel-1 RGB | Pixel-level labels: oil, look-alike, sea, land, ship |
| **Zenodo Oil Spill Part I** | Sentinel-1 | 1,200 images | 2048×2048 TIFF Sigma0 | Training + validation, ground truth masks |
| **Zenodo Oil Spill Part II** | Sentinel-1 | 685 images | 2048×2048 TIFF | Oil-free + look-alike images |
| **Zenodo Oil Spill Part III** | Sentinel-1 | 450 images | 2048×2048 TIFF | Test set: 150 look-alike, 150 oil-free, 150 oil spill |
| **Egyptian Dataset** | Suez Canal | 1,500 incidents | Sentinel-1 | Localized dataset, MIoU 0.7872 |
| **Mediterranean YOLOv4** | Literature | 5,930 images | Sentinel-1 | 9,768 oil spills (2015-2018) |
| **Peruvian Dataset** | Peru waters | 2,112 patches | 512×512 from 40 scenes | 2014-2024, harmonized với CleanSeaNet |

---

### 2.5. TorchGeo - Deep Learning cho Viễn thám

#### 2.5.1. Tổng quan TorchGeo

[TorchGeo](https://torchgeo.readthedocs.io/) là thư viện PyTorch cho deep learning với dữ liệu địa không gian (geospatial data), cung cấp:

- **Datasets**: 70+ datasets viễn thám chuẩn
- **Pre-trained Models**: Weights cho nhiều loại sensor
- **Transforms**: Augmentation phù hợp với ảnh vệ tinh
- **Samplers**: Xử lý ảnh lớn (tiling, sampling)

#### 2.5.2. Các Model có trong TorchGeo

TorchGeo cung cấp **20+ model architectures**:

**A. Classification & Feature Extraction:**

| Model | Class | Mô tả | Ứng dụng Ship/Oil Spill |
|-------|-------|-------|-------------------------|
| **ResNet** | `ResNet18`, `ResNet50`, `ResNet101`, `ResNet152` | Residual networks với skip connections | Backbone cho detection, classification |
| **Vision Transformer (ViT)** | `ViTSmall14`, `ViTBase14`, `ViTLarge14`, `ViTHuge14` | Transformer-based image classification | Classification, feature extraction |
| **Swin Transformer** | `SwinV2-B`, etc. | Hierarchical vision transformer | Multi-scale feature extraction |

**B. Semantic Segmentation:**

| Model | Class | Mô tả | Ứng dụng |
|-------|-------|-------|----------|
| **U-Net** | `Unet` | Encoder-decoder với skip connections | Oil spill segmentation |
| **FCN** | `FCN` | Fully Convolutional Network | General segmentation |
| **FarSeg** | `FarSeg` | Foreground-Aware Relation Network | High-resolution segmentation |

**C. Change Detection:**

| Model | Class | Mô tả |
|-------|-------|-------|
| **ChangeStar** | `ChangeStar` | Siamese-based change detection |
| **Be The Change (BTC)** | `BTC` | Change detection network |
| **ChangeViT** | `ChangeViT` | Vision Transformer cho change detection |
| **FC-Siamese Networks** | `FCSiamConc`, `FCSiamDiff` | Fully convolutional siamese |

**D. Foundation Models (Transfer Learning):**

| Model | Class | Đặc điểm |
|-------|-------|----------|
| **Copernicus-FM** | `Copernicus_FM` | Explicit spatial/temporal/spectral support |
| **CROMA** | `CROMA` | Vision Transformer variant |
| **DOFA** | `DOFA` | Masked autoencoder approach |
| **Scale-MAE** | `ScaleMAE` | Scale-flexible masked autoencoder |
| **Panopticon** | `Panopticon` | Dynamic support foundation model |

**E. Specialized Models:**

| Model | Class | Mô tả |
|-------|-------|-------|
| **Aurora** | `Aurora` | Weather/atmospheric forecasting |
| **EarthLoc** | `EarthLoc` | Geo-localization |
| **ConvLSTM** | `ConvLSTM` | Temporal sequence modeling |
| **L-TAE** | `LTAE` | Temporal attention encoder |
| **MOSAIKS** | `MOSAIKS` | Multi-spectral representation learning |

#### 2.5.3. Pre-trained Weights theo Sensor

**Rất quan trọng cho Ship Detection và Oil Spill Detection:**

| Sensor | Loại | Pre-trained Models | Channels | Ứng dụng |
|--------|------|---------------------|----------|----------|
| **Sentinel-1** | SAR | ResNet50, ViT (S/B/L/H), DINOv2 | 2 (VV, VH) | **Ship detection, Oil spill detection** |
| **Sentinel-2** | Multispectral | ResNet18/50/152, ViT, U-Net, EarthLoc | 13 bands | Land cover, coastal monitoring |
| **Landsat** | Multispectral | ResNet18/50, ViT | 7-11 bands | Large-scale analysis |
| **NAIP** | Aerial RGB | Swin V2-B | 3 (RGB) | High-resolution detection |

**Chi tiết Sentinel-1 Pre-trained Weights:**

| Model | Weight Name | Đặc điểm |
|-------|-------------|----------|
| ResNet50 | `Sentinel1_All_Moco_Weights` | MoCo self-supervised |
| ViT-Small-14 | `Sentinel1_All_SatlasPretrain_Weights` | Satlas pretrained |
| ViT-Base-14 | `Sentinel1_All_SatlasPretrain_Weights` | Satlas pretrained |
| ViT-Large-14 | `Sentinel1_All_SatlasPretrain_Weights` | Satlas pretrained |
| ViT-Huge-14 | `Sentinel1_All_SatlasPretrain_Weights` | Satlas pretrained |
| DINOv2 variants | Multiple | Self-supervised, versatile |

#### 2.5.4. Utility Functions

```python
from torchgeo.models import get_model, get_model_weights, list_models

# List tất cả models available
models = list_models()
print(models)  # ['resnet18', 'resnet50', 'vit_small_patch14_dinov2', ...]

# Get model với pretrained weights
model = get_model('resnet50', weights='Sentinel1_All_Moco_Weights')

# Get weights enum
weights = get_model_weights('resnet50')
```

#### 2.5.5. Đề xuất Model cho Ship Detection và Oil Spill Detection

**Ship Detection (Object Detection):**

| Nhiệm vụ | Model đề xuất | Backbone từ TorchGeo | Ghi chú |
|----------|---------------|----------------------|---------|
| **SAR Ship Detection** | YOLOv8 / Faster R-CNN | ResNet50 (Sentinel1_All_Moco_Weights) | Transfer learning từ TorchGeo |
| **Classification** | ViT-Base-14 | Sentinel1_All_SatlasPretrain_Weights | Vessel type classification |
| **Multi-scale** | FPN + ResNet50 | Sentinel1 weights | Handle ship size variation |

**Oil Spill Detection (Segmentation):**

| Nhiệm vụ | Model đề xuất | Backbone từ TorchGeo | Ghi chú |
|----------|---------------|----------------------|---------|
| **Binary Segmentation** | U-Net | ResNet50 Sentinel1 weights | Oil vs background |
| **Multi-class Segmentation** | DeepLabV3+ | ResNet50/EfficientNet | Oil, look-alike, water, land |
| **Foundation Model Transfer** | Fine-tune DOFA/CROMA | Pre-trained on multiple sensors | Khi ít labeled data |

---

## 3. Tài liệu Tham khảo

### Papers
- [Ship Detection with Deep Learning in Optical Remote-Sensing Images: A Survey](https://www.mdpi.com/2072-4292/16/7/1145)
- [Deep CNNs for Ship Detection Using DOTA and TGRS-HRRSD](https://www.sciencedirect.com/science/article/abs/pii/S027311772401055X)
- [Automated Oil Spill Detection Using Deep Learning and SAR](https://www.nature.com/articles/s41598-025-03028-1)
- [Oil Spill Identification from Satellite Images Using DNNs](https://www.mdpi.com/2072-4292/11/15/1762)
- [Improved Semantic Segmentation Based on DeepLabv3+](https://www.nature.com/articles/s41598-024-60375-1)
- [U-Net Ensemble for Enhanced Semantic Segmentation](https://www.mdpi.com/2072-4292/16/12/2077)

### Datasets
- [Satellite Imagery Datasets Containing Ships (GitHub)](https://github.com/jasonmanesis/Satellite-Imagery-Datasets-Containing-Ships)
- [xView3-SAR: Detecting Dark Fishing Activity](https://proceedings.neurips.cc/paper_files/paper/2022/file/f4d4a021f9051a6c18183b059117e8b5-Paper-Datasets_and_Benchmarks.pdf)
- [Zenodo Sentinel-1 SAR Oil Spill Dataset Part I](https://zenodo.org/records/8346860)
- [Zenodo Sentinel-1 SAR Oil Spill Dataset Part II](https://zenodo.org/records/8253899)
- [Zenodo Sentinel-1 SAR Oil Spill Dataset Part III](https://zenodo.org/records/13761290)
- [ShipRSImageNet (GitHub)](https://github.com/zzndream/ShipRSImageNet)
- [HRSC2016-MS Dataset](https://datasetninja.com/hrsc2016-ms)

### Libraries & Tools
- [TorchGeo Models Documentation](https://torchgeo.readthedocs.io/en/stable/api/models.html)
- [Segmentation Models PyTorch](https://smp.readthedocs.io/en/latest/models.html)
- [Ship Detection on SAR Data (GitHub)](https://github.com/jasonmanesis/Ship-Detection-on-Remote-Sensing-Synthetic-Aperture-Radar-Data)
- [MMRotate for Oriented Detection](https://github.com/open-mmlab/mmrotate)

### Commercial/Government Resources
- [Copernicus Data Space (Sentinel-1)](https://dataspace.copernicus.eu/)
- [EMSA CleanSeaNet](https://www.emsa.europa.eu/csn-menu.html)

---

## 4. Các câu hỏi cần làm rõ

1. **Phạm vi chương**: Chương này thuộc báo cáo/luận văn nào? Cần biết context để điều chỉnh độ sâu kỹ thuật.
2. **Ảnh vệ tinh cụ thể**: Dự án sử dụng Sentinel-1 SAR hay optical imagery?
3. **Bài toán ưu tiên**: Ship detection hay oil spill detection là focus chính?
4. **Phần thực nghiệm**: Có cần viết phần so sánh thực nghiệm các model không?
5. **TorchGeo chi tiết**: Có cần code examples cho việc sử dụng TorchGeo models không?

---

*Generated: 2025-12-19*
*Research scope: CNN/Deep Learning in Remote Sensing for Ship and Oil Spill Detection*
