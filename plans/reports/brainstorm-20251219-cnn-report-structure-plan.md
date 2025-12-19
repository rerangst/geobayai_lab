# Plan: Báo cáo CNN/Deep Learning trong Viễn thám

## Tóm tắt Yêu cầu

| Yêu cầu | Chi tiết |
|---------|----------|
| **Định dạng** | Nhiều file .md riêng biệt → Pipeline VitePress + Pandoc → DOCX |
| **Ngôn ngữ** | Tiếng Việt, giữ nguyên thuật ngữ tiếng Anh (polygon, bounding box, encoder, decoder...) |
| **Độ dài mỗi mục** | >2000 từ |
| **Văn phong** | Học thuật, lời văn chi tiết, hạn chế code |
| **Hình ảnh** | Ảnh từ dataset + Sơ đồ mô tả (Mermaid) |
| **Tích hợp** | Vào `docs/` folder → VitePress → Pandoc DOCX |

---

## Cấu trúc Thư mục Đề xuất

```
docs/
├── cnn-remote-sensing/                    # Folder mới cho báo cáo CNN
│   ├── README.md                          # Mục lục chương
│   │
│   ├── 01-introduction/
│   │   └── gioi-thieu-cnn-deep-learning.md
│   │
│   ├── 02-cnn-fundamentals/
│   │   ├── kien-truc-cnn-co-ban.md
│   │   └── backbone-networks-resnet-vgg-efficientnet.md
│   │
│   ├── 03-cnn-satellite-methods/
│   │   ├── phan-loai-anh-classification.md
│   │   ├── phat-hien-doi-tuong-object-detection.md
│   │   ├── phan-doan-ngu-nghia-segmentation.md
│   │   └── instance-segmentation.md
│   │
│   ├── 04-ship-detection/
│   │   ├── dac-diem-bai-toan-ship-detection.md
│   │   ├── cac-model-phat-hien-tau.md
│   │   ├── quy-trinh-ship-detection-pipeline.md
│   │   └── datasets-ship-detection.md
│   │
│   ├── 05-oil-spill-detection/
│   │   ├── dac-diem-bai-toan-oil-spill.md
│   │   ├── cac-model-phat-hien-dau-loang.md
│   │   ├── quy-trinh-oil-spill-pipeline.md
│   │   └── datasets-oil-spill-detection.md
│   │
│   ├── 06-torchgeo-models/
│   │   ├── tong-quan-torchgeo.md
│   │   ├── classification-models.md
│   │   ├── segmentation-models.md
│   │   ├── change-detection-models.md
│   │   └── pretrained-weights-sensors.md
│   │
│   └── 07-conclusion/
│       └── ket-luan-va-huong-phat-trien.md
│
└── images/
    └── cnn-remote-sensing/               # Folder ảnh minh họa
        ├── diagrams/                     # Sơ đồ Mermaid export PNG
        ├── datasets/                     # Ảnh từ dataset
        │   ├── ship-detection/
        │   └── oil-spill/
        └── architectures/                # Sơ đồ kiến trúc model
```

---

## Nội dung Chi tiết Từng File

### Chương 1: Giới thiệu (1 file, ~2500 từ)

#### `01-introduction/gioi-thieu-cnn-deep-learning.md`

**Outline:**
1. Bối cảnh: Deep Learning trong viễn thám
2. Tại sao CNN phù hợp với ảnh vệ tinh
3. Lịch sử phát triển: từ LeNet → ResNet → Transformer
4. Các bài toán chính: Classification, Detection, Segmentation
5. Mục tiêu báo cáo: Tập trung Ship Detection + Oil Spill Detection

**Hình ảnh yêu cầu:**
- Sơ đồ timeline phát triển CNN (Mermaid)
- Ví dụ ảnh vệ tinh với các bài toán khác nhau

---

### Chương 2: Kiến trúc CNN (2 files, ~5000 từ)

#### `02-cnn-fundamentals/kien-truc-cnn-co-ban.md` (~2500 từ)

**Outline:**
1. Khái niệm Convolutional Neural Network
2. Các thành phần cơ bản:
   - Convolutional Layer: filter, stride, padding
   - Pooling Layer: Max Pooling, Average Pooling
   - Activation Function: ReLU, Sigmoid, Softmax
   - Fully Connected Layer
3. Quá trình forward propagation
4. Training: Backpropagation, Loss function, Optimizer
5. Overfitting và Regularization: Dropout, Batch Normalization

**Hình ảnh yêu cầu:**
- Sơ đồ kiến trúc CNN cơ bản (Mermaid)
- Minh họa convolution operation
- Minh họa pooling

#### `02-cnn-fundamentals/backbone-networks-resnet-vgg-efficientnet.md` (~2500 từ)

**Outline:**
1. VGGNet: Modular convolution, deep architecture
2. ResNet: Residual connections, skip connections
3. EfficientNet: Compound scaling
4. Swin Transformer: Hierarchical vision transformer
5. So sánh hiệu năng các backbone

**Hình ảnh yêu cầu:**
- Sơ đồ so sánh kiến trúc VGG, ResNet, EfficientNet
- Bảng so sánh accuracy/parameters/FLOPs

---

### Chương 3: Phương pháp CNN với ảnh vệ tinh (4 files, ~10000 từ)

#### `03-cnn-satellite-methods/phan-loai-anh-classification.md` (~2500 từ)

**Outline:**
1. Định nghĩa bài toán Classification
2. Output: Class label + confidence score
3. Ứng dụng trong viễn thám:
   - Land Cover Classification
   - Scene Classification
   - Vessel Type Classification
4. Kiến trúc điển hình
5. Loss function: CrossEntropy, Focal Loss
6. Metrics: Accuracy, Precision, Recall, F1

**Hình ảnh yêu cầu:**
- Ví dụ ảnh vệ tinh với scene classification
- Confusion matrix ví dụ

#### `03-cnn-satellite-methods/phat-hien-doi-tuong-object-detection.md` (~2500 từ)

**Outline:**
1. Định nghĩa bài toán Object Detection
2. Output: Bounding box + class + confidence
3. Two-stage detectors: Faster R-CNN, Cascade R-CNN
4. One-stage detectors: YOLO, SSD, RetinaNet
5. Feature Pyramid Network (FPN)
6. Anchor-based vs Anchor-free
7. Loss: Classification loss + Regression loss
8. Metrics: mAP, IoU, Precision-Recall curve

**Hình ảnh yêu cầu:**
- Sơ đồ so sánh two-stage vs one-stage
- Ví dụ ảnh với bounding box từ dataset
- Minh họa IoU

#### `03-cnn-satellite-methods/phan-doan-ngu-nghia-segmentation.md` (~2500 từ)

**Outline:**
1. Định nghĩa Semantic Segmentation
2. Output: Pixel-level mask
3. Kiến trúc Encoder-Decoder:
   - Encoder: Feature extraction
   - Decoder: Upsampling, resolution recovery
   - Skip connections
4. Các model phổ biến:
   - U-Net: Architecture, skip connections
   - DeepLabV3+: ASPP, atrous convolution
   - FPN: Multi-scale features
   - PSPNet: Pyramid Pooling
5. Ứng dụng: Oil spill, flood mapping, urban extraction
6. Metrics: IoU, mIoU, Dice Score, Pixel Accuracy

**Hình ảnh yêu cầu:**
- Sơ đồ U-Net architecture (Mermaid)
- Sơ đồ DeepLabV3+ với ASPP
- Ví dụ segmentation mask từ dataset

#### `03-cnn-satellite-methods/instance-segmentation.md` (~2500 từ)

**Outline:**
1. Khác biệt với Semantic Segmentation
2. Output: Instance mask + bounding box + class
3. Mask R-CNN: Architecture
4. YOLACT, SOLOv2
5. Ứng dụng: Đếm từng tàu, từng tòa nhà
6. So sánh với Detection + Segmentation riêng

**Hình ảnh yêu cầu:**
- So sánh semantic vs instance segmentation
- Ví dụ instance mask

---

### Chương 4: Phát hiện Tàu (4 files, ~10000 từ)

#### `04-ship-detection/dac-diem-bai-toan-ship-detection.md` (~2500 từ)

**Outline:**
1. Giới thiệu bài toán Ship Detection
2. Tầm quan trọng: Hàng hải, an ninh, IUU fishing
3. Thách thức:
   - Kích thước đa dạng (small object detection)
   - Môi trường phức tạp: sóng, bọt, nhiễu SAR
   - Tàu gần bờ, cầu cảng
   - Oriented bounding box
   - Dense ships, occlusion
4. SAR vs Optical imagery
5. Ưu điểm SAR: All-weather, day-night

**Hình ảnh yêu cầu:**
- Ảnh SAR với tàu từ Sentinel-1
- Ảnh optical với tàu từ WorldView
- Minh họa các thách thức

#### `04-ship-detection/cac-model-phat-hien-tau.md` (~2500 từ)

**Outline:**
1. Two-stage detectors cho Ship Detection:
   - Faster R-CNN + FPN + ResNet
   - Rotated Faster R-CNN
   - Cascade R-CNN
2. One-stage detectors (YOLO family):
   - YOLOv5, YOLOv7, YOLOv8
   - YOLOv9, YOLOv10/11
   - AC-YOLO (2025, SAR-specific)
3. Specialized architectures:
   - HO-ShipNet
   - MSS-Net
   - YOLO-SD
4. Rotated/Oriented detection: R-CNN với RoI Rotate
5. So sánh hiệu năng

**Hình ảnh yêu cầu:**
- Bảng so sánh models
- Sơ đồ YOLO architecture

#### `04-ship-detection/quy-trinh-ship-detection-pipeline.md` (~2500 từ)

**Outline:**
1. Data Acquisition:
   - SAR: Sentinel-1, TerraSAR-X, ICEYE
   - Optical: WorldView, Planet
2. Preprocessing:
   - SAR: Speckle filtering, Sigma0 calibration
   - Optical: Atmospheric correction
   - Tiling strategies
3. Data Augmentation: Mosaic, MixUp, RandomFlip
4. Model Training:
   - Backbone selection
   - Loss function: Focal Loss, CIoU
   - Optimizer, learning rate
5. Inference:
   - Sliding window
   - TTA (Test Time Augmentation)
   - NMS (Non-Maximum Suppression)
6. Post-processing: AIS correlation, tracking

**Hình ảnh yêu cầu:**
- Flowchart pipeline hoàn chỉnh (Mermaid)
- Ví dụ preprocessing SAR

#### `04-ship-detection/datasets-ship-detection.md` (~2500 từ)

**Outline:**
1. SAR Datasets:
   - SSDD (SAR Ship Detection Dataset)
   - HRSID
   - xView3-SAR
2. Optical Datasets:
   - HRSC2016
   - ShipRSImageNet
   - DOTA (ships category)
   - FGSD
   - xView1
3. So sánh chi tiết: số lượng, resolution, annotation type
4. Cách download và sử dụng
5. Benchmark results

**Hình ảnh yêu cầu:**
- Bảng so sánh datasets
- Sample images từ mỗi dataset
- Distribution chart

---

### Chương 5: Phát hiện Dầu loang (4 files, ~10000 từ)

#### `05-oil-spill-detection/dac-diem-bai-toan-oil-spill.md` (~2500 từ)

**Outline:**
1. Giới thiệu bài toán Oil Spill Detection
2. Tầm quan trọng: Môi trường, sinh thái, pháp luật
3. Thách thức:
   - Look-alikes: sóng thấp, tảo, film sinh học
   - Shape không đều, thay đổi theo thời gian
   - Kích thước đa dạng
   - Điều kiện môi trường: gió, sóng
4. Tại sao dùng SAR:
   - All-weather
   - Contrast rõ (dark spot)
   - Sentinel-1 miễn phí
5. VV vs VH polarization

**Hình ảnh yêu cầu:**
- Ảnh SAR với dầu loang từ Sentinel-1
- So sánh oil spill vs look-alike
- Minh họa cơ chế SAR detect oil

#### `05-oil-spill-detection/cac-model-phat-hien-dau-loang.md` (~2500 từ)

**Outline:**
1. Semantic Segmentation models:
   - U-Net (IoU 96%)
   - DeepLabV3+ (98.14% accuracy)
   - Dual Attention U-Net
   - FPN
2. Two-stage approach: Classification → Segmentation
3. Hybrid CNN-Transformer
4. Detection models (bounding box): YOLOv4, Faster R-CNN
5. So sánh hiệu năng
6. Backbone selection cho SAR

**Hình ảnh yêu cầu:**
- Sơ đồ U-Net cho oil spill
- Sơ đồ Dual Attention mechanism
- Bảng so sánh models

#### `05-oil-spill-detection/quy-trinh-oil-spill-pipeline.md` (~2500 từ)

**Outline:**
1. SAR Data Acquisition: Sentinel-1 GRD
2. SAR Preprocessing:
   - Orbit correction
   - Thermal noise removal
   - Radiometric calibration → Sigma0 (dB)
   - Speckle filtering: Lee, Frost, Gamma-MAP
   - Geometric terrain correction
   - Land masking
3. Traditional detection baseline: CFAR, thresholding
4. Deep Learning detection:
   - Model: DeepLabV3+ / U-Net
   - Input: 2048×2048 tiles
   - Output: Binary/Multi-class mask
5. Post-processing:
   - Morphological operations
   - Connected component analysis
   - Confidence thresholding
6. Look-alike discrimination
7. Output & Reporting

**Hình ảnh yêu cầu:**
- Flowchart pipeline hoàn chỉnh (Mermaid)
- Ví dụ SAR preprocessing steps
- Ví dụ output mask

#### `05-oil-spill-detection/datasets-oil-spill-detection.md` (~2500 từ)

**Outline:**
1. EMSA CleanSeaNet dataset
2. Zenodo Sentinel-1 SAR Oil Spill Dataset (Part I, II, III)
3. Egyptian/Mediterranean datasets
4. So sánh chi tiết
5. Annotation format: pixel-level mask
6. Cách download và sử dụng
7. Benchmark results

**Hình ảnh yêu cầu:**
- Bảng so sánh datasets
- Sample images với ground truth
- Geographic coverage map

---

### Chương 6: TorchGeo Models (5 files, ~12500 từ)

#### `06-torchgeo-models/tong-quan-torchgeo.md` (~2500 từ)

**Outline:**
1. Giới thiệu TorchGeo
2. Tính năng chính:
   - Datasets: 70+ datasets viễn thám
   - Pre-trained Models
   - Transforms
   - Samplers
3. Cài đặt và sử dụng cơ bản
4. Integration với PyTorch ecosystem
5. So sánh với các thư viện khác

**Hình ảnh yêu cầu:**
- Sơ đồ kiến trúc TorchGeo
- Code snippet (minimal, focus on usage)

#### `06-torchgeo-models/classification-models.md` (~2500 từ)

**Outline:**
1. ResNet trong TorchGeo:
   - ResNet18, ResNet50, ResNet101, ResNet152
   - Pre-trained weights theo sensor
2. Vision Transformer (ViT):
   - ViTSmall14, ViTBase14, ViTLarge14, ViTHuge14
   - Satlas pretrained
3. Swin Transformer
4. Utility functions: get_model(), list_models()
5. Ứng dụng cho Ship/Oil Spill classification

**Hình ảnh yêu cầu:**
- Bảng models và weights
- Sơ đồ ViT architecture

#### `06-torchgeo-models/segmentation-models.md` (~2500 từ)

**Outline:**
1. U-Net trong TorchGeo
2. FCN (Fully Convolutional Network)
3. FarSeg (Foreground-Aware Relation Network)
4. Backbone selection
5. Ứng dụng cho Oil Spill segmentation

**Hình ảnh yêu cầu:**
- Sơ đồ U-Net with backbone
- So sánh FCN vs U-Net

#### `06-torchgeo-models/change-detection-models.md` (~2500 từ)

**Outline:**
1. ChangeStar: Siamese-based
2. Be The Change (BTC)
3. ChangeViT: Vision Transformer cho change detection
4. FC-Siamese Networks: FCSiamConc, FCSiamDiff
5. Ứng dụng: Damage assessment, temporal analysis

**Hình ảnh yêu cầu:**
- Sơ đồ Siamese architecture
- Ví dụ change detection output

#### `06-torchgeo-models/pretrained-weights-sensors.md` (~2500 từ)

**Outline:**
1. Sentinel-1 weights (SAR):
   - ResNet50 Moco
   - ViT Satlas
   - DINOv2 variants
2. Sentinel-2 weights (Multispectral)
3. Landsat weights
4. NAIP weights (Aerial)
5. Foundation Models: DOFA, CROMA, Scale-MAE
6. Cách sử dụng pretrained weights
7. Transfer learning strategies

**Hình ảnh yêu cầu:**
- Bảng weights theo sensor
- Sơ đồ transfer learning

---

### Chương 7: Kết luận (1 file, ~2500 từ)

#### `07-conclusion/ket-luan-va-huong-phat-trien.md`

**Outline:**
1. Tóm tắt nội dung báo cáo
2. So sánh Ship Detection vs Oil Spill Detection
3. Xu hướng phát triển:
   - Transformer-based models
   - Foundation Models
   - Self-supervised learning
   - Edge deployment
4. Thách thức tương lai
5. Khuyến nghị cho nghiên cứu/ứng dụng tại Việt Nam

**Hình ảnh yêu cầu:**
- Sơ đồ xu hướng phát triển
- Timeline tương lai

---

## Yêu cầu Hình ảnh

### Loại 1: Ảnh từ Dataset (Cần download/source)

| File | Ảnh cần | Nguồn |
|------|---------|-------|
| Ship Detection | SAR ship image | Sentinel-1 / SSDD / HRSID |
| Ship Detection | Optical ship image | HRSC2016 / xView1 |
| Oil Spill | SAR oil spill image | Zenodo dataset / CleanSeaNet |
| Oil Spill | Look-alike comparison | Literature / CleanSeaNet |

### Loại 2: Sơ đồ Mermaid (Tạo mới)

| Nội dung | Type | File đặt |
|----------|------|----------|
| CNN architecture cơ bản | Flowchart | kien-truc-cnn-co-ban.md |
| U-Net architecture | Flowchart | phan-doan-ngu-nghia-segmentation.md |
| DeepLabV3+ ASPP | Flowchart | phan-doan-ngu-nghia-segmentation.md |
| Ship Detection Pipeline | Flowchart | quy-trinh-ship-detection-pipeline.md |
| Oil Spill Pipeline | Flowchart | quy-trinh-oil-spill-pipeline.md |
| Two-stage vs One-stage | Flowchart | phat-hien-doi-tuong-object-detection.md |
| Siamese Network | Flowchart | change-detection-models.md |

---

## Cập nhật Config & Build Script

### VitePress Config Update

Thêm sidebar mới cho `cnn-remote-sensing/`:

```javascript
'/cnn-remote-sensing/': [
  {
    text: 'Tổng quan',
    items: [
      { text: 'Mục lục', link: '/cnn-remote-sensing/' }
    ]
  },
  {
    text: 'Chương 1: Giới thiệu',
    items: [...]
  },
  // ... các chương khác
]
```

### Build Script Update

Thêm các file mới vào `scripts/build-docx.sh`:

```bash
pandoc \
    docs/cnn-remote-sensing/README.md \
    docs/cnn-remote-sensing/01-introduction/*.md \
    docs/cnn-remote-sensing/02-cnn-fundamentals/*.md \
    docs/cnn-remote-sensing/03-cnn-satellite-methods/*.md \
    docs/cnn-remote-sensing/04-ship-detection/*.md \
    docs/cnn-remote-sensing/05-oil-spill-detection/*.md \
    docs/cnn-remote-sensing/06-torchgeo-models/*.md \
    docs/cnn-remote-sensing/07-conclusion/*.md \
    -o output/cnn-remote-sensing.docx \
    ...
```

---

## Ước tính Workload

| Chương | Số file | Số từ | Thời gian ước tính |
|--------|---------|-------|-------------------|
| 1. Giới thiệu | 1 | 2,500 | 1 session |
| 2. Kiến trúc CNN | 2 | 5,000 | 2 sessions |
| 3. Phương pháp CNN | 4 | 10,000 | 3-4 sessions |
| 4. Ship Detection | 4 | 10,000 | 3-4 sessions |
| 5. Oil Spill Detection | 4 | 10,000 | 3-4 sessions |
| 6. TorchGeo Models | 5 | 12,500 | 4-5 sessions |
| 7. Kết luận | 1 | 2,500 | 1 session |
| **Tổng** | **21 files** | **52,500 từ** | **~18-20 sessions** |

---

## Quy tắc Viết

### Thuật ngữ giữ nguyên tiếng Anh

- polygon, bounding box, mask
- encoder, decoder, backbone
- feature map, pooling, stride, padding
- loss function, optimizer
- ground truth, inference
- precision, recall, accuracy, IoU, mAP
- batch, epoch, learning rate
- overfitting, underfitting
- SAR, GRD, VV, VH polarization
- speckle, Sigma0, dB

### Văn phong

- Dùng thể bị động khi mô tả phương pháp
- Câu văn dài, chi tiết, học thuật
- Tránh bullet points quá nhiều, ưu tiên đoạn văn
- Giải thích concept trước khi đi vào chi tiết
- Liên kết giữa các section

---

## Hướng dẫn Thu thập Hình ảnh từ Dataset

### 1. Ship Detection - SAR Images

#### SSDD (SAR Ship Detection Dataset)
- **Download**: [GitHub - SSDD](https://github.com/TianwenZhang0825/Official-SSDD)
- **Format**: JPG/PNG, đã có annotation
- **Sample images**: Sentinel-1, TerraSAR-X, RadarSat-2
- **Cách dùng**: Chọn 5-10 ảnh representative, có tàu rõ ràng

#### HRSID (High-Resolution SAR Images Dataset)
- **Download**: [GitHub - HRSID](https://github.com/chaozhong2010/HRSID)
- **Format**: 800×800 JPEG
- **Số lượng**: 5,604 images
- **Cách dùng**: Chọn ảnh có nhiều tàu, ảnh single ship, ảnh near-shore

#### xView3-SAR (Từ project hiện tại)
- **Download**: [xView3 IUU Challenge](https://iuu.xview.us/download)
- **Format**: GeoTIFF (cần convert)
- **Cách dùng**: Crop sample areas, export PNG

### 2. Ship Detection - Optical Images

#### HRSC2016
- **Download**: [HRSC2016 Official](https://www.kaggle.com/datasets/guofeng/hrsc2016)
- **Format**: BMP với XML annotation
- **Cách dùng**: Chọn ảnh có rotated bounding box rõ ràng

#### ShipRSImageNet
- **Download**: [GitHub - ShipRSImageNet](https://github.com/zzndream/ShipRSImageNet)
- **Format**: PNG/JPG với JSON annotation
- **Cách dùng**: Chọn ảnh multi-class ships

### 3. Oil Spill Detection - SAR Images

#### Zenodo Sentinel-1 Oil Spill Dataset
- **Download Part I**: [Zenodo Record 8346860](https://zenodo.org/records/8346860)
- **Download Part II**: [Zenodo Record 8253899](https://zenodo.org/records/8253899)
- **Download Part III**: [Zenodo Record 13761290](https://zenodo.org/records/13761290)
- **Format**: TIFF 2048×2048, Sigma0 (dB)
- **Cách dùng**:
  - Chọn ảnh oil spill rõ ràng
  - Chọn ảnh look-alike để so sánh
  - Chọn ground truth mask tương ứng

#### EMSA CleanSeaNet (Nếu có access)
- **Access**: Cần đăng ký qua EMSA
- **Alternative**: Tìm published examples trong papers

### 4. Sentinel-1 Raw Data (Tự download)

#### Copernicus Data Space
- **URL**: https://dataspace.copernicus.eu/
- **Bước 1**: Đăng ký tài khoản miễn phí
- **Bước 2**: Search → Sentinel-1 → GRD product
- **Bước 3**: Filter theo vùng biển Việt Nam hoặc vùng có oil spill
- **Bước 4**: Download và process bằng SNAP/Python

#### Python Script xử lý Sentinel-1
```python
# Sử dụng sentinelsat hoặc eodag để download
# Sử dụng snappy hoặc rasterio để process
# Export crop area sang PNG
```

### 5. Cấu trúc Thư mục Ảnh

```
docs/images/cnn-remote-sensing/
├── datasets/
│   ├── ship-detection/
│   │   ├── sar/
│   │   │   ├── ssdd-sample-01.png
│   │   │   ├── hrsid-sample-01.png
│   │   │   └── sentinel1-ship-example.png
│   │   └── optical/
│   │       ├── hrsc2016-sample-01.png
│   │       └── shiprsnet-sample-01.png
│   └── oil-spill/
│       ├── oil-spill-sample-01.png
│       ├── oil-spill-mask-01.png
│       ├── lookalike-sample-01.png
│       └── comparison-oil-vs-lookalike.png
├── diagrams/
│   ├── cnn-architecture.png
│   ├── unet-architecture.png
│   ├── ship-detection-pipeline.png
│   └── oil-spill-pipeline.png
└── architectures/
    ├── resnet-block.png
    ├── yolo-architecture.png
    └── deeplabv3-aspp.png
```

### 6. Yêu cầu Ảnh tối thiểu

| Chương | Số ảnh tối thiểu | Loại ảnh |
|--------|------------------|----------|
| Ship Detection | 8-10 | SAR + Optical, with/without bbox |
| Oil Spill Detection | 6-8 | Oil spill, look-alike, mask, comparison |
| CNN Architecture | 4-5 | Diagrams (Mermaid/PNG) |
| TorchGeo | 2-3 | Examples output |

### 7. Quy cách Ảnh

| Thuộc tính | Giá trị |
|------------|---------|
| **Format** | PNG (ưu tiên) hoặc JPG |
| **Resolution** | 800-1200px width |
| **Naming** | kebab-case, descriptive |
| **Caption** | Chuẩn bị caption tiếng Việt cho mỗi ảnh |

---

## Next Steps

1. ✅ **Confirm plan** với user
2. **Tạo folder structure** trong `docs/`
3. **Thu thập hình ảnh** từ datasets (hướng dẫn ở trên)
4. **Viết từng file** theo thứ tự chương (bắt đầu Chương 1)
5. **Tạo sơ đồ Mermaid** trong quá trình viết
6. **Update VitePress config**
7. **Update build script**
8. **Test build** VitePress + DOCX

---

*Generated: 2025-12-19*
*Updated: Thêm hướng dẫn thu thập hình ảnh*
