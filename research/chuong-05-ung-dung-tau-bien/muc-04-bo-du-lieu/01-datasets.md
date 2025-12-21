# Chương 5: Datasets cho Ship Detection

## 5.19. Tổng quan về Ship Detection Datasets

Datasets đóng vai trò quan trọng trong việc training và benchmarking các models ship detection. Các datasets chất lượng cao cung cấp diverse samples, accurate annotations, và challenging scenarios để phát triển và đánh giá algorithms. Trong phần này, chúng ta sẽ phân tích chi tiết các datasets phổ biến nhất cho ship detection từ cả SAR và optical imagery.

Các datasets có thể được phân loại theo nhiều tiêu chí: loại ảnh (SAR vs optical), loại annotation (horizontal bbox, oriented bbox, segmentation mask), số lượng classes (binary vs multi-class), và nguồn gốc (academic vs competition).

## 5.20. SAR Ship Detection Datasets

### 5.20.1. SSDD (SAR Ship Detection Dataset)

SSDD là một trong những datasets SAR ship detection phổ biến nhất, được sử dụng rộng rãi làm benchmark.

| Thuộc tính | Giá trị |
|------------|---------|
| **Nguồn vệ tinh** | Sentinel-1, TerraSAR-X, RadarSat-2 |
| **Số lượng ảnh** | 1,160 |
| **Số lượng ships** | 2,456 |
| **Resolution** | 1-15 meters |
| **Polarization** | HH, HV, VV, VH |
| **Annotation** | Horizontal bounding box |
| **Classes** | 1 (ship) |

**Đặc điểm:**
- Multi-sensor: Ảnh từ nhiều vệ tinh SAR khác nhau
- Multi-resolution: Resolution đa dạng từ 1m đến 15m
- Multi-polarization: Nhiều chế độ phân cực
- Ships in various environments: Offshore, nearshore, ports

**Phiên bản:**
- SSDD original: Horizontal bounding boxes
- SSDD-OBB: Oriented bounding boxes (rotated version)

**Download:**
Repository GitHub và các sources học thuật.

### 5.20.2. HRSID (High-Resolution SAR Images Dataset)

HRSID là dataset SAR ship detection với high resolution và số lượng lớn.

| Thuộc tính | Giá trị |
|------------|---------|
| **Nguồn vệ tinh** | Sentinel-1, TerraSAR-X |
| **Số lượng ảnh** | 5,604 |
| **Số lượng ships** | 16,951 |
| **Kích thước ảnh** | 800×800 pixels |
| **Resolution** | 0.5-3 meters |
| **Annotation** | Instance segmentation (polygon) |

**Đặc điểm:**
- Instance-level annotation: Polygon masks cho mỗi ship
- High quality: Careful annotation process
- Challenging scenarios: Dense ships, nearshore, various sizes

**Ưu điểm:**
- Lớn nhất trong các SAR ship datasets
- Instance segmentation annotations cho phép đánh giá segmentation models
- High resolution images

### 5.20.3. xView3-SAR

xView3-SAR là dataset từ xView3 challenge (**Chương 4.3**), focus vào maritime detection cho anti-IUU fishing.

| Thuộc tính | Giá trị |
|------------|---------|
| **Nguồn vệ tinh** | Sentinel-1 |
| **Số lượng scenes** | 991 full-size scenes |
| **Kích thước trung bình** | 29,400 × 24,400 pixels |
| **Tổng pixels** | 1,400 gigapixels |
| **Số maritime objects** | 243,018 |
| **Coverage** | 43.2 million km² |

**Object Categories:**
- Fishing vessels
- Non-fishing vessels
- Fixed infrastructure (offshore platforms)

**Ancillary Data:**
- Bathymetry
- Wind speed/quality
- AIS tracks
- VMS data

**Đặc điểm:**
- Largest SAR vessel detection dataset
- Multi-modal: SAR + ancillary data
- Real-world scenarios: IUU fishing detection
- Competition-grade quality

**Challenges:**
- Very large images: Need tiling strategies
- Label noise: Some unlabeled vessels
- Class imbalance: More non-fishing than fishing

### 5.20.4. AIR-SARShip

Dataset SAR ship detection từ Chinese satellites.

| Thuộc tính | Giá trị |
|------------|---------|
| **Nguồn** | Gaofen-3 |
| **Số lượng ảnh** | 300+ |
| **Resolution** | 1-3 meters |
| **Annotation** | Bounding box |

**Đặc điểm:**
- Chinese SAR satellite data
- Variety of scenarios

### 5.20.5. OpenSARShip

Dataset SAR ship detection với multi-class labels.

| Thuộc tính | Giá trị |
|------------|---------|
| **Nguồn** | Sentinel-1 |
| **Số lượng chips** | 11,346 |
| **Classes** | 17 ship types |

**Ship Types:** Cargo, Tanker, Fishing, Passenger, Tug, etc.

**Đặc điểm:**
- Multi-class classification
- AIS-derived labels
- Useful cho ship type classification research

## 5.21. Optical Ship Detection Datasets

### 5.21.1. HRSC2016 (High Resolution Ship Collection 2016)

HRSC2016 là dataset optical ship detection phổ biến với oriented annotations.

| Thuộc tính | Giá trị |
|------------|---------|
| **Số lượng ảnh** | 1,070 |
| **Số lượng ships** | 2,976 |
| **Resolution** | 0.4-2 meters |
| **Kích thước ảnh** | ~1000×600 pixels (varied) |
| **Annotation** | 3 levels: HBB, OBB, Segmentation |

**Annotation Levels:**
1. **Level 1:** Horizontal bounding box
2. **Level 2:** Oriented bounding box (rotated)
3. **Level 3:** Pixel-level segmentation

**Class Hierarchy:**
- Ship (root)
  - Ship category (warship, carrier, submarine, etc.)
    - Ship type (specific types)

**Đặc điểm:**
- Multi-level annotations
- Oriented bounding boxes
- Fine-grained classification possible
- Standard benchmark cho oriented ship detection

### 5.21.2. HRSC2016-MS (Multi-Scale Extension)

Extended version của HRSC2016 với focus vào multi-scale ships.

| Thuộc tính | Giá trị |
|------------|---------|
| **Số lượng ảnh** | 1,680 |
| **Số lượng ships** | 7,655 |
| **Đặc điểm** | Rich multi-scale ship objects |

**Cải tiến:**
- Thêm 610 ảnh mới
- Nhiều ships nhỏ và multi-scale scenarios
- Better cho evaluating multi-scale detection

### 5.21.3. ShipRSImageNet

Dataset lớn với hierarchical ship classification.

| Thuộc tính | Giá trị |
|------------|---------|
| **Số lượng ảnh** | 3,435 |
| **Số lượng ships** | 17,573 |
| **Resolution** | Varied (from multiple sources) |
| **Annotation** | HBB và OBB |

**Classification Hierarchy (4 levels):**
- Level 0: Ship (root)
- Level 1: Category (Military, Civilian)
- Level 2: Subcategory
- Level 3: 49 specific ship types + 1 Dock

**Sources:**
Aggregated từ DOTA, HRSC2016, NWPU VHR-10, etc.

**Đặc điểm:**
- Largest optical ship dataset
- Rich class taxonomy
- Suitable cho fine-grained classification

### 5.21.4. FGSD (Fine-Grained Ship Detection)

Dataset cho fine-grained ship detection và classification.

| Thuộc tính | Giá trị |
|------------|---------|
| **Số lượng ảnh** | 2,612 |
| **Số lượng ships** | 5,634 |
| **Classes** | 43 ship types |
| **Locations** | 17 major ports, 4 countries |

**Đặc điểm:**
- Fine-grained classes
- Real port scenarios
- Diverse geographic locations

### 5.21.5. DOTA (Ships Category)

DOTA (Dataset for Object Detection in Aerial Images) chứa ships như một trong 15 categories.

| Thuộc tính (Ships) | Giá trị |
|------------|---------|
| **Tổng ảnh DOTA** | 2,806 |
| **Ships instances** | ~10,000 |
| **Resolution** | 0.1-2 meters |
| **Annotation** | Oriented bounding box |

**Đặc điểm:**
- Part of larger aerial detection dataset
- Standard benchmark cho oriented detection
- High quality annotations
- DOTA-v1.0, v1.5, v2.0 versions

### 5.21.6. xView1 (Ships Category)

xView1 dataset với ships là một trong 60 object classes.

| Thuộc tính | Giá trị |
|------------|---------|
| **Nguồn** | WorldView-3 |
| **Resolution** | 0.3 meters |
| **Ship instances** | ~100,000+ |
| **Ship classes** | 10 maritime vessel types |

**Ship Types trong xView1:**
- Motorboat
- Sailboat
- Tugboat
- Barge
- Fishing Vessel
- Ferry
- Yacht
- Container Ship
- Oil Tanker
- Engineering Vessel

**Đặc điểm:**
- Very high resolution
- Multiple ship classes
- Horizontal bounding boxes
- Large-scale dataset

## 5.22. So sánh Datasets

### 5.22.1. Bảng So sánh Tổng hợp

| Dataset | Type | Images | Ships | Annotation | Classes | Resolution |
|---------|------|--------|-------|------------|---------|------------|
| SSDD | SAR | 1,160 | 2,456 | HBB | 1 | 1-15m |
| HRSID | SAR | 5,604 | 16,951 | Polygon | 1 | 0.5-3m |
| xView3-SAR | SAR | 991 | 243,018 | Point | 3 | 5-40m |
| HRSC2016 | Optical | 1,070 | 2,976 | HBB/OBB/Seg | Multi | 0.4-2m |
| ShipRSImageNet | Optical | 3,435 | 17,573 | HBB/OBB | 50 | Varied |
| DOTA-Ship | Optical | ~500 | ~10,000 | OBB | 1 | 0.1-2m |
| xView1-Ship | Optical | ~200 | ~100,000 | HBB | 10 | 0.3m |

### 5.22.2. Recommendations theo Use Case

**SAR Ship Detection Research:**
- SSDD: Standard benchmark, moderate size
- HRSID: Larger scale, instance segmentation
- xView3-SAR: Largest, real-world challenge

**Oriented Detection:**
- HRSC2016: Standard benchmark
- DOTA-Ship: Part of broader benchmark
- ShipRSImageNet: Large scale với OBB

**Ship Classification:**
- OpenSARShip: SAR multi-class
- ShipRSImageNet: Hierarchical classes
- FGSD: Fine-grained types

**High Resolution Analysis:**
- xView1: 0.3m resolution
- HRSC2016: Sub-meter resolution

## 5.23. Cách Sử dụng Datasets

### 5.23.1. Download và Setup

**SSDD:**
```
GitHub repositories
Academic requests
```

**HRSID:**
```
GitHub: https://github.com/chaozhong2010/HRSID
```

**xView3:**
```
Official: https://iuu.xview.us/download
Requires registration
```

**HRSC2016:**
```
Kaggle: https://www.kaggle.com/datasets/guofeng/hrsc2016
Official academic channels
```

### 5.23.2. Data Splits

Hầu hết datasets cung cấp official train/val/test splits:

**SSDD:**
- Train: 80%
- Test: 20%
(Varied splits trong different papers)

**HRSC2016:**
- Train: 436 images
- Val: 181 images
- Test: 444 images

**xView3-SAR:**
- Train: 583 scenes
- Validation: 60 scenes
- Public Test: 205 scenes
- Holdout Test: 143 scenes

### 5.23.3. Evaluation Metrics

**Standard Metrics:**
- mAP@0.5: Most common
- mAP@0.75: Higher quality threshold
- mAP@[0.5:0.95]: COCO-style

**Dataset-specific:**
- xView3: Aggregate metric combining detection và classification
- DOTA: AP per category

### 5.23.4. Baseline Results

**SSDD (mAP@0.5):**
- Faster R-CNN: ~85%
- YOLOv5: ~93%
- YOLOv8: ~95%

**HRSC2016 (mAP):**
- Rotated FRCNN: ~89%
- S²A-Net: ~95%
- Oriented R-CNN: ~96%

**xView3 (Aggregate Score):**
- Baseline: ~0.2
- Top solutions: ~0.6

## 5.24. Tạo Custom Dataset

### 5.24.1. Khi nào cần Custom Dataset

- Specific geographic region (ví dụ: biển Việt Nam)
- Specific ship types không có trong existing datasets
- Different sensor không được cover
- Proprietary data requirements

### 5.24.2. Annotation Guidelines

**Bounding Box:**
- Tight box bao quanh ship
- Include ship wake nếu visible và consistent
- Handle partial ships (ship cắt bởi image edge)

**Oriented Bounding Box:**
- Align với ship's long axis
- Consistent convention cho angle (ví dụ: bow direction)

**Quality Control:**
- Multiple annotators
- Inter-annotator agreement check
- Expert review cho difficult cases

### 5.24.3. Tools cho Annotation

- **LabelImg:** Simple bounding box annotation
- **CVAT:** Comprehensive, supports OBB
- **Labelbox:** Cloud-based, collaboration features
- **QGIS:** GIS-native, good cho geospatial data

### 5.24.4. Data Augmentation cho Small Datasets

Khi custom dataset nhỏ:
- Extensive augmentation
- Transfer learning từ larger datasets (hoặc TorchGeo weights - **Chương 3.5**)
- Pseudo-labeling với model trained trên public data
- Synthetic data generation

---

## Kết chương

Bài toán ship detection minh họa ứng dụng object detection trong viễn thám. Chương này đã trình bày:
- Đặc điểm và thách thức của bài toán
- Các mô hình áp dụng (tham chiếu Chương 3)
- Pipeline hoàn chỉnh từ data acquisition đến deployment
- Datasets phổ biến cho training và benchmarking

Tương tự, **Chương 6** sẽ trình bày một ứng dụng khác - phát hiện dầu loang - sử dụng các kỹ thuật semantic segmentation đã học ở các chương trước.
