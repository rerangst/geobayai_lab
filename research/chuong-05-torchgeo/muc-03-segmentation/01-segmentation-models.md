# Chương 5: Segmentation Models trong TorchGeo

## 6.20. Tổng quan Segmentation trong TorchGeo

Semantic segmentation là task quan trọng trong remote sensing, cho phép pixel-level classification của ảnh vệ tinh. Các ứng dụng bao gồm land cover mapping, crop field delineation, building footprint extraction, flood mapping, và nhiều tasks khác yêu cầu chi tiết về spatial extent của các features.

TorchGeo cung cấp comprehensive support cho segmentation tasks thông qua pre-trained encoders, benchmark datasets, và training utilities được thiết kế đặc biệt cho geospatial data.

## 6.21. Segmentation Architectures

### 6.21.1. U-Net trong TorchGeo

U-Net là architecture cơ bản và phổ biến nhất cho remote sensing segmentation.

**Implementation:**
TorchGeo sử dụng U-Net implementations từ segmentation_models_pytorch library:
- Various encoder backbones
- Skip connections preserve spatial information
- Configurable depth và features

**Encoder Options:**
- ResNet (18, 34, 50, 101, 152)
- EfficientNet (B0-B7)
- VGG
- DenseNet
- MobileNet

**Pre-trained Encoders:**
Encoders có thể được initialized với:
- TorchGeo remote sensing weights (SSL4EO, SatMAE)
- ImageNet weights
- Random initialization

**Configuration:**
```
model = UNet(
    encoder_name="resnet50",
    encoder_weights="ssl4eo_moco",  # TorchGeo specific
    in_channels=13,  # Sentinel-2 bands
    classes=10,  # Number of classes
    activation=None,  # For training
)
```

### 6.21.2. DeepLabV3+

DeepLabV3+ với atrous spatial pyramid pooling cho multi-scale context.

**Key Components:**
- Encoder backbone (ResNet, Xception)
- ASPP module với multiple dilation rates
- Simple decoder với skip connection

**Advantages cho Remote Sensing:**
- Captures objects ở multiple scales
- Good boundary delineation
- Efficient inference

**TorchGeo Integration:**
```
model = DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="ssl4eo",
    in_channels=13,
    classes=10,
)
```

### 6.21.3. Feature Pyramid Network (FPN)

FPN-based segmentation cho multi-scale features.

**Architecture:**
- Bottom-up pathway (encoder)
- Top-down pathway với lateral connections
- Prediction từ merged features

**Use Cases:**
- Multi-scale object segmentation
- Objects ranging từ buildings đến large water bodies

### 6.21.4. PSPNet

Pyramid Scene Parsing Network với pyramid pooling module.

**Key Feature:**
- Global context aggregation
- Multiple pooling scales (1×1, 2×2, 3×3, 6×6)
- Good cho understanding scene layout

### 6.21.5. MANet (Multi-scale Attention Network)

Attention-based segmentation architecture.

**Components:**
- Multi-scale feature extraction
- Attention mechanisms
- Feature aggregation

## 6.22. Pre-trained Weights cho Segmentation

### 6.22.1. Encoder Pre-training

Segmentation models sử dụng pre-trained encoders:

**SSL4EO Weights:**
Available cho encoders:
- ResNet-18, 34, 50 với MoCo pre-training
- Sentinel-1 và Sentinel-2 data
- Significant improvement over ImageNet

**Usage:**
```
# Load encoder với TorchGeo weights
encoder = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO)
# Use as backbone cho segmentation model
```

### 6.22.2. Full Model Weights

Some datasets provide full segmentation model weights:

**ChesapeakeCVPR:**
- U-Net trained on land cover
- Available cho evaluation

**Custom Training:**
Most use cases require training segmentation head from scratch với pre-trained encoder.

### 6.22.3. Transfer Learning Strategy

**Two-stage Approach:**
1. Load pre-trained encoder
2. Initialize decoder randomly
3. Fine-tune entire model

**Layer-wise Learning Rates:**
- Lower LR cho pre-trained encoder
- Higher LR cho decoder
- Gradual unfreezing

## 6.23. Segmentation Datasets

### 6.23.1. ChesapeakeCVPR

| Attribute | Value |
|-----------|-------|
| **Source** | NAIP + Landsat + other |
| **Classes** | 7 land cover classes |
| **Coverage** | Chesapeake Bay watershed, USA |
| **Resolution** | 1m (NAIP), 30m (Landsat) |

**Classes:**
Water, Tree Canopy / Forest, Low Vegetation / Field, Barren Land, Impervious (other), Impervious (road), No Data

**Usage:**
```
from torchgeo.datasets import ChesapeakeCVPR
from torchgeo.samplers import RandomGeoSampler

dataset = ChesapeakeCVPR(root="data", crs=CRS, res=1.0)
sampler = RandomGeoSampler(dataset, size=256, length=1000)
```

### 6.23.2. LandCover.ai

| Attribute | Value |
|-----------|-------|
| **Source** | Aerial orthophotos |
| **Classes** | 4 classes + background |
| **Coverage** | Poland |
| **Resolution** | 0.25m - 0.5m |

**Classes:**
Building, Woodland, Water, Road, Background

**Đặc điểm:**
- High-resolution imagery
- Building và road extraction
- European urban/rural mix

### 6.23.3. GeoNRW

| Attribute | Value |
|-----------|-------|
| **Source** | Aerial imagery |
| **Classes** | 10 classes |
| **Coverage** | North Rhine-Westphalia, Germany |
| **Resolution** | 1m |

**Classes:**
Forest, Water, Agricultural, Urban Fabric, Industrial, Road, Railway, etc.

### 6.23.4. SpaceNet

SpaceNet provides building footprint segmentation datasets:

| Dataset | Coverage | Buildings |
|---------|----------|-----------|
| SpaceNet 1 | Rio de Janeiro | 382k |
| SpaceNet 2 | Las Vegas, Paris, Shanghai, Khartoum | 858k |
| SpaceNet 4 | Atlanta | 126k |
| SpaceNet 7 | Global urban areas | Temporal |

**Use Cases:**
- Building footprint extraction
- Urban mapping
- Change detection (SpaceNet 7)

### 6.23.5. INRIA Aerial Image Labeling

| Attribute | Value |
|-----------|-------|
| **Task** | Building segmentation |
| **Coverage** | 5 cities (US và Austria) |
| **Resolution** | 0.3m |
| **Area** | 810 km² |

**Đặc điểm:**
- Binary segmentation (building/not building)
- High-resolution RGB
- Diverse urban environments

### 6.23.6. Potsdam và Vaihingen

ISPRS 2D Semantic Labeling datasets:

| Dataset | Resolution | Classes |
|---------|------------|---------|
| Potsdam | 5cm | 6 |
| Vaihingen | 9cm | 6 |

**Classes:**
Impervious Surface, Building, Low Vegetation, Tree, Car, Clutter

**Đặc điểm:**
- Very high resolution
- Urban semantic segmentation
- Standard benchmark cho aerial imagery

## 6.24. SemanticSegmentationTask

### 6.24.1. Overview

TorchGeo cung cấp `SemanticSegmentationTask` Lightning module cho streamlined training.

**Features:**
- Configurable encoder và decoder
- Support cho various loss functions
- Standard metrics (IoU, Dice, accuracy)
- Integration với TorchGeo datasets

### 6.24.2. Configuration

**Architecture Selection:**
```
task = SemanticSegmentationTask(
    model="unet",
    backbone="resnet50",
    weights="ssl4eo",
    in_channels=4,  # RGBIR
    num_classes=7,
    loss="ce",  # or "focal", "dice"
    lr=1e-4,
)
```

**Loss Functions:**
- Cross Entropy: Standard cho multi-class
- Focal Loss: Handles class imbalance
- Dice Loss: Optimizes overlap metric
- Combination: CE + Dice common

### 6.24.3. Metrics

**Per-pixel Metrics:**
- Overall Accuracy
- Per-class Accuracy
- Mean Accuracy

**IoU-based Metrics:**
- Per-class IoU
- Mean IoU (mIoU)
- Frequency-weighted IoU

**Dice Score:**
- Per-class Dice
- Mean Dice

## 6.25. Training Best Practices

### 6.25.1. Data Preprocessing

**Normalization:**
- Use sensor-specific statistics
- TorchGeo provides per-band mean/std
- Consistent normalization train/test

**Patch Sampling:**
For large rasters:
- RandomGeoSampler cho training
- GridGeoSampler cho inference
- Appropriate patch size (256, 512)
- Overlap cho inference

### 6.25.2. Class Imbalance

Remote sensing often has severe class imbalance:

**Solutions:**
- Class weighting in loss function
- Focal Loss
- Over-sampling rare classes
- Data augmentation cho minority classes
- Dice/IoU loss

**Computing Weights:**
```
# Inverse frequency weighting
class_weights = 1.0 / class_frequencies
# Or median frequency balancing
class_weights = median_freq / class_frequencies
```

### 6.25.3. Augmentation

**Geometric:**
- Random flip (horizontal, vertical)
- Random rotation (90°, 180°, 270°, or arbitrary)
- Random scale
- Random crop

**Photometric:**
- Brightness và contrast adjustment
- Gaussian noise
- Blur

**Important:**
- Apply same transforms to image và mask
- Preserve spatial alignment
- Use nearest neighbor for mask interpolation

### 6.25.4. Multi-scale Training

**Input Scales:**
- Train với multiple input resolutions
- Improves scale invariance
- Helps với objects của different sizes

**Multi-scale Inference:**
- Process at multiple scales
- Merge predictions
- Improves accuracy với computation cost

## 6.26. Inference Strategies

### 6.26.1. Sliding Window

For large images:
1. Divide into overlapping patches
2. Run model on each patch
3. Merge predictions

**Overlap Handling:**
- Average probabilities in overlap regions
- Or max probability
- Or learned fusion

### 6.26.2. Test Time Augmentation (TTA)

Apply augmentations at inference:
- Horizontal/vertical flips
- 90° rotations
- Average predictions

**Benefits:**
- Improved accuracy (1-2% typical)
- More robust predictions
- Smoother boundaries

### 6.26.3. Post-processing

**Morphological Operations:**
- Remove small isolated regions
- Fill holes
- Smooth boundaries

**CRF (Conditional Random Field):**
- Refine boundaries
- Spatial consistency
- Consider image evidence

**Vector Conversion:**
- Convert raster mask to vector polygons
- Simplification cho cleaner shapes
- Export to GIS formats

## 6.27. Benchmark Results

### 6.27.1. ChesapeakeCVPR Results

| Model | Encoder | Pre-training | mIoU |
|-------|---------|--------------|------|
| U-Net | ResNet-50 | ImageNet | 62.3 |
| U-Net | ResNet-50 | SSL4EO | 66.8 |
| DeepLabV3+ | ResNet-50 | SSL4EO | 68.2 |
| FPN | ResNet-50 | SSL4EO | 67.5 |

### 6.27.2. LandCover.ai Results

| Model | Encoder | mIoU |
|-------|---------|------|
| U-Net | EfficientNet-B4 | 78.5 |
| FPN | ResNet-101 | 79.2 |
| DeepLabV3+ | Xception | 80.1 |

### 6.27.3. Potsdam Results

| Model | OA | mIoU |
|-------|----|----- |
| U-Net | 89.2 | 76.8 |
| DeepLabV3+ | 90.5 | 79.3 |
| HRNet | 91.2 | 81.5 |

**Observations:**
- Domain-specific pre-training (SSL4EO) consistently helps
- DeepLabV3+ generally strong performer
- HRNet excellent for high-resolution imagery

## 6.28. Advanced Topics

### 6.28.1. Instance Segmentation

Beyond semantic segmentation:
- Mask R-CNN cho instance-level
- Separate each object instance
- Building instance extraction

TorchGeo supports through integration với detectron2 hoặc mmdetection.

### 6.28.2. Panoptic Segmentation

Combining semantic và instance:
- "Stuff" classes (background, water)
- "Things" classes (buildings, vehicles)
- Unified output

### 6.28.3. Multi-temporal Segmentation

Using time series:
- Stack multiple dates
- Recurrent architectures (ConvLSTM)
- Attention over time

**Applications:**
- Crop mapping với growing season
- Change detection
- Seasonal land cover

### 6.28.4. Multi-modal Fusion

Combining data sources:

**Early Fusion:**
- Concatenate inputs (optical + SAR)
- Single network processes all

**Late Fusion:**
- Separate encoders
- Merge at decision level

**Mid-level Fusion:**
- Separate encoders
- Merge features before decoder

TorchGeo IntersectionDataset facilitates loading aligned multi-modal data.

### 6.28.5. Uncertainty Quantification

Estimating prediction confidence:
- Monte Carlo Dropout
- Deep Ensembles
- Evidential Deep Learning

Important cho:
- Identifying unreliable predictions
- Active learning
- Decision support

## 6.29. Use Cases

### 6.29.1. Land Cover Mapping

**Task:**
Classify every pixel into land cover categories.

**Approach:**
- Sentinel-2 data
- U-Net với SSL4EO pre-trained encoder
- 10-15 classes typically

**Challenges:**
- Class confusion (e.g., grass vs crop)
- Seasonal variation
- Cloud contamination

### 6.29.2. Building Footprint Extraction

**Task:**
Segment building pixels từ aerial/satellite imagery.

**Approach:**
- High-resolution imagery (0.3-1m)
- Binary hoặc instance segmentation
- Post-process to polygons

**Challenges:**
- Dense urban areas
- Varying building sizes
- Shadows và occlusion

### 6.29.3. Agricultural Field Delineation

**Task:**
Identify boundaries của agricultural fields.

**Approach:**
- Multi-temporal Sentinel-2
- Edge detection hoặc semantic segmentation
- Vectorization

**Challenges:**
- Fuzzy field boundaries
- Seasonal changes
- Fragmented landscapes

### 6.29.4. Flood Mapping

**Task:**
Identify inundated areas từ satellite imagery.

**Approach:**
- SAR data (Sentinel-1) cho all-weather
- Binary segmentation (water/non-water)
- Rapid response requirement

**Challenges:**
- Urban flooding (mixed pixels)
- Vegetation trong water
- False positives (shadows, wet surfaces)

### 6.29.5. Road Network Extraction

**Task:**
Extract road network từ imagery.

**Approach:**
- High-resolution imagery
- Semantic segmentation + skeletonization
- Graph extraction cho network topology

**Challenges:**
- Occluded by trees
- Varying road widths
- Junctions và complex topology

Segmentation trong TorchGeo provides essential tools cho converting satellite imagery into actionable spatial information, với pre-trained models significantly accelerating development cycles.

