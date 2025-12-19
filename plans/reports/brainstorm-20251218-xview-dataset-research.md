# xView Challenge Series: Comprehensive Dataset Research Report

## Executive Summary

This report analyzes the xView Challenge Series (xView1, xView2, xView3) - three major satellite imagery computer vision competitions organized by the Defense Innovation Unit (DIU). Each challenge targets different applications: object detection, building damage assessment, and maritime vessel detection. The report examines dataset characteristics, creation methodologies, winning solutions, and feasibility of replicating similar datasets.

---

## 1. Challenge Overview

| Aspect | xView1 (2018) | xView2 (2019) | xView3 (2021-2022) |
|--------|---------------|---------------|---------------------|
| **Focus** | Object Detection | Building Damage Assessment | Maritime Vessel Detection |
| **Imagery** | WorldView-3 (Optical) | Maxar Open Data (Optical) | Sentinel-1 (SAR) |
| **Resolution** | 0.3m GSD | <0.8m GSD | 5-40m GSD |
| **Task** | Multi-class detection (60 classes) | Localization + Damage classification | Detection + Vessel classification |
| **Submissions** | 2,000+ | 2,000+ | 1,900 registrants |
| **Prize Pool** | $100,000 | N/A | $150,000 |
| **Organizers** | DIU, NGA | DIU, CMU SEI, CrowdAI | DIU, Global Fishing Watch |

---

## 2. Dataset Specifications

### 2.1 xView1 Dataset

| Metric | Value |
|--------|-------|
| **Total Objects** | 1,000,000+ instances |
| **Object Classes** | 60 fine-grained classes |
| **Coverage Area** | ~1,400 km² |
| **Image Count** | 847 images (train) |
| **Avg Image Size** | 3,316 × 2,911 pixels |
| **Annotation Type** | Horizontal bounding boxes |
| **Spectral Bands** | 3-band (RGB) or 8-band |
| **Source Satellite** | WorldView-3 |

**Class Categories:**
- Vehicles (cars, trucks, buses, aircraft, ships)
- Buildings and structures
- Equipment and machinery
- Railway vehicles
- Towers and infrastructure

**Key Challenges:**
- Severe class imbalance (Small Cars: 200-300K instances vs Railway Vehicles: ~100)
- Small object detection at high resolution
- Multi-scale recognition (objects vary 1-100+ pixels)
- Fine-grained classification (80%+ classes are fine-grained)

### 2.2 xView2 (xBD) Dataset

| Metric | Value |
|--------|-------|
| **Building Annotations** | 850,736 polygons |
| **Coverage Area** | 45,362 km² |
| **Total Images** | 22,068 (1024×1024 RGB) |
| **Disaster Events** | 19 natural disasters |
| **Training Split** | 632,228 polygons / 18,336 images |
| **Test Split** | 109,724 polygons / 1,866 images |
| **Holdout Split** | 108,784 polygons / 1,866 images |
| **Source** | Maxar/DigitalGlobe Open Data Program |

**Joint Damage Scale (4 levels):**
| Level | Description | Count |
|-------|-------------|-------|
| 0 | No Damage | 313,033 |
| 1 | Minor Damage | 36,860 |
| 2 | Major Damage | 29,904 |
| 3 | Destroyed | 31,560 |

**Additional Labels:**
- Environmental factors: smoke, fire, flood water, pyroclastic flow, lava
- Pre/post disaster image pairs

### 2.3 xView3-SAR Dataset

| Metric | Value |
|--------|-------|
| **Total Images** | 991 full-size scenes |
| **Avg Image Size** | 29,400 × 24,400 pixels |
| **Total Pixels** | 1,400 gigapixels |
| **Maritime Objects** | 243,018 verified |
| **Coverage Area** | 43.2 million km² |
| **Source Satellite** | Sentinel-1 (C-band SAR) |
| **Polarization** | VV and VH |

**Object Categories:**
- Fishing vessels
- Non-fishing vessels
- Fixed infrastructure (offshore platforms)

**Ancillary Data:**
- Bathymetry
- Wind speed/quality
- AIS tracks
- VMS data

---

## 3. Dataset Creation Methodology

### 3.1 xView1 Creation Process

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  WorldView-3    │───▶│  QGIS-based     │───▶│  3-Stage QC     │
│  Acquisition    │    │  Annotation     │    │  Process        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Steps:**
1. **Image Collection**: WorldView-3 satellite captures at 0.3m GSD
2. **Preprocessing**: Image tiling and geospatial alignment
3. **Annotation**: QGIS software with professional annotators
4. **Quality Control**: Three-stage verification process
5. **Gold Standard**: Expert validation with ground truth samples

**Key Innovations:**
- Novel geospatial category detection process
- Hierarchical class taxonomy for 60 object types
- Bounding box standardization across diverse regions

### 3.2 xView2 (xBD) Creation Process

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Imagery    │───▶│   Triage &   │───▶│   Polygon    │───▶│   Damage     │
│   Sourcing   │    │   Selection  │    │   Annotation │    │   Labeling   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
        │                                                           │
        └───────────────────────────────────────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Expert QC       │
                    │  (2-3% error     │
                    │   correction)    │
                    └──────────────────┘
```

**Detailed Steps:**
1. **Imagery Sourcing**: Maxar/DigitalGlobe Open Data Program (post-disaster releases)
2. **Triage**: Manual review to identify usable areas with actual damage
3. **Polygon Annotation**: Building footprints drawn on pre-disaster imagery
4. **Damage Classification**:
   - Overlay pre-imagery polygons on post-imagery
   - Classify each building using Joint Damage Scale
   - Multiple annotation rounds with expert review
5. **Quality Control**:
   - Disaster response expert validation (NASA, FEMA, CAL FIRE, CA Air National Guard)
   - ~2-3% mislabel correction rate
6. **Image Alignment**: Pixel-shift correction for re-projection issues

**Collaborating Organizations:**
- Defense Innovation Unit (DIU)
- Carnegie Mellon University SEI
- CrowdAI, Inc.
- Australian Geospatial Intelligence Organisation (AGO)

### 3.3 xView3-SAR Creation Process

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Geographic  │───▶│  SAR Image   │───▶│  CFAR Auto   │───▶│  AIS/VMS     │
│  Selection   │    │  Processing  │    │  Detection   │    │  Correlation │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
        │                                                           │
        └───────────────────────────────────────────────────────────┘
                              ▼
                    ┌─────────────────────┐
                    │  Manual Labeling +  │
                    │  Annotation Merge   │
                    └─────────────────────┘
```

**9-Step Workflow:**
1. **Geographic Selection**: Strategic areas with comprehensive SAR + AIS coverage
   - European waters (North Sea, Bay of Biscay, Iceland, Adriatic)
   - West Africa (high IUU activity regions)
2. **Raw Imagery Processing**: ESA Sentinel-1 Toolbox + GDAL library
3. **Ancillary Data Processing**: Bathymetry, wind speed/quality
4. **CFAR Detection**: Constant False Alarm Rate algorithm for automated detection
5. **AIS Correlation**: Match SAR detections with Automatic Identification System tracks
6. **Vessel Classification**: Using AIS data to characterize vessel type/activity
7. **Manual Labeling**: Human annotators for unlabeled objects
8. **Annotation Merge**: Combine automated + manual labels
9. **Data Partitioning**: Train/validation/test split

**Ground Truth Definition:**
- Human-labeled objects (high/medium confidence)
- High-confidence AIS correlations
- Center-pixel location only (no bounding boxes)

---

## 4. Top 5 Winning Solutions Analysis

### 4.1 xView1 Challenge Winners

#### 🥇 1st Place: Reduced Focal Loss (Nikolay Sergievskiy et al.)

| Aspect | Details |
|--------|---------|
| **Architecture** | FPN Faster R-CNN + ResNet-50 backbone |
| **Framework** | PyTorch Detectron |
| **Key Innovation** | Reduced Focal Loss function |
| **mAP Score** | 31.74 (public) / 29.32 (private) |

**Technical Approach:**
- Modified Focal Loss to maintain RPN recall while handling class imbalance
- Soft weighting mechanism: constant weight below threshold, scaled focal weight above
- Training: 700×700 crops with 80px overlap, rotational augmentation (10°, 90°, 180°, 270°)
- Final dataset: 63,535 augmented images from 846 originals

**Data Augmentation:**
- Random flip
- Color jittering
- Scale jittering (500-900 pixels)
- Random undersampling of prevalent classes

#### 🥈 5th Place: CMU SEI Team (Ritwik Gupta, Alex Fulton, Kaylene Stocking)

| Aspect | Details |
|--------|---------|
| **Architecture** | SSD variant (best performance) |
| **Alternatives Tested** | RetinaNet, Faster R-CNN, YOLOv3 |
| **Feature Extraction** | DenseNet/RetinaNet + shallow satellite-specific CNN |

**Technical Approach:**
- Dual-CNN strategy: pre-trained deep CNN + shallow satellite-specific CNN
- Experimented with YOLO using large anchor box configurations
- SSD ultimately provided best mAP results

### 4.2 xView2 Challenge Winners

#### 🥇 1st Place: UNet Siamese Network

| Aspect | Details |
|--------|---------|
| **Architecture** | UNet-like Encoder-Decoder (Siamese) |
| **Encoders** | ResNet34, SE-ResNeXt50, SeNet154, DPN92 |
| **Training Time** | ~7 days on 2× Titan V GPUs |
| **Performance** | 266% more accurate than baseline |

**Two-Stage Pipeline:**
1. **Localization Stage**:
   - Train on pre-disaster images only (avoid noise from post-disaster variations)
   - Binary building segmentation
   - Loss: Dice + Focal

2. **Classification Stage**:
   - Siamese network sharing localization weights
   - Process pre/post images through identical pathways
   - Concatenate decoder features for damage classification
   - Loss: Dice + Focal + CrossEntropyLoss (weighted for damage classes 2-4)

**Key Innovations:**
- Shared-weight Siamese architecture handles nadir shifts/misalignment
- Morphological dilation (5×5 kernel) for bolder predictions
- 4× TTA (original + flips + 180° rotation)
- 2× oversampling for damage classes 2-4

#### 🥈 2nd Place Solution

| Aspect | Details |
|--------|---------|
| **Training** | Separate localization and classification networks |
| **Precision** | Mixed-precision training (Apex O1) |
| **Loss** | Multiclass FocalLossWithDice |

#### 🥉 5th Place: Dual-HRNet

| Aspect | Details |
|--------|---------|
| **Architecture** | Dual High-Resolution Network |
| **Strengths** | Good performance for both localization and classification |
| **Competition Size** | Top 5 among 3,500+ participants |

### 4.3 xView3 Challenge Winners

#### 🥇 1st Place: CircleNet (Eugene Khvedchenya / BloodAxe)

| Aspect | Details |
|--------|---------|
| **Architecture** | CircleNet (CenterNet + U-Net inspired) |
| **Backbones** | EfficientNet B4, B5, V2S |
| **Output Stride** | Stride-2 (high-resolution) |
| **Ensemble** | 12 models (3×4 scheme) |
| **Final Score** | 0.617 (holdout) - 3× baseline |

**Technical Innovations:**
- **High-Resolution Output**: Stride-2 predictions (F1: 0.9999 vs 0.9672 at stride-16)
- **Label Noise Handling**:
  - Label smoothing (0.05)
  - Shannon's entropy regularization for missing labels
- **Custom SAR Normalization**: Sigmoid activation instead of linear scaling
- **Reduced Focal Loss**: Fixed 3-pixel encoding radius for objectness

**Inference:**
- 2048×2048 tiles with 1536-pixel stride overlap
- Left-right flip TTA
- Balanced sampling across vessel categories

**Model Outputs:**
1. Objectness heatmaps
2. Object length (pixels)
3. Vessel classification
4. Fishing-activity classification

#### 🥈 2nd Place Solution

| Aspect | Details |
|--------|---------|
| **Architecture** | UNet-like CNN |
| **Backbones** | EfficientNet V2, ResNest, NFNet L0 |
| **Approach** | Binary segmentation + regression |

#### 🥉 3rd Place Solution

| Aspect | Details |
|--------|---------|
| **Repository** | DIUx-xView/xView3_third_place |
| **Documentation** | PDF writeup available |

#### 4th Place: Allen AI

| Aspect | Details |
|--------|---------|
| **Repository** | allenai/sar_vessel_detect |
| **Focus** | Sentinel-1 vessel detection |

---

## 5. Feasibility Analysis: Creating Similar Datasets

### 5.1 Resource Requirements Comparison

| Resource | xView1-like | xView2-like | xView3-like |
|----------|-------------|-------------|-------------|
| **Imagery Cost** | $$$$ (Commercial) | $$ (Open Data) | Free (Sentinel-1) |
| **Annotation Complexity** | High (60 classes) | Very High (polygons + damage) | Medium (point + class) |
| **Expert Requirements** | Moderate | Very High (disaster experts) | Moderate (maritime) |
| **Infrastructure** | Standard GPU | Standard GPU | High storage (TB-scale) |

### 5.2 Imagery Acquisition Options

#### Option A: High-Resolution Optical (xView1/xView2 style)
| Source | Resolution | Cost | Accessibility |
|--------|------------|------|---------------|
| **WorldView-3 (Maxar)** | 0.3m | $2,900/100km² tasking | Commercial license required |
| **Maxar Open Data** | 0.5-0.8m | Free (disasters) | CC 4.0 license |
| **Planet** | 3-5m | ~$5/km²/year | Subscription model |

#### Option B: SAR Imagery (xView3 style)
| Source | Resolution | Cost | Accessibility |
|--------|------------|------|---------------|
| **Sentinel-1** | 5-40m | Free | Open Access (ESA) |
| **ICEYE** | 0.25-1m | Commercial | High cost |
| **Capella Space** | 0.5m | Commercial | Enterprise pricing |

### 5.3 Annotation Cost Estimates

| Task Type | Cost Range | Notes |
|-----------|------------|-------|
| **Bounding Box (simple)** | $0.02-0.05/object | Basic labeling |
| **Bounding Box (satellite)** | $0.10-0.30/object | Domain expertise |
| **Polygon Segmentation** | $0.50-2.00/polygon | Complex shapes |
| **Damage Assessment** | $1.00-5.00/building | Expert + multi-stage QC |
| **SAR Object Detection** | $0.30-1.00/object | Specialized SAR knowledge |

#### Estimated Total Costs (Similar Scale)

| Dataset Type | Objects | Annotation Cost | Imagery Cost | Total Estimate |
|--------------|---------|-----------------|--------------|----------------|
| xView1-like (1M objects) | 1,000,000 | $100K-300K | $200K-500K | **$300K-800K** |
| xView2-like (850K polygons) | 850,736 | $400K-1M | $50K (Open Data) | **$450K-1M** |
| xView3-like (243K objects) | 243,018 | $75K-250K | Free (Sentinel) | **$75K-250K** |

### 5.4 Annotation Platforms & Tools

| Platform | Satellite Support | Cost Model | Quality |
|----------|-------------------|------------|---------|
| **Scale AI** | Yes (Remotasks) | Enterprise | Variable (crowdsourced) |
| **Labelbox** | Yes (geospatial) | $10/hr base | High (managed) |
| **CVAT** | Limited | Free (open-source) | Self-managed |
| **GroundWork (Element84)** | Native GeoTIFF | SaaS | High |
| **QGIS** | Excellent | Free | Expert-dependent |

### 5.5 Feasibility Matrix

| Factor | xView1-like | xView2-like | xView3-like |
|--------|:-----------:|:-----------:|:-----------:|
| **Imagery Accessibility** | ⚠️ Commercial | ✅ Open Data | ✅ Free |
| **Annotation Tooling** | ✅ Mature | ⚠️ Specialized | ⚠️ SAR expertise |
| **Expert Requirements** | ⚠️ Domain knowledge | ❌ Disaster experts | ⚠️ Maritime/SAR |
| **Ground Truth Availability** | ⚠️ Manual only | ✅ Event-based | ✅ AIS/VMS data |
| **Scalability** | ⚠️ Cost-limited | ✅ Good | ✅ Excellent |
| **Time to Create** | 12-18 months | 6-12 months | 3-6 months |

**Legend:** ✅ Favorable | ⚠️ Moderate | ❌ Challenging

---

## 6. Recommendations

### 6.1 For Creating xView1-like Dataset (Object Detection)
1. **Start with Maxar Open Data** for initial prototyping (free, disaster imagery)
2. Consider **Planet imagery** for cost-effective large-scale coverage
3. Focus on **fewer classes initially** (10-20) to manage annotation complexity
4. Use **active learning** to reduce annotation burden

### 6.2 For Creating xView2-like Dataset (Damage Assessment)
1. **Leverage Maxar Open Data Program** - primary imagery source for disasters
2. **Partner with disaster response agencies** for expert annotation validation
3. Develop **standardized damage scale** before annotation begins
4. Implement **multi-stage QC** similar to xBD process

### 6.3 For Creating xView3-like Dataset (Maritime Detection)
1. **Best feasibility** due to free Sentinel-1 data
2. **Leverage AIS/VMS** for automated ground truth generation
3. Use **CFAR algorithm** for initial detection, then manual refinement
4. Consider **regional focus** (e.g., Vietnam's EEZ) for manageable scope

---

## 7. Key Takeaways

### Dataset Design Principles
1. **Class imbalance** is inevitable - plan mitigation strategies upfront
2. **Multi-stage QC** with expert validation significantly improves quality
3. **Hybrid annotation** (automated + manual) scales better than pure manual
4. **Pre/post imagery pairs** enable change detection but add complexity

### Winning Solution Patterns
1. **Encoder-Decoder architectures** dominate (UNet, FPN)
2. **ResNet family** remains strong backbone choice
3. **Custom loss functions** address dataset-specific challenges
4. **Test-time augmentation** provides consistent gains
5. **Ensemble methods** typically win competitions

### Feasibility Summary
- **Most Feasible**: xView3-like (SAR maritime detection) - free imagery, AIS ground truth
- **Moderate Feasibility**: xView2-like (damage assessment) - open data available post-disaster
- **Least Feasible**: xView1-like (general detection) - high commercial imagery costs

---

## Sources

### Official Resources
- [xView Dataset Official](https://xviewdataset.org/)
- [xView3 IUU Challenge](https://iuu.xview.us/)
- [DIU xView Challenge Series](https://www.diu.mil/ai-xview-challenge)
- [CMU SEI xView2 Project](https://www.sei.cmu.edu/projects/xview-2-challenge/)

### Papers
- [xView: Objects in Context in Overhead Imagery (arXiv:1802.07856)](https://arxiv.org/abs/1802.07856)
- [Creating xBD: A Dataset for Assessing Building Damage (CVPR 2019)](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf)
- [xView3-SAR: Detecting Dark Fishing Activity (NeurIPS 2022)](https://arxiv.org/abs/2206.00897)
- [Reduced Focal Loss: 1st Place xView Solution (arXiv:1903.01347)](https://arxiv.org/abs/1903.01347)

### GitHub Repositories
- [xView2 1st Place](https://github.com/DIUx-xView/xView2_first_place)
- [xView3 1st Place](https://github.com/BloodAxe/xView3-The-First-Place-Solution)
- [xView3 2nd Place](https://github.com/DIUx-xView/xView3_second_place)
- [xView3 3rd Place](https://github.com/DIUx-xView/xView3_third_place)
- [xView2 5th Place (Dual-HRNet)](https://github.com/DIUx-xView/xView2_fifth_place)

### Data Access
- [Copernicus Data Space (Sentinel-1)](https://dataspace.copernicus.eu/)
- [Maxar Open Data Program](https://www.maxar.com/open-data)
- [NASA Earthdata (Maxar CSDA)](https://www.earthdata.nasa.gov/about/csda/vendor-maxar)

---

*Report generated: 2024-12-18*
*Research scope: xView Challenge Series (2018-2022)*
