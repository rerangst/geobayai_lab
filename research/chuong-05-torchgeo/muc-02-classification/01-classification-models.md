# Chương 5: Classification Models trong TorchGeo

## 6.11. Tổng quan Classification trong TorchGeo

Classification là một trong những tasks phổ biến nhất trong remote sensing, từ land use/land cover classification đến scene classification và crop type mapping. TorchGeo cung cấp comprehensive support cho classification tasks thông qua pre-trained models, benchmark datasets, và training utilities.

Trong phần này, chúng ta sẽ phân tích chi tiết các classification models có sẵn trong TorchGeo, cách sử dụng pre-trained weights, và best practices cho training classification models trên satellite imagery.

## 6.12. Available Architectures

### 6.12.1. ResNet Family

ResNet (Residual Networks) là backbone phổ biến nhất cho remote sensing classification trong TorchGeo.

**Variants có sẵn:**
- ResNet-18: 11M parameters, lightweight
- ResNet-34: 21M parameters
- ResNet-50: 25M parameters, best balance
- ResNet-101: 44M parameters
- ResNet-152: 60M parameters, highest capacity

**Modifications cho Remote Sensing:**
TorchGeo ResNet được modified để handle multi-spectral input:
- First convolutional layer adapted cho variable input channels
- Supports 1 to any number of bands
- Maintains compatibility với pre-trained weights

**Usage:**
```
from torchgeo.models import resnet50, ResNet50_Weights

# Load với pre-trained weights
weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
model = resnet50(weights=weights)

# Or for custom input channels
model = resnet50(in_chans=13)  # 13-band Sentinel-2
```

### 6.12.2. Vision Transformer (ViT)

Vision Transformers đã được adapted cho remote sensing trong TorchGeo.

**Variants:**
- ViT-Small: 22M parameters
- ViT-Base: 86M parameters
- ViT-Large: 307M parameters

**Pre-training:**
ViT models trong TorchGeo có weights từ:
- SSL4EO self-supervised learning
- SatMAE masked autoencoder
- DINO distillation

**Patch Processing:**
ViT processes images as sequences of patches:
- Default patch size: 16×16 pixels
- Position embeddings cho spatial awareness
- Self-attention captures global context

**Advantages cho Remote Sensing:**
- Better global context than CNNs
- Handles variable object scales
- Strong transfer learning performance

### 6.12.3. Swin Transformer

Swin Transformer với hierarchical feature maps được support.

**Key Features:**
- Shifted window attention
- Multi-scale representations
- Efficient computation

**Variants:**
- Swin-Tiny: 28M parameters
- Swin-Small: 50M parameters
- Swin-Base: 88M parameters

**Remote Sensing Benefits:**
- Hierarchical features suit multi-scale objects
- Local+global attention
- Strong performance on benchmarks

### 6.12.4. EfficientNet

EfficientNet với compound scaling trong TorchGeo.

**Variants:**
- EfficientNet-B0 đến B7
- Increasing size và accuracy

**Advantages:**
- Excellent accuracy/efficiency trade-off
- Mobile-friendly options (B0-B2)
- Scalable cho different compute budgets

### 6.12.5. Other Architectures

TorchGeo cũng supports:
- VGG (11, 13, 16, 19)
- DenseNet
- MobileNet variants
- RegNet

## 6.13. Pre-trained Weights

### 6.13.1. SSL4EO Weights

SSL4EO (Self-Supervised Learning for Earth Observation) cung cấp weights trained trên European satellite data.

**SSL4EO-S12:**
- Dataset: Sentinel-1 và Sentinel-2 imagery
- Coverage: Europe, seasonal variations
- Size: 200k+ image triplets

**Pre-training Methods:**

**MoCo v2 (Momentum Contrast):**
- Contrastive learning approach
- Instance discrimination task
- Strong feature learning

**DINO (Distillation with No Labels):**
- Self-distillation from Vision Transformers
- Teacher-student framework
- Good for ViT models

**MAE (Masked Autoencoder):**
- Reconstruction-based pre-training
- Masks patches và reconstructs
- Data-efficient

**Available Weights:**

| Model | Method | Input Bands | Weight Name |
|-------|--------|-------------|-------------|
| ResNet-18 | MoCo | Sentinel-2 All | SENTINEL2_ALL_MOCO |
| ResNet-50 | MoCo | Sentinel-2 All | SENTINEL2_ALL_MOCO |
| ResNet-50 | MoCo | Sentinel-2 RGB | SENTINEL2_RGB_MOCO |
| ResNet-50 | MoCo | Sentinel-1 | SENTINEL1_ALL_MOCO |
| ViT-S | DINO | Sentinel-2 | SENTINEL2_ALL_DINO |
| ViT-B | MAE | Sentinel-2 | SENTINEL2_ALL_MAE |

### 6.13.2. SatMAE Weights

SatMAE (Satellite Masked Autoencoder) specifically designed cho satellite imagery.

**Key Features:**
- Temporal encoding
- Multi-spectral support
- Global scale pre-training

**Performance:**
SatMAE shows strong transfer performance:
- Outperforms ImageNet pretrained on most benchmarks
- Particularly good for multi-spectral data
- Better data efficiency

### 6.13.3. ImageNet Weights

Standard ImageNet weights cũng available:
- Baseline comparison
- Good for RGB-only tasks
- Wide architecture support

**Adaptation for Multi-spectral:**
Khi sử dụng ImageNet weights với multi-spectral data:
- Modify first conv layer
- Options: Duplicate RGB weights, random initialize extra channels, or learned adaptation

### 6.13.4. Domain-specific Weights

**Million-AID:**
- Aerial image classification dataset
- 1 million images, 51 classes
- Good for high-resolution aerial tasks

**BigEarthNet:**
- Multi-label Sentinel-2 classification
- 590k patches, 19/43 labels
- European land cover

## 6.14. Classification Datasets

### 6.14.1. EuroSAT

| Attribute | Value |
|-----------|-------|
| **Source** | Sentinel-2 |
| **Classes** | 10 land use classes |
| **Samples** | 27,000 patches |
| **Patch Size** | 64×64 pixels |
| **Bands** | 13 multispectral |

**Classes:**
Industrial, Residential, Annual Crop, Permanent Crop, River, Sea/Lake, Herbaceous Vegetation, Highway, Pasture, Forest

**Usage trong TorchGeo:**
```
from torchgeo.datasets import EuroSAT
from torchgeo.datamodules import EuroSATDataModule

dataset = EuroSAT(root="data", split="train", download=True)
# Or with DataModule
dm = EuroSATDataModule(root="data", batch_size=32)
```

### 6.14.2. UC Merced Land Use

| Attribute | Value |
|-----------|-------|
| **Source** | USGS aerial imagery |
| **Classes** | 21 land use classes |
| **Samples** | 2,100 images |
| **Size** | 256×256 pixels |
| **Resolution** | 0.3m |

**Classes:**
Agricultural, Airplane, Baseball Diamond, Beach, Buildings, Chaparral, Dense Residential, Forest, Freeway, Golf Course, Harbor, Intersection, Medium Residential, Mobile Home Park, Overpass, Parking Lot, River, Runway, Sparse Residential, Storage Tanks, Tennis Courts

### 6.14.3. BigEarthNet

| Attribute | Value |
|-----------|-------|
| **Source** | Sentinel-2 |
| **Patches** | 590,326 |
| **Size** | 120×120 pixels |
| **Labels** | Multi-label (43 classes) |
| **Coverage** | 10 European countries |

**Đặc điểm:**
- Large-scale dataset
- Multi-label classification
- Updated label scheme (19 classes) for cleaner taxonomy
- Sentinel-1 version also available

### 6.14.4. PatternNet

| Attribute | Value |
|-----------|-------|
| **Classes** | 38 classes |
| **Samples** | 30,400 images |
| **Size** | 256×256 pixels |
| **Resolution** | ~0.06m to ~5m |

**Diverse classes:**
Từ natural (beach, forest, wetland) đến man-made (airport, bridge, stadium).

### 6.14.5. RESISC45

| Attribute | Value |
|-----------|-------|
| **Classes** | 45 classes |
| **Samples** | 31,500 images |
| **Size** | 256×256 pixels |
| **Source** | Google Earth |

**Comprehensive coverage:**
Wide range of scene types cho robust evaluation.

## 6.15. ClassificationTask trong TorchGeo

### 6.15.1. Overview

TorchGeo cung cấp `ClassificationTask` Lightning module cho streamlined training.

**Key Features:**
- Configurable model và backbone
- Support cho multi-class và multi-label
- Standard metrics tracking
- Learning rate scheduling

### 6.15.2. Configuration Options

**Model Selection:**
- Backbone architecture (resnet50, vit_small_patch16_224, etc.)
- Pre-trained weights source
- Number of output classes

**Training:**
- Learning rate và scheduler
- Loss function (CE, BCE, Focal)
- Optimizer (Adam, SGD, AdamW)

**Data:**
- Batch size
- Number of workers
- Augmentation pipeline

### 6.15.3. Multi-label Classification

Cho datasets như BigEarthNet với multiple labels per image:
- BCE loss thay vì Cross Entropy
- Sigmoid activation thay vì Softmax
- Threshold tuning cho optimal F1

### 6.15.4. Metrics

**Classification Metrics:**
- Overall Accuracy
- Per-class Accuracy
- F1 Score (macro, micro, weighted)
- Precision và Recall
- Confusion Matrix

## 6.16. Training Best Practices

### 6.16.1. Data Preprocessing

**Normalization:**
TorchGeo provides per-band statistics cho common sensors:
- Sentinel-2 mean/std
- Landsat mean/std
- Sensor-specific normalization

**Band Selection:**
Not all bands equally useful:
- RGB (B4, B3, B2) cho visual
- Near-infrared bands cho vegetation
- SWIR cho moisture
- Task-dependent selection

### 6.16.2. Augmentation

**Recommended Augmentations:**
- Random horizontal/vertical flip
- Random rotation
- Random crop và resize
- Color jittering (for RGB)

**Multi-spectral Considerations:**
- Maintain band relationships
- Avoid augmentations that break physical meaning
- Consider sensor-specific constraints

### 6.16.3. Transfer Learning Strategy

**Fine-tuning:**
1. Load pre-trained weights
2. Replace classification head
3. Freeze backbone initially
4. Gradually unfreeze layers
5. Lower learning rate for pre-trained layers

**Linear Probing:**
- Freeze all pre-trained layers
- Only train classification head
- Quick baseline evaluation
- Test feature quality

### 6.16.4. Hyperparameter Tuning

**Learning Rate:**
- Lower for pre-trained: 1e-4 to 1e-3
- Higher for scratch: 1e-3 to 1e-2
- Warmup recommended

**Batch Size:**
- Larger batches: More stable, need higher LR
- Typical: 32-128 depending on GPU memory

**Epochs:**
- Dataset size dependent
- Early stopping based on validation
- Typically 50-200 epochs

## 6.17. Benchmark Results

### 6.17.1. EuroSAT Results

| Model | Pre-training | Accuracy |
|-------|--------------|----------|
| ResNet-50 | ImageNet | 95.5% |
| ResNet-50 | SSL4EO MoCo | 97.2% |
| ViT-B | SSL4EO MAE | 97.8% |
| Swin-T | SSL4EO | 97.5% |

### 6.17.2. UC Merced Results

| Model | Pre-training | Accuracy |
|-------|--------------|----------|
| ResNet-50 | ImageNet | 96.8% |
| ResNet-50 | Million-AID | 98.1% |
| ViT-B | SSL4EO | 98.4% |

### 6.17.3. BigEarthNet Results

| Model | Pre-training | mAP |
|-------|--------------|-----|
| ResNet-50 | ImageNet | 75.2 |
| ResNet-50 | SSL4EO | 79.8 |
| ViT-B | SSL4EO MAE | 82.3 |

**Observations:**
- Domain-specific pre-training consistently outperforms ImageNet
- Transformers show advantage on large datasets
- Multi-spectral pre-training crucial for Sentinel-2 tasks

## 6.18. Advanced Topics

### 6.18.1. Few-shot Classification

Khi labeled data hạn chế:
- Linear probing với strong pre-trained features
- Meta-learning approaches
- Semi-supervised methods

TorchGeo pre-trained models đặc biệt valuable cho few-shot scenarios.

### 6.18.2. Temporal Classification

For time-series classification:
- Stack temporal features
- Use recurrent architectures on top
- Temporal attention mechanisms

### 6.18.3. Multi-modal Fusion

Combine different data sources:
- Early fusion: Concatenate inputs
- Late fusion: Combine predictions
- Mid-level fusion: Merge features

TorchGeo supports loading multiple datasets với IntersectionDataset.

### 6.18.4. Uncertainty Estimation

Quantify prediction confidence:
- Monte Carlo Dropout
- Deep Ensembles
- Evidential Deep Learning

Important cho operational remote sensing applications.

## 6.19. Use Cases

### 6.19.1. Land Use/Land Cover Mapping

**Application:**
Classify land areas into categories (urban, forest, water, agriculture, etc.)

**Approach:**
- EuroSAT hoặc similar dataset cho training
- ResNet-50 với SSL4EO weights
- Fine-tune for target region

### 6.19.2. Crop Type Classification

**Application:**
Identify crop types from satellite imagery

**Approach:**
- Temporal stack of images (growing season)
- Multi-spectral bands including red edge, NIR
- Domain-specific pre-training valuable

### 6.19.3. Disaster Assessment

**Application:**
Classify damage levels sau natural disasters

**Approach:**
- Pre/post event image comparison
- Fine-grained classification (intact, damaged, destroyed)
- Transfer learning từ related tasks

### 6.19.4. Infrastructure Monitoring

**Application:**
Detect và classify infrastructure (airports, ports, power plants)

**Approach:**
- High-resolution imagery
- Fine-grained categories
- Object-oriented classification

Classification trong TorchGeo provides powerful tools cho diverse remote sensing applications, với pre-trained models significantly reducing data và compute requirements cho new tasks.

