# Pre-trained Weights và Sensor Support trong TorchGeo

## 6.40. Tổng quan Pre-trained Weights

Pre-trained weights là một trong những đóng góp quan trọng nhất của TorchGeo cho remote sensing community. Thay vì training models từ scratch hoặc sử dụng ImageNet pretrained weights không phù hợp với satellite imagery, TorchGeo cung cấp weights được train đặc biệt trên dữ liệu vệ tinh, cho performance tốt hơn đáng kể trên downstream tasks.

Trong phần này, chúng ta sẽ phân tích chi tiết các pre-trained weights có sẵn, sensor support, và cách sử dụng hiệu quả cho các ứng dụng khác nhau.

## 6.41. SSL4EO Pre-trained Weights

### 6.41.1. Tổng quan SSL4EO

SSL4EO (Self-Supervised Learning for Earth Observation) là initiative để tạo pre-trained models cho Earth Observation data sử dụng self-supervised learning.

**SSL4EO-S12:**
Dataset và weights cho Sentinel-1 và Sentinel-2:
- 200,000+ image triplets
- European coverage
- Seasonal variations captured
- Multiple pre-training methods

**Tại sao Self-supervised:**
- Không cần labeled data (expensive và limited cho remote sensing)
- Learns general representations
- Transfers well to various tasks
- Scales với available data

### 6.41.2. MoCo Pre-training

Momentum Contrast (MoCo) cho remote sensing:

**Method:**
- Contrastive learning framework
- Query và key encoders
- Momentum update cho key encoder
- Large negative sample queue

**SSL4EO MoCo Weights:**

| Model | Input | Weight Name |
|-------|-------|-------------|
| ResNet-18 | Sentinel-2 All Bands | SENTINEL2_ALL_MOCO |
| ResNet-50 | Sentinel-2 All Bands | SENTINEL2_ALL_MOCO |
| ResNet-50 | Sentinel-2 RGB Only | SENTINEL2_RGB_MOCO |
| ResNet-50 | Sentinel-1 All Bands | SENTINEL1_ALL_MOCO |

**Usage:**
```python
from torchgeo.models import ResNet50_Weights, resnet50

# Load với Sentinel-2 MoCo weights
weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
model = resnet50(weights=weights)

# Access transforms
transform = weights.transforms()
```

### 6.41.3. MAE Pre-training

Masked Autoencoder pre-training cho Vision Transformers:

**Method:**
- Mask random patches of image
- Reconstruct masked patches
- Learns strong representations

**SSL4EO MAE Weights:**

| Model | Input | Weight Name |
|-------|-------|-------------|
| ViT-Small | Sentinel-2 | SENTINEL2_ALL_MAE |
| ViT-Base | Sentinel-2 | SENTINEL2_ALL_MAE |

**Advantages:**
- Excellent cho ViT architectures
- Data-efficient training
- Strong transfer performance

### 6.41.4. DINO Pre-training

Self-Distillation với No Labels:

**Method:**
- Student-teacher framework
- Teacher is momentum-updated student
- Knowledge distillation without labels

**SSL4EO DINO Weights:**

| Model | Input | Weight Name |
|-------|-------|-------------|
| ViT-Small | Sentinel-2 | SENTINEL2_ALL_DINO |
| ViT-Base | Sentinel-2 | SENTINEL2_ALL_DINO |

**Characteristics:**
- Good for ViT models
- Learns semantic features
- Works well on diverse tasks

## 6.42. SatMAE Weights

### 6.42.1. Overview

SatMAE (Satellite Masked Autoencoder) specifically designed cho satellite imagery:

**Key Features:**
- Temporal encoding cho time series
- Multi-spectral positional embeddings
- Global-scale pre-training
- Designed cho fMoW dataset

### 6.42.2. Available Weights

| Model | Dataset | Resolution |
|-------|---------|------------|
| ViT-Large | fMoW RGB | Various |
| ViT-Large | fMoW Temporal | Multi-date |

### 6.42.3. Temporal Capabilities

SatMAE handles temporal dimension:
- Multiple dates as input
- Temporal position encoding
- Good for change detection và time series

## 6.43. Sensor-specific Weights

### 6.43.1. Sentinel-2 Weights

**Available Pre-training:**
- SSL4EO MoCo (ResNet)
- SSL4EO MAE (ViT)
- SSL4EO DINO (ViT)
- SatMAE (ViT)
- BigEarthNet supervised

**Band Support:**
- All 13 bands: Complete spectral information
- RGB only: B4, B3, B2 (10m resolution)
- RGB + NIR: B4, B3, B2, B8

**Usage với All Bands:**
```python
# ResNet-50 với 13-band Sentinel-2
weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
model = resnet50(weights=weights, in_chans=13)
```

### 6.43.2. Sentinel-1 Weights

**Available Pre-training:**
- SSL4EO MoCo

**Polarization Support:**
- VV + VH: Dual polarization
- Single polarization options

**SAR-specific Considerations:**
- Log-scale normalization
- Speckle handling
- Different value distributions than optical

**Usage:**
```python
# ResNet-50 cho Sentinel-1
weights = ResNet50_Weights.SENTINEL1_ALL_MOCO
model = resnet50(weights=weights, in_chans=2)  # VV, VH
```

### 6.43.3. Landsat Weights

**Current Status:**
- Fewer dedicated weights than Sentinel
- Can use Sentinel weights với adaptation
- Some benchmark-specific weights

**Approach:**
- Use Sentinel-2 weights (similar bands)
- Adapt first layer for Landsat bands
- Fine-tune on Landsat data

### 6.43.4. High-resolution Imagery

For aerial và very-high-resolution satellite:

**Options:**
- NAIP-specific training
- ImageNet weights (often sufficient for RGB)
- Million-AID pre-training

**Considerations:**
- Usually RGB or RGBIR
- Higher spatial resolution
- Different object scales

## 6.44. Weight Comparison và Selection

### 6.44.1. Performance Comparison

**EuroSAT (Sentinel-2 Classification):**

| Weights | ResNet-50 Accuracy |
|---------|-------------------|
| Random Init | 89.2% |
| ImageNet | 95.5% |
| SSL4EO MoCo | 97.2% |
| SSL4EO MAE (ViT) | 97.8% |

**Key Insight:** Domain-specific pre-training outperforms ImageNet by 1.5-2%.

### 6.44.2. Selection Guidelines

**Sentinel-2 Tasks:**
- First choice: SSL4EO MoCo/MAE weights
- Alternative: BigEarthNet pre-trained

**Sentinel-1 Tasks:**
- Use: SSL4EO Sentinel-1 weights
- Critical: Match polarization channels

**High-resolution RGB:**
- ImageNet weights often sufficient
- Million-AID for aerial specific
- Fine-tuning usually needed

**Multi-sensor:**
- Separate encoders với sensor-specific weights
- Fusion at feature level

### 6.44.3. When to Use What

| Scenario | Recommended Weights |
|----------|---------------------|
| Sentinel-2 classification | SSL4EO MoCo/MAE |
| Sentinel-2 segmentation | SSL4EO MoCo encoder |
| Sentinel-1 classification | SSL4EO S1 MoCo |
| SAR ship detection | SSL4EO S1 or ImageNet |
| High-res buildings | ImageNet or Million-AID |
| Temporal analysis | SatMAE |

## 6.45. Adapting Weights cho Different Inputs

### 6.45.1. Channel Mismatch

When input channels differ từ pre-trained weights:

**Fewer Channels:**
```python
# Pre-trained on 13 bands, using 4 (RGBIR)
# Option 1: Select matching weights
model = resnet50(weights=weights)
# Manually select first conv weights for bands

# Option 2: Average redundant channels
# Pre-trained weights: 13 channels
# New input: 4 channels
# Average groups of weights
```

**More Channels:**
```python
# Pre-trained on 3 (RGB), using 13
# Duplicate và tile RGB weights
# Or initialize extra channels randomly
```

**TorchGeo Approach:**
TorchGeo models often handle this automatically với in_chans parameter.

### 6.45.2. Resolution Mismatch

When spatial resolution differs:

**Interpolation:**
- Bilinear resize input to expected resolution
- Or modify patch/window sizes

**Considerations:**
- Very different resolutions may need different architectures
- Feature scales may not transfer well

### 6.45.3. Value Range Adaptation

Different sensors have different value ranges:

**Normalization:**
- Use sensor-specific statistics
- Match distribution của pre-training data
- TorchGeo provides normalization stats

## 6.46. Fine-tuning Strategies

### 6.46.1. Full Fine-tuning

Update all parameters:
- Best when sufficient data available
- Highest performance potential
- Risk of overfitting on small datasets

```python
model = resnet50(weights=ssl4eo_weights)
optimizer = Adam(model.parameters(), lr=1e-4)
```

### 6.46.2. Linear Probing

Freeze pre-trained layers, only train classifier:
- Quick baseline
- Tests feature quality
- Works với minimal data

```python
model = resnet50(weights=ssl4eo_weights)
# Freeze backbone
for param in model.parameters():
    param.requires_grad = False
# Unfreeze classifier
for param in model.fc.parameters():
    param.requires_grad = True
```

### 6.46.3. Progressive Unfreezing

Gradually unfreeze layers:
1. Train classifier only
2. Unfreeze last block
3. Unfreeze more blocks
4. Fine-tune entire model

**Benefits:**
- Stable training
- Preserves pre-trained features
- Good for limited data

### 6.46.4. Layer-wise Learning Rates

Different learning rates for different depths:
- Lower LR for early (pre-trained) layers
- Higher LR for later layers và new heads

```python
optimizer = Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
```

## 6.47. Practical Considerations

### 6.47.1. Memory và Compute

**Model Sizes:**

| Model | Parameters | Memory (inference) |
|-------|------------|-------------------|
| ResNet-18 | 11M | ~200MB |
| ResNet-50 | 25M | ~400MB |
| ViT-Base | 86M | ~700MB |
| ViT-Large | 307M | ~2.5GB |

**Considerations:**
- Edge deployment: ResNet-18, MobileNet
- Server deployment: ResNet-50, ViT-Base
- Research: ViT-Large

### 6.47.2. Inference Speed

**Typical FPS (GPU):**

| Model | Input Size | FPS |
|-------|------------|-----|
| ResNet-18 | 224×224 | ~500 |
| ResNet-50 | 224×224 | ~200 |
| ViT-Base | 224×224 | ~100 |

**Considerations:**
- Real-time applications: ResNet-18, MobileNet
- Batch processing: ResNet-50, ViT-Base acceptable

### 6.47.3. Data Requirements

**Guidelines:**

| Data Amount | Strategy |
|-------------|----------|
| <100 samples | Linear probing only |
| 100-1000 | Progressive unfreezing |
| 1000-10000 | Full fine-tuning với care |
| >10000 | Full fine-tuning |

Pre-trained weights reduce data requirements by 10x or more.

## 6.48. Accessing Weights

### 6.48.1. TorchGeo Weight Enum

```python
from torchgeo.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ViTSmall16_Weights,
)

# List available weights
print(ResNet50_Weights.__members__)
```

### 6.48.2. Loading Weights

```python
# Method 1: Through model factory
model = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO)

# Method 2: Load và apply manually
weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
model = resnet50()
model.load_state_dict(weights.get_state_dict(progress=True))

# Method 3: Custom loading
state_dict = torch.load("path/to/weights.pth")
model.load_state_dict(state_dict)
```

### 6.48.3. Weight Metadata

Weights include metadata:

```python
weights = ResNet50_Weights.SENTINEL2_ALL_MOCO

# Access metadata
print(weights.meta)
# {'bands': ['B01', 'B02', ...], 'mean': [...], 'std': [...]}

# Get transforms
transform = weights.transforms()
```

## 6.49. Future Directions

### 6.49.1. More Sensors

Expanding support cho:
- MODIS
- VIIRS
- Pléiades, WorldView
- Commercial SAR (ICEYE, Capella)
- Hyperspectral

### 6.49.2. Foundation Models

Trend toward large foundation models:
- IBM/NASA Prithvi
- Google/DeepMind Earth models
- Generalist remote sensing models

### 6.49.3. Multi-modal Pre-training

Joint pre-training on:
- Optical + SAR
- Satellite + text
- Image + location

### 6.49.4. Efficient Pre-training

Reducing compute requirements:
- Knowledge distillation
- Efficient architectures
- Progressive training

## 6.50. Summary và Recommendations

### 6.50.1. Quick Start Recommendations

**Sentinel-2 RGB:**
- ResNet-50 + SSL4EO MoCo
- Good balance of performance và efficiency

**Sentinel-2 All Bands:**
- ResNet-50 + SSL4EO MoCo (all bands)
- Use all spectral information

**Sentinel-1 SAR:**
- ResNet-50 + SSL4EO S1 MoCo
- Essential cho SAR applications

**Vision Transformer:**
- ViT-Base + SSL4EO MAE
- Best overall performance

### 6.50.2. Best Practices

1. **Always use domain-specific weights** when available
2. **Match input channels** to pre-training
3. **Use appropriate normalization** statistics
4. **Start với linear probing** to test
5. **Progressive unfreezing** for limited data
6. **Monitor validation metrics** for overfitting

### 6.50.3. Common Mistakes

- Using ImageNet weights without adaptation for multi-spectral
- Ignoring normalization differences
- Full fine-tuning với insufficient data
- Wrong input channel configuration
- Mismatched transforms

TorchGeo pre-trained weights represent significant advancement cho remote sensing deep learning, enabling practitioners to achieve strong performance với reduced data và compute requirements.

