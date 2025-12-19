# Chương 5: Change Detection Models trong TorchGeo

## 6.30. Tổng quan Change Detection

Change detection là task xác định và phân tích những thay đổi trên bề mặt Trái Đất qua thời gian bằng cách so sánh ảnh vệ tinh từ các thời điểm khác nhau. Đây là một trong những ứng dụng quan trọng nhất của remote sensing, với applications trong urban expansion monitoring, disaster assessment, deforestation tracking, agricultural monitoring, và nhiều lĩnh vực khác.

TorchGeo cung cấp support cho change detection thông qua datasets, data loading utilities cho temporal data, và integration với các deep learning approaches cho bi-temporal và multi-temporal change detection.

## 6.31. Formulations của Change Detection

### 6.31.1. Binary Change Detection

Đơn giản nhất - xác định change hay no-change cho mỗi pixel:
- Input: Two images (pre và post)
- Output: Binary mask (0 = no change, 1 = change)

**Pros:**
- Simple formulation
- Straightforward evaluation
- Less annotation effort

**Cons:**
- Không phân loại loại change
- May miss subtle changes
- Limited information for analysis

### 6.31.2. Semantic Change Detection

Phân loại loại change cho mỗi pixel:
- Input: Two images
- Output: Multi-class mask indicating change type

**Example Classes:**
- No change
- Building construction
- Building demolition
- Vegetation gain
- Vegetation loss
- Water body change

**Pros:**
- Rich information
- Actionable insights
- Better understanding of dynamics

**Cons:**
- More annotation required
- Class imbalance issues
- More complex models

### 6.31.3. From-to Change Detection

Identify both source và destination land cover:
- Input: Two images
- Output: Change matrix (from class A to class B)

**Example:**
- Forest → Urban
- Agricultural → Forest
- Water → Land (or vice versa)

**Pros:**
- Complete change characterization
- Supports transition analysis
- Links với land cover classification

**Cons:**
- N×N possible transitions cho N classes
- Sparse annotations
- Complex training

### 6.31.4. Multi-temporal Change Detection

Extend beyond bi-temporal to time series:
- Input: Sequence of images (3 or more dates)
- Output: Change points, trajectories, or trends

**Applications:**
- Continuous monitoring
- Trend analysis
- Seasonal pattern detection

## 6.32. Deep Learning Architectures

### 6.32.1. Siamese Networks

Siamese architecture xử lý hai images với shared weights:

**Architecture:**
```
Image_t1 → Encoder → Features_t1
Image_t2 → Encoder → Features_t2
(Features_t1, Features_t2) → Comparison Module → Change Map
```

**Key Components:**
- **Shared Encoder:** Same CNN/Transformer processes both images
- **Comparison Module:** Computes difference/similarity
- **Decoder:** Generates change map

**Comparison Strategies:**
- Feature concatenation: [F1, F2]
- Feature difference: |F1 - F2|
- Feature correlation: F1 * F2
- Learned comparison

**Advantages:**
- Weight sharing ensures consistent feature extraction
- Naturally handles temporal comparison
- Flexible comparison module

### 6.32.2. UNet-based Change Detection

Adapting U-Net cho change detection:

**Early Fusion:**
- Concatenate t1 và t2 images as input (6 channels for RGB)
- Single U-Net processes concatenated input
- Output is change map

**Late Fusion:**
- Separate encoders cho t1 và t2
- Merge features at bottleneck or decoder
- Share decoder

**Skip Connection Modifications:**
- Temporal attention trong skip connections
- Difference features in skip connections

### 6.32.3. FC-EF và FC-Siam-Conc/Diff

Fully Convolutional architectures specifically designed cho change detection:

**FC-EF (Early Fusion):**
- Concatenate images as input
- Standard FCN architecture
- Simple but effective

**FC-Siam-Conc (Siamese Concatenation):**
- Siamese encoder
- Concatenate features at each level
- Shared decoder

**FC-Siam-Diff (Siamese Difference):**
- Siamese encoder
- Compute difference at each level
- Shared decoder

### 6.32.4. Attention-based Models

**DTCDSCN (Dual-Task Change Detection with Semantic Consistency):**
- Dual attention modules
- Semantic consistency constraint
- Multi-task learning

**BIT (Binary Transformer):**
- Transformer-based change detection
- Self-attention for temporal comparison
- State-of-the-art performance

### 6.32.5. STANet (Spatial-Temporal Attention Network)

Combines spatial và temporal attention:
- PAM (Position Attention Module)
- BAM (Basic Attention Module)
- Multi-scale features

## 6.33. Change Detection Datasets

### 6.33.1. OSCD (Onera Satellite Change Detection)

| Attribute | Value |
|-----------|-------|
| **Source** | Sentinel-2 |
| **Pairs** | 24 image pairs |
| **Size** | Variable (600×600 to 10000×10000) |
| **Classes** | Binary (change/no-change) |
| **Change Type** | Urban change |

**Đặc điểm:**
- Urban areas worldwide
- Multi-spectral (13 bands)
- Challenging scenarios

**Usage trong TorchGeo:**
```
from torchgeo.datasets import OSCD

dataset = OSCD(root="data", split="train", download=True)
sample = dataset[0]
# sample["image1"], sample["image2"], sample["mask"]
```

### 6.33.2. LEVIR-CD

| Attribute | Value |
|-----------|-------|
| **Source** | Google Earth |
| **Pairs** | 637 image pairs |
| **Size** | 1024×1024 pixels |
| **Resolution** | 0.5m |
| **Classes** | Binary (building change) |

**Đặc điểm:**
- Building construction và demolition
- High resolution
- Large dataset
- 10-year time span

### 6.33.3. WHU Building Change Detection

| Attribute | Value |
|-----------|-------|
| **Source** | Aerial imagery |
| **Pairs** | 1 pair |
| **Size** | 32507×15354 pixels |
| **Resolution** | 0.075m |
| **Classes** | Binary |

**Đặc điểm:**
- Very high resolution
- Building change only
- Single large area

### 6.33.4. SECOND (Semantic Change Detection Dataset)

| Attribute | Value |
|-----------|-------|
| **Source** | Aerial imagery |
| **Pairs** | 4662 pairs |
| **Size** | 512×512 pixels |
| **Classes** | 6 semantic changes |

**Classes:**
- No change
- Change to low vegetation
- Change to trees
- Change to water
- Change to playgrounds
- Change to buildings

### 6.33.5. xView2

| Attribute | Value |
|-----------|-------|
| **Source** | WorldView, GeoEye |
| **Task** | Building damage assessment |
| **Classes** | 4 damage levels |
| **Events** | Multiple natural disasters |

**Classes:**
- No damage
- Minor damage
- Major damage
- Destroyed

**Đặc điểm:**
- Real disaster events
- Pre và post-disaster imagery
- Building-level damage assessment

### 6.33.6. SpaceNet 7

| Attribute | Value |
|-----------|-------|
| **Source** | Planet |
| **Task** | Monthly building change |
| **Duration** | 24 months |
| **Coverage** | 100 global sites |

**Đặc điểm:**
- Multi-temporal (not just bi-temporal)
- Monthly observations
- Building tracking over time

## 6.34. Training Change Detection Models

### 6.34.1. Data Loading

**Bi-temporal Loading:**
```
# Load paired images
sample = {
    "image1": preprocess(load(path_t1)),
    "image2": preprocess(load(path_t2)),
    "mask": load(change_mask)
}
```

**TorchGeo Approach:**
- Sử dụng custom dataset hoặc adapt existing
- Handle temporal alignment
- Consistent preprocessing

### 6.34.2. Augmentation

**Paired Augmentation:**
- Apply same geometric transform to both images
- Ensures spatial alignment maintained
- Photometric can differ (different acquisition conditions)

**Recommended Augmentations:**
- Random flip (same for both)
- Random rotation (same for both)
- Random crop (same for both)
- Color jittering (separate for each, mimics condition variation)

### 6.34.3. Loss Functions

**Binary Cross Entropy:**
Standard cho binary change detection.

**Focal Loss:**
Handles severe class imbalance (most pixels unchanged).

**Dice Loss:**
Better for imbalanced segmentation.

**Combined Loss:**
```
L = α * BCE + β * Dice
```

### 6.34.4. Class Imbalance

Change detection has extreme imbalance (typically <5% change):

**Strategies:**
- Focal Loss với appropriate γ
- Class weighting
- Over-sampling change areas
- Patch selection biased toward change

### 6.34.5. Evaluation Metrics

**Pixel-level:**
- Precision, Recall, F1
- IoU (Intersection over Union)
- Overall Accuracy
- Kappa coefficient

**Object-level:**
- Detection rate
- False alarm rate
- Per-change accuracy

## 6.35. Pre-trained Models cho Change Detection

### 6.35.1. Transfer từ Classification/Segmentation

Using pre-trained encoders:
- Load SSL4EO hoặc similar weights
- Use as encoder in Siamese/U-Net architecture
- Fine-tune for change detection

**Approach:**
```
# Pre-trained encoder
encoder = resnet50(weights=SSL4EO_WEIGHTS)

# Siamese change detection
class SiameseCD(nn.Module):
    def __init__(self, encoder):
        self.encoder = encoder
        self.decoder = build_decoder()

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        diff = torch.abs(f1 - f2)
        return self.decoder(diff)
```

### 6.35.2. Change Detection Specific Pre-training

Emerging approaches:
- Self-supervised với temporal data
- Contrastive learning across time
- Masked autoencoder for change

### 6.35.3. Available Implementations

TorchGeo integrates với:
- **OpenCD:** Open source change detection toolbox
- **Custom implementations:** For common architectures
- **torchvision compatibility:** Leverage existing models

## 6.36. Benchmark Results

### 6.36.1. OSCD Results

| Model | F1 | IoU |
|-------|----|----|
| FC-EF | 23.0 | 13.0 |
| FC-Siam-Diff | 28.5 | 16.6 |
| U-Net CD | 35.2 | 21.4 |
| BIT | 52.1 | 35.2 |

### 6.36.2. LEVIR-CD Results

| Model | F1 | IoU |
|-------|----|----|
| FC-Siam-Conc | 83.4 | 71.6 |
| FC-Siam-Diff | 86.3 | 75.9 |
| DTCDSCN | 87.8 | 78.3 |
| BIT | 89.3 | 80.7 |

### 6.36.3. xView2 Results

| Model | F1 Localization | F1 Classification |
|-------|-----------------|-------------------|
| U-Net baseline | 75.0 | 42.0 |
| HRNet | 82.5 | 53.2 |
| Competition winners | 88.0 | 65.0 |

## 6.37. Advanced Topics

### 6.37.1. Multi-temporal Change Detection

Beyond bi-temporal:

**Time Series Processing:**
- 3D convolutions
- Recurrent networks (LSTM, GRU)
- Temporal attention

**Architectures:**
- ConvLSTM-based change detection
- Temporal Transformer
- U-TAE (U-Net with Temporal Attention)

**Applications:**
- Continuous monitoring
- Change trajectory analysis
- Anomaly detection in time series

### 6.37.2. Multi-sensor Change Detection

Using different sensors at t1 và t2:

**Challenges:**
- Different spectral bands
- Different resolutions
- Different viewing geometries

**Approaches:**
- Sensor-specific encoders
- Domain adaptation
- Learned alignment

### 6.37.3. Unsupervised Change Detection

When labels unavailable:

**Deep Learning Approaches:**
- Autoencoder-based (reconstruction error as change)
- Deep clustering
- Self-supervised with pseudo-labels

**Traditional (for comparison):**
- Image differencing
- CVA (Change Vector Analysis)
- PCA-based methods

### 6.37.4. Weakly Supervised Change Detection

With limited labels:

**Approaches:**
- Image-level labels (change exists or not)
- Partial pixel labels
- Active learning for annotation
- Semi-supervised methods

## 6.38. Use Cases

### 6.38.1. Urban Expansion Monitoring

**Objective:**
Track city growth over time.

**Approach:**
- Annual or multi-year intervals
- Building và impervious surface detection
- Semantic change (vegetation → urban)

**Data:**
- Sentinel-2 hoặc high-resolution
- LEVIR-CD type training

### 6.38.2. Disaster Damage Assessment

**Objective:**
Rapidly assess damage after disasters.

**Approach:**
- Pre-event baseline imagery
- Post-event rapid acquisition
- Building damage classification

**Data:**
- xView2 training
- Various satellite sources
- Rapid response requirement

### 6.38.3. Deforestation Monitoring

**Objective:**
Detect forest loss.

**Approach:**
- Regular (monthly/quarterly) monitoring
- Binary change (forest → non-forest)
- Alert generation

**Data:**
- Sentinel-1 (SAR, all-weather)
- Sentinel-2 (optical)
- Landsat (historical)

### 6.38.4. Agricultural Change

**Objective:**
Monitor agricultural practices và land use.

**Approach:**
- Seasonal comparison
- Crop rotation detection
- Field boundary changes

**Data:**
- Multi-temporal Sentinel-2
- Crop type ground truth
- Agricultural calendars

### 6.38.5. Coastal Erosion Monitoring

**Objective:**
Track shoreline changes.

**Approach:**
- Annual hoặc seasonal comparison
- Land-water boundary detection
- Long-term trend analysis

**Data:**
- Sentinel-2, Landsat
- Historical archives
- Tide-normalized selection

## 6.39. Implementation trong TorchGeo

### 6.39.1. Loading Change Detection Data

```python
# Using OSCD dataset
from torchgeo.datasets import OSCD

dataset = OSCD(root="data", split="train")

for sample in dataset:
    image1 = sample["image1"]  # Pre-change
    image2 = sample["image2"]  # Post-change
    mask = sample["mask"]      # Change mask
```

### 6.39.2. Custom Change Detection Dataset

```python
class CustomChangeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.pairs = load_pairs(root)
        self.transform = transform

    def __getitem__(self, idx):
        t1_path, t2_path, mask_path = self.pairs[idx]

        sample = {
            "image1": load_image(t1_path),
            "image2": load_image(t2_path),
            "mask": load_mask(mask_path)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
```

### 6.39.3. Model Training

```python
# Simple change detection training
model = SiameseUNet(encoder="resnet50", pretrained=True)
criterion = FocalLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs):
    for batch in dataloader:
        img1, img2, mask = batch["image1"], batch["image2"], batch["mask"]

        pred = model(img1, img2)
        loss = criterion(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Change detection trong TorchGeo enables powerful temporal analysis of Earth observation data, supporting critical applications từ urban planning to disaster response.

