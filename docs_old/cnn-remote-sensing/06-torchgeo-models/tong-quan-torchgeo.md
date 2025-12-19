# Tổng quan về TorchGeo

## 6.1. Giới thiệu TorchGeo

TorchGeo là một thư viện Python mã nguồn mở được phát triển bởi Microsoft Research, cung cấp datasets, samplers, transforms, và pre-trained models cho geospatial machine learning với PyTorch. TorchGeo ra đời từ nhận thức rằng dữ liệu địa lý (geospatial data) có những đặc điểm riêng biệt khác với natural images thông thường, và việc áp dụng trực tiếp các phương pháp computer vision tiêu chuẩn thường không đạt hiệu quả tối ưu.

TorchGeo được thiết kế để bridge the gap giữa remote sensing community và deep learning community, cung cấp các công cụ chuẩn hóa cho việc làm việc với satellite imagery, aerial imagery, và các dạng geospatial data khác. Thư viện được tích hợp chặt chẽ với PyTorch ecosystem, tận dụng các thành phần như torchvision, Lightning, và các thư viện phổ biến khác.

## 6.2. Tại sao cần TorchGeo

### 6.2.1. Đặc điểm Riêng biệt của Geospatial Data

Dữ liệu địa lý khác biệt cơ bản với natural images ở nhiều khía cạnh:

**1. Số lượng Bands/Channels:**
Trong khi natural images có 3 channels (RGB), satellite imagery có thể có từ vài đến hàng trăm bands spectral. Sentinel-2 có 13 bands, Landsat 8 có 11 bands, các hyperspectral sensors có thể có hàng trăm bands. Các mô hình tiêu chuẩn (ResNet, VGG, etc.) được thiết kế cho 3-channel input và cần được adapted.

**2. Data Types:**
- Natural images: 8-bit unsigned integer (0-255)
- Satellite imagery: 16-bit, 32-bit, floating point; có thể có negative values (calibrated reflectance, temperature)
- DEM: Floating point elevation values
- SAR: Complex numbers, logarithmic scale

**3. Spatial Resolution và Extent:**
Satellite images có thể cover hàng nghìn kilometers với trillions of pixels. Xử lý toàn bộ image không khả thi - cần tiling và sampling strategies.

**4. Coordinate Reference Systems (CRS):**
Geospatial data có thông tin về vị trí địa lý, được lưu trong các CRS khác nhau (UTM, WGS84, etc.). Việc align data từ nhiều sources yêu cầu xử lý CRS cẩn thận.

**5. Temporal Dimension:**
Nhiều ứng dụng remote sensing yêu cầu time series analysis. Data từ các thời điểm khác nhau cần được aligned và combined.

**6. Multi-modal Nature:**
Combination của optical, SAR, DEM, và các data types khác là common. Fusion strategies cần được hỗ trợ.

### 6.2.2. Limitations của Existing Tools

**Torchvision:**
- Designed cho natural images (RGB, 8-bit)
- Transforms assume 3 channels
- Datasets assume standard image formats
- No support cho geospatial metadata

**Scikit-learn, TensorFlow:**
- Similar limitations
- No native geospatial support

**GDAL/Rasterio:**
- Excellent cho reading/writing geospatial data
- No ML integration
- Need bridge to PyTorch

TorchGeo fills this gap bằng việc providing:
- Geospatial-aware datasets và data loading
- Transforms that work với multi-spectral data
- Samplers for large raster data
- Pre-trained models cho remote sensing
- Integration với existing tools (rasterio, pyproj)

## 6.3. Kiến trúc TorchGeo

### 6.3.1. Core Components

TorchGeo được tổ chức thành các modules chính:

**1. Datasets:**
- `GeoDataset`: Base class cho geospatial datasets với CRS và bounds
- `NonGeoDataset`: Wrapper cho standard image datasets
- Built-in datasets: Sentinel, Landsat, NAIP, và nhiều benchmarks

**2. Samplers:**
- `RandomGeoSampler`: Random sampling từ large rasters
- `GridGeoSampler`: Systematic grid sampling
- `PreChippedGeoSampler`: For pre-tiled data
- Handle overlapping data sources với different extents

**3. Transforms:**
- Augmentations compatible với multi-spectral data
- Kornia-based transforms
- Indices computation (NDVI, NDWI, etc.)

**4. Models:**
- Pre-trained backbones cho remote sensing
- Task-specific heads (classification, segmentation)
- Support cho various architectures

**5. Trainers:**
- Lightning-based training loops
- Classification, segmentation, object detection
- Handles common training patterns

### 6.3.2. Dataset Abstraction

**GeoDataset:**
Core abstraction cho geospatial data. Mỗi GeoDataset có:
- `crs`: Coordinate reference system
- `res`: Spatial resolution
- `bounds`: Geographic extent (BoundingBox)
- `__getitem__`: Returns data for a given bounding box query

GeoDatasets có thể được combined using set operations:
- Union (`|`): Combine datasets, return first available
- Intersection (`&`): Return only where all datasets overlap

**IntersectionDataset và UnionDataset:**
```
# Combine Sentinel-2 và DEM data
dataset = sentinel2 & dem  # Only where both available
dataset = sentinel2 | dem  # Either one
```

### 6.3.3. Sampling Strategy

Cho large raster data, TorchGeo provides samplers:

**RandomGeoSampler:**
- Sample random patches từ dataset extent
- Respects data availability (only samples where data exists)
- Configurable patch size

**GridGeoSampler:**
- Systematic sampling on regular grid
- Useful cho inference over entire area
- Configurable stride/overlap

**Batch Sampling:**
- Multiple patches per batch
- Handles variable-size inputs
- Efficient data loading

## 6.4. Datasets có sẵn trong TorchGeo

### 6.4.1. Satellite Imagery Datasets

**Sentinel-1:**
- C-band SAR data
- Dual polarization (VV, VH)
- GRD products support

**Sentinel-2:**
- 13 spectral bands
- 10m, 20m, 60m resolutions
- Surface reflectance products

**Landsat:**
- Landsat 7, 8, 9 support
- Multiple products (Surface Reflectance, etc.)
- Historical data access

**NAIP (National Agriculture Imagery Program):**
- High-resolution aerial imagery (1m)
- RGBIR bands
- Continental US coverage

### 6.4.2. Benchmark Datasets

TorchGeo includes nhiều benchmark datasets cho evaluation:

**Classification:**
- EuroSAT: 10 land use classes, Sentinel-2
- UC Merced: 21 land use classes, aerial imagery
- PatternNet: 38 classes, high-resolution satellite
- BigEarthNet: Large-scale multi-label, Sentinel-2

**Segmentation:**
- ChesapeakeCVPR: Land cover, high-resolution
- LandCover.ai: Poland land cover
- NAIP ChesapeakeCVPR
- GeoNRW: Germany land cover

**Change Detection:**
- OSCD: Onera Satellite Change Detection
- SpaceNet: Urban change

**Object Detection:**
- xView: 60 object classes
- SpaceNet: Building footprints

### 6.4.3. DEM và Auxiliary Data

**SRTM (Shuttle Radar Topography Mission):**
- Global elevation data
- ~30m resolution

**Aster Global DEM:**
- Global coverage
- ~30m resolution

**Auxiliary:**
- Administrative boundaries
- Climate data
- Various derived products

## 6.5. Pre-trained Models

### 6.5.1. Self-supervised Pre-training

TorchGeo cung cấp weights từ self-supervised pre-training trên satellite data:

**SSL4EO-S12:**
- Pre-trained trên Sentinel-1 và Sentinel-2
- Multiple architectures (ResNet, ViT)
- MoCo v2, DINO, MAE strategies
- Designed cho European data

**SatMAE:**
- Masked autoencoder pre-training
- Multi-spectral support
- Good transfer learning performance

**GASSL:**
- Geography-aware self-supervised learning
- Considers spatial relationships
- Temporal consistency

### 6.5.2. Supervised Pre-trained Models

Models pre-trained trên labeled remote sensing datasets:

**Million-AID Pre-training:**
- Large-scale aerial image dataset
- Classification-based pre-training

**BigEarthNet Pre-training:**
- Multi-label classification
- Sentinel-2 data
- European coverage

### 6.5.3. Available Architectures

TorchGeo supports nhiều architectures:

**CNNs:**
- ResNet (18, 34, 50, 101, 152)
- VGG (11, 13, 16, 19)
- EfficientNet (B0-B7)
- MobileNet

**Vision Transformers:**
- ViT (Base, Large, Huge)
- Swin Transformer
- DeiT

**Segmentation:**
- U-Net với various encoders
- DeepLabV3+
- FPN-based architectures

## 6.6. Integration với PyTorch Ecosystem

### 6.6.1. PyTorch Lightning

TorchGeo provides Lightning modules cho common tasks:

**ClassificationTask:**
```
Handles image classification với:
- Configurable backbone
- Multi-class và multi-label support
- Standard metrics (accuracy, F1)
```

**SemanticSegmentationTask:**
```
For pixel-wise classification:
- Encoder-decoder architectures
- IoU, Dice metrics
- Class weighting
```

**ObjectDetectionTask:**
```
For bounding box prediction:
- Faster R-CNN based
- mAP evaluation
```

### 6.6.2. Torchvision Compatibility

TorchGeo extends torchvision:
- Compatible model APIs
- Extended transforms
- Interoperable datasets

### 6.6.3. Kornia Integration

Augmentations powered by Kornia:
- Differentiable transforms
- GPU-accelerated
- Batch processing

## 6.7. Ưu điểm của TorchGeo

### 6.7.1. Chuẩn hóa

TorchGeo cung cấp:
- Standard interfaces cho geospatial data
- Reproducible data loading
- Consistent preprocessing
- Documented best practices

### 6.7.2. Pre-trained Weights

Weights trained trên satellite data thường outperform ImageNet pretrained:
- Domain-specific features
- Multi-spectral understanding
- Sensor-specific patterns

### 6.7.3. Ease of Use

Simplified workflow:
- Few lines of code to load data
- Standard training loops
- Integration với familiar tools

### 6.7.4. Extensibility

Easy to extend:
- Custom datasets following base classes
- New transforms
- Custom models

### 6.7.5. Community và Support

Active development:
- Regular updates
- Growing community
- Documentation và tutorials
- Research backing (Microsoft)

## 6.8. Limitations và Considerations

### 6.8.1. Learning Curve

- Geospatial concepts (CRS, projections) cần được understand
- Dataset abstraction có thể complex cho beginners
- Different từ standard PyTorch patterns

### 6.8.2. Data Access

- Many datasets require download (large)
- Some require registration/agreements
- Cloud storage not always straightforward

### 6.8.3. Specific Use Cases

- May not cover all sensors/products
- Custom preprocessing sometimes needed
- Edge cases may require manual handling

### 6.8.4. Performance

- Large data handling có overhead
- Memory management cho big rasters
- Optimization may be needed cho production

## 6.9. Getting Started

### 6.9.1. Installation

```
pip install torchgeo
```

Hoặc với conda:
```
conda install -c conda-forge torchgeo
```

### 6.9.2. Basic Usage Example

**Loading a dataset:**
```python
from torchgeo.datasets import EuroSAT

dataset = EuroSAT(root="data", download=True)
sample = dataset[0]
# sample["image"], sample["label"]
```

**Using pre-trained model:**
```python
from torchgeo.models import ResNet50_Weights

weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
model = resnet50(weights=weights)
```

**Training với Lightning:**
```python
from torchgeo.trainers import ClassificationTask
from pytorch_lightning import Trainer

task = ClassificationTask(model="resnet50", ...)
trainer = Trainer()
trainer.fit(task, datamodule)
```

### 6.9.3. Documentation và Resources

- Official docs: https://torchgeo.readthedocs.io/
- GitHub: https://github.com/microsoft/torchgeo
- Tutorials và examples trong repository
- Research papers citing TorchGeo

## 6.10. TorchGeo trong Research

### 6.10.1. Publications

TorchGeo được sử dụng và cited trong nhiều research papers:
- Land cover classification studies
- Change detection research
- Environmental monitoring
- Agricultural applications
- Urban analysis

### 6.10.2. Benchmarking

Standardized benchmarks enable:
- Fair comparison between methods
- Reproducible results
- Community progress tracking

### 6.10.3. New Model Development

Researchers sử dụng TorchGeo để:
- Develop new architectures
- Test new pre-training strategies
- Evaluate on standard benchmarks
- Share weights và code

TorchGeo đã trở thành một công cụ quan trọng trong remote sensing và deep learning community, facilitating research và applications trong geospatial machine learning.

