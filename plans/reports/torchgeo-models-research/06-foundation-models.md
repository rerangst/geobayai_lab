# Foundation Models Research: TorchGeo & Prithvi

**Research Date:** 2025-12-20
**Status:** Complete

## Executive Summary

Two critical frameworks power geospatial deep learning: **TorchGeo** (2111.08872) provides unified infrastructure for datasets/samplers/transforms, enabling reproducible benchmarking. **Prithvi-EO** (2310.18660) introduces transformer-based foundation models pre-trained on 1TB+ multispectral imagery, achieving SOTA on 6+ downstream tasks. Combined, they represent the foundation → application pipeline: TorchGeo standardizes data handling; Prithvi provides pre-trained weights for transfer learning.

## 1. TorchGeo: Infrastructure Layer

### Key Contributions
- First library integrating geospatial data with PyTorch ecosystem
- Unified API: datasets, samplers, transforms, pre-trained models for multispectral satellite imagery
- On-the-fly preprocessing (eliminates storage-intensive pre-processing)
- Pre-trained models using all Sentinel-2 bands (enables transfer learning on limited labeled data)
- Reproducible benchmarking across 15+ standard datasets

### Architecture & Design Philosophy
- **Modular design**: Aligns with PyTorch conventions (Dataset, DataLoader, Transform)
- **Composability**: Generic geospatial data sources + benchmark datasets
- **Multispectral support**: Handles N-band satellite imagery (not limited to RGB)
- **Geospatial-aware samplers**: Spatial sampling patterns for remote sensing
- **Memory efficiency**: On-the-fly processing without massive disk footprint

### PyTorch Ecosystem Integration
- Native PyTorch Dataset/DataLoader compatibility
- Composable transforms for geospatial preprocessing
- Compatible with standard PyTorch training loops
- Support for distributed training via PyTorch DistributedDataParallel
- Pre-trained model weights compatible with standard nn.Module architecture

### Benchmark Results & Applications
- **Tasks supported**: Classification, regression, segmentation, object detection, change detection
- **Datasets**: RESISC45 (scene classification), NWPU-RESISC45, UC Merced, Eurosat, BIGEARTHNET
- **Performance**: Competitive ImageNet pre-training impact; domain-specific pre-training beneficial
- **Real applications**: Land cover mapping, deforestation/flood monitoring, disaster response, urban planning, climate research, precision agriculture

## 2. Prithvi-EO: Foundation Model Layer

### Key Innovations
- **First geospatial foundation model**: 100M-600M parameter transformer pre-trained on NASA HLS data (4.2M temporal time series)
- **Temporal architecture**: Video format input (B,C,T,H,W) with spatial+temporal attention
- **Self-supervised learning**: Masked AutoEncoder (MAE) with ViT backbone
- **Scale**: Prithvi-EO-2.0 (600M params) 6x larger than v1; trained on 1TB+ multispectral imagery
- **Open source**: Available via Hugging Face; integrated with TerraTorch fine-tuning toolkit

### Architecture Details
- **Encoder-decoder**: Asymmetric MAE with Vision Transformer (ViT-L/ViT-H backbone)
- **3D embeddings**: 3D patch + positional encodings for spatiotemporal data
- **Geospatial context**: Lat/lon + date metadata via 2D sin/cos encoding (weighted sum fusion)
- **Robustness**: Drop mechanism handles missing data during pre-training
- **Models**: 300M (ViT-L) & 600M (ViT-H) parameter variants

### Multi-modal Capabilities
- **Primary**: Pre-trained on optical HLS (Harmonized Landsat + Sentinel-2, 30m resolution, 10-year temporal coverage)
- **Adaptable**: Demonstrated integration of SAR (Radar Vegetation Index) + MSI (Multi-Spectral Instrument) for Above Ground Biomass estimation
- **Flexibility**: Can incorporate atmospheric/land surface variables (MERRA-2) for tasks like GPP prediction
- **Multi-temporal**: Handles time series for change detection, crop monitoring, temporal dynamics

### Transfer Learning Strategy
- **Fine-tuning framework**: TerraTorch toolkit simplifies adaptation to diverse Earth Observation tasks
- **Task-specific heads**: Linear layer (classification), U-Net (segmentation), regression (pixel-wise)
- **Training**: Standard hyperparameter tuning with appropriate loss functions (cross-entropy, MSE, etc.)
- **Efficiency**: Pre-training accelerates convergence vs. random initialization

### Benchmark Results (GEO-Bench evaluation)
| Task | Metric | Prithvi-EO v1 | Prithvi-EO 600M | Improvement |
|------|--------|--------------|-----------------|-------------|
| Flood Detection | IoU water | 79.6% | 83.1% | +3.5pp |
| Wildfire Scars | IoU burn | 76.8% | 83.2% | +6.4pp |
| Landslide Detection | mIoU/F1 | baseline | superior | ++ |
| Crop Segmentation | mIoU/accuracy | - | 50.7% / 68.8% | SOTA |
| Land Cover (Sen4Map) | multi-fraction | competitive | superior | ++ |
| AGB Estimation | multi-modal SAR+MSI | - | adaptive | multi-modal |
| GPP Estimation | R² vs RF | up to 20% improvement | consistent best | ++ |

**Overall**: 600M outperforms 6+ competing geospatial foundation models across resolutions (0.1m–15m).

## 3. Integration: TorchGeo + Prithvi Workflow

```
1. Data Ingestion (TorchGeo)
   └─> Load benchmark dataset or custom geospatial data

2. Preprocessing (TorchGeo)
   └─> On-the-fly transforms, multispectral handling

3. Model Selection (Prithvi)
   └─> Load pre-trained 100M/300M/600M from Hugging Face

4. Fine-tuning (TerraTorch)
   └─> Attach task-specific head, train on labeled data

5. Evaluation (TorchGeo)
   └─> Benchmark on standard datasets, reproducible metrics
```

## Key Design Choices

**TorchGeo**:
- Why modularity? Reduces learning curve; enables gradual adoption
- Why on-the-fly? Eliminates 100GB+ preprocessing; real-time experimentation
- Why multispectral? Satellite data has 10-13 bands; RGB insufficient

**Prithvi**:
- Why temporal? Remote sensing is inherently time-series (crop growth, flood dynamics, seasonal change)
- Why MAE? Self-supervised learning scales to unlabeled satellite data (petabytes available)
- Why large? Bigger models (600M) capture complex Earth dynamics better than small models

## Unresolved Questions

1. **Computational requirements**: Exact FLOPS for Prithvi fine-tuning on different dataset sizes?
2. **Data drift**: How well do Prithvi models trained on CONUS generalize to Africa/Asia/South America?
3. **Real-time inference**: Edge deployment capabilities for drone/UAV imagery?
4. **Multi-temporal alignment**: Best practices for aligning Sentinel-2 time series with irregular acquisition patterns?

## Sources

- [TorchGeo Paper (arXiv:2111.08872)](https://arxiv.org/abs/2111.08872)
- [TorchGeo GitHub](https://github.com/torchgeo/torchgeo)
- [PyTorch Blog: Geospatial DL with TorchGeo](https://pytorch.org/blog/geospatial-deep-learning-with-torchgeo/)
- [Prithvi Paper (arXiv:2310.18660)](https://arxiv.org/abs/2310.18660)
- [Prithvi Hugging Face Models](https://huggingface.co/ibm-nasa-geospatial)
- [IBM Research: Prithvi-EO-2.0 Release](https://research.ibm.com/blog/prithvi2-geospatial)
- [NASA Earthdata: Prithvi AI Foundation Model](https://www.earthdata.nasa.gov/news/nasa-ibm-openly-release-geospatial-ai-foundation-model-nasa-earth-observation-data)
