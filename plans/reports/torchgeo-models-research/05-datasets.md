# TorchGeo Remote Sensing Benchmark Datasets

## 1. EuroSAT

**Dataset Statistics**
- Samples: 27,000 patches (2,000–3,000 per class)
- Classes: 10 (agriculture, artificial, industrial, herbaceous vegetation, highway, pastures, permanent crops, residential, river, forest)
- Resolution: 64×64 pixels, 10m ground resolution
- Coverage: 34 European countries (Austria, Belgium, France, Germany, Italy, Poland, etc.)

**Sensor Source**: Sentinel-2 (13 spectral bands, freely available via Copernicus)

**Task Type**: Multi-class land use/land cover classification

**Access**: GitHub ([phelber/EuroSAT](https://github.com/phelber/EuroSAT)), MIT licensed. Both RGB and multispectral versions available.

**Baseline Results**: 98.57% overall accuracy (CNN). Fine-tuned ResNet-50/GoogLeNet achieve ~2% improvement over random initialization. Using additional spectral indices: 99.58% accuracy.

---

## 2. BigEarthNet

**Dataset Statistics**
- Samples: 590,326 image patches (v1); 549,488 Sentinel-1/S-2 pairs (v2.0)
- Classes: 43 multi-label (imbalanced, CORINE LC 2018 level-3)
- Resolution: 1.2×1.2 km ground coverage; 120×120 px (10m bands), 60×60 px (20m), 20×20 px (60m)
- Coverage: 10 European countries (Austria, Belgium, Finland, Ireland, Kosovo, Lithuania, Luxembourg, Portugal, Serbia, Switzerland)

**Sensor Source**: Sentinel-1 SAR + Sentinel-2 MSI (13 bands), atmospherically corrected Level 2A

**Task Type**: Multi-label multi-class land cover classification

**Access**: [BigEarthNet official](https://bigearth.net/), TensorFlow Datasets, Radiant MLHub

**Baseline Results**: EfficientNet-b0 with channel attention (4.5% higher F-Score vs ResNet50 baseline). RGB-only CNNs underperform MSI models. Models trained from scratch on BigEarthNet outperform ImageNet pretrained models.

---

## 3. LandCover.ai

**Dataset Statistics**
- Samples: 41 orthophotos (33 @ 25cm, 8 @ 50cm resolution)
- Classes: 4 (building 1.85 km², woodland 72.02 km², water 13.15 km², road 3.5 km²)
- Resolution: 25cm or 50cm per pixel
- Coverage: 216.27 km² rural Poland (EPSG:2180)

**Sensor Source**: Orthophotos (RGB/GeoTIFF) from Head Office of Geodesy and Cartography, Poland (2015–2018 flights)

**Task Type**: Semantic segmentation (4-class land cover)

**Access**: [LandCover.ai website](https://landcover.ai.linuxpolska.com/), CC-BY-NC-SA 4.0 license

**Baseline Results**: DeepLabv3+ achieves 90.18% mean Intersection over Union (mIoU) on test set. Diverse optical conditions (saturation, sunlight angles, vegetation seasons) improve robustness.

---

## 4. OSCD (Onera Satellite Change Detection)

**Dataset Statistics**
- Samples: 24 image pairs (14 training, 10 test)
- Spatial resolution: 10m (RGB), 20m, 60m (multispectral)
- Coverage: Global (Brazil, USA, Europe, Middle East, Asia)
- Temporal span: 2015–2018

**Sensor Source**: Sentinel-2 (13 multispectral bands, registered pairs)

**Task Type**: Binary change detection (semantic change segmentation)

**Access**: [IEEE DataPort](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection), CC-BY-NC-SA license. RGB and full 13-band MS versions.

**Baseline Results**: Early Fusion & Siamese CNN architectures proposed in [Daudt et al. 2018](https://ieeexplore.ieee.org/document/8518015/). Residual blocks improve training performance. Fusion of Sentinel-1 SAR + Sentinel-2 MSI enhanced urban change detection.

---

## 5. LEVIR-CD

**Dataset Statistics**
- Samples: 637 bitemporal VHR image pairs (445 train, 128 val, 64 test)
- Resolution: 0.5m/pixel (1024×1024 px patches)
- Changes annotated: 31,333 individual building change instances (binary labels)
- Temporal span: 2002–2018 (5–14 year gaps)
- Coverage: 20 regions in Texas, USA (Austin, Lakeway, Buda, Kyle, Dripping Springs, etc.)

**Sensor Source**: Google Earth VHR satellite imagery

**Task Type**: Binary building change detection (growth & decline)

**Access**: [LEVIR official](https://justchenhao.github.io/LEVIR/), [GitHub](https://github.com/justchenhao/LEVIR), academic-only usage

**Baseline Results**: Double-annotated by experts for high quality. Supports villa residences, tall apartments, garages, warehouses. Benchmark architectures available for semantic change detection.

---

## 6. xView2

**Dataset Statistics**
- Training: 8,399 pairs (Tier1: 2,799, Tier3: 5,600), total 22,068 images (1024×1024 RGB)
- Test: 933 pairs (2.6 GB)
- Disasters: Wildfires, landslides, dam collapses, volcanic eruptions, earthquakes, tsunamis, storms, floods
- Coverage: Global disaster sites

**Sensor Source**: High-resolution RGB satellite imagery (pre/post disaster pairs)

**Task Type**: Building localization + damage assessment (multi-class: no damage, minor, major, destroyed)

**Access**: [xView2 Challenge website](https://www.sei.cmu.edu/projects/xview-2-challenge/), CC license, registration required. Includes bounding boxes, environmental metadata (fire, water, smoke).

**Baseline Results**: Joint Damage Scale (0–4) developed with disaster response experts. Models used operationally for post-wildfire damage assessment (Australia, California). Multi-stage pipeline: building detection + damage classification.

---

## Summary Table

| Dataset | Samples | Classes | Resolution | Sensor | Task |
|---------|---------|---------|-----------|--------|------|
| EuroSAT | 27k | 10 | 10m | Sentinel-2 | Classification |
| BigEarthNet | 590k | 43 (ML) | 10-60m | S1/S2 | Multi-label classification |
| LandCover.ai | 41 orthos | 4 | 25-50cm | Orthophoto | Segmentation |
| OSCD | 24 pairs | 2 | 10-60m | Sentinel-2 | Change detection |
| LEVIR-CD | 637 pairs | 2 | 0.5m | Google Earth | Building change detection |
| xView2 | 8,399 pairs | 4 | High-res RGB | Satellite | Damage assessment |

---

## Key Observations

- **Scale spectrum**: EuroSAT (small, balanced) → BigEarthNet (massive, imbalanced) → xView2/LEVIR-CD (specialized)
- **Sensor diversity**: Free Sentinel data (EuroSAT, BigEarthNet, OSCD) vs premium VHR (LEVIR-CD, xView2)
- **Task complexity**: Single-label → multi-label → change detection → damage classification
- **Accessibility**: Most public via GitHub/MLHub; xView2 requires registration
- **Baseline progression**: CNN→ResNet→EfficientNet→Vision Transformers (per BigEarthNet benchmarking)

---

## Sources

- [EuroSAT GitHub](https://github.com/phelber/EuroSAT)
- [EuroSAT arXiv](https://arxiv.org/abs/1709.00029)
- [BigEarthNet official](https://bigearth.net/)
- [BigEarthNet arXiv](https://arxiv.org/abs/1902.06148)
- [BigEarthNet-MM arXiv](https://arxiv.org/abs/2105.07921)
- [LandCover.ai website](https://landcover.ai.linuxpolska.com/)
- [LandCover.ai arXiv](https://arxiv.org/abs/2005.02264)
- [OSCD official](https://rcdaudt.github.io/oscd/)
- [OSCD IEEE DataPort](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection)
- [LEVIR-CD official](https://justchenhao.github.io/LEVIR/)
- [LEVIR-CD GitHub](https://github.com/justchenhao/LEVIR)
- [xView2 Challenge](https://www.sei.cmu.edu/projects/xview-2-challenge/)
- [xView2 DIU](https://www.diu.mil/latest/assessing-building-damage-from-satellite-imagery)
