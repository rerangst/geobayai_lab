# xView2/xBD Dataset Image Sources & Documentation

Research conducted: 2025-12-19

## Overview

The xView2 Challenge dataset (xBD - xView Building Damage) contains satellite imagery before and after natural disasters with building damage classification. This document catalogs image sources, visualization resources, and download locations.

---

## Primary Research Papers & Figures

### 1. CVPR 2019 Paper: "Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery"

**Paper Details**
- Authors: Ritwik Gupta, Bryce Goodman, Nirav Patel, Ricky Hosfelt, Sandra Sajeev, Eric Heim, Jigar Doshi, Keane Lucas, Howie Choset, Matthew Gaston
- Published: CVPR 2019 Workshops (cv4gc)
- Date: November 21, 2019

**Available Formats**
| Format | URL | License |
|--------|-----|---------|
| PDF (Direct) | https://arxiv.org/pdf/1911.09296 | arXiv Open Access |
| PDF (CVF) | https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf | Open Access |
| HTML (ar5iv) | https://ar5iv.labs.arxiv.org/html/1911.09296 | Open Access |
| HTML (CVF) | https://openaccess.thecvf.com/content_CVPRW_2019/html/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.html | Open Access |

**Paper Contains**
- Figure 1: Example disaster events (Hurricane Harvey, Palu Tsunami, Mexico City Earthquake, Santa Rosa Fire)
- Figure 2: Pre- and post-disaster satellite imagery pairs (4 events)
- Figure 3: Geographic distribution map - disaster types and locations worldwide
- Figure 4: Joint Damage Scale classification descriptions (0-3 scale)
- Figure 5: Building polygon annotations on pre-disaster imagery
- Figure 6: Area coverage distribution across disasters
- Figure 7: Positive vs. negative imagery counts by disaster
- Figure 8: Building polygon density per disaster event
- Figure 9: Damage classification label distribution (heavily imbalanced - 8x more "no damage")
- Figure 10: Baseline classification model architecture diagram

**Image Accessibility**: Figures embedded in PDF/HTML. Use ar5iv HTML version for responsive viewing.

---

## Official Dataset Source

### xView2 Challenge Website

**Main URL**: https://xview2.org/

**Dataset Page**: https://xview2.org/dataset

**Status**: Official repository (requires JavaScript, registration for download)

**Dataset Contents**
- 22,068 satellite images at 1024×1024 resolution
- 11,034 pre/post-disaster image pairs
- 850,736 annotated building polygons
- Coverage: 45,362 km² across 15 countries
- 19 natural disasters, 6 disaster types
- Ground sampling distance (GSD): <0.8 meters
- Source: Maxar/DigitalGlobe Open Data Program

**Download Requirements**
- Registration required
- ~10 GB compressed, ~11 GB uncompressed
- Two training tiers: Tier 1 (2,799 pairs) + Tier 3 (5,600 pairs) = 8,399 total
- Test set: 933 image pairs

**Disaster Types Included**
| Disaster Type | Count | Examples |
|---------------|-------|----------|
| Earthquake/Tsunami | 4 | Palu Tsunami, Mexico Earthquake, Lombok Earthquake, Sulawesi Tsunami |
| Wildfire | 3 | Santa Rosa Fire, Socal Fire, Pinery Bushfire |
| Flooding | 2 | Midwest Flooding, Nepal Flooding, India Monsoon |
| Volcanic Eruption | 1 | Guatemala Volcano, Lower Puna Eruption |
| Wind/Hurricane | 5 | Hurricane Harvey, Hurricane Florence, Hurricane Michael, Hurricane Matthew, Joplin Tornado |
| Landslide/Other | 4 | Additional events |

---

## Satellite Image Metadata

**Imagery Source**: Maxar/DigitalGlobe Open Data Program
- URL: https://www.digitalglobe.com/ecosystem/open-data
- License: Various (check individual event availability)
- Sensors: Multiple high-resolution optical satellites
- Resolution: ~0.3-0.8 meters GSD

**Image Specifications**
- Format: PNG, RGB (3-channel)
- Resolution: 1024×1024 pixels
- Color Depth: 24-bit RGB
- Off-nadir angles: Variable (realistic satellite acquisition)
- Sun elevation angles: Variable

**Annotation Format**
- Building polygons in WKT notation
- Damage labels: 0 (no damage), 1 (minor), 2 (major), 3 (destroyed)
- Additional labels: fire, water, smoke, lava (environmental factors)
- Georeferencing metadata included

---

## Damage Classification Scale

### Joint Damage Scale (4-Level Ordinal System)

**Color Scheme** (standard visualization)
| Level | Value | Color | Description |
|-------|-------|-------|-------------|
| No Damage | 0 | Green | Undisturbed, no structural damage, no burn marks |
| Minor Damage | 1 | Blue | Partial burns, water surrounding, roof elements missing, visible cracks |
| Major Damage | 2 | Orange | Partial wall/roof collapse, significant damage encroaching |
| Destroyed | 3 | Red | Complete collapse, structure uninhabitable |

**Label Distribution in xBD**
- No Damage: 313,033 polygons (84%)
- Minor Damage: 36,860 polygons (5%)
- Major Damage: 29,904 polygons (4%)
- Destroyed: 31,560 polygons (4%)
- Unclassified: 14,011 polygons (3%)
- **Imbalance Challenge**: 8x more "no damage" than other categories

---

## Alternative Access Points & Mirrors

### 1. Hugging Face
**URL**: https://huggingface.co/datasets/danielz01/xView2
- Creator: Chenhui Zhang (@danielz01)
- Access: Requires login + accept terms and conditions
- Status: ~104 monthly downloads
- Format: Parquet
- Size: 1K-10K entries

### 2. Roboflow Universe
**URL**: https://universe.roboflow.com/ozu/xview2
- Status: Public preview available
- Sample Size: 520 open-source images
- Classes: undamaged, minor-damage, major-damage, destroyed
- License: CC BY 4.0
- Includes: Pre-trained XView2 model + API

### 3. Kaggle
- Searchable via Kaggle dataset exploration
- May contain community kernels and subsets
- Useful for quick experimentation

### 4. TorchGeo Library
**URL**: https://torchgeo.readthedocs.io/
- Python library for geospatial deep learning
- Includes xView2 dataset loader
- Challenge training set: ~7.8 GB
- Challenge test set: ~2.6 GB

### 5. Earth Observation Database
**URL**: https://eod-grss-ieee.com/dataset-detail/MHpyVXNmV0dxaEtWWVBaNzlpckJPUT09
- Metadata and index: xBD (xView2)

---

## Baseline & Reference Implementations

### Official Baseline Repository
**GitHub**: https://github.com/DIUx-xView/xView2_baseline
- Language: Python 3.6+
- Architecture: U-Net for localization, ResNet50 for classification
- Authors: CMU SEI
- Includes: Data preparation, training, inference scripts
- License: Check repository

**Output Specification**
- Grayscale PNG format
- Pixel values: 0-4 (no building, no damage, minor, major, destroyed)

### Top Challenge Solutions
- **1st Place**: https://github.com/DIUx-xView/xView2_first_place
  - Siamese Neural Networks
  - Full image (1024×1024) inference with 4 TTA

- **2nd Place**: https://github.com/ethanweber/xview2
  - Detectron2-based (Facebook)
  - Multi-temporal fusion

- **Visualization Tool**: ethanweber/xview2 includes notebook for visualizing predictions

### Toolkit & Utilities
- **xview2-toolkit**: https://github.com/ashnair1/xview2-toolkit
  - Annotation visualization
  - MS-COCO format conversion
  - Segmentation map generation

---

## Related Papers & Research

### Extended Works
1. **[2212.13876] xFBD: Focused Building Damage Dataset and Analysis**
   - URL: https://arxiv.org/pdf/2212.13876
   - Focused subset with different methodology

2. **[2405.04800v1] DeepDamageNet**
   - URL: https://arxiv.org/html/2405.04800v1
   - Two-step model for multi-disaster assessment

3. **Building Damage Assessment Papers**
   - ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S0034425721003564
   - Taylor & Francis: https://www.tandfonline.com/doi/full/10.1080/17538947.2024.2302577

### Benchmark Collections
- **Satellite Image Deep Learning Datasets**: https://github.com/satellite-image-deep-learning/datasets
- **NAD Benchmarks**: https://roc-hci.github.io/NADBenchmarks/

---

## Dataset Statistics & Insights

### Coverage by Disaster
| Event | Region | Type | Coverage | Building Count |
|-------|--------|------|----------|-----------------|
| Hurricane Harvey | Houston, TX | Wind | ~8000 km² | 100,000+ |
| Palu Tsunami | Indonesia | Tsunami | <1000 km² | 100,000+ |
| Mexico Earthquake | Mexico City | Earthquake | <1000 km² | 100,000+ |
| Pinery Bushfire | Australia | Fire | ~8000 km² | Large |

### Model Performance Baseline
- Localization F1 (U-Net): 0.80
- Localization IoU: 0.66
- Classification F1 (ResNet50): Varies by damage class
- Combined F1 (weighted): ~0.71 (IBM approach)

### Challenge Application
- Real-world use: California wildfire damage assessment
- Processing time: 10-20 minutes per large area (vs 1-2 days manual)
- Deployed by: California National Guard

---

## Image Download Instructions

### Method 1: Official xView2 Portal
1. Navigate to https://xview2.org/dataset
2. Create account and login
3. Accept challenge terms
4. Download data (10 GB compressed)
5. Extract to working directory

### Method 2: Maxar Open Data STAC
1. Access STAC catalog: https://maxar-opendata.s3.amazonaws.com/events/catalog.json
2. Use GIS tools or STAC client to fetch individual events
3. Filter by disaster event
4. Download GeoTIFF or COG format

### Method 3: Roboflow (Quick Start)
1. Visit https://universe.roboflow.com/ozu/xview2
2. Browse 520 sample images
3. Download with API key for programmatic access

### Method 4: TorchGeo (Python)
```python
from torchgeo.datasets import XView2
dataset = XView2(root="path/to/data", download=True)
```

### Method 5: Hugging Face (Python)
```python
from datasets import load_dataset
ds = load_dataset("danielz01/xView2")  # Requires login/acceptance
```

---

## Image Licensing & Attribution

### Satellite Imagery
- **Source**: Maxar/DigitalGlobe Open Data Program
- **License**: Varies per event (typically public domain for crisis events)
- **Attribution**: DigitalGlobe/Maxar required
- **Terms**: Check specific event in Open Data Program catalog

### Dataset Annotations
- **License**: Creative Commons (varies by version)
- **Citation**: Gupta, R., et al. (2019). "Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery"
- **Challenge Data**: xView2 Challenge terms apply

### Research Use
- **Academic**: Permitted with attribution
- **Commercial**: Check licensing terms before deployment
- **Redistribution**: Allowed under CC BY 4.0 (Roboflow subset)

---

## Format Specifications

### Pre-Disaster & Post-Disaster Images
- **Container**: PNG or GeoTIFF
- **Bands**: RGB (3 channels)
- **Resolution**: 1024×1024 pixels
- **Bit Depth**: 8-bit per channel
- **Color Space**: sRGB

### Ground Truth Annotations
- **Format**: JSON (polygon vertices) + CSV (damage labels)
- **Polygon Notation**: WKT (Well-Known Text)
- **Damage Values**: 0-3 (ordinal scale)
- **Metadata**: Georeferencing, sensor info, timestamp

### Output Format (Predictions)
- **Container**: PNG (grayscale)
- **Values**: 0-4 (pixel-wise damage classification)
- **Resolution**: 1024×1024 pixels

---

## Key Statistics Summary

| Metric | Value |
|--------|-------|
| Total Images | 22,068 |
| Image Pairs | 11,034 |
| Building Polygons | 850,736 |
| Geographic Coverage | 45,362 km² |
| Countries | 15+ |
| Disaster Events | 19 |
| Disaster Types | 6 |
| GSD (Resolution) | <0.8 meters |
| Image Size | 1024×1024 pixels |
| Color Format | RGB (24-bit) |
| Damage Classes | 4 levels (0-3) |
| Download Size | ~10 GB (compressed), ~11 GB (uncompressed) |
| Training Pairs | 8,399 (Tier1 + Tier3) |
| Test Pairs | 933 |

---

## Recommended Citation

```bibtex
@inproceedings{gupta2019xbd,
  title={Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery},
  author={Gupta, Ritwik and Goodman, Bryce and Patel, Nirav and Hosfelt, Ricky and Sajeev, Sandra and Heim, Eric and Doshi, Jigar and Lucas, Keane and Choset, Howie and Gaston, Matthew},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={18--26},
  year={2019}
}
```

---

## Important Notes

### Data Imbalance
- "No damage" class heavily overrepresented (84% of labels)
- Imbalance factor: ~8x more "no damage" than other classes
- Poses challenge for class-balanced training

### Visual Similarity Challenge
- Minor vs. major damage can have subtle visual differences
- Models often confuse adjacent damage classes
- Joint Damage Scale designed as practical trade-off

### Real-World Application
- Successfully used by California National Guard for wildfire assessment
- Demonstrated 10-20 minute assessment vs. 1-2 days manual analysis
- High-impact humanitarian/disaster response tool

### Access Limitations
- Full dataset requires registration at xView2.org
- Some mirrors/subsets publicly available (Roboflow, HuggingFace)
- Roboflow subset (520 images) under CC BY 4.0 license

---

## Unresolved Questions

1. **Exact GSD by Event**: Paper states "<0.8 meters" GSD but specific resolution per disaster varies
2. **Sensor Specifications**: Which satellites exactly? (Likely DigitalGlobe WorldView series, specifics not documented)
3. **Temporal Gap**: Exact time between pre- and post-disaster capture not always specified
4. **Quality Assurance**: Annotation inter-rater agreement / QA metrics not provided in CVPR paper
5. **Update Frequency**: Whether new disasters have been added to dataset post-2019
6. **Raw Image URLs**: Specific CDN/S3 paths for individual images not publicly listed (must download via portal)

---

## Last Updated

Research Date: 2025-12-19

All URLs verified as active at time of documentation. Note that websites and data availability may change.
