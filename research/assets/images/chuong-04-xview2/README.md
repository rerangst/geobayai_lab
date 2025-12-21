# xView2/xBD Dataset Documentation Index

Comprehensive research and documentation for the xView2/xBD satellite imagery dataset for building damage assessment after natural disasters.

---

## Quick Navigation

### For First-Time Users
1. Start with **[download-guide.md](./download-guide.md)** - Fastest way to access data
2. See **[image-reference-catalog.md](./image-reference-catalog.md)** - Understanding damage classification (4 levels, colors, examples)
3. Reference **[image-sources.md](./image-sources.md)** - Complete source catalog

### For Researchers
1. **[image-sources.md](./image-sources.md)** - Academic sources, paper figures, citations
2. **[image-reference-catalog.md](./image-reference-catalog.md)** - Visual characteristics, quality metrics, disaster patterns
3. **[download-guide.md](./download-guide.md)** - Batch download scripts and data access

### For Developers
1. **[download-guide.md](./download-guide.md)** - Code examples (Python, bash)
2. **[image-reference-catalog.md](./image-reference-catalog.md)** - Format specifications (PNG, JSON, CSV)
3. **[image-sources.md](./image-sources.md)** - Integration points (TorchGeo, Roboflow, HuggingFace)

---

## Document Overview

### 1. image-sources.md (389 lines)
**Complete Image Source Catalog**

Primary purpose: Identify and catalog all image sources, papers, and data access points

Contains:
- CVPR 2019 paper details with 10 figure descriptions
- Official dataset specifications (850,736 buildings, 45,362 km²)
- Damage classification scale (4 levels with visual descriptions)
- 5 alternative access methods (Roboflow, TorchGeo, HuggingFace, Maxar STAC)
- Satellite imagery metadata (GSD, resolution, sensors)
- Download instructions for each access method
- Licensing & attribution requirements
- Related research papers (5+ citations)
- Complete dataset statistics
- Unresolved questions (7 items)

**Best for**:
- Finding authoritative sources
- Understanding dataset scope and specifications
- Academic citations and references
- Comprehensive dataset overview

---

### 2. download-guide.md (400 lines)
**Image Download & Access Methods**

Primary purpose: Practical guide for accessing and downloading images

Contains:
- 4 public access methods (no/minimal registration)
- Official portal step-by-step download (full dataset)
- JSON annotation format documentation
- CSV metadata format reference
- PNG image specifications (1024×1024, RGB)
- Output format for predictions (grayscale 0-4)
- 3 complete bash/Python download scripts
- TorchGeo Python integration code
- Hugging Face dataset loader
- Image loading examples (OpenCV, JSON parsing)
- Troubleshooting guide (8 issues + solutions)
- Storage requirements by tier (4GB-11GB)
- Legal and attribution guidance

**Best for**:
- Getting started quickly
- Downloading images (multiple methods)
- Data processing and loading
- Integration with ML pipelines
- Troubleshooting download/access issues

---

### 3. image-reference-catalog.md (483 lines)
**Image Types, Formats, and Visual Reference**

Primary purpose: Detailed visual and technical documentation of images

Contains:
- Pre-disaster satellite image specifications
- Post-disaster satellite image characteristics
- Building polygon annotation format (GeoJSON)
- 4-level damage scale with visual examples:
  - Level 0 (Green): No Damage - 313,033 polygons (84%)
  - Level 1 (Blue): Minor Damage - 36,860 polygons (5%)
  - Level 2 (Orange): Major Damage - 29,904 polygons (4%)
  - Level 3 (Red): Destroyed - 31,560 polygons (4%)
- Color visualization codes (RGB hex values)
- Disaster-specific damage patterns (earthquake, fire, flood, wind, volcano)
- Geographic distribution across events
- Image quality metrics (GSD, off-nadir, sun elevation)
- Annotation quality statistics
- Class imbalance challenges and solutions
- Performance baselines by class
- PNG/JSON/CSV format specifications
- Visual artifacts and limitations
- When/why to use xBD dataset

**Best for**:
- Understanding damage classification visually
- Learning disaster-specific patterns
- Format/specification reference
- Quality and limitation assessment
- Model performance expectations

---

## Dataset at a Glance

| Metric | Value |
|--------|-------|
| **Total Images** | 22,068 (1024×1024 RGB) |
| **Image Pairs** | 11,034 (pre/post-disaster) |
| **Building Polygons** | 850,736 |
| **Geographic Coverage** | 45,362 km² across 15+ countries |
| **Disaster Events** | 19 (across 6 types) |
| **Damage Classes** | 4 levels (0-3 scale) |
| **Ground Sampling Distance** | <0.8 meters |
| **Download Size** | 10 GB compressed, 11 GB uncompressed |
| **Training Pairs** | 8,399 (Tier 1 + Tier 3) |
| **Test Pairs** | 933 |

---

## Damage Classification Quick Reference

### Color-Coded Scale
```
Level 0 (No Damage)  → GREEN   (#00AA00) - 84% of labels
Level 1 (Minor)      → BLUE    (#0000FF) - 5% of labels
Level 2 (Major)      → ORANGE  (#FF8800) - 4% of labels
Level 3 (Destroyed)  → RED     (#FF0000) - 4% of labels
```

### Visual Characteristics

**Green (No Damage)**: Building stands, roof intact, colors match pre-disaster

**Blue (Minor Damage)**: Some roof elements missing, visible cracks, building recognizable

**Orange (Major Damage)**: Partial collapse, debris visible, significant structural damage

**Red (Destroyed)**: Complete collapse, only rubble visible, foundation-level destruction

---

## File Organization

```
/home/tchatb/sen_doc/docs/assets/images/chuong-04-xview2/
│
├── README.md                        ← You are here
├── image-sources.md                 ← Source catalog & references
├── download-guide.md                ← Download methods & code
├── image-reference-catalog.md       ← Visual reference & formats
│
├── dataset/                         ← For downloaded images
│   ├── [pre-disaster .png files]
│   ├── [post-disaster .png files]
│   └── [annotation .json files]
│
└── solutions/                       ← Reference implementations
    └── [example code/notebooks]
```

---

## Access Methods Summary

### Method 1: Roboflow (Easiest - No Login)
- **Images**: 520 samples
- **License**: CC BY 4.0
- **Access**: https://universe.roboflow.com/ozu/xview2
- **Time**: <5 minutes
- **Best for**: Quick experimentation

### Method 2: TorchGeo (Python Library)
- **Images**: Full dataset auto-download
- **Installation**: `pip install torchgeo`
- **Access**: Automatic on first use
- **Time**: 15-30 minutes
- **Best for**: ML/PyTorch integration

### Method 3: xView2 Official Portal (Complete Dataset)
- **Images**: 11,034 pairs (full dataset)
- **Access**: https://xview2.org/dataset
- **Registration**: Required + license acceptance
- **Time**: 1-2 hours download
- **Best for**: Production/research use

### Method 4: Hugging Face
- **Images**: Full dataset (parquet format)
- **Access**: https://huggingface.co/datasets/danielz01/xView2
- **Registration**: HF account + license acceptance
- **Time**: Variable
- **Best for**: ML pipelines with HF integration

### Method 5: Maxar STAC Catalog (Advanced)
- **Images**: Original satellite data
- **Format**: GeoTIFF/COG
- **Access**: API-based, direct S3
- **Registration**: Optional
- **Time**: Custom (per image)
- **Best for**: GIS professionals, custom analysis

---

## Documentation Statistics

| Document | Lines | Focus | Audience |
|----------|-------|-------|----------|
| image-sources.md | 389 | Academic sources, dataset specs, citations | Researchers, academicians |
| download-guide.md | 400 | Practical access methods, code examples | Developers, practitioners |
| image-reference-catalog.md | 483 | Visual reference, formats, specifications | Data scientists, ML engineers |
| **Total** | **1272** | Comprehensive xBD documentation | All users |

---

## Key Statistics

### Dataset Composition
- **No Damage**: 313,033 polygons (84%) - Severely imbalanced
- **Minor Damage**: 36,860 polygons (5%)
- **Major Damage**: 29,904 polygons (4%)
- **Destroyed**: 31,560 polygons (4%)
- **Unclassified**: 14,011 polygons (3%)

### Disaster Types
- Earthquake/Tsunami (4 events)
- Wildfire (3 events)
- Flooding (2+ events)
- Volcanic Eruption (2 events)
- Wind/Hurricane (5 events)
- Landslide/Other (4 events)

### Model Performance Baselines
- **Best Localization F1**: 0.80 (U-Net baseline)
- **Best Classification F1**: 0.66 (ResNet50 baseline)
- **Combined Weighted F1**: ~0.71 (best challenge submissions)
- **No Damage F1**: 0.81 (easiest class)
- **Minor Damage F1**: 0.42 (hardest class - high confusion)

---

## Important Limitations

1. **Class Imbalance**: 84% of labels are "no damage" (9.9:1 destroyed ratio)
2. **Visual Ambiguity**: Minor vs. major damage have subtle differences
3. **RGB-Only**: No multispectral or thermal bands
4. **Realistic Angles**: Off-nadir and varied sun elevation (not nadir/ideal)
5. **Subjective Boundaries**: Annotation guidelines have inherent subjectivity

---

## Real-World Impact

**California Wildfire Response**:
- **Time Saved**: ~98% reduction (10-20 minutes vs 1-2 days per area)
- **Deployment**: California National Guard actively using xView2 models
- **Coverage**: Successfully assessing large-scale wildfire damage
- **Accuracy**: Models provide actionable intelligence for emergency response

---

## Getting Started Checklist

- [ ] Read this README
- [ ] Choose access method (Roboflow for quick start, xView2 portal for complete)
- [ ] Follow **download-guide.md** for your chosen method
- [ ] Review **image-reference-catalog.md** to understand damage levels
- [ ] Reference **image-sources.md** for papers/citations
- [ ] Load sample image pair and inspect with provided code
- [ ] Review damage classification examples
- [ ] Consider class imbalance for your use case

---

## Citation

**Paper**: Gupta, R., et al. (2019). "Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery." In CVPR Workshops, pp. 18-26.

**BibTeX**:
```bibtex
@inproceedings{gupta2019xbd,
  title={Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery},
  author={Gupta, Ritwik and Goodman, Bryce and Patel, Nirav and Hosfelt, Ricky and Sajeev, Sandra and Heim, Eric and Doshi, Jigar and Lucas, Keane and Choset, Howie and Gaston, Matthew},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={18--26},
  year={2019}
}
```

**Attribution**: Satellite imagery sourced from Maxar/DigitalGlobe Open Data Program

---

## Quick Links

### Official Resources
- [xView2 Challenge Portal](https://xview2.org/)
- [xView2 Dataset Page](https://xview2.org/dataset)
- [CVPR 2019 Paper](https://arxiv.org/abs/1911.09296)
- [Maxar Open Data](https://www.digitalglobe.com/ecosystem/open-data)

### Alternative Access
- [Roboflow Universe](https://universe.roboflow.com/ozu/xview2)
- [Hugging Face Dataset](https://huggingface.co/datasets/danielz01/xView2)
- [TorchGeo Documentation](https://torchgeo.readthedocs.io/)

### Reference Implementations
- [Official Baseline](https://github.com/DIUx-xView/xView2_baseline)
- [1st Place Solution](https://github.com/DIUx-xView/xView2_first_place)
- [2nd Place Solution](https://github.com/ethanweber/xview2)
- [Toolkit & Utilities](https://github.com/ashnair1/xview2-toolkit)

### Research Papers
- [xBD Paper (arXiv)](https://arxiv.org/abs/1911.09296)
- [xFBD Related Work](https://arxiv.org/pdf/2212.13876)
- [DeepDamageNet](https://arxiv.org/html/2405.04800v1)

---

## Document Maintenance

**Last Updated**: 2025-12-19
**Research Scope**: Complete xBD/xView2 dataset documentation
**All URLs**: Verified active as of research date
**Sources**: 13+ primary sources catalogued

**Notes**: No actual image downloads performed (protected by registration). All URLs, access methods, specifications, and metadata documented for reference.

---

## Navigation Tips

- **Need a specific image type?** → See image-reference-catalog.md
- **Want to download data?** → See download-guide.md
- **Looking for papers/citations?** → See image-sources.md
- **Understanding damage levels?** → See image-reference-catalog.md (damage scale section)
- **Python code examples?** → See download-guide.md (scripts section)
- **Complete dataset specs?** → See image-sources.md (dataset statistics table)
