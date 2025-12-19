# xView3-SAR Image Sources and References

Research compiled on 2025-12-19 documenting available image resources for xView3-SAR dataset documentation.

## Research Summary

This document catalogs authoritative sources for SAR imagery, vessel detection examples, and technical visualizations relevant to the xView3-SAR dataset documentation. The dataset consists of nearly 1,000 Sentinel-1 SAR images (average 29,400 × 24,400 pixels) with 220,000+ vessel annotations across maritime regions with high IUU (illegal, unreported, unregulated) fishing activity.

---

## Primary Document Sources

### 1. xView3 Dataset Whitepaper (Official)
**Source:** https://iuu.xview.us/xview3_dataset_whitepaper.pdf
**Contains:**
- Figure 1: Examples of VH and VV polarization bands from xView3 scene
- Figure 2: Sample annotations over xView3 scene
- Dataset specifications and processing methodology
- Metadata about VV, VH, bathymetry, wind speed, wind direction rasters

**Access:** PDF downloadable from official xView3 website
**License:** Check xView3 website for terms
**Suggested Use:** Official reference material for dataset visualization

---

### 2. NeurIPS 2022 Paper (xView3-SAR)
**Source:** https://proceedings.neurips.cc/paper_files/paper/2022/file/f4d4a021f9051a6c18183b059117e8b5-Paper-Datasets_and_Benchmarks.pdf
**Authors:** Paolo et al. (2022)
**Contains:**
- Figure 1: VH and VV band examples
- Figure 2: Dataset creation process diagram
- Figure 3: Geographic distribution map showing European waters (North Sea, Bay of Biscay, Iceland, Adriatic) and West African coast
- Figure 4: Sample annotations and vessel detection examples
- Technical specifications for SAR imagery processing

**Citation:** xView3-SAR: Detecting Dark Fishing Activity Using Synthetic Aperture Radar Imagery. Proceedings of the 36th International Conference on Neural Information Processing Systems (NeurIPS 2022)

**arXiv Version:** https://arxiv.org/abs/2206.00897
**License:** Check NeurIPS/arXiv terms
**Suggested Use:** Peer-reviewed reference for technical accuracy

---

### 3. AI2 Skylight Team Whitepaper
**Source:** https://github.com/allenai/sar_vessel_detect/blob/main/whitepaper.pdf
**Description:** xView3 competition submission documenting vessel detection approach using Sentinel-1 SAR imagery
**Contains:** Technical approach, methods, and visualizations from winning solution variant
**License:** Check repository license (likely Apache 2.0)
**Suggested Use:** Advanced technical reference for model implementation

---

## GitHub Reference Repositories

### 4. Official xView3 Reference Implementation
**Source:** https://github.com/DIUx-xView/xview3-reference
**Contains:**
- `train_reference.ipynb`: Jupyter notebook with visual examples and sample data
- `metric.py`: Scoring metrics with visualization support
- Example output visualizations and performance feedback
- Data processing code with example outputs

**Repository Stars:** Community-maintained
**License:** Check repository license
**Download:** Images/visualizations embedded in Jupyter notebooks

**Suggested Use:** Practical implementation examples with actual data visualization code

---

### 5. xView3 Challenge Winner Solutions
**Repository:** https://github.com/DIUx-xView
**Contains Multiple Solutions:**

#### First Place Solution
**URL:** https://github.com/BloodAxe/xView3-The-First-Place-Solution
**Includes:** CircleNet model, visualization outputs, detection examples
**License:** Check repository

#### Third Place Solution
**URL:** https://github.com/DIUx-xView/xView3_third_place
**Document:** `./doc/xView3_competition_third_place_solution.pdf`
**Includes:** Detailed methodology with figures and diagrams

#### Community Solutions
**URL:** https://github.com/naivelogic/xview3_ship_detection
**Contains:** Detection examples, validation visualizations

---

## SAR Dataset Sources

### 6. HRSID (High-Resolution SAR Images Dataset)
**Source:** https://github.com/chaozhong2010/HRSID
**Specs:** 5,604 high-resolution SAR images, 16,951 ship instances
**Resolutions:** 0.5m, 1m, 3m per pixel
**Format:** JPG (standard), PNG (high-fidelity)
**Download:** Google Drive, Baidu Cloud
**Includes:** Instance segmentation masks, semantic segmentation examples
**License:** IEEE Access paper - check repository for terms
**Contact:** chaozhong2010@163.com

**Suggested Use:** High-quality SAR imagery examples complementary to xView3

---

### 7. SAR-Ship-Dataset (CAESAR-Radi)
**Source:** https://github.com/CAESAR-Radi/SAR-Ship-Dataset
**Specs:** 39,729 ship chips (256×256 pixels)
**Source Data:** 102 Gaofen-3 images, 108 Sentinel-1 images
**Features:** Multi-scale and small object detection examples
**Download:** Google Drive, Baidu Netdisk
**License:** Reference IEEE journal - check repository

**Suggested Use:** Reference for small object detection in SAR

---

### 8. Sentinel-1 Open Data Sources

#### Copernicus Data Space Ecosystem
**URL:** https://dataspace.copernicus.eu/data-collections/sentinel-data/sentinel-1
**Products:** RAW, GRD, SLC in native format
**Access:** Free, global coverage since 2014
**Resolution:** 5-40m spatial resolution
**Suggested Use:** Source for additional SAR imagery examples

#### Google Earth Engine
**URL:** https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
**Contains:** Sentinel-1 GRD scenes in cloud-optimized format
**Resolutions:** 10m, 25m, 40m
**Polarizations:** VV, HH, VV+VH, HH+HV combinations
**Access:** Programmatic via Python/JavaScript API

#### NASA Earthdata / ASF DAAC
**URL:** https://www.earthdata.nasa.gov/data/platforms/space-based-platforms/sentinel-1
**Access Methods:**
- Vertex search application
- Programmatic via asf_search
- Earthdata Search interface
**Format:** Standard Sentinel-1 products

#### AWS Open Data Registry
**URL:** https://registry.opendata.aws/sentinel-1/
**Format:** Cloud-optimized GeoTIFF
**Coverage:** Global archive from mission start to present
**Updates:** New data within hours of availability

---

## Educational and Tutorial Resources

### 9. Digital Earth Africa - Ship Detection Tutorial
**URL:** https://docs.digitalearthafrica.org/en/latest/sandbox/notebooks/Real_world_examples/Ship_detection_with_radar.html
**Example:** Suez Canal ship detection (March 2021)
**Content:** Practical detection workflow with visualizations
**Visualization:** Detection overlays, threshold optimization examples

---

### 10. ESA Sentinel-1 User Guides
**Maritime Monitoring:** https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/applications/maritime-monitoring
**Polarimetry:** https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/product-overview/polarimetry
**Content:** Technical documentation, best practices for maritime detection

---

### 11. EODAG Tutorial - Gulf of Trieste Ship Detection
**URL:** https://eodag.readthedocs.io/en/stable/notebooks/tutos/tuto_ship_detection.html
**Example:** Sentinel-1 SAR ship detection with adaptive thresholding
**Visualizations:** Detection algorithm examples, threshold optimization

---

### 12. MATLAB/Simulink Example
**URL:** https://www.mathworks.com/help/images/ship-detection-from-sentinel1-c-band-sar-data-using-yolov2-object-detection.html
**Dataset:** Large-Scale SAR Ship Detection Dataset v1.0 (LS-SSDD-v1.0)
**Specs:** 15 large-scale images (24,000×16,000 pixels), Sentinel-1
**Polarizations:** VV and VH dual-pol
**Format:** 24-bit grayscale JPG

---

## Specialized Research Papers

### 13. Multi-Task Deep Learning for Ship Identification
**Source:** https://www.mdpi.com/2072-4292/11/24/2997
**Content:** Ship detection, classification, length estimation from Sentinel-1
**Figures:** VV/VH backscatter examples, multi-task network architecture

---

### 14. Feature Visualization Analysis (VH vs VV)
**Source:** https://www.mdpi.com/2072-4292/13/6/1184
**Title:** Ship Detection and Feature Visualization Analysis Based on Lightweight CNN in VH and VV Polarization Images
**Content:** Comparative analysis of polarization channels for ship detection

---

## Key Technical Content Descriptions

### SAR Polarization Visualization
**Content Found In:** xView3 whitepaper (Figure 1), NeurIPS paper (Figures 1-2)

**VV Polarization (Vertical-Vertical):**
- Shows sea surface features (wind, waves, oil slicks)
- Better for indirect vessel detection via wake patterns
- Higher surface clutter from sea roughness
- Useful context for ship characterization

**VH Polarization (Vertical-Horizontal):**
- Better ship-to-sea contrast
- Clearer vessel point targets
- Lower background clutter
- Optimal for direct vessel detection
- Shows vessel shape and skeleton more clearly

**Recommended Visualization:** Composite RGB images (VV=Red, VH=Green, VV/VH ratio=Blue)

---

### Geographic Coverage Examples

**European Waters (from xView3):**
- North Sea: diverse maritime traffic, fishing and cargo vessels
- Bay of Biscay: fishing activity
- Iceland waters: maritime infrastructure
- Adriatic Sea: mixed maritime activity

**African Coast (from xView3):**
- West African offshore: high IUU fishing activity
- Coastal waters with oil development infrastructure

---

### Dark Vessel Detection Context

**Key Concept:** Vessels not broadcasting AIS signals ("dark vessels")
**Importance:** Combat IUU fishing (illegal, unreported, unregulated)
**Detection Capability:** SAR works day/night, all-weather conditions
**Dataset Approach:** Hybrid labeling combining AIS-to-SAR matching + human expert verification

---

## Dataset Statistics and Specifications

**xView3-SAR Dimensions:**
- Scene Count: 991 full-size SAR images
- Average Scene Size: 29,400 × 24,400 pixels
- Total Coverage: ~1,422 gigapixels (4.7× larger than MS-COCO)
- Total Coverage Area: 80+ million km²
- Vessel Annotations: 220,000+ instances
- Polarizations: VV and VH (dual-pol)
- Resolution: 20m pixel spacing
- Ancillary Data: Bathymetry, wind speed, wind direction, land/ice mask

**Available Bands:**
- VH and VV polarization rasters (10m spacing)
- Bathymetry raster (500m spacing)
- Wind speed raster (500m spacing)
- Wind direction raster (500m spacing)
- Land/ice mask (500m spacing)

---

## Unresolved Questions

1. **Exact Figure URLs in PDFs:** The whitepaper and NeurIPS PDF links work but individual figure URLs cannot be extracted directly—figures exist within PDFs only
2. **Direct SAR Image Downloads:** Most example visualizations are embedded in notebooks/papers; raw TIFF/GeoTIFF files require download from data portals
3. **License Details:** Specific CC/copyright status for figures in academic papers needs verification from publishers
4. **Sentinel-1 Raw Data Access:** Full resolution Sentinel-1 IW mode data requires account creation on data portals; processing steps needed for visualization

---

## Recommended Next Steps for Image Collection

1. **Download PDFs and extract figures:**
   - xView3 whitepaper (Figures 1-2)
   - NeurIPS paper (Figures 1-4)
   - AI2 Skylight whitepaper

2. **Clone GitHub repositories and extract notebook outputs:**
   - xview3-reference for visualization code
   - HRSID for example SAR imagery
   - Challenge solutions for detection overlays

3. **Register and download from data portals:**
   - Copernicus Data Space for Sentinel-1 samples
   - Google Earth Engine (Python API) for scenes
   - AWS S3 open data for cloud-optimized files

4. **Processing pipeline:**
   - Use GDAL/rasterio to load and visualize GeoTIFF data
   - Create composite RGB from VV/VH bands
   - Generate annotation overlays using detected bbox coordinates
   - Normalize for web display (PNG/JPEG compression)

---

## Summary of Image Categories Available

| Category | Source | Format | License |
|----------|--------|--------|---------|
| VV/VH Polarization Examples | xView3 whitepaper, NeurIPS paper | PDF figures | Academic/DIU |
| Geographic Coverage Maps | NeurIPS Figure 4 | PDF figure | NeurIPS |
| Vessel Detection Examples | Challenge solutions repos | Jupyter outputs | Per repo license |
| High-Res SAR Samples | HRSID, SAR-Ship-Dataset | JPG/PNG | Academic license |
| Sentinel-1 Raw Data | Copernicus/NASA/AWS | GeoTIFF | CC BY 4.0 (ESA) |
| Processing Tutorials | Digital Earth Africa, EODAG | Notebooks | Open source |
| Backscatter Visualizations | Research papers | Scientific diagrams | Academic papers |

---

## References

All sources referenced in this document are listed with URLs above. Key primary sources:

- [xView3 Official Website](https://iuu.xview.us/)
- [xView3 Dataset Whitepaper](https://iuu.xview.us/xview3_dataset_whitepaper.pdf)
- [NeurIPS 2022 Paper - xView3-SAR](https://proceedings.neurips.cc/paper_files/paper/2022/file/f4d4a021f9051a6c18183b059117e8b5-Paper-Datasets_and_Benchmarks.pdf)
- [DIUx-xView GitHub Organization](https://github.com/DIUx-xView)
- [Global Fishing Watch](https://globalfishingwatch.org/)
- [Copernicus Sentinel-1 Data](https://dataspace.copernicus.eu/data-collections/sentinel-data/sentinel-1)

---

**Document Status:** Research phase complete - awaiting download/processing phase
**Last Updated:** 2025-12-19
**Format:** Markdown
**Scope:** Image sources inventory for xView3-SAR documentation
