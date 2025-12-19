# xView2 Image Download & Access Guide

## Quick Reference: Image Access Methods

### Public Access (No Registration)

#### 1. Roboflow Universe (520 Sample Images)
```bash
# Via web browser
https://universe.roboflow.com/ozu/xview2

# Via Roboflow API
import roboflow
rf = roboflow.Roboflow(api_key="YOUR_KEY")
project = rf.workspace("ozu").project("xview2")
dataset = project.version(1).download("coco")
```

**What You Get**:
- 520 pre-processed sample images
- Pre/post disaster pairs
- Damage classifications (0-3)
- License: CC BY 4.0
- No registration needed for browser access

---

#### 2. TorchGeo (Python Library)
```bash
pip install torchgeo

# Python code
from torchgeo.datasets import XView2
from torch.utils.data import DataLoader

# Auto-downloads on first access
dataset = XView2(root="/data/xview2", split="train", download=True)
loader = DataLoader(dataset, batch_size=32)

for batch in loader:
    pre_image = batch['pre_image']      # Pre-disaster RGB
    post_image = batch['post_image']    # Post-disaster RGB
    damage_label = batch['damage']      # 0-3 damage levels
```

**Coverage**:
- Training set: ~7.8 GB (9,168 pairs available)
- Test set: ~2.6 GB (933 pairs)
- Auto-handles decompression and normalization

---

### Registration Required (Official Dataset)

#### 3. xView2 Official Portal (Full Dataset: 11,034 Pairs)

**Step-by-Step**:
1. Visit: https://xview2.org/dataset
2. Click "Register" or "Login"
3. Create account (email, password)
4. Accept Challenge Terms of Use
5. Navigate to Downloads
6. Select desired tier(s):
   - **Tier 1**: 2,799 pairs (smaller, easier to download)
   - **Tier 3**: 5,600 pairs (larger, more data)
   - Test set: 933 pairs (no labels, for evaluation)

**Download Structure**:
```
xview2_data/
├── train/
│   ├── aleppo_b/
│   │   ├── aleppo_b_00000000_pre.png   (1024×1024 RGB)
│   │   ├── aleppo_b_00000000_post.png  (1024×1024 RGB)
│   │   └── aleppo_b_00000000_buildings.json  (polygons + damage)
│   ├── guatemala-volcano/
│   ├── hurricane-florence/
│   └── ... [more disasters]
├── test/
│   └── [similar structure, no labels]
└── metadata.json
```

**Size**: 10 GB compressed, 11 GB uncompressed

**Disaster Events in Official Dataset**:
- aleppo_b (building)
- guatemala-volcano
- hurricane-florence
- hurricane-harvey
- hurricane-matthew
- hurricane-michael
- joplin-tornado
- lower-puna-eruption
- mexico-earthquake
- midwest-flooding
- nepal-flooding
- palu-tsunami
- pinery-bushfire
- santa-rosa-wildfire
- socal-fire

---

#### 4. Hugging Face Datasets (Parquet Format)

**Note**: Requires login + license acceptance

```python
from datasets import load_dataset

# Must be logged in: huggingface-cli login
ds = load_dataset("danielz01/xView2")

# Access image data
for sample in ds["train"]:
    pre_disaster = sample["pre_image"]
    post_disaster = sample["post_image"]
    damage_map = sample["damage_labels"]
    print(damage_map)  # 0-3 values
```

**Advantages**: Parquet format, integrated with ML pipelines
**Disadvantages**: Limited documentation, requires HF account

---

## Metadata & Annotation Format

### JSON Annotation Structure

**File**: `disaster_name_imageid_buildings.json`

```json
{
  "features": [
    {
      "properties": {
        "uid": "polygon_id_1",
        "damage": 0,
        "class": "building"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [[x1, y1], [x2, y2], [x3, y3], ...]
        ]
      }
    },
    {
      "properties": {
        "uid": "polygon_id_2",
        "damage": 2,
        "class": "building"
      },
      "geometry": { ... }
    }
  ]
}
```

**Damage Values**:
- 0: No Damage
- 1: Minor Damage
- 2: Major Damage
- 3: Destroyed

**Polygon Format**: GeoJSON with pixel coordinates (not lat/lon)

---

### CSV Metadata (Sample)

```csv
ImageID,DisasterType,PreImageDate,PostImageDate,OffNadir,SunElevation,SatelliteSensor
guatemala-volcano_00000000,volcano,2018-06-01,2018-06-04,28.5,45.2,DigitalGlobe
hurricane-harvey_00000001,wind,2017-08-25,2017-08-31,15.3,52.1,DigitalGlobe
...
```

---

## Image File Specifications

### Pre-Disaster Image
- **Filename**: `{disaster}_{imageid}_pre.png`
- **Dimensions**: 1024×1024 pixels
- **Format**: PNG, 8-bit RGB (24-bit color)
- **Color Space**: sRGB
- **Compression**: PNG lossless

### Post-Disaster Image
- **Filename**: `{disaster}_{imageid}_post.png`
- **Identical specs to pre-disaster**

### Output Prediction Format (for submissions)
- **Format**: Grayscale PNG (single channel)
- **Filename**: `{imageid}_damage.png`
- **Dimensions**: 1024×1024 pixels
- **Values**: 0-4 (0=background, 1-4=damage levels)
- **Data Type**: uint8

---

## Batch Download Scripts

### Script 1: Download Roboflow Subset (Easiest)
```bash
#!/bin/bash
# Requires: curl, unzip

cd ~/data
curl -L "https://universe.roboflow.com/api/dataset/xview2/download/coco" \
  -o xview2_roboflow.zip
unzip xview2_roboflow.zip
ls -la xview2/
```

**Result**: 520 sample images in `xview2/` directory

---

### Script 2: Download via xView2 Portal (Manual)
```bash
#!/bin/bash
# Downloads file(s) from xView2 portal via login token

# Note: Must manually obtain download URLs from portal
# 1. Login to https://xview2.org/dataset
# 2. Copy download links (may have short expiration)
# 3. Use with wget/curl as shown

DOWNLOAD_URL="[paste from portal]"
OUTPUT_FILE="xview2_tier1.zip"

# Download with progress
wget --progress=bar:force -O "$OUTPUT_FILE" "$DOWNLOAD_URL"

# Extract
unzip -q "$OUTPUT_FILE" -d ./xview2_data/

# Verify structure
find ./xview2_data -name "*_pre.png" | head -5
```

---

### Script 3: Dataset.py for xBD (Utility)
```python
#!/usr/bin/env python3
"""
Download and verify xBD dataset integrity
"""

import os
import json
from pathlib import Path
from collections import defaultdict

def verify_xbd_structure(root_dir):
    """Check dataset completeness"""
    stats = defaultdict(int)
    disasters = set()

    root = Path(root_dir)

    for disaster_dir in root.glob("train/*/"):
        disaster = disaster_dir.name
        disasters.add(disaster)

        pre_images = list(disaster_dir.glob("*_pre.png"))
        post_images = list(disaster_dir.glob("*_post.png"))
        labels = list(disaster_dir.glob("*_buildings.json"))

        stats[disaster] = {
            'pre': len(pre_images),
            'post': len(post_images),
            'labels': len(labels),
            'verified': len(pre_images) == len(post_images) == len(labels)
        }

    print(f"Found {len(disasters)} disasters")
    print(f"Total verified image pairs: {sum(s['pre'] for s in stats.values())}")

    for disaster, counts in sorted(stats.items()):
        status = "✓" if counts['verified'] else "✗"
        print(f"{status} {disaster:30s} {counts['pre']:4d} pairs")

if __name__ == "__main__":
    verify_xbd_structure("./xview2_data")
```

---

## Direct Image Access Patterns

### Maxar STAC Catalog (for Advanced Users)
```python
import requests
from urllib.parse import urljoin

# STAC catalog endpoint
MAXAR_STAC = "https://maxar-opendata.s3.amazonaws.com/events/catalog.json"

# Fetch catalog
catalog = requests.get(MAXAR_STAC).json()

# Find hurricane-harvey event
for collection in catalog['links']:
    if 'harvey' in collection['rel'].lower():
        event_url = urljoin(MAXAR_STAC, collection['href'])
        event_data = requests.get(event_url).json()

        # Access imagery
        for item in event_data['links']:
            if 'thumbnail' in item.get('rel', ''):
                print(item['href'])  # Direct image URL
```

---

## Storage Requirements

| Dataset Tier | Compressed | Uncompressed | Approx Time (100Mbps) |
|--------------|-----------|-------------|-------|
| Tier 1 (2,799 pairs) | ~4 GB | ~4.5 GB | 5-6 min |
| Tier 3 (5,600 pairs) | ~7 GB | ~7.5 GB | 9-11 min |
| Test Set (933 pairs) | ~1.5 GB | ~1.6 GB | 2 min |
| Full Dataset (8,399) | ~10 GB | ~11 GB | 13-16 min |
| Roboflow Sample | ~100 MB | ~110 MB | <1 min |

---

## Processing After Download

### Load Images (OpenCV)
```python
import cv2
import numpy as np

# Read pre/post pair
pre = cv2.imread('guatemala-volcano_00000000_pre.png')
post = cv2.imread('guatemala-volcano_00000000_post.png')

# BGR to RGB
pre_rgb = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
post_rgb = cv2.cvtColor(post, cv2.COLOR_BGR2RGB)

print(f"Shape: {pre_rgb.shape}")  # (1024, 1024, 3)
print(f"Type: {pre_rgb.dtype}")   # uint8
```

### Load Annotations (JSON)
```python
import json

with open('guatemala-volcano_00000000_buildings.json') as f:
    data = json.load(f)

for feature in data['features']:
    damage_level = feature['properties']['damage']  # 0-3
    polygon = feature['geometry']['coordinates'][0]  # [(x1,y1), (x2,y2), ...]

    print(f"Damage: {damage_level}, Vertices: {len(polygon)}")
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Dataset not found" (TorchGeo) | Ensure torch, torchvision installed; check disk space |
| "403 Forbidden" (xView2.org) | Login token expired; login again, re-request links |
| "Polygon parsing error" | Check JSON format; ensure valid GeoJSON structure |
| "Memory error loading full dataset" | Use data loader batching; process by disaster subset |
| "Missing post-disaster images" | Dataset incomplete; verify download integrity |
| "Wrong image dimensions" | Some sensors produce non-1024×1024; check metadata |

---

## Legal & Attribution

**Always Include**:
- Paper citation (Gupta et al. 2019)
- Maxar/DigitalGlobe attribution
- Challenge terms acknowledgment

**Suggested Attribution**:
> Satellite imagery sourced from Maxar/DigitalGlobe Open Data Program. Building damage annotations from the xBD dataset ([Gupta et al., 2019](https://arxiv.org/abs/1911.09296)).

---

## References

- Official: https://xview2.org/dataset
- Paper: https://arxiv.org/abs/1911.09296
- Roboflow: https://universe.roboflow.com/ozu/xview2
- TorchGeo: https://torchgeo.readthedocs.io/en/v0.5.0/api/datasets.html#torchgeo.datasets.xView2
- Maxar STAC: https://maxar-opendata.s3.amazonaws.com/events/catalog.json
