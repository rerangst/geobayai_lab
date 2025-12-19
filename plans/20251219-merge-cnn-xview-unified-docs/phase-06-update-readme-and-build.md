# Phase 6: Update README and Build Scripts

## Objective
Update project README, create unified build script, update VitePress config.

---

## 1. Create Unified README

Create `docs/unified-remote-sensing/README.md`:

```markdown
# Ung dung Deep Learning trong Vien tham va xView Challenges

## Gioi thieu

Tai lieu thong nhat ve:
1. **CNN/Deep Learning** - Kien truc, phuong phap cho anh ve tinh
2. **xView Challenges** - 3 cuoc thi computer vision (2018-2022)
3. **Ung dung thuc te** - Ship Detection, Building Damage, Oil Spill Detection

---

## Muc luc

### Chuong 1: Tong quan
- [Gioi thieu tong quan](./01-tong-quan/gioi-thieu-tong-quan.md)

### Chuong 2: Bo du lieu xView
- [xView1: Object Detection Dataset](./02-xview-datasets/01-xview1-object-detection.md)
- [xView2: Building Damage Dataset](./02-xview-datasets/02-xview2-building-damage.md)
- [xView3: Maritime SAR Dataset](./02-xview-datasets/03-xview3-maritime-sar.md)

### Chuong 3: Kien truc CNN co ban
- [Kien truc CNN co ban](./03-cnn-fundamentals/01-kien-truc-cnn-co-ban.md)
- [Backbone Networks](./03-cnn-fundamentals/02-backbone-networks.md)

### Chuong 4: Phuong phap CNN voi anh ve tinh
- [Phan loai anh](./04-cnn-satellite-methods/01-classification.md)
- [Phat hien doi tuong](./04-cnn-satellite-methods/02-object-detection.md)
- [Phan doan ngu nghia](./04-cnn-satellite-methods/03-semantic-segmentation.md)
- [Instance Segmentation](./04-cnn-satellite-methods/04-instance-segmentation.md)

### Chuong 5: Phat hien tau bien (Ship Detection)
- [Dac diem bai toan](./05-ship-detection/01-dac-diem-bai-toan.md)
- [Cac model](./05-ship-detection/02-cac-model.md)
- [Pipeline](./05-ship-detection/03-pipeline.md)
- [Datasets](./05-ship-detection/04-datasets.md)
- [xView3 Winners](./05-ship-detection/05-xview3-winner-1st-circlenet.md) (5 solutions)

### Chuong 6: Danh gia thiet hai toa nha (Building Damage)
- [xView2 Winners](./06-building-damage/01-xview2-winner-1st-siamese-unet.md) (5 solutions)

### Chuong 7: Phat hien doi tuong (Object Detection - xView1)
- [xView1 Winners](./07-object-detection-xview1/01-xview1-winner-1st-focal-loss.md) (5 solutions)

### Chuong 8: Phat hien vet dau loang (Oil Spill Detection)
- [Dac diem bai toan](./08-oil-spill-detection/01-dac-diem-bai-toan.md)
- [Cac model](./08-oil-spill-detection/02-cac-model.md)
- [Pipeline](./08-oil-spill-detection/03-pipeline.md)
- [Datasets](./08-oil-spill-detection/04-datasets.md)

### Chuong 9: TorchGeo Models
- [Tong quan TorchGeo](./09-torchgeo-models/01-tong-quan.md)
- [Classification Models](./09-torchgeo-models/02-classification-models.md)
- [Segmentation Models](./09-torchgeo-models/03-segmentation-models.md)
- [Change Detection Models](./09-torchgeo-models/04-change-detection-models.md)
- [Pretrained Weights](./09-torchgeo-models/05-pretrained-weights.md)

### Chuong 10: Ket luan
- [Ket luan va huong phat trien](./10-ket-luan/ket-luan-va-huong-phat-trien.md)

---

## Thong tin

| Thuoc tinh | Gia tri |
|------------|---------|
| **Tong so chuong** | 10 |
| **Tong so file** | 39+ |
| **Ngon ngu** | Tieng Viet (giu nguyen thuat ngu tieng Anh) |
| **Pham vi** | CNN/Deep Learning, xView Challenges, Ship/Oil/Building Detection |

---

## Tai lieu tham khao

- TorchGeo: https://torchgeo.readthedocs.io/
- xView Challenge: https://xviewdataset.org/
- Copernicus: https://dataspace.copernicus.eu/
```

---

## 2. Create Unified Build Script

Create `scripts/build-unified-docx.sh`:

```bash
#!/bin/bash
# Build DOCX from Unified Remote Sensing Markdown files
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Building unified-remote-sensing.docx...${NC}"

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
}

check_command pandoc

mkdir -p output
DATE=$(date +%Y-%m-%d)

MERMAID_FLAG=""
if command -v mermaid-filter &> /dev/null; then
    MERMAID_FLAG="-F mermaid-filter"
fi

pandoc \
    docs/unified-remote-sensing/README.md \
    docs/unified-remote-sensing/01-tong-quan/gioi-thieu-tong-quan.md \
    docs/unified-remote-sensing/02-xview-datasets/01-xview1-object-detection.md \
    docs/unified-remote-sensing/02-xview-datasets/02-xview2-building-damage.md \
    docs/unified-remote-sensing/02-xview-datasets/03-xview3-maritime-sar.md \
    docs/unified-remote-sensing/03-cnn-fundamentals/01-kien-truc-cnn-co-ban.md \
    docs/unified-remote-sensing/03-cnn-fundamentals/02-backbone-networks.md \
    docs/unified-remote-sensing/04-cnn-satellite-methods/01-classification.md \
    docs/unified-remote-sensing/04-cnn-satellite-methods/02-object-detection.md \
    docs/unified-remote-sensing/04-cnn-satellite-methods/03-semantic-segmentation.md \
    docs/unified-remote-sensing/04-cnn-satellite-methods/04-instance-segmentation.md \
    docs/unified-remote-sensing/05-ship-detection/01-dac-diem-bai-toan.md \
    docs/unified-remote-sensing/05-ship-detection/02-cac-model.md \
    docs/unified-remote-sensing/05-ship-detection/03-pipeline.md \
    docs/unified-remote-sensing/05-ship-detection/04-datasets.md \
    docs/unified-remote-sensing/05-ship-detection/05-xview3-winner-1st-circlenet.md \
    docs/unified-remote-sensing/05-ship-detection/06-xview3-winner-2nd-selim.md \
    docs/unified-remote-sensing/05-ship-detection/07-xview3-winner-3rd-tumenn.md \
    docs/unified-remote-sensing/05-ship-detection/08-xview3-winner-4th-ai2.md \
    docs/unified-remote-sensing/05-ship-detection/09-xview3-winner-5th-kohei.md \
    docs/unified-remote-sensing/06-building-damage/01-xview2-winner-1st-siamese-unet.md \
    docs/unified-remote-sensing/06-building-damage/02-xview2-winner-2nd-selim.md \
    docs/unified-remote-sensing/06-building-damage/03-xview2-winner-3rd-eugene.md \
    docs/unified-remote-sensing/06-building-damage/04-xview2-winner-4th-zheng.md \
    docs/unified-remote-sensing/06-building-damage/05-xview2-winner-5th-dual-hrnet.md \
    docs/unified-remote-sensing/07-object-detection-xview1/01-xview1-winner-1st-focal-loss.md \
    docs/unified-remote-sensing/07-object-detection-xview1/02-xview1-winner-2nd-adelaide.md \
    docs/unified-remote-sensing/07-object-detection-xview1/03-xview1-winner-3rd-usf.md \
    docs/unified-remote-sensing/07-object-detection-xview1/04-xview1-winner-4th-studio-mapp.md \
    docs/unified-remote-sensing/07-object-detection-xview1/05-xview1-winner-5th-cmu-sei.md \
    docs/unified-remote-sensing/08-oil-spill-detection/01-dac-diem-bai-toan.md \
    docs/unified-remote-sensing/08-oil-spill-detection/02-cac-model.md \
    docs/unified-remote-sensing/08-oil-spill-detection/03-pipeline.md \
    docs/unified-remote-sensing/08-oil-spill-detection/04-datasets.md \
    docs/unified-remote-sensing/09-torchgeo-models/01-tong-quan.md \
    docs/unified-remote-sensing/09-torchgeo-models/02-classification-models.md \
    docs/unified-remote-sensing/09-torchgeo-models/03-segmentation-models.md \
    docs/unified-remote-sensing/09-torchgeo-models/04-change-detection-models.md \
    docs/unified-remote-sensing/09-torchgeo-models/05-pretrained-weights.md \
    docs/unified-remote-sensing/10-ket-luan/ket-luan-va-huong-phat-trien.md \
    -o output/unified-remote-sensing.docx \
    --toc \
    --toc-depth=3 \
    $MERMAID_FLAG \
    --metadata title="Ung dung Deep Learning trong Vien tham va xView Challenges" \
    --metadata author="Research Team" \
    --metadata date="$DATE" \
    --standalone

echo -e "${GREEN}Done! Output: output/unified-remote-sensing.docx${NC}"
```

---

## 3. Update package.json

Add new script to `package.json`:

```json
{
  "scripts": {
    "build:unified": "bash scripts/build-unified-docx.sh"
  }
}
```

---

## 4. Update Project README

Update `/home/tchatb/sen_doc/README.md` to add unified docs section:

**Add after ## Documentation section:**

```markdown
## Unified Documentation (NEW)

The unified documentation merges CNN remote sensing fundamentals with xView challenge solutions.

```bash
# Build unified DOCX
npm run build:unified
```

Structure:
- 10 chapters covering CNN theory + xView practical solutions
- 39+ markdown files
- Vietnamese language with English technical terms
```

---

## 5. Make Script Executable

```bash
chmod +x scripts/build-unified-docx.sh
```

---

## Verification

```bash
# Test build script syntax
bash -n scripts/build-unified-docx.sh

# Verify README exists
cat docs/unified-remote-sensing/README.md | head -20

# Count total files
find docs/unified-remote-sensing -name "*.md" | wc -l
# Expected: ~40 files
```

## Git Commit

```bash
git add docs/unified-remote-sensing/README.md
git add scripts/build-unified-docx.sh
git add package.json
git add README.md
git commit -m "docs: add unified docs README and build script"
```

---

## Post-Implementation Cleanup (Optional)

After verifying unified docs work correctly:

```bash
# Option A: Keep original docs (recommended initially)
# No action needed

# Option B: Archive original docs
# mkdir -p docs/_archive
# mv docs/cnn-remote-sensing docs/_archive/
# mv docs/xview-challenges docs/_archive/
```
