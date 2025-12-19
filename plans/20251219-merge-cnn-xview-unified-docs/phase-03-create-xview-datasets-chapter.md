# Phase 3: Create xView Datasets Chapter

## Objective
Consolidate all 3 xView dataset specifications into single reference chapter.

## Source Files
| File | Challenge | Content |
|------|-----------|---------|
| `docs/xview-challenges/xview1/dataset-xview1-detection.md` | xView1 | Object detection dataset |
| `docs/xview-challenges/xview2/dataset-xview2-xbd-building-damage.md` | xView2 | Building damage dataset |
| `docs/xview-challenges/xview3/dataset-xview3-sar-maritime.md` | xView3 | Maritime SAR dataset |

## Target Directory
`docs/unified-remote-sensing/02-xview-datasets/`

## Operations

### 1. Copy dataset files with renumbered names
```bash
cp docs/xview-challenges/xview1/dataset-xview1-detection.md \
   docs/unified-remote-sensing/02-xview-datasets/01-xview1-object-detection.md

cp docs/xview-challenges/xview2/dataset-xview2-xbd-building-damage.md \
   docs/unified-remote-sensing/02-xview-datasets/02-xview2-building-damage.md

cp docs/xview-challenges/xview3/dataset-xview3-sar-maritime.md \
   docs/unified-remote-sensing/02-xview-datasets/03-xview3-maritime-sar.md
```

### 2. Create chapter README
Create `docs/unified-remote-sensing/02-xview-datasets/README.md`:

```markdown
# Chuong 2: Bo du lieu xView

## Tong quan

xView Challenge Series la chuoi 3 cuoc thi computer vision tren anh ve tinh do Defense Innovation Unit (DIU) to chuc tu 2018-2022.

| Challenge | Nam | Nhiem vu | Anh ve tinh | Quy mo |
|-----------|-----|----------|-------------|--------|
| xView1 | 2018 | Object Detection | WorldView-3 (0.3m) | 1M+ objects, 60 classes |
| xView2 | 2019 | Building Damage | Maxar (<0.8m) | 850K buildings, 19 disasters |
| xView3 | 2021-22 | Maritime Detection | Sentinel-1 SAR | 243K objects, 43.2M km2 |

## Tai lieu

1. [xView1: Object Detection Dataset](./01-xview1-object-detection.md)
2. [xView2: Building Damage Dataset](./02-xview2-building-damage.md)
3. [xView3: Maritime SAR Dataset](./03-xview3-maritime-sar.md)

## Luu y

Chuong nay chi chua thong so ky thuat cua bo du lieu. Cac giai phap dat giai duoc trinh bay trong:
- Chuong 5: Ship Detection (xView3 winners)
- Chuong 6: Building Damage (xView2 winners)
- Chuong 7: Object Detection (xView1 winners)
```

## Verification
```bash
ls -la docs/unified-remote-sensing/02-xview-datasets/
# Expected: 4 files (README.md + 3 dataset files)
```

## Git Commit
```bash
git add docs/unified-remote-sensing/02-xview-datasets/
git commit -m "docs: create Chapter 2 - xView datasets reference"
```
