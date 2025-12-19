# Phase 1: Create Unified Directory Structure

## Objective
Create 10-chapter directory hierarchy for unified documentation.

## Commands

```bash
# Create base directory
mkdir -p docs/unified-remote-sensing

# Create all chapter directories
mkdir -p docs/unified-remote-sensing/01-tong-quan
mkdir -p docs/unified-remote-sensing/02-xview-datasets
mkdir -p docs/unified-remote-sensing/03-cnn-fundamentals
mkdir -p docs/unified-remote-sensing/04-cnn-satellite-methods
mkdir -p docs/unified-remote-sensing/05-ship-detection
mkdir -p docs/unified-remote-sensing/06-building-damage
mkdir -p docs/unified-remote-sensing/07-object-detection-xview1
mkdir -p docs/unified-remote-sensing/08-oil-spill-detection
mkdir -p docs/unified-remote-sensing/09-torchgeo-models
mkdir -p docs/unified-remote-sensing/10-ket-luan
```

## Directory Mapping

| New Chapter | Source Content |
|-------------|----------------|
| 01-tong-quan | CNN 01-introduction + new overview |
| 02-xview-datasets | xView1/2/3 dataset-*.md files |
| 03-cnn-fundamentals | CNN 02-cnn-fundamentals/* |
| 04-cnn-satellite-methods | CNN 03-cnn-satellite-methods/* |
| 05-ship-detection | CNN 04-ship-detection + xView3 winners |
| 06-building-damage | xView2 winners (5 files) |
| 07-object-detection-xview1 | xView1 winners (5 files) |
| 08-oil-spill-detection | CNN 05-oil-spill-detection/* |
| 09-torchgeo-models | CNN 06-torchgeo-models/* |
| 10-ket-luan | CNN 07-conclusion/* |

## Verification

```bash
# Verify structure created
ls -la docs/unified-remote-sensing/
# Expected: 10 directories
```

## Git Commit

```bash
git add docs/unified-remote-sensing/
git commit -m "chore: create unified remote sensing directory structure"
```
