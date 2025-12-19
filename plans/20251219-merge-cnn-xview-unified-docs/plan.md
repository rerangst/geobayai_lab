# Implementation Plan: Merge CNN & xView Docs into Unified Structure

**Date:** 2025-12-19 | **Status:** Ready for Implementation

## Objective
Merge 21 CNN remote sensing docs + 18 xView challenge docs into unified 10-chapter structure preserving all content.

## Target Structure
```
docs/unified-remote-sensing/
├── 01-tong-quan/                    # Merged CNN intro + general overview
├── 02-xview-datasets/               # xView 1,2,3 dataset specs ONLY
├── 03-cnn-fundamentals/             # CNN architecture + backbones
├── 04-cnn-satellite-methods/        # Classification, detection, segmentation
├── 05-ship-detection/               # CNN ship detection + xView3 winners
├── 06-building-damage/              # xView2 winners (NEW section)
├── 07-object-detection-xview1/      # xView1 winners
├── 08-oil-spill-detection/          # CNN oil spill detection
├── 09-torchgeo-models/              # TorchGeo integration
└── 10-ket-luan/                     # Conclusion
```

## Phase Overview

| Phase | Description | Files Affected |
|-------|-------------|----------------|
| 1 | Create directory structure | 10 new directories |
| 2 | Merge Chapter 1 (overview) | 2 source -> 1 merged |
| 3 | Create xView datasets chapter | 3 dataset files moved |
| 4 | Reorganize CNN chapters (3-4) | 6 files moved |
| 5 | Integrate xView winners | 15 winner files redistributed |
| 6 | Update README + build scripts | 3 config files |

## Key Decisions
1. **Preserve bilingual content** - Keep Vietnamese + English terminology
2. **No content deletion** - All 39 files preserved
3. **Winner solutions grouped by task** - Not by competition
4. **Datasets separated from methods** - Chapter 2 is reference-only

## Execution Order
1. phase-01-create-unified-structure.md
2. phase-02-merge-chapter-1-overview.md
3. phase-03-create-xview-datasets-chapter.md
4. phase-04-reorganize-cnn-chapters.md
5. phase-05-integrate-xview-winners.md
6. phase-06-update-readme-and-build.md

## Risk Mitigation
- Git commit after each phase for rollback capability
- Verify all links/references after move operations
- Test build scripts before final commit
