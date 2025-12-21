# Phase 3: TorchGeo Models Expansion

## Context
- **Parent Plan:** [plan.md](./plan.md)
- **Dependencies:** Phase 0 (papers folder moved)
- **Blockers:** None after Phase 0

## Parallelization
- **Concurrent with:** Phases 1, 2, 4
- **Blocks:** Phases 5, 6, 7

## Overview
Transform Ch5 (TorchGeo) → Ch3 (Model Architecture). Include papers/ folder. Add narrative transitions.

## File Ownership (Exclusive)

| Current Path | New Path | Action |
|-------------|----------|--------|
| `chuong-05-torchgeo/muc-01-tong-quan/01-tong-quan.md` | `chuong-03-kien-truc-model/muc-01-tong-quan/01-torchgeo-overview.md` | Rename + Enhance |
| `chuong-05-torchgeo/muc-02-classification/01-classification-models.md` | `chuong-03-kien-truc-model/muc-02-classification/01-classification-models.md` | Rename + Enhance |
| `chuong-05-torchgeo/muc-03-segmentation/01-segmentation-models.md` | `chuong-03-kien-truc-model/muc-03-segmentation/01-segmentation-models.md` | Rename + Enhance |
| `chuong-05-torchgeo/muc-04-change-detection/01-change-detection-models.md` | `chuong-03-kien-truc-model/muc-04-change-detection/01-change-detection-models.md` | Rename + Enhance |
| `chuong-05-torchgeo/muc-05-pretrained-weights/01-pretrained-weights.md` | `chuong-03-kien-truc-model/muc-05-pretrained-weights/01-pretrained-weights.md` | Rename + Enhance |
| `chuong-05-torchgeo/papers/*` | `chuong-03-kien-truc-model/papers/*` | Move (28 PDFs) |

Total: 5 markdown files + 28 paper PDFs + 2 scripts

## Implementation Steps

### 1. Create New Directory Structure
```bash
mkdir -p research/chuong-03-kien-truc-model/{muc-01-tong-quan,muc-02-classification,muc-03-segmentation,muc-04-change-detection,muc-05-pretrained-weights}
```

### 2. Move and Rename Files
Move each file to new location with git mv

### 3. Enhance Content Per File

**01-torchgeo-overview.md:**
- Add TorchGeo architecture diagram
- Explain dataset/dataloader/trainer pattern
- Add comparison with other geo libraries

**01-classification-models.md:**
- ResNet, EfficientNet specifics for RS
- ViT-based classifiers

**01-segmentation-models.md:**
- U-Net, FPN, DeepLabV3
- Swin-based segmentation

**01-change-detection-models.md:**
- Siamese networks
- Bitemporal fusion strategies

**01-pretrained-weights.md:**
- SSL4EO weights
- SatMAE, Prithvi foundation models

### 4. Update Internal Links
All internal references updated to chuong-03-*

## Narrative Transition Requirements

**Đầu chương 3:**
```markdown
## Giới thiệu

Dựa trên nền tảng lý thuyết CNN đã trình bày ở **Chương 2**, chương này
giới thiệu các kiến trúc mô hình cụ thể được triển khai trong thư viện
TorchGeo - công cụ chuyên biệt cho xử lý ảnh viễn thám.
```

**Cuối chương 3:**
```markdown
## Kết chương

Các kiến trúc mô hình được trình bày trong chương này sẽ được minh họa
qua các giải pháp chiến thắng trong **Chương 4 - xView Challenges**,
và ứng dụng thực tế trong **Chương 5** (phát hiện tàu) và **Chương 6**
(phát hiện dầu loang).
```

## Success Criteria
- [ ] All 5 markdown files moved to chuong-03-kien-truc-model/
- [ ] Papers folder moved with all 28 PDFs
- [ ] Old chuong-05-torchgeo/ directory removed
- [ ] Model architecture diagrams added (Mermaid)
- [ ] Forward refs to Ch4 (xView applications)
- [ ] Backward refs to Ch2 (theory foundations)
- [ ] **Narrative transitions added**

## Conflict Prevention
- Do NOT modify other chapters during rename
- Remove old directory only after git mv complete
- Phase 8 handles VitePress config update
