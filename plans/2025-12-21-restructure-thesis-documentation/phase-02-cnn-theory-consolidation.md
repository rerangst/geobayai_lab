# Phase 2: CNN Theory Consolidation

## Context
- **Parent Plan:** [plan.md](./plan.md)
- **Dependencies:** Phase 0 (cnn-basics images renamed)
- **Blockers:** None after Phase 0

## Parallelization
- **Concurrent with:** Phases 1, 3, 4
- **Blocks:** Phases 5, 6, 7

## Overview
Enhance CNN theory with remote sensing context. Add ViT/Transformer. Update image paths. Add narrative transitions.

## File Ownership (Exclusive)

| File | Action |
|------|--------|
| `research/chuong-02-co-so-ly-thuyet/muc-01-kien-truc-cnn/01-kien-truc-co-ban.md` | Enhance |
| `research/chuong-02-co-so-ly-thuyet/muc-01-kien-truc-cnn/02-backbone-networks.md` | Enhance |
| `research/chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/01-phan-loai-anh.md` | Minor updates |
| `research/chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/02-phat-hien-doi-tuong.md` | Minor updates |
| `research/chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/03-phan-doan-ngu-nghia.md` | Minor updates |
| `research/chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/04-instance-segmentation.md` | Minor updates |

Total: 6 files

## Implementation Steps

### 1. Review All 6 Files
- Identify duplicate content
- Note missing modern architectures

### 2. Enhance 01-kien-truc-co-ban.md
Add sections:
- Remote sensing image characteristics (multispectral, SAR)
- Multi-scale feature challenges

### 3. Enhance 02-backbone-networks.md
Add:
- Vision Transformer (ViT) basics
- Swin Transformer for dense prediction
- Self-supervised pre-training (MAE, MoCo)

### 4. Update Method Files (01-04)
- Add forward references to Ch3 (TorchGeo models)
- Add forward references to Ch4 (xView solutions)
- Remove any duplicated architecture details

### 5. Add Comparison Table
Table comparing CNN vs Transformer for remote sensing tasks

## Narrative Transition Requirements

**Đầu chương 2:**
```markdown
## Giới thiệu

Sau khi đã nắm được bối cảnh và mục tiêu nghiên cứu ở **Chương 1**,
chương này trình bày cơ sở lý thuyết về mạng CNN và các phương pháp
xử lý ảnh viễn thám. Đây là nền tảng kiến thức cần thiết để hiểu các
kiến trúc mô hình cụ thể được giới thiệu ở các chương tiếp theo.
```

**Cuối chương 2:**
```markdown
## Kết chương

Chương này đã trình bày các kiến thức nền tảng về CNN và các bài toán
xử lý ảnh. **Chương 3** sẽ giới thiệu các kiến trúc mô hình cụ thể
được triển khai trong thư viện TorchGeo - công cụ chuyên biệt cho
deep learning trong viễn thám.
```

## Success Criteria
- [ ] ViT/Transformer section added
- [ ] Remote sensing context integrated
- [ ] Image paths updated (cnn-basics → chuong-02-cnn)
- [ ] No duplicate content between files
- [ ] Forward references to Ch3, Ch4 added
- [ ] Vietnamese academic style
- [ ] **Narrative transitions added**

## Conflict Prevention
- Do NOT modify chuong-03, 04, 05, 06, 07
- Cross-refs use new chapter numbers (Ch3=TorchGeo, Ch4=xView)
- Update image paths per Phase 0 renaming
