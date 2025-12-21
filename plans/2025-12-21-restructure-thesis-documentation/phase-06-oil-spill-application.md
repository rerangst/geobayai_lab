# Phase 6: Oil Spill Application

## Context
- **Parent Plan:** [plan.md](./plan.md)
- **Dependencies:** Phases 1, 2, 3, 4 complete
- **Blockers:** Must wait for Ch3 (models), Ch4 (xView) to be finalized

## Parallelization
- **Concurrent with:** Phase 5
- **Blocks:** Phase 7

## Overview
Transform current Ch4 (Oil Spill Detection) into new Ch6. Reframe as application chapter with cross-references to segmentation models (Ch3) and damage detection approaches (Ch4.2).

## File Ownership (Exclusive)

| Current Path | New Path | Action |
|-------------|----------|--------|
| `chuong-04-phat-hien-dau-loang/muc-01-dac-diem-bai-toan/01-dac-diem.md` | `chuong-06-ung-dung-dau-loang/muc-01-dac-diem/01-dac-diem.md` | Rename + Add refs |
| `chuong-04-phat-hien-dau-loang/muc-02-mo-hinh/01-cac-mo-hinh.md` | `chuong-06-ung-dung-dau-loang/muc-02-mo-hinh/01-mo-hinh-ung-dung.md` | Rename + Refactor |
| `chuong-04-phat-hien-dau-loang/muc-03-quy-trinh/01-pipeline.md` | `chuong-06-ung-dung-dau-loang/muc-03-pipeline/01-pipeline.md` | Rename + Add refs |
| `chuong-04-phat-hien-dau-loang/muc-04-bo-du-lieu/01-datasets.md` | `chuong-06-ung-dung-dau-loang/muc-04-datasets/01-datasets.md` | Rename |

Total: 4 files

## Implementation Steps

### 1. Create New Directory Structure
```bash
mkdir -p research/chuong-06-ung-dung-dau-loang/{muc-01-dac-diem,muc-02-mo-hinh,muc-03-pipeline,muc-04-datasets}
```

### 2. Move Files
Use git mv for all 4 files

### 3. Add Cross-References

**01-dac-diem.md:**
- Reference SAR polarimetry from Ch2
- Note similarity to change detection problem

**01-mo-hinh-ung-dung.md:**
- Refactor to reference Ch3 segmentation models
- Add: "Xem kien truc U-Net tai Muc 3.3"
- Focus on oil spill-specific adaptations
- Reference xView2 damage segmentation techniques (Ch4.2)

**01-pipeline.md:**
- Reference TorchGeo segmentation trainers
- Add pre/post processing specific to oil detection

**01-datasets.md:**
- Compare SAR oil spill datasets
- Note differences from optical imagery

### 4. Add Application-Specific Content
- Environmental monitoring context
- Vietnamese coastal considerations
- Multi-temporal analysis for spill tracking

## Narrative Transition Requirements

**Đầu chương 6:**
```markdown
## Giới thiệu

Tiếp nối bài toán object detection ở **Chương 5**, chương này trình bày
ứng dụng semantic segmentation vào phát hiện dầu loang trên ảnh SAR.
Kỹ thuật segmentation từ **Chương 3** và phương pháp damage assessment
từ xView2 (**Chương 4**) sẽ được áp dụng vào bối cảnh giám sát môi
trường biển.
```

**Cuối chương 6:**
```markdown
## Kết chương

Hai chương ứng dụng (phát hiện tàu và dầu loang) đã minh họa cách áp
dụng các kiến trúc deep learning vào bài toán viễn thám thực tế.
**Chương 7** sẽ tổng kết các nội dung chính và đề xuất hướng phát triển
trong tương lai.
```

## Success Criteria
- [ ] All 4 files moved to chuong-06-ung-dung-dau-loang/
- [ ] Old chuong-04-phat-hien-dau-loang/ removed
- [ ] Cross-refs to Ch3.3 (segmentation) added
- [ ] Cross-refs to Ch4.2 (xView2 damage) added
- [ ] No duplicated segmentation model content
- [ ] **Narrative transitions added**

## Conflict Prevention
- Do NOT modify Ch3, Ch4 files
- Only add backward references
- Phase 8 handles VitePress config
