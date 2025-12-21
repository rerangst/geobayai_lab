# Phase 5: Ship Detection Application

## Context
- **Parent Plan:** [plan.md](./plan.md)
- **Dependencies:** Phases 1, 2, 3, 4 complete
- **Blockers:** Must wait for Ch3 (models), Ch4 (xView) to be finalized

## Parallelization
- **Concurrent with:** Phase 6
- **Blocks:** Phase 7

## Overview
Transform current Ch3 (Ship Detection) into new Ch5. Reframe as application chapter with cross-references to model architectures (Ch3) and xView3 solutions (Ch4).

## File Ownership (Exclusive)

| Current Path | New Path | Action |
|-------------|----------|--------|
| `chuong-03-phat-hien-tau-bien/muc-01-dac-diem-bai-toan/01-dac-diem.md` | `chuong-05-ung-dung-tau-bien/muc-01-dac-diem/01-dac-diem.md` | Rename + Add refs |
| `chuong-03-phat-hien-tau-bien/muc-02-mo-hinh/01-cac-mo-hinh.md` | `chuong-05-ung-dung-tau-bien/muc-02-mo-hinh/01-mo-hinh-ung-dung.md` | Rename + Refactor |
| `chuong-03-phat-hien-tau-bien/muc-03-quy-trinh/01-pipeline.md` | `chuong-05-ung-dung-tau-bien/muc-03-pipeline/01-pipeline.md` | Rename + Add refs |
| `chuong-03-phat-hien-tau-bien/muc-04-bo-du-lieu/01-datasets.md` | `chuong-05-ung-dung-tau-bien/muc-04-datasets/01-datasets.md` | Rename |

Total: 4 files

## Implementation Steps

### 1. Create New Directory Structure
```bash
mkdir -p research/chuong-05-ung-dung-tau-bien/{muc-01-dac-diem,muc-02-mo-hinh,muc-03-pipeline,muc-04-datasets}
```

### 2. Move Files
Use git mv for all 4 files

### 3. Add Cross-References

**01-dac-diem.md:**
- Reference SAR characteristics from Ch2
- Link to xView3 dataset (Ch4.3)

**01-mo-hinh-ung-dung.md:**
- Refactor to reference Ch3 models instead of repeating
- Add: "Xem Chi tiet kien truc tai Chuong 3"
- Focus on application-specific adaptations

**01-pipeline.md:**
- Reference TorchGeo trainers from Ch3
- Add inference pipeline diagram

**01-datasets.md:**
- Cross-reference xView3 dataset from Ch4.3
- Compare with other maritime datasets

### 4. Add Application-Specific Content
- Real-world deployment considerations
- Vietnamese maritime context
- Computational requirements

## Narrative Transition Requirements

**Đầu chương 5:**
```markdown
## Giới thiệu

Chương này áp dụng các kiến trúc object detection từ **Chương 3** và các
kỹ thuật từ cuộc thi xView3-SAR (**Chương 4**) vào bài toán thực tế:
phát hiện tàu biển trên ảnh viễn thám. Đây là ứng dụng quan trọng trong
giám sát hàng hải và chống đánh bắt bất hợp pháp (IUU fishing).
```

**Cuối chương 5:**
```markdown
## Kết chương

Bài toán phát hiện tàu biển minh họa ứng dụng object detection trong
viễn thám. Tương tự, **Chương 6** sẽ trình bày một ứng dụng khác - phát
hiện dầu loang - sử dụng các kỹ thuật semantic segmentation đã học ở
các chương trước.
```

## Success Criteria
- [ ] All 4 files moved to chuong-05-ung-dung-tau-bien/
- [ ] Old chuong-03-phat-hien-tau-bien/ removed
- [ ] Cross-refs to Ch3 (models) added
- [ ] Cross-refs to Ch4.3 (xView3) added
- [ ] No duplicated model architecture content
- [ ] **Narrative transitions added**

## Conflict Prevention
- Do NOT modify Ch3, Ch4 files
- Only add backward references, not forward
- Phase 8 handles VitePress config
