# Phase 4: xView Reorganization

## Context
- **Parent Plan:** [plan.md](./plan.md)
- **Dependencies:** Phase 0 (image folders renamed)
- **Blockers:** None after Phase 0

## Parallelization
- **Concurrent with:** Phases 1, 2, 3
- **Blocks:** Phases 5, 6, 7

## Overview
Ch6 (xView) → Ch4. Add comparative analysis. Update image paths. Add narrative transitions.

## File Ownership (Exclusive)

| Current Path | New Path | Action |
|-------------|----------|--------|
| `chuong-06-xview-challenges/00-gioi-thieu-xview.md` | `chuong-04-xview-challenges/00-gioi-thieu.md` | Rename |
| `chuong-06-xview-challenges/muc-01-xview1-object-detection/*` | `chuong-04-xview-challenges/muc-01-xview1/*` | Rename (6 files) |
| `chuong-06-xview-challenges/muc-02-xview2-building-damage/*` | `chuong-04-xview-challenges/muc-02-xview2/*` | Rename (6 files) |
| `chuong-06-xview-challenges/muc-03-xview3-maritime/*` | `chuong-04-xview-challenges/muc-03-xview3/*` | Rename (6 files) |

Total: 19 files

## Implementation Steps

### 1. Create New Directory Structure
```bash
mkdir -p research/chuong-04-xview-challenges/{muc-01-xview1,muc-02-xview2,muc-03-xview3}
```

### 2. Move Files
Use git mv for all 19 files

### 3. Enhance Introduction (00-gioi-thieu.md)
Add:
- Comparative table: xView1 vs xView2 vs xView3
- Common techniques across challenges
- Evolution of approaches (2018-2022)

### 4. Streamline Solution Files
For each giai-*.md:
- Ensure consistent structure
- Add model architecture reference to Ch3
- Add technique comparison table

### 5. Add Cross-Challenge Analysis
New section in 00-gioi-thieu.md:
- Common winning patterns
- Transfer learning trends
- Ensemble strategies

### 6. Update Internal Links
All links updated to chuong-04-*

## Narrative Transition Requirements

**Đầu chương 4:**
```markdown
## Giới thiệu

Các kiến trúc mô hình đã giới thiệu ở **Chương 3** được kiểm chứng qua
loạt cuộc thi xView do Defense Innovation Unit tổ chức. Chương này phân
tích 3 cuộc thi và 15 giải pháp chiến thắng, minh họa cách áp dụng
các kỹ thuật deep learning vào bài toán viễn thám thực tế.
```

**Cuối chương 4:**
```markdown
## Kết chương

Các kỹ thuật và bài học từ xView Challenges sẽ được áp dụng cụ thể vào
hai bài toán thực tế: **Chương 5** trình bày phát hiện tàu biển (liên
quan đến xView1/xView3), và **Chương 6** trình bày phát hiện dầu loang
(ứng dụng kỹ thuật segmentation tương tự xView2).
```

## Success Criteria
- [ ] All 19 files moved to chuong-04-xview-challenges/
- [ ] Old chuong-06-xview-challenges/ removed
- [ ] Image paths updated to new folder names
- [ ] Comparative analysis table added
- [ ] Consistent structure across all solution files
- [ ] Backward refs to Ch3 (model architectures)
- [ ] **Narrative transitions added**

## Conflict Prevention
- Image folders already renamed in Phase 0
- Update all image paths in markdown files to new locations
- Phase 8 handles VitePress config
