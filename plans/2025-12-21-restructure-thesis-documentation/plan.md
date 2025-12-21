# Plan: Restructure Vietnamese Thesis Documentation

## Overview
Reorganize 7-chapter thesis with TorchGeo/xView before applications. Add narrative transitions between chapters.

## Current vs Target Structure

| Current | Target | Transformation |
|---------|--------|----------------|
| Ch1: Gioi thieu | Ch1: Gioi thieu | Expanded + transition |
| Ch2: Co so ly thuyet | Ch2: CNN + Vien tham | Enhanced |
| Ch5: TorchGeo | Ch3: Kien truc Model | Major rewrite + papers |
| Ch6: xView | Ch4: xView Challenges | Reorganized |
| Ch3: Tau bien | Ch5: Tau bien | Add cross-refs |
| Ch4: Dau loang | Ch6: Dau loang | Add cross-refs |
| Ch7: Ket luan | Ch7: Ket luan | Synthesis |

## Asset Renaming (Per User Request)

| Current Path | New Path |
|-------------|----------|
| `assets/images/xview1/` | `assets/images/chuong-04-xview1/` |
| `assets/images/xview2/` | `assets/images/chuong-04-xview2/` |
| `assets/images/xview3/` | `assets/images/chuong-04-xview3/` |
| `assets/images/cnn-basics/` | `assets/images/chuong-02-cnn/` |
| `chuong-05-torchgeo/papers/` | `chuong-03-kien-truc-model/papers/` |

## Dependency Graph

```
Phase 0 (Assets) ──┐
Phase 1 (Ch1) ─────┼──► Phase 5 (Ch5) ──┐
Phase 2 (Ch2) ─────┤                    ├──► Phase 7 (Ch7) ──► Phase 8 (Config)
Phase 3 (Ch3) ─────┼──► Phase 6 (Ch6) ──┘
Phase 4 (Ch4) ─────┘
```

## Execution Strategy

| Wave | Phases | Parallel | Notes |
|------|--------|----------|-------|
| 0 | 0 | No | Assets first (path dependencies) |
| 1 | 1,2,3,4 | Yes | Content restructure |
| 2 | 5,6 | Yes | Applications |
| 3 | 7 | No | Conclusion |
| 4 | 8 | No | VitePress config |

## File Ownership Matrix

| Phase | Exclusive Files | Action |
|-------|-----------------|--------|
| 0 | assets/images/*, papers/ | Rename folders |
| 1 | chuong-01-gioi-thieu/* | Rewrite + transition |
| 2 | chuong-02-co-so-ly-thuyet/* | Enhance + transition |
| 3 | chuong-05-torchgeo/* -> chuong-03-* | Rename + papers + transition |
| 4 | chuong-06-xview-challenges/* -> chuong-04-* | Rename + transition |
| 5 | chuong-03-phat-hien-tau-bien/* -> chuong-05-* | Rename + refs + transition |
| 6 | chuong-04-phat-hien-dau-loang/* -> chuong-06-* | Rename + refs + transition |
| 7 | chuong-07-ket-luan/* | Synthesis |
| 8 | .vitepress/config.mjs, index.md | Update nav + paths |

## Writing Guidelines: Narrative Transitions

**Mỗi chương/mục PHẢI có:**
1. **Lời dẫn đầu chương** - Giới thiệu tổng quan nội dung chương
2. **Kết nối với chương trước** - "Dựa trên cơ sở lý thuyết ở Chương N..."
3. **Dẫn dắt sang chương sau** - "Chương tiếp theo sẽ trình bày..."

**Template:**
```markdown
## Giới thiệu Chương

Chương này trình bày [nội dung chính]. Dựa trên [kiến thức từ chương trước],
chúng ta sẽ tìm hiểu [chi tiết]. Nội dung được chia thành:
- Mục 1: ...
- Mục 2: ...

## [Nội dung cuối chương]
...
Những kiến thức này sẽ được ứng dụng trong Chương X, nơi chúng ta...
```

## Phase Files
- [Phase 0: Asset Reorganization](./phase-00-asset-reorganization.md)
- [Phase 1: Introduction Rewrite](./phase-01-introduction-rewrite.md)
- [Phase 2: CNN Theory Consolidation](./phase-02-cnn-theory-consolidation.md)
- [Phase 3: TorchGeo Models Expansion](./phase-03-torchgeo-models-expansion.md)
- [Phase 4: xView Reorganization](./phase-04-xview-reorganization.md)
- [Phase 5: Ship Detection Application](./phase-05-ship-detection-application.md)
- [Phase 6: Oil Spill Application](./phase-06-oil-spill-application.md)
- [Phase 7: Conclusion Synthesis](./phase-07-conclusion-synthesis.md)
- [Phase 8: Navigation Config](./phase-08-navigation-config.md)

## Success Criteria
- [ ] All files restructured to new chapter numbering
- [ ] Asset folders renamed per new structure
- [ ] Papers folder moved to Ch3
- [ ] All image paths updated in markdown files
- [ ] Narrative transitions added to all chapters
- [ ] No broken internal links
- [ ] VitePress builds without errors
