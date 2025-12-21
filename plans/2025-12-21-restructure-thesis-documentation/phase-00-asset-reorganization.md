# Phase 0: Asset Reorganization

## Context
- **Parent Plan:** [plan.md](./plan.md)
- **Dependencies:** None (runs first)
- **Blockers:** Blocks all content phases (1-8)

## Parallelization
- **Concurrent with:** None
- **Blocks:** Phases 1-8 (all content phases depend on correct asset paths)

## Overview
Rename asset folders and move papers to match new chapter structure. Must complete before content restructuring.

## File Ownership (Exclusive)

### Image Folders
| Current Path | New Path | Files |
|-------------|----------|-------|
| `assets/images/xview1/` | `assets/images/chuong-04-xview1/` | 21 files |
| `assets/images/xview2/` | `assets/images/chuong-04-xview2/` | 14 files |
| `assets/images/xview3/` | `assets/images/chuong-04-xview3/` | 18 files |
| `assets/images/cnn-basics/` | `assets/images/chuong-02-cnn/` | 16 files |

### Papers Folder
| Current Path | New Path | Files |
|-------------|----------|-------|
| `chuong-05-torchgeo/papers/` | `chuong-03-kien-truc-model/papers/` | 28 PDFs + 2 scripts |

## Implementation Steps

### 1. Rename Image Folders
```bash
cd research/assets/images
git mv xview1 chuong-04-xview1
git mv xview2 chuong-04-xview2
git mv xview3 chuong-04-xview3
git mv cnn-basics chuong-02-cnn
```

### 2. Move Papers Folder
```bash
# Papers will move with Phase 3 chapter rename
# Or create target first:
mkdir -p research/chuong-03-kien-truc-model
git mv research/chuong-05-torchgeo/papers research/chuong-03-kien-truc-model/papers
```

### 3. Update Image Paths in Markdown Files
Search and replace in all .md files:
| Old Pattern | New Pattern |
|-------------|-------------|
| `assets/images/xview1/` | `assets/images/chuong-04-xview1/` |
| `assets/images/xview2/` | `assets/images/chuong-04-xview2/` |
| `assets/images/xview3/` | `assets/images/chuong-04-xview3/` |
| `assets/images/cnn-basics/` | `assets/images/chuong-02-cnn/` |

### 4. Update Papers References
Update any markdown files referencing papers/ folder.

## Success Criteria
- [ ] All 4 image folders renamed
- [ ] Papers folder in new location
- [ ] All image paths in .md files updated
- [ ] No broken image links
- [ ] Git history preserved (git mv)

## Conflict Prevention
- Complete this phase BEFORE starting content phases
- Use git mv to preserve history
- Run link checker after completion
- Phase 8 will handle VitePress config updates for new paths

## Verification
```bash
# Check for old paths still referenced
grep -r "assets/images/xview1" research/
grep -r "assets/images/cnn-basics" research/
# Should return 0 results after update
```
