# Phase 8: Navigation Config Update

## Context
- **Parent Plan:** [plan.md](./plan.md)
- **Dependencies:** Phases 1-7 complete
- **Blockers:** Must wait for all file moves finalized

## Parallelization
- **Concurrent with:** None
- **Blocks:** None (final phase)

## Overview
Update VitePress config for new chapter structure. Ensure asset paths work. Validate all links.

## File Ownership (Exclusive)

| File | Action |
|------|--------|
| `research/.vitepress/config.mjs` | Complete rewrite of sidebar |
| `research/index.md` | Update chapter links |

Total: 2 files

## Implementation Steps

### 1. Update config.mjs Sidebar

New sidebar structure:
```javascript
sidebar: [
  {
    text: 'Chuong 1: Gioi thieu',
    items: [...]
  },
  {
    text: 'Chuong 2: Co so ly thuyet',
    items: [...] // 6 files
  },
  {
    text: 'Chuong 3: Kien truc Model (TorchGeo)',
    items: [...] // 5 files
  },
  {
    text: 'Chuong 4: xView Challenges',
    items: [...] // 19 files with nested structure
  },
  {
    text: 'Chuong 5: Phat hien tau bien',
    items: [...] // 4 files
  },
  {
    text: 'Chuong 6: Phat hien dau loang',
    items: [...] // 4 files
  },
  {
    text: 'Chuong 7: Ket luan',
    items: [...] // 1 file
  }
]
```

### 2. Update Nav Links
```javascript
nav: [
  { text: 'Trang chu', link: '/' },
  { text: 'Gioi thieu', link: '/chuong-01-gioi-thieu/...' },
  { text: 'TorchGeo', link: '/chuong-03-kien-truc-model/...' },
  { text: 'xView', link: '/chuong-04-xview-challenges/...' },
]
```

### 3. Update index.md
- Update chapter listing
- Fix any broken quick links

### 4. Verify Asset Paths
Confirm all renamed asset folders are accessible:
```bash
# Check image paths in built HTML
grep -r "chuong-02-cnn" research/.vitepress/dist/
grep -r "chuong-04-xview" research/.vitepress/dist/
```

### 5. Validate All Links
Run VitePress build:
```bash
cd research && npm run build
```
Fix any dead link errors

### 5. Test Local Preview
```bash
npm run dev
```
Manually verify navigation

## Success Criteria
- [ ] VitePress builds without errors
- [ ] No dead link warnings
- [ ] Sidebar shows correct 7-chapter structure
- [ ] Nav links work
- [ ] Local preview navigable

## Conflict Prevention
- Only modify .vitepress/config.mjs and index.md
- Do NOT modify content files
- All path updates based on actual file locations from Phases 1-7

## Rollback Plan
If build fails:
1. Check git diff for phase 1-7 file moves
2. Verify all paths in config match actual files
3. Fix paths iteratively
