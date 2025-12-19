# Scripts and Templates Analysis Report

## Overview
Documentation project using **VitePress** (web) + **Pandoc** (DOCX generation) with automated CI/CD via GitHub Actions.

---

## Scripts Summary

### 1. `scripts/create-template.py` (Python)
**Purpose**: Generate Vietnamese thesis formatting template for DOCX output

**Key Features**:
- Creates `templates/reference.docx` with Vietnamese thesis standards
- Requires: `python-docx` library
- Configures:
  - Font: Times New Roman (body 13pt, headings 14-18pt)
  - Margins: Left 3.5cm, Right 2cm, Top/Bottom 3cm (Vietnamese standard)
  - Page size: A4 (210 × 297 mm)
  - Line spacing: 1.5 lines (thesis requirement)
  - Heading styles: 4 levels (H1-H4)
  - TOC support (styles 1-3)

**Usage**: `python scripts/create-template.py`

**Output**: `templates/reference.docx` (binary Word document)

---

### 2. `scripts/build-unified-docx.sh` (Bash)
**Purpose**: Convert all 39 markdown files into single unified DOCX document

**Key Features**:
- Requires: Pandoc (checked at runtime)
- Processes 39 markdown files across 7 chapters:
  - Chapter 1: Introduction (CNN/Deep Learning)
  - Chapter 2: Theoretical Foundations (CNN Architecture, Image Processing)
  - Chapter 3: Ship Detection (Maritime)
  - Chapter 4: Oil Spill Detection (Environmental)
  - Chapter 5: TorchGeo (Classification, Segmentation, Change Detection)
  - Chapter 6: xView Challenges (xView1, xView2, xView3 + 5 winners each)
  - Chapter 7: Conclusions

**Conversion Options**:
- `--toc` + `--toc-depth=3`: Generates table of contents
- `--metadata`: Sets title, author, date
- `--standalone`: Creates complete document

**Output**: `output/thesis-remote-sensing.docx`

**Status Colors**: Uses ANSI colors for CLI feedback (GREEN, YELLOW, RED)

---

### 3. `scripts/update-headers.sh` (Bash)
**Purpose**: Normalize document headers to Vietnamese thesis numbering format

**Header Format**:
- H1: `# Chương X: Title` (Chapter X)
- H2: `X.Y. Title` (Section numbering)
- H3: `X.Y.Z. Title` (Subsection numbering)

**Processing**:
- Updates all 7 chapters (`chuong-01` through `chuong-07`)
- Skips files that already have "Chương" prefix (idempotent)
- Uses `sed` for in-place replacement of first H1 only
- H2 numbering marked as complex (case-by-case handling noted)

**Current Limitations**:
- Only implements H1 updates fully
- H2/H3 numbering requires manual or enhanced logic

---

## Build Generation Tools

### Primary Tools

| Tool | Purpose | Version |
|------|---------|---------|
| **VitePress** | Web documentation generator | ^1.5.0 |
| **Pandoc** | Markdown to DOCX converter | (system) |
| **Mermaid** | Diagram rendering | ^11.0.0 |
| **VitePress Mermaid Plugin** | Mermaid integration | ^2.0.17 |
| **python-docx** | DOCX template creation | (pip install) |

### Optional Tools (NOT currently used)
- `mermaid-filter`: Pandoc filter for diagrams
- `@mermaid-js/mermaid-cli`: CLI diagram renderer

---

## Templates

### Template File: `templates/reference.docx`
**Type**: Binary Word document (generated, not tracked as source)

**Purpose**: Reference template for Pandoc DOCX conversion

**Specifications**:
- Vietnamese thesis format (standard Vietnamese university requirement)
- Font: Times New Roman 13pt body text
- Heading hierarchy: 18pt (H1) → 16pt (H2) → 14pt (H3) → 13pt (H4)
- Margins: 3.5cm left, 2cm right, 3cm top/bottom (asymmetric for binding)
- Line spacing: 1.5 (standard for academic work)
- Includes sample Vietnamese content for verification

**Regeneration**:
- Deleted and regenerated via `scripts/create-template.py`
- Ensures consistent formatting across builds

---

## Document Structure

### Markdown Organization
```
docs/
├── .vitepress/          # VitePress config
├── xview-challenges/    # Main content (web structure)
├── chuong-*/            # Thesis structure (DOCX chapters)
│   └── muc-*/           # Sections (Vietnamese: "mục")
│       └── *.md         # Individual articles
└── index.md             # Homepage
```

### File Naming Conventions
- **Web**: Short English names (`dataset-xview1-detection.md`)
- **Thesis**: Vietnamese chapter/section names (`chuong-06-xview-challenges/muc-01-xview1-object-detection/`)

---

## Automation Workflows

### 1. Local Development
```bash
npm run docs:dev       # VitePress dev server (hot reload)
npm run docs:build     # Build static site for GitHub Pages
npm run docs:preview   # Preview production build
npm run build:docx     # Generate DOCX (requires pandoc)
```

### 2. CI/CD Pipeline (`.github/workflows/deploy.yml`)

**Trigger**: Push to `main` branch (docs/, templates/, package.json, workflow changes)

**Job 1: Deploy Web**
1. Checkout code
2. Setup Node.js 20
3. npm install dependencies
4. `npm run docs:build` → Generate static HTML
5. Upload artifact to GitHub Pages
6. Deploy to GitHub Pages (automatic)

**Job 2: Build DOCX**
1. Checkout code
2. Install Pandoc (apt-get)
3. Run Pandoc with:
   - Input: 8 markdown files (README + xView1/2/3 + winners)
   - Reference: `templates/reference.docx`
   - Options: TOC, metadata, standalone
4. Output: `output/xview-research.docx`
5. Upload as 30-day artifact
6. Create GitHub release if tagged (automatic)

**Parallelization**: Both jobs run simultaneously (independent)

---

## Automation Features

### Error Handling
- Bash: `set -e` (fail on error)
- Python: Try/except for missing dependencies with helpful error messages
- CI/CD: Conditional steps (e.g., release only on tags)

### Cache/Performance
- GitHub Actions: npm cache enabled
- Concurrent: Pages group set to single concurrent run

### Output Artifacts
- **Web**: Deployed to GitHub Pages (automatic)
- **DOCX**: 30-day artifact retention, release on git tags
- File size reporting: `du -h` for DOCX build output

---

## Key Integration Points

1. **Template Reference**: Pandoc uses `templates/reference.docx` as style reference
2. **Markdown Processing**: Both VitePress and Pandoc consume same markdown (dual output)
3. **Mermaid Integration**: VitePress plugin handles diagrams; Pandoc shows as code blocks
4. **Vietnamese Support**: Font configuration in template ensures proper Vietnamese character rendering
5. **CI/CD Artifacts**: Artifacts available for download; releases trigger on version tags

---

## Summary

| Aspect | Technology |
|--------|-----------|
| **Web Output** | VitePress ^1.5.0 → Static HTML → GitHub Pages |
| **DOCX Output** | Pandoc → Vietnamese thesis format template |
| **Diagrams** | Mermaid ^11.0.0 (web + code blocks in DOCX) |
| **CI/CD** | GitHub Actions (2 parallel jobs) |
| **Automation** | Bash scripts + Python utilities |
| **Format** | Vietnamese academic thesis standards |

---

## Unresolved Questions

- H2/H3 header numbering in `update-headers.sh` marked as "complex" - implementation strategy unclear
- Whether mermaid-filter or mermaid-cli will be integrated for diagram support in DOCX
- Artifact retention and release strategy for long-term documentation versions
