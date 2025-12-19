# Ứng Dụng Deep Learning trong Viễn Thám

## Project Overview

This project contains comprehensive research documentation on applying Deep Learning and Convolutional Neural Networks (CNN) to remote sensing satellite imagery analysis. The documentation is written primarily in Vietnamese with technical terms preserved in English, covering both theoretical foundations and practical applications through three major international competitions.

**Type:** Scientific research documentation (39 Markdown files, 7 chapters)
**Language:** Vietnamese + English (technical terms)
**Format:** Web (VitePress) + DOCX (Pandoc + Vietnamese thesis formatting)
**License:** MIT

## Quick Start

### Web Development
```bash
npm install
npm run docs:dev        # Development server at http://localhost:5173
npm run docs:build      # Production build
npm run docs:preview    # Preview built site
```

### DOCX Generation
```bash
# Prerequisites
pip install python-docx
npm install -g mermaid-filter @mermaid-js/mermaid-cli

# Build
npm run build:docx
```

## Documentation Structure

**7 Chapters covering:**

| Chapter | Content | Files |
|---------|---------|-------|
| 1. Giới thiệu | CNN & Deep Learning fundamentals | 1 |
| 2. Cơ sở lý thuyết | CNN architectures, image processing | 6 |
| 3. Phát hiện tàu biển | Ship detection case study | 4 |
| 4. Phát hiện dầu loang | Oil spill detection case study | 4 |
| 5. TorchGeo | Remote sensing library overview | 5 |
| 6. xView Challenges | 3 competitions, 15 winning solutions | 18 |
| 7. Kết luận | Conclusions and future directions | 1 |

**Live:** https://tchatb.github.io/sen_doc/

## Key Resources

### Documentation Files
- **project-overview-pdr.md**: Research objectives, document writing standards, image/diagram guidelines, QA procedures
- **codebase-summary.md**: Complete content organization, chapter-by-chapter breakdown, statistics
- **docs-standards.md**: Document structure standards, heading hierarchy, formatting rules, quality checklist
- **system-architecture.md**: Build pipeline, deployment architecture, CI/CD workflows

### Build Tools
- **VitePress 1.5.0:** Static site generation from Markdown
- **Pandoc:** Convert Markdown to DOCX with Vietnamese formatting
- **Mermaid:** Diagram generation (flowcharts, architecture diagrams)
- **GitHub Pages:** Free hosting for web documentation
- **GitHub Actions:** Automated build and deployment

## Writing Standards

### Language
- **Primary:** Academic Vietnamese (học thuật)
- **Technical terms:** Keep in English (CNN, ResNet, SAR, etc.)
- **Consistency:** Unified terminology across all chapters
- **No duplication:** Cross-reference instead of repeating content

### Content Organization
- **H1:** Chapter title only (one per file)
- **H2:** Main sections (numbered: 1.1, 1.2, etc.)
- **H3:** Subsections (numbered: 1.1.1, 1.1.2, etc.)
- **Cross-references:** Link to other chapters using relative paths
- **Bibliography:** IEEE/APA format, complete author/year/title info

### Images & Diagrams
- **Standard size:** 800px width (responsive layout)
- **Format:** PNG for diagrams, JPEG for satellite imagery
- **Diagrams:** Mermaid-based (flowchart, architecture, sequence)
- **AI-generated images:** For complex 3D architectures or visualizations
- **Captions:** Numbered (Hình 2.1, Hình 2.2, etc.) with description

## Document Generation Pipeline

```
Markdown Files → VitePress → Static HTML Site → GitHub Pages
    ↓
Markdown → Mermaid Filter → Pandoc → DOCX with Template
```

**Vietnamese Thesis Format (DOCX):**
- Font: Times New Roman 13pt
- Line spacing: 1.5
- Margins: Left 3.5cm, Right 2cm, Top/Bottom 3cm
- Page size: A4

## CI/CD Workflow

**Triggers:** Push to `main` branch

**Automated Steps:**
1. Install dependencies
2. Build VitePress static site
3. Deploy to GitHub Pages
4. Generate DOCX artifact (optional)
5. Upload release artifacts

**Manual Actions:**
- Trigger DOCX build on demand
- Download DOCX from workflow artifacts
- Create releases with version tags

## Data Organization

```
docs/
├── chuong-01-gioi-thieu/           # Chapter 1 (1 file)
├── chuong-02-co-so-ly-thuyet/      # Chapter 2 (6 files)
├── chuong-03-phat-hien-tau-bien/   # Chapter 3 (4 files)
├── chuong-04-phat-hien-dau-loang/  # Chapter 4 (4 files)
├── chuong-05-torchgeo/             # Chapter 5 (5 files)
├── chuong-06-xview-challenges/     # Chapter 6 (18 files)
├── chuong-07-ket-luan/             # Chapter 7 (1 file)
├── assets/images/                  # Images by challenge
├── .vitepress/config.mjs           # VitePress configuration
└── index.md                        # Homepage
```

## Content Verification

Quality assurance includes:
- **Grammar & spelling:** Vietnamese academic style
- **Cross-references:** Verify all internal links
- **Terminology:** Consistent glossary usage
- **Duplicates:** Identify and consolidate redundant content
- **Images:** All images properly sized and credited
- **Links:** Automated checking of external/internal links

## Future Enhancements

- Additional remote sensing case studies
- Code examples and Jupyter notebooks
- Interactive visualizations
- Expanded challenge analysis
- Video tutorials (optional)

## Contributing

Follow the guidelines in `docs/docs-standards.md` for:
- Document structure and formatting
- Writing conventions
- Image standards
- Quality checklist

## License

MIT
