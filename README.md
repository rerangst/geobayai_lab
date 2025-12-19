# xView Challenge Research Documentation

Comprehensive research documentation on the xView Challenge Series - satellite imagery computer vision competitions organized by the Defense Innovation Unit (DIU).

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run docs:dev

# Build for production
npm run docs:build

# Preview production build
npm run docs:preview
```

## Documentation

Visit the live documentation at: https://tchatb.github.io/sen_doc/

### Contents

| Challenge | Year | Focus | Documents |
|-----------|------|-------|-----------|
| xView1 | 2018 | Object Detection | 1 dataset + 5 winners |
| xView2 | 2019 | Building Damage | 1 dataset + 5 winners |
| xView3 | 2021-22 | Maritime Detection | 1 dataset + 5 winners |

## Build DOCX

Generate a Word document with Vietnamese thesis formatting:

```bash
# Install dependencies
pip install python-docx

# Create reference template (if not exists)
python scripts/create-template.py

# Build DOCX (requires pandoc + mermaid-filter)
npm run build:docx
```

### DOCX Requirements

- [Pandoc](https://pandoc.org/installing.html)
- [mermaid-filter](https://github.com/raghur/mermaid-filter): `npm install -g mermaid-filter`
- [@mermaid-js/mermaid-cli](https://github.com/mermaid-js/mermaid-cli): `npm install -g @mermaid-js/mermaid-cli`

## Project Structure

```
sen_doc/
├── docs/                    # Documentation source
│   ├── .vitepress/         # VitePress config
│   ├── xview-challenges/   # Challenge documentation
│   │   ├── xview1/        # Object detection
│   │   ├── xview2/        # Building damage
│   │   └── xview3/        # Maritime detection
│   └── index.md           # Home page
├── templates/
│   └── reference.docx     # Vietnamese thesis template
├── scripts/
│   ├── build-docx.sh     # DOCX build script
│   └── create-template.py # Template generator
├── .github/workflows/
│   └── deploy.yml        # CI/CD workflow
└── package.json
```

## CI/CD

On push to `main`:
1. **Web**: VitePress builds and deploys to GitHub Pages
2. **DOCX**: Pandoc generates Word document (available as artifact)

## Vietnamese Thesis Format

The DOCX output follows Vietnamese thesis standards:
- Font: Times New Roman 13pt
- Line spacing: 1.5
- Margins: Left 3.5cm, Right 2cm, Top/Bottom 3cm
- Page size: A4

## License

MIT
