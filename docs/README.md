# Tài liệu Dự án

Folder này chứa tài liệu tổng quan và hướng dẫn cho dự án nghiên cứu.

## Nội dung

| File | Mô tả |
|------|-------|
| [project-overview-pdr.md](./project-overview-pdr.md) | Tổng quan dự án, PDR, quy trình viết tài liệu |
| [codebase-summary.md](./codebase-summary.md) | Tóm tắt cấu trúc nội dung nghiên cứu |
| [docs-standards.md](./docs-standards.md) | Tiêu chuẩn văn phong học thuật, format tài liệu |
| [system-architecture.md](./system-architecture.md) | Kiến trúc hệ thống build (VitePress, Pandoc) |

## Cấu trúc Dự án

```
sen_doc/
├── docs/               ← Tài liệu dự án (bạn đang ở đây)
│   ├── project-overview-pdr.md
│   ├── codebase-summary.md
│   ├── docs-standards.md
│   └── system-architecture.md
├── research/           ← Nội dung nghiên cứu (7 chương, 39 files)
│   ├── chuong-01-gioi-thieu/
│   ├── chuong-02-co-so-ly-thuyet/
│   ├── chuong-03-phat-hien-tau-bien/
│   ├── chuong-04-phat-hien-dau-loang/
│   ├── chuong-05-torchgeo/
│   ├── chuong-06-xview-challenges/
│   └── chuong-07-ket-luan/
├── scripts/            ← Build scripts
├── output/             ← DOCX output
└── plans/              ← Kế hoạch phát triển
```

## Build Commands

```bash
# Web (VitePress)
npm run docs:dev      # Dev server
npm run docs:build    # Build static site

# DOCX (Pandoc)
npm run build:docx    # Build thesis-remote-sensing.docx
```
