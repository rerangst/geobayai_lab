# Plan: Tổ chức lại tài liệu theo cấu trúc luận văn VN

**Date:** 2025-12-19 | **Status:** Ready for Implementation

## Mục tiêu
Chuyển đổi 39 files docs hiện tại → cấu trúc luận văn chuẩn VN với đánh số 1./1.1./1.1.1.

## Cấu trúc mục tiêu

```
docs/
├── README.md (mục lục)
├── chuong-01-gioi-thieu/
│   └── muc-01-tong-quan/
│       └── 01-gioi-thieu-de-tai.md
├── chuong-02-co-so-ly-thuyet/
│   ├── muc-01-kien-truc-cnn/
│   │   ├── 01-cnn-co-ban.md
│   │   └── 02-backbone-networks.md
│   └── muc-02-phuong-phap-xu-ly-anh/
│       ├── 01-phan-loai.md
│       ├── 02-phat-hien-doi-tuong.md
│       ├── 03-phan-doan-ngu-nghia.md
│       └── 04-instance-segmentation.md
├── chuong-03-phat-hien-tau-bien/
│   └── muc-01-ship-detection/
├── chuong-04-phat-hien-dau-loang/
│   └── muc-01-oil-spill-detection/
├── chuong-05-torchgeo/
│   └── muc-01-models/
├── chuong-06-xview-challenges/
│   ├── muc-01-xview1/
│   ├── muc-02-xview2/
│   └── muc-03-xview3/
└── chuong-07-ket-luan/
```

## Phases

| Phase | Description | Files |
|-------|-------------|-------|
| [Phase 1](./phase-01-create-structure.md) | Tạo cấu trúc thư mục mới | 20 folders |
| [Phase 2](./phase-02-migrate-cnn-docs.md) | Di chuyển + đổi tên CNN docs | 21 files |
| [Phase 3](./phase-03-migrate-xview-docs.md) | Di chuyển + đổi tên xView docs | 18 files |
| [Phase 4](./phase-04-update-headers.md) | Cập nhật section numbering | 39 files |
| [Phase 5](./phase-05-update-readme-build.md) | Cập nhật README + build scripts | 3 files |

## File Mapping (39 files)

**Chương 1-5:** CNN docs (21 files)
**Chương 6:** xView challenges (18 files)
**Chương 7:** Kết luận (1 file - từ CNN conclusion)

## Notes
- Giữ nguyên nội dung, chỉ đổi tên + cấu trúc
- Backup trước khi thực hiện
- Commit sau mỗi phase
