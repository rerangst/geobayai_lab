# Phase 4: Cập nhật Section Headers

**Parent:** [plan.md](./plan.md) | **Status:** Pending

## Overview
Cập nhật numbering trong tất cả 39 files theo hệ thống 1. / 1.1. / 1.1.1.

## Quy tắc đánh số

| Level | Format | Ví dụ |
|-------|--------|-------|
| Chương | **Chương X:** | Chương 1: Giới thiệu |
| Mục | **X.Y.** | 1.1. Tổng quan về CNN |
| Tiểu mục | **X.Y.Z.** | 1.1.1. Lịch sử phát triển |

## Mapping Chương → Số

| Folder | Chương số |
|--------|-----------|
| chuong-01-gioi-thieu | 1 |
| chuong-02-co-so-ly-thuyet | 2 |
| chuong-03-phat-hien-tau-bien | 3 |
| chuong-04-phat-hien-dau-loang | 4 |
| chuong-05-torchgeo | 5 |
| chuong-06-xview-challenges | 6 |
| chuong-07-ket-luan | 7 |

## Header Update Rules

1. **Title (H1):** Thêm "Chương X: " prefix
   - `# Giới thiệu CNN` → `# Chương 1: Giới thiệu CNN`

2. **Section (H2):** Đổi thành X.Y.
   - `## 1.1. Tổng quan` → giữ nguyên nếu đã đúng format
   - `## Overview` → `## 1.1. Tổng quan`

3. **Subsection (H3):** Đổi thành X.Y.Z.
   - `### Details` → `### 1.1.1. Chi tiết`

## Implementation Strategy

Sử dụng script để:
1. Đọc file
2. Parse H1, H2, H3 headers
3. Replace với numbering chuẩn
4. Ghi lại file

```bash
# Pseudo-script
for file in docs/chuong-*/muc-*/*.md; do
  chapter=$(extract_chapter_num "$file")
  section=$(extract_section_num "$file")
  update_headers "$file" "$chapter" "$section"
done
```

## Todo
- [ ] Tạo script update-headers.sh
- [ ] Test với 1 file mẫu
- [ ] Chạy cho tất cả 39 files
- [ ] Review kết quả

## Success Criteria
- Tất cả H1 có format "Chương X: Title"
- Tất cả H2 có format "X.Y. Title"
- Tất cả H3 có format "X.Y.Z. Title"
