# Phase 5: Cập nhật README và Build Scripts

**Parent:** [plan.md](./plan.md) | **Status:** Pending

## Overview
Cập nhật README.md với mục lục mới và sửa build scripts.

## Files cần cập nhật

### 1. docs/README.md
Viết lại hoàn toàn với mục lục theo cấu trúc mới:

```markdown
# Nghiên cứu Ứng dụng Deep Learning trong Viễn thám

## Mục lục

### Chương 1: Giới thiệu
- [1.1. Tổng quan](./chuong-01-gioi-thieu/muc-01-tong-quan/01-gioi-thieu-cnn-deep-learning.md)

### Chương 2: Cơ sở lý thuyết
- [2.1. Kiến trúc CNN](./chuong-02-co-so-ly-thuyet/muc-01-kien-truc-cnn/)
  - [2.1.1. Kiến trúc cơ bản](...)
  - [2.1.2. Backbone Networks](...)
- [2.2. Phương pháp xử lý ảnh](./chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/)
  - [2.2.1. Phân loại ảnh](...)
  - [2.2.2. Phát hiện đối tượng](...)
  - [2.2.3. Phân đoạn ngữ nghĩa](...)
  - [2.2.4. Instance Segmentation](...)

### Chương 3: Phát hiện tàu biển
...

### Chương 4: Phát hiện dầu loang
...

### Chương 5: TorchGeo
...

### Chương 6: xView Challenges
...

### Chương 7: Kết luận
...
```

### 2. scripts/build-unified-docx.sh
Cập nhật file paths theo cấu trúc mới:

```bash
pandoc \
    docs/README.md \
    docs/chuong-01-gioi-thieu/muc-01-tong-quan/01-gioi-thieu-cnn-deep-learning.md \
    docs/chuong-02-co-so-ly-thuyet/muc-01-kien-truc-cnn/01-kien-truc-co-ban.md \
    docs/chuong-02-co-so-ly-thuyet/muc-01-kien-truc-cnn/02-backbone-networks.md \
    # ... all 39 files in order
```

### 3. Xóa scripts cũ (optional)
- scripts/build-cnn-docx.sh (deprecated)
- scripts/build-docx.sh (deprecated)

## Todo
- [ ] Viết docs/README.md mới
- [ ] Cập nhật scripts/build-unified-docx.sh
- [ ] Test build DOCX
- [ ] Xóa scripts cũ nếu không cần

## Success Criteria
- README.md hiển thị đúng mục lục
- Build DOCX thành công
- Tất cả links hoạt động
