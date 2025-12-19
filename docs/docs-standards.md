# Document Structure Standards - Tiêu Chuẩn Cấu Trúc Tài Liệu

## 1. Phân Cấp Heading (Heading Hierarchy)

### 1.1. Quy Tắc Cơ Bản
```
# Heading 1 (H1) - Tiêu đề Chương
## Heading 2 (H2) - Tiêu đề Mục chính
### Heading 3 (H3) - Tiêu đề Mục con
#### Heading 4 (H4) - Tiêu đề Sub-section (tối đa)
```

### 1.2. Quy Định Cụ Thể

**H1 - Chương (Chapter):**
- Sử dụng format: `# Chương N: Tên chương`
- Ví dụ: `# Chương 1: Giới thiệu`
- Chỉ 1 H1 mỗi file Markdown

**H2 - Mục chính (Main Section):**
- Sử dụng format: `## N.M. Tên mục` (N=chapter, M=section number)
- Ví dụ: `## 1.1. Bối cảnh`, `## 2.3. Kiến trúc CNN`
- Đối với mục không có số: `## Giới thiệu` hoặc `## Kết luận`
- Tối đa 5-7 H2 mỗi chương

**H3 - Mục con (Subsection):**
- Sử dụng format: `### N.M.K. Tên mục con`
- Ví dụ: `### 1.1.1. Định nghĩa CNN`
- Tương tự với H2, đánh số hoặc không

**H4 - Sub-section (hiếm khi sử dụng):**
- Chỉ sử dụng nếu cần độ chi tiết rất cao
- Format: `#### Tên detail` (không cần đánh số)

### 1.3. Ví Dụ Cấu Trúc Một Chương

```markdown
# Chương 2: Cơ Sở Lý Thuyết

## 2.1. Kiến Trúc CNN

### 2.1.1. Thành Phần Cơ Bản
Nội dung...

### 2.1.2. Hàm Activation
Nội dung...

## 2.2. Phương Pháp Xử Lý Ảnh

### 2.2.1. Phân Loại Ảnh
Nội dung...

### 2.2.2. Phát Hiện Đối Tượng
Nội dung...
```

---

## 2. Cấu Trúc Nội Dung (Content Structure)

### 2.1. Phần Mở Đầu (Opening Paragraph)
- **Yêu cầu:** Mỗi mục H2 nên bắt đầu với 1-2 đoạn giới thiệu
- **Nội dung:** Giải thích mục đích của mục, vì sao nó quan trọng
- **Độ dài:** 2-4 câu, ~100-150 từ
- **Ví dụ:**
  ```markdown
  ## 2.3. Phát Hiện Đối Tượng

  Phát hiện đối tượng là một trong những bài toán quan trọng nhất
  trong xử lý ảnh viễn thám, với ứng dụng rộng rãi từ giám sát giao
  thông, quản lý đô thị, đến bảo vệ an ninh hàng hải. Trong phần này,
  chúng ta sẽ tìm hiểu các kiến trúc mạng nơ-ron chính được sử dụng
  cho bài toán này.
  ```

### 2.2. Khoảng Trống Giữa Đoạn
- **Quy tắc:** 1 dòng trống giữa các đoạn
- **Không được:** Không để 2-3 dòng trống liên tiếp
- **Paragraph dài:** Nếu đoạn dài quá 5 câu, chia thành 2 đoạn

### 2.3. Liệt Kê (Lists)

**Bullet points (unordered list):**
```markdown
- Điểm thứ nhất
- Điểm thứ hai
- Điểm thứ ba
```

**Numbered list (ordered list):**
```markdown
1. Bước đầu tiên
2. Bước thứ hai
3. Bước thứ ba
```

**Quy tắc:**
- Sử dụng khi liệt kê 3+ items
- Cùng cấp độ chi tiết trong 1 list
- Nếu sub-item, indent thêm 2 spaces:
  ```markdown
  1. Item chính
     - Sub-item
     - Sub-item
  2. Item chính khác
  ```

### 2.4. Mô Tả Kỹ Thuật (Technical Descriptions)

**Công thức toán học:**
- Sử dụng LaTeX inline: `$formula$` cho công thức nhỏ
- Sử dụng LaTeX block cho công thức lớn:
  ```
  $$
  equation
  $$
  ```
- Ví dụ:
  ```markdown
  Hàm ReLU được định nghĩa là: $f(x) = \max(0, x)$
  ```

**Code snippets:**
- Sử dụng code block với ngôn ngữ: `` ```python ` ``
- Highlight important lines
- Giải thích output
- Ví dụ:
  ```markdown
  Để load một mô hình ResNet-50:

  ```python
  import torchvision.models as models
  model = models.resnet50(pretrained=True)
  ```
  ```

**Thuật ngữ kỹ thuật:**
- Lần đầu xuất hiện: Cung cấp định nghĩa hoặc công thức
- Lần tiếp theo: Sử dụng trực tiếp (có thể tham chiếu đến lần đầu)
- Ví dụ:
  ```markdown
  Phép tích chập (convolution) được định nghĩa toán học là: $y = w * x + b$
  ...
  Phép tích chập đó là thao tác cơ bản nhất...
  ```

---

## 3. Hình Ảnh và Diagram (Images and Diagrams)

### 3.1. Cách Nhúng Hình Ảnh

**Syntax:**
```markdown
![Alt text describing the image](path/to/image.png "Caption if needed")
```

**Quy tắc:**
- **Alt text:** Luôn cung cấp, mô tả hình ảnh bằng tiếng Việt
- **Path:** Relative path từ `docs/` (ví dụ: `../assets/images/xview1/image.png`)
- **Caption:** Tùy chọn, sử dụng title attribute nếu cần
- **Format:** PNG cho diagram, JPEG cho ảnh vệ tinh

**Ví dụ:**
```markdown
![Kiến trúc CNN với 3 lớp convolution](../assets/images/cnn-architecture.png)
```

### 3.2. Diagram Mermaid

**Syntax:**
```markdown
​```mermaid
graph LR
    A[Input] --> B[Processing]
    B --> C[Output]
​```
```

**Quy tắc:**
- Tên node bằng tiếng Việt (hoặc tiếng Anh nếu là thuật ngữ chính thức)
- Mô tả mối quan hệ rõ ràng bằng arrows
- Không quá 15 nodes (chia thành nhiều diagram nếu cần)

**Các loại diagram được hỗ trợ:**
- `graph LR` (left to right) - hướng ngang
- `graph TD` (top down) - hướng dọc
- `flowchart TD` - tương tự graph TD
- `sequenceDiagram` - biểu đồ trình tự
- Xem thêm: https://mermaid.js.org/

**Ví dụ:**
```markdown
​```mermaid
graph TD
    A["Ảnh vệ tinh"] --> B["Tiền xử lý"]
    B --> C["CNN Backbone"]
    C --> D["Feature Pyramid"]
    D --> E["Detection Head"]
    E --> F["Bounding Boxes"]
    style F fill:#90EE90
​```
```

### 3.3. Vị Trí Hình Ảnh

**Quy tắc:**
- Hình ảnh nên đặt gần nội dung liên quan
- Không để hình ảnh đơn lẻ (luôn có text giải thích trước)
- Tối thiểu 1-2 câu text trước hình ảnh
- Tối thiểu 1 câu text sau hình ảnh (để liên kết với hình)

**Ví dụ cấu trúc:**
```markdown
### 2.1.1. Kiến Trúc CNN

Một mạng Convolutional Neural Network điển hình bao gồm ba thành phần
chính: các lớp convolution để trích xuất đặc trưng, các lớp pooling
để giảm kích thước, và các lớp fully connected để phân loại. Sơ đồ
dưới đây minh họa kiến trúc cơ bản:

![Sơ đồ kiến trúc CNN cơ bản](../assets/images/cnn-basic.png)

Như hình vẽ trên, dữ liệu đầu vào được xử lý theo các lớp tuần tự,
mỗi lớp chiết xuất các đặc trưng ở mức độ trừu tượng khác nhau.
```

### 3.4. Caption và Numbering

**Quy tắc:**
- Hình ảnh quan trọng nên có numbering: "Hình 2.1: Tiêu đề hình"
- Tham chiếu trong text: "Hình 2.1 cho thấy..."
- Format trong markdown:

```markdown
**Hình 2.1:** Kiến trúc CNN cơ bản
![CNN architecture](../assets/images/cnn-basic.png)

Hình 2.1 trên minh họa...
```

---

## 4. Bảng (Tables)

### 4.1. Syntax Markdown Table

```markdown
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Row 1, Col 1 | Row 1, Col 2 | Row 1, Col 3 |
| Row 2, Col 1 | Row 2, Col 2 | Row 2, Col 3 |
```

### 4.2. Quy Tắc Bảng

- **Alignment:** Canh lề trái (mặc định), canh phải, canh giữa
  ```markdown
  | Trái | Giữa | Phải |
  |:-----|:----:|-----:|
  ```
- **Độ rộng:** Tối thiểu 3 cột, tối đa 6-7 cột (nếu quá, chia thành 2 bảng)
- **Heading:** Luôn có heading row
- **Số hàng:** Tối đa 10-15 hàng (nếu quá, có thể tách hoặc tham chiếu dataset)
- **Text dài:** Nếu cell content quá dài (>50 ký tự), sử dụng dòng mới hoặc tối giản

### 4.3. Ví Dụ

```markdown
| Model | Year | Accuracy | Parameters |
|-------|------|----------|-----------|
| AlexNet | 2012 | 63.3% | 60M |
| VGG-16 | 2014 | 71.3% | 138M |
| ResNet-50 | 2015 | 76.1% | 25.5M |
| EfficientNet-B0 | 2019 | 77.1% | 5.3M |

**Bảng 2.1:** So sánh các mô hình CNN trên ImageNet
```

---

## 5. Tham Chiếu Nội Bộ (Internal References)

### 5.1. Cross-Chapter References

**Format:**
```markdown
Như đã trình bày ở [Chương 2, Mục 2.1.1](../chuong-02-co-so-ly-thuyet/muc-01-kien-truc-cnn/01-kien-truc-co-ban.md#211-định-nghĩa-cnn), ...
```

**Quy tắc:**
- Sử dụng markdown links với đường dẫn tệp đầy đủ
- Sử dụng anchor (#heading) để liên kết đến heading cụ thể
- Anchor format: `#heading` (lowercase, dấu gạch ngang thay khoảng trắng)

### 5.2. Same-File References

**Format:**
```markdown
Xem [Mục 2.3](#23-phát-hiện-đối-tượng) để biết thêm chi tiết.
```

**Quy tắc:**
- Sử dụng `#heading` anchor
- Không cần đường dẫn tệp (trong cùng file)

### 5.3. Tài Liệu Tham Khảo (Bibliography)

**Format (trong nội dung):**
```markdown
Theo nghiên cứu của [LeCun et al., 1998], CNN được sử dụng lần đầu tiên...
```

**Format (danh sách tài liệu):**
Ở cuối chương hoặc file, tạo section:
```markdown
## Tài Liệu Tham Khảo

[1] LeCun, Y., Haffner, P., Bottou, L., & Bengio, Y. (1998). "Object recognition with gradient-based learning." In Shape, contour and grouping in computer vision (pp. 319-345). Springer, Berlin, Heidelberg.

[2] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet classification with deep convolutional neural networks." Advances in neural information processing systems, 25.
```

**Quy tắc:**
- Sử dụng format IEEE hoặc APA
- Liệt kê theo thứ tự xuất hiện hoặc theo chữ cái
- Luôn cung cấp đầy đủ: Tác giả, năm, tiêu đề, source

---

## 6. Khối Thông Tin Đặc Biệt (Special Content Blocks)

### 6.1. Blockquote

**Syntax:**
```markdown
> Trích dẫn từ nguồn khác
> Tiếp tục trích dẫn
```

**Sử dụng khi:**
- Trích dẫn trực tiếp từ paper hoặc sách
- Highlight một ý tưởng quan trọng
- Cảnh báo hoặc ghi chú từ tác giả

### 6.2. Highlight/Emphasis

**Syntax:**
```markdown
**Chữ đậm** cho khái niệm quan trọng
*Chữ nghiêng* cho thuật ngữ lần đầu hoặc emphasis
```

**Quy tắc:**
- Chữ đậm: Khái niệm, định nghĩa lần đầu
- Chữ nghiêng: Emphasis, tiêu đề hình/bảng
- Không lạm dụng (tối đa 3 lần per paragraph)

### 6.3. Note/Warning Boxes

Sử dụng Blockquote + emoji (nếu được phép):
```markdown
> **Lưu ý:** Điều này rất quan trọng vì...
```

Hoặc plain text bold:
```markdown
**Lưu ý:** Điều này rất quan trọng vì...
```

---

## 7. Line Breaks và Spacing

### 7.1. Quy Tắc Dòng Trống
- **Giữa paragraphs:** 1 dòng trống
- **Giữa heading và content:** 0 dòng trống (trực tiếp sau heading)
- **Giữa heading và heading:** 1 dòng trống (ngoài dòng trống của markdown)
- **Giữa list và text:** 1 dòng trống

### 7.2. Hard Line Break
```markdown
Text với
(2 spaces + enter) để break dòng nhưng không tạo paragraph mới
```

### 7.3. Horizontal Rule
```markdown
---
```
Sử dụng hiếm khi để tách các phần chính

---

## 8. Code Style Guide

### 8.1. Inline Code
```markdown
Lệnh `python script.py` dùng để chạy script
Hàm `conv2d()` thực hiện phép tích chập
```

**Quy tắc:**
- Backticks đơn cho code inline
- Không thêm space giữa backticks và code

### 8.2. Code Block

**Với syntax highlight:**
```markdown
​```python
import torch
model = torch.hub.load(...)
​```
```

**Các ngôn ngữ được hỗ trợ:**
- `python`, `javascript`, `bash`, `sql`, `yaml`, `json`, `xml`, `markdown`, etc.

**Quy tắc:**
- Chỉ định ngôn ngữ cho syntax highlight
- Comment giải thích các dòng quan trọng
- Giữ code snippet dưới 20 dòng (nếu dài, cắt hoặc cung cấp link)

### 8.3. Output từ Code

```markdown
**Output:**
```
tensor([[1., 2.], [3., 4.]])
```
```

---

## 9. Metadata và Front Matter (VitePress)

### 9.1. VitePress Front Matter

Ở đầu mỗi file Markdown:
```yaml
---
title: Tên trang
description: Mô tả ngắn
lang: vi-VN
---
```

**Quy tắc:**
- Luôn include `title` và `description`
- `lang: vi-VN` cho tất cả files (đã thiết lập global)
- Tùy chọn: `tags`, `keywords`, `author`

### 9.2. Ví Dụ

```markdown
---
title: Kiến trúc CNN Cơ Bản
description: Tìm hiểu chi tiết các thành phần và hoạt động của Convolutional Neural Network
---

# Kiến Trúc CNN Cơ Bản

Nội dung...
```

---

## 10. Kiểm Tra Chất Lượng (Quality Checklist)

### 10.1. Trước khi Commit

**Nội dung:**
- [ ] Không có typo hoặc lỗi grammar (sử dụng spell checker)
- [ ] Tất cả cross-references hợp lệ
- [ ] Không có nội dung lặp lại
- [ ] Tất cả công thức toán học được render chính xác
- [ ] Tất cả hình ảnh load đúng

**Cấu trúc:**
- [ ] Heading hierarchy đúng (H1 > H2 > H3)
- [ ] Không bỏ lỡn heading level (H1 > H3)
- [ ] Có H2 introductory paragraph trước sub-sections
- [ ] Lists được format đúng

**Hình ảnh:**
- [ ] Alt text có ý nghĩa (tiếng Việt)
- [ ] Hình ảnh có caption hoặc giải thích trong text
- [ ] Đường dẫn hình ảnh đúng (relative path)
- [ ] Kích thước hình ảnh phù hợp (800px hoặc tương đương)

**Tham chiếu:**
- [ ] Tất cả tài liệu tham khảo được liệt kê đầy đủ
- [ ] Format tài liệu tham khảo đồng nhất
- [ ] Không có broken links (nội bộ và ngoài)

### 10.2. Build & Preview

```bash
npm run docs:dev        # Local development server
npm run docs:build      # Build static site
npm run docs:preview    # Preview built site
```

**Kiểm tra:**
- [ ] Site loads without errors
- [ ] Navigation works (sidebar, navbar)
- [ ] Search function works
- [ ] Mermaid diagrams render correctly
- [ ] All images display properly
- [ ] Links navigate correctly

---

## 11. Ví Dụ Hoàn Chỉnh

### 11.1. Template cho Một Mục (Section)

```markdown
---
title: Tên Mục
description: Mô tả ngắn
---

# Chương N: Tên Chương

## N.M. Tên Mục Chính

Giới thiệu mục (1-2 đoạn, ~150 từ). Giải thích vì sao mục này quan
trọng và các nội dung chính sẽ được trình bày.

### N.M.1. Tên Mục Con

Mô tả khái niệm đầu tiên. Công thức: $y = f(x)$

Tiếp tục giải thích...

**Hình N.1:** Tiêu đề hình
![Alt text describing image](../assets/images/image.png)

Như hình trên, ...

### N.M.2. Tên Mục Con Khác

Một khái niệm khác:

- Điểm 1
- Điểm 2
- Điểm 3

**Bảng N.1:** Tiêu đề bảng
| Cột 1 | Cột 2 |
|-------|-------|
| Dữ liệu | Dữ liệu |

Code example:

​```python
# Code mẫu
print("Hello")
​```

## Tài Liệu Tham Khảo

[1] Author, Y. (Year). "Title". Journal Name.
```

### 11.2. Kiểm Tra Từ Checklist

- [x] Có H1 duy nhất (tên chương)
- [x] Có H2 introductory (N.M. + text)
- [x] H3 sub-sections được đánh số
- [x] Hình ảnh có caption + giải thích
- [x] Bảng có tiêu đề
- [x] Code block có language highlight
- [x] Tài liệu tham khảo ở cuối
- [x] Không có typo
- [x] Links hợp lệ

---

## 12. Chú Ý Đặc Biệt cho Viễn Thám

### 12.1. Thuật Ngữ Đặc Thù
- **GSD:** Ground Sample Distance
- **SAR:** Synthetic Aperture Radar
- **Sentinel-1/2:** Tên vệ tinh cụ thể (không dịch)
- **RGB/NIR/SWIR:** Tên kênh phổ (giữ nguyên tiếng Anh)

### 12.2. Viết Tắt
- Lần đầu: Viết đầy đủ + viết tắt trong ngoặc `Convolutional Neural Network (CNN)`
- Lần tiếp: Chỉ viết tắt `CNN`
- Duy trì danh sách viết tắt trong project overview

### 12.3. Hình Ảnh Vệ Tinh
- Luôn credit source (Sentinel-1, WorldView-3, etc.)
- Cung cấp metadata: độ phân giải, sensor, ngày capture
- Có legend cho các kênh phổ sử dụng

---

**Last Updated:** 2024-12-19
**Standard Version:** v1.0.0
