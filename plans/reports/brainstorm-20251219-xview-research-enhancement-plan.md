# Brainstorm: Kế hoạch Nâng cấp Tài liệu xView Research

**Ngày:** 2024-12-19
**Chủ đề:** Bổ sung chi tiết và hình ảnh minh họa cho tài liệu nghiên cứu xView

---

## 1. Phân tích Vấn đề Hiện tại

### 1.1 Thiếu sót Nội dung

| Vấn đề | Mô tả | Mức độ |
|--------|-------|--------|
| **Không có hình ảnh** | Toàn bộ 18 tài liệu chỉ có text + mermaid diagrams | Nghiêm trọng |
| **Nội dung quá ngắn** | Mỗi mục cần ít nhất 2000 từ, hiện tại chỉ vài trăm | Nghiêm trọng |
| **Quá nhiều code/bảng** | Thiếu lời văn giải thích, phân tích chuyên sâu | Cao |
| **Thiếu ví dụ trực quan** | Không có sample images từ dataset | Cao |
| **Bố cục không đồng nhất** | Mỗi file có cấu trúc khác nhau | Trung bình |

### 1.2 Yêu cầu Nội dung Mới

| Tiêu chí | Yêu cầu |
|----------|---------|
| **Độ dài mỗi section** | Tối thiểu 2000 từ |
| **Phong cách viết** | Lời văn mô tả chi tiết, hạn chế code snippets |
| **Ngôn ngữ** | Tiếng Việt, giữ nguyên thuật ngữ kỹ thuật tiếng Anh |
| **Thuật ngữ** | KHÔNG dịch: polygon, bounding box, feature map, backbone, encoder, decoder, loss function, epoch, batch size, learning rate, v.v. |
| **Hình ảnh** | Bắt buộc có hình minh họa từ dataset thực tế + sơ đồ architecture |

### 1.3 Thống kê Hiện tại vs Yêu cầu

| Loại tài liệu | Hiện tại | Yêu cầu mới | Cần bổ sung |
|---------------|----------|-------------|-------------|
| Dataset docs | 380-485 dòng (~1500 từ) | 6 sections × 2000 từ = 12000 từ | +10500 từ |
| Winner docs | 129-390 dòng (~800 từ) | 6 sections × 2000 từ = 12000 từ | +11200 từ |
| **Tổng 18 files** | ~20000 từ | ~216000 từ | **+196000 từ** |

---

## 2. Nguồn Hình ảnh Có sẵn

### 2.1 xView1 Dataset
- **Official:** https://xviewdataset.org/
- **Dataset Ninja:** https://datasetninja.com/xview (sample images, poster, heatmaps)
- **ArXiv Paper:** https://ar5iv.labs.arxiv.org/html/1802.07856 (figures)

### 2.2 xView2/xBD Dataset
- **Official:** https://xview2.org/
- **CVPR Paper PDF:** Có nhiều hình minh họa before/after damage
- **GitHub:** https://github.com/DIUx-xView/xView2_baseline

### 2.3 xView3-SAR Dataset
- **Official:** https://iuu.xview.us/
- **NeurIPS Paper:** https://arxiv.org/abs/2206.00897
- **Sample SAR images** từ Sentinel-1

---

## 3. Đề xuất Bố cục Thống nhất và Hướng dẫn Nội dung

### 3.1 Quy tắc Viết Chung

**Nguyên tắc thuật ngữ - KHÔNG DỊCH các từ sau:**
- Computer Vision: feature map, backbone, encoder, decoder, skip connection, attention mechanism
- Object Detection: bounding box, anchor box, IoU, NMS, RPN, FPN, ROI pooling
- Deep Learning: loss function, gradient, epoch, batch size, learning rate, optimizer, regularization
- Data: annotation, label, ground truth, dataset, training set, validation set, test set
- Image: pixel, resolution, GSD, RGB, multispectral, panchromatic
- Geometry: polygon, coordinate, centroid, overlap, tile, patch
- Architecture: UNet, ResNet, EfficientNet, transformer, head, neck
- Metrics: precision, recall, F1 score, mAP, accuracy

**Phong cách viết:**
- Mỗi section tối thiểu 2000 từ văn xuôi
- Giải thích chi tiết lý do, bối cảnh, ý nghĩa
- Phân tích so sánh với các phương pháp khác
- Hạn chế code blocks, ưu tiên mô tả bằng lời
- Sử dụng hình ảnh minh họa kèm caption chi tiết

---

### 3.2 Template cho Dataset Documentation (6 sections × 2000+ từ)

#### Section 1: Giới thiệu và Bối cảnh (~2500 từ)

**Nội dung bắt buộc:**
- Lịch sử ra đời của dataset, tổ chức phát triển, mục tiêu ban đầu
- Bối cảnh ngành công nghiệp và nhu cầu thực tiễn dẫn đến việc tạo dataset
- So sánh với các dataset tồn tại trước đó, giải thích tại sao cần dataset mới
- Mô tả chi tiết về cuộc thi liên quan (prize, participants, timeline)
- Tác động của dataset đến cộng đồng nghiên cứu và ứng dụng thực tế
- Các bài báo khoa học quan trọng sử dụng dataset này

**Hình ảnh yêu cầu:**
- Hình 1: Overview grid hiển thị đa dạng các loại ảnh trong dataset
- Hình 2: Timeline phát triển dataset và các milestone quan trọng

---

#### Section 2: Thông số Kỹ thuật và Thống kê (~2500 từ)

**Nội dung bắt buộc:**
- Mô tả chi tiết nguồn ảnh vệ tinh (tên vệ tinh, thông số kỹ thuật, độ phân giải)
- Giải thích ý nghĩa của Ground Sample Distance (GSD) và tầm quan trọng
- Phân tích phân bố địa lý của các ảnh (regions, countries, terrains)
- Thống kê chi tiết về số lượng object theo từng class
- Phân tích class imbalance và các thách thức liên quan
- Mô tả kích thước ảnh, định dạng file, metadata đi kèm
- So sánh quy mô với các dataset khác trong cùng lĩnh vực

**Hình ảnh yêu cầu:**
- Hình 3: Biểu đồ phân bố class (bar chart hoặc pie chart)
- Hình 4: Bản đồ địa lý hiển thị coverage area
- Hình 5: Histogram kích thước object

---

#### Section 3: Hệ thống Phân loại và Annotation (~2500 từ)

**Nội dung bắt buộc:**
- Mô tả chi tiết taxonomy của các class (parent classes, subclasses)
- Giải thích lý do phân loại theo cách này, dựa trên tiêu chuẩn nào
- Mô tả định dạng annotation (bounding box, polygon, segmentation mask)
- Phân tích ưu nhược điểm của định dạng annotation được chọn
- Giải thích các trường dữ liệu trong file annotation
- Mô tả hệ thống class ID và mapping với tên class
- Thảo luận về các trường hợp khó phân loại và cách xử lý

**Hình ảnh yêu cầu:**
- Hình 6: Sơ đồ cây phân loại class (hierarchy diagram)
- Hình 7: Ví dụ annotation overlay trên ảnh gốc
- Hình 8: So sánh các loại object khó phân biệt

---

#### Section 4: Quy trình Tạo Dataset (~2500 từ)

**Nội dung bắt buộc:**
- Mô tả chi tiết từng bước trong pipeline thu thập và xử lý ảnh
- Giải thích tiêu chí lựa chọn ảnh (quality, diversity, coverage)
- Mô tả quy trình annotation (tools, annotators, guidelines)
- Phân tích quy trình quality control (multi-stage review, inter-annotator agreement)
- Mô tả cách xử lý edge cases và ambiguous objects
- Giải thích về gold standard validation và expert review
- Thảo luận về thời gian và resources cần thiết để tạo dataset

**Hình ảnh yêu cầu:**
- Hình 9: Flowchart quy trình tạo dataset end-to-end
- Hình 10: Screenshot công cụ annotation (QGIS hoặc tương tự)
- Hình 11: Ví dụ về quality control (before/after correction)

---

#### Section 5: Thách thức Computer Vision (~2500 từ)

**Nội dung bắt buộc:**
- Phân tích chi tiết từng thách thức kỹ thuật đặc thù của dataset
- Thách thức 1: Multi-scale object detection (objects từ 3px đến 100+px)
- Thách thức 2: Class imbalance (tỷ lệ có thể lên đến 3000:1)
- Thách thức 3: Fine-grained classification (phân biệt các subclass tương tự)
- Thách thức 4: Varying imaging conditions (lighting, weather, season)
- Thách thức 5: Dense object clustering (objects chồng chéo, gần nhau)
- Đề xuất các hướng tiếp cận để giải quyết từng thách thức
- Tham khảo các phương pháp đã được chứng minh hiệu quả

**Hình ảnh yêu cầu:**
- Hình 12: Ví dụ về multi-scale challenge (cùng scene, nhiều kích thước object)
- Hình 13: Ví dụ về fine-grained classes khó phân biệt
- Hình 14: Ví dụ về varying imaging conditions

---

#### Section 6: Hướng dẫn Sử dụng và Tài nguyên (~2000 từ)

**Nội dung bắt buộc:**
- Hướng dẫn download và chuẩn bị data
- Mô tả cấu trúc thư mục sau khi giải nén
- Các preprocessing steps khuyến nghị (tiling, normalization, augmentation)
- Thảo luận về train/val/test splits và cross-validation strategies
- Lưu ý về licensing và ethical considerations
- Danh sách các baseline models và pretrained weights
- Các tutorial và code examples có sẵn
- Cách trích dẫn dataset trong publications

**Hình ảnh yêu cầu:**
- Hình 15: Sơ đồ cấu trúc thư mục
- Hình 16: Ví dụ về preprocessing pipeline

---

### 3.3 Template cho Winner Solution (6 sections × 2000+ từ)

#### Section 1: Giới thiệu Team và Thành tích (~2000 từ)

**Nội dung bắt buộc:**
- Thông tin chi tiết về team/tác giả (background, affiliation, expertise)
- Lịch sử tham gia các cuộc thi trước đó và thành tích
- Bối cảnh cuộc thi (số participants, difficulty level, prize)
- Điểm số chi tiết (public LB, private LB, individual metrics)
- So sánh với baseline và các giải pháp khác
- Mô tả tổng quan approach trong 2-3 đoạn văn
- Tóm tắt các đổi mới chính của giải pháp

**Hình ảnh yêu cầu:**
- Hình 1: Leaderboard screenshot hoặc ranking visualization
- Hình 2: High-level architecture overview

---

#### Section 2: Phân tích Vấn đề và Đổi mới (~2500 từ)

**Nội dung bắt buộc:**
- Phân tích sâu về các thách thức cụ thể của dataset/task
- Mô tả tại sao các phương pháp thông thường không đủ tốt
- Giải thích chi tiết ý tưởng đổi mới chính (key innovation)
- Cơ sở lý thuyết đằng sau đổi mới (mathematical intuition, prior work)
- So sánh với các phương pháp alternative và lý do chọn approach này
- Thảo luận về trade-offs và limitations của approach
- Các thí nghiệm ablation chứng minh hiệu quả của innovation

**Hình ảnh yêu cầu:**
- Hình 3: Diagram minh họa key innovation
- Hình 4: So sánh trước/sau khi áp dụng innovation
- Hình 5: Ablation study results visualization

---

#### Section 3: Kiến trúc Mô hình Chi tiết (~2500 từ)

**Nội dung bắt buộc:**
- Mô tả tổng quan kiến trúc (encoder-decoder, multi-stage, etc.)
- Giải thích chi tiết từng component (backbone, neck, head)
- Lý do chọn backbone cụ thể (ResNet, EfficientNet, etc.) và pretrained weights
- Mô tả các modifications so với architecture gốc
- Giải thích về feature pyramid và multi-scale processing
- Thảo luận về attention mechanisms nếu có
- Mô tả output heads và prediction format
- Số lượng parameters, FLOPs, inference time

**Hình ảnh yêu cầu:**
- Hình 6: Architecture diagram chi tiết với tensor dimensions
- Hình 7: Feature map visualization ở các stages
- Hình 8: Comparison với baseline architecture

---

#### Section 4: Chiến lược Training (~2500 từ)

**Nội dung bắt buộc:**
- Mô tả chi tiết loss functions và lý do chọn (Focal Loss, Dice Loss, etc.)
- Giải thích về class weighting và handling imbalance
- Mô tả data augmentation pipeline (geometric, color, cutout, mixup, etc.)
- Giải thích các augmentation đặc thù cho satellite imagery
- Optimizer selection và learning rate scheduling
- Batch size, number of epochs, early stopping criteria
- Multi-stage training strategy nếu có
- Hardware requirements và training time
- Techniques để đảm bảo reproducibility

**Hình ảnh yêu cầu:**
- Hình 9: Training loss/metric curves
- Hình 10: Augmentation examples visualization
- Hình 11: Learning rate schedule diagram

---

#### Section 5: Inference và Post-processing (~2000 từ)

**Nội dung bắt buộc:**
- Mô tả inference pipeline (tiling, overlap, stitching)
- Giải thích Test-Time Augmentation (TTA) strategy
- Mô tả ensemble methods (model diversity, averaging/voting)
- Post-processing techniques (NMS, morphological operations)
- Threshold selection và tuning strategies
- Inference speed và optimization techniques
- Memory management cho large images

**Hình ảnh yêu cầu:**
- Hình 12: Tiled inference visualization
- Hình 13: Ensemble architecture diagram
- Hình 14: Post-processing effects (before/after)

---

#### Section 6: Kết quả, Phân tích và Bài học (~2500 từ)

**Nội dung bắt buộc:**
- Detailed breakdown of performance metrics
- Analysis by class, by disaster type, by geographic region
- Error analysis: false positives, false negatives patterns
- Comparison with other top solutions
- What worked well và why
- What didn't work và lessons learned
- Suggestions for future improvements
- Practical deployment considerations
- Code availability và reproducibility notes

**Hình ảnh yêu cầu:**
- Hình 15: Confusion matrix hoặc per-class performance
- Hình 16: Example predictions (good và bad cases)
- Hình 17: Failure cases analysis

---

## 4. Ví dụ Phong cách Viết Mới

### 4.1 Ví dụ ĐÚNG (Văn xuôi chi tiết, giữ thuật ngữ)

> Quá trình annotation của xView1 Dataset đòi hỏi sự tham gia của đội ngũ annotators chuyên nghiệp sử dụng phần mềm QGIS. Mỗi annotator được giao nhiệm vụ vẽ bounding box xung quanh từng object nhìn thấy được trong ảnh vệ tinh. Điều quan trọng cần lưu ý là bounding box phải được vẽ theo trục ngang (axis-aligned), nghĩa là các cạnh của box luôn song song với các cạnh của ảnh, thay vì oriented bounding box có thể xoay theo hướng của object.
>
> Quy trình quality control được thiết kế theo mô hình ba giai đoạn nhằm đảm bảo độ chính xác cao nhất. Trong giai đoạn đầu tiên, supervisor sẽ review ngẫu nhiên một phần annotation của mỗi annotator để phát hiện các lỗi hệ thống. Giai đoạn thứ hai bao gồm cross-validation, trong đó một annotator khác sẽ kiểm tra lại công việc của đồng nghiệp. Cuối cùng, ở giai đoạn thứ ba, các chuyên gia (domain experts) sẽ đối chiếu annotation với gold standard dataset để đánh giá inter-annotator agreement.
>
> Một trong những thách thức lớn nhất trong quá trình annotation là xử lý các trường hợp occlusion, khi một object bị che khuất một phần bởi object khác hoặc bởi yếu tố môi trường như bóng cây, mây, hoặc công trình xây dựng. Hướng dẫn annotation quy định rằng chỉ phần visible của object mới được bao gồm trong bounding box, điều này dẫn đến một số box có kích thước nhỏ bất thường so với kích thước thực của object.

### 4.2 Ví dụ SAI (Quá nhiều code, thiếu giải thích)

```
# SAI - Không nên viết như thế này:

| Field | Type |
|-------|------|
| bounds_imcoords | string |
| type_id | int |

Định dạng: `"xmin,ymin,xmax,ymax"`
```

### 4.3 Danh sách Thuật ngữ KHÔNG DỊCH

| Lĩnh vực | Thuật ngữ giữ nguyên |
|----------|---------------------|
| **Geometry** | polygon, bounding box, centroid, vertex, edge, coordinate, tile, patch, overlap |
| **Model** | backbone, encoder, decoder, head, neck, feature map, skip connection, attention |
| **Training** | loss function, optimizer, learning rate, epoch, batch size, gradient, regularization |
| **Detection** | anchor box, RPN, FPN, NMS, IoU, ROI pooling, proposal |
| **Data** | annotation, label, ground truth, dataset, split, augmentation, preprocessing |
| **Metrics** | precision, recall, F1 score, mAP, accuracy, confusion matrix |
| **Image** | pixel, resolution, GSD, RGB, multispectral, panchromatic, channel |

---

## 5. Kế hoạch Chi tiết

### Phase 1: Thu thập Hình ảnh (3 ngày)

| Task | Nguồn | Output |
|------|-------|--------|
| Download xView1 samples | datasetninja, arXiv | 10-15 images |
| Download xView2 samples | CVPR paper, GitHub | 10-15 images (before/after pairs) |
| Download xView3 samples | NeurIPS paper, iuu.xview.us | 10-15 SAR images |
| Extract architecture diagrams | Winner papers/repos | 15 diagrams |
| Create custom diagrams | Draw.io/Figma | 10 diagrams |

**Thư mục lưu trữ:**
```
docs/
├── assets/
│   ├── images/
│   │   ├── xview1/
│   │   │   ├── dataset/
│   │   │   └── solutions/
│   │   ├── xview2/
│   │   └── xview3/
│   └── diagrams/
```

### Phase 2: Chuẩn hóa Bố cục (2 ngày)

| Task | Files | Action |
|------|-------|--------|
| Cập nhật dataset docs | 3 files | Thêm sections theo template |
| Cập nhật winner docs | 15 files | Đồng bộ structure |
| Thêm placeholder images | 18 files | `![Alt](./assets/images/...)` |

### Phase 3: Bổ sung Nội dung (5 ngày)

**Priority 1 - Dataset Docs (đã khá đầy đủ):**
- Thêm hình minh họa
- Bổ sung phần challenges visualization

**Priority 2 - xView1 Winners (thiếu nhiều nhất):**
- #2 Adelaide: Tìm paper/presentation
- #3 USF: Liên hệ hoặc suy luận từ context
- #4 Studio Mapp: Tương tự

**Priority 3 - xView2/3 Winners (khá tốt):**
- Thêm hình minh họa
- Chi tiết hóa architecture sections

### Phase 4: Tạo Biểu đồ (3 ngày)

| Loại | Số lượng | Tool |
|------|----------|------|
| Architecture diagrams | 15 | Mermaid/Draw.io |
| Pipeline flowcharts | 6 | Mermaid |
| Performance charts | 6 | Chart.js/matplotlib |
| Class distribution | 3 | matplotlib |

---

## 5. Yêu cầu Kỹ thuật

### 5.1 Image Hosting Options

| Option | Pros | Cons |
|--------|------|------|
| **GitHub repo (docs/assets)** | Simple, version controlled | Repo size limit |
| **GitHub releases** | Larger files OK | Not in repo |
| **External CDN** | Fast, unlimited | External dependency |

**Khuyến nghị:** GitHub repo + optimize images (WebP, max 500KB/image)

### 5.2 VitePress Image Support

```js
// docs/.vitepress/config.mjs
export default {
  markdown: {
    image: {
      lazyLoading: true
    }
  }
}
```

### 5.3 Image Optimization

```bash
# Batch convert to WebP
for f in *.png; do
  cwebp -q 80 "$f" -o "${f%.png}.webp"
done
```

---

## 6. Timeline Tổng thể (Điều chỉnh cho yêu cầu mới)

### 6.1 Phân bổ theo File

| File | Độ dài mục tiêu | Thời gian ước tính |
|------|-----------------|-------------------|
| Dataset doc (mỗi file) | 12,000 từ + 16 hình | 2-3 ngày |
| Winner doc (mỗi file) | 12,000 từ + 17 hình | 2-3 ngày |

### 6.2 Timeline Chi tiết

```
Phase 1: Chuẩn bị (1 tuần)
├── Thu thập hình ảnh từ sources
├── Tạo folder structure
└── Chuẩn hóa template

Phase 2: Dataset Documentation (2 tuần)
├── Tuần 1: xView1 Dataset (12,000 từ)
├── Tuần 1: xView2/xBD Dataset (12,000 từ)
└── Tuần 2: xView3-SAR Dataset (12,000 từ)

Phase 3: xView1 Winner Solutions (2 tuần)
├── Tuần 3: #1 Reduced Focal Loss, #2 Adelaide
├── Tuần 3: #3 USF, #4 Studio Mapp
└── Tuần 4: #5 CMU SEI

Phase 4: xView2 Winner Solutions (2 tuần)
├── Tuần 5: #1 Siamese UNet, #2 Selim
├── Tuần 5: #3 Eugene, #4 Z-Zheng
└── Tuần 6: #5 Dual-HRNet

Phase 5: xView3 Winner Solutions (2 tuần)
├── Tuần 7: #1 CircleNet, #2 Selim
├── Tuần 7: #3 Tumenn, #4 AI2 Skylight
└── Tuần 8: #5 Kohei

Phase 6: Review & Deploy (1 tuần)
├── Quality check
├── Image optimization
└── Final deployment

TỔNG: ~10 tuần
```

---

## 7. Danh sách Hình ảnh Cần Có

### 7.1 Dataset Documentation

| File | Images Needed |
|------|---------------|
| `dataset-xview1-detection.md` | sample_grid.webp, class_distribution.webp, annotation_example.webp |
| `dataset-xview2-xbd-building-damage.md` | before_after_pairs.webp, damage_scale.webp, disaster_map.webp |
| `dataset-xview3-sar-maritime.md` | sar_sample.webp, vessel_types.webp, dark_fishing_example.webp |

### 7.2 Winner Solutions

| Challenge | Common Images |
|-----------|---------------|
| xView1 (5 winners) | architecture_{1-5}.webp, detection_examples_{1-5}.webp |
| xView2 (5 winners) | siamese_arch.webp, damage_prediction.webp, localization_result.webp |
| xView3 (5 winners) | circlenet_arch.webp, sar_detection.webp, fishing_classification.webp |

---

## 8. Rủi ro và Giải pháp

| Rủi ro | Xác suất | Giải pháp |
|--------|----------|-----------|
| Không tìm được source images | Trung bình | Tạo placeholder diagrams |
| xView1 #2-4 thiếu thông tin | Cao | Ghi chú "Limited documentation" |
| Repo size quá lớn | Thấp | Git LFS hoặc external hosting |
| Mermaid không render trong DOCX | Đã xảy ra | Tạo PNG exports song song |

---

## 9. Success Criteria

### 9.1 Tiêu chí Nội dung (Bắt buộc)

| Tiêu chí | Yêu cầu | Đo lường |
|----------|---------|----------|
| **Độ dài mỗi section** | ≥ 2000 từ | Word count tool |
| **Tổng độ dài mỗi file** | ≥ 12000 từ (6 sections) | Word count tool |
| **Phong cách viết** | Văn xuôi, hạn chế code | Manual review |
| **Thuật ngữ** | Giữ nguyên tiếng Anh | Manual review |
| **Hình ảnh mỗi file** | ≥ 10 hình | Count images |

### 9.2 Tiêu chí Kỹ thuật

- [ ] Bố cục 18 files đồng nhất theo template
- [ ] Tất cả hình ảnh từ nguồn thực tế (dataset hoặc papers)
- [ ] Images optimized < 500KB each
- [ ] VitePress build thành công
- [ ] DOCX xuất được với hình ảnh embedded

### 9.3 Ước tính Khối lượng

| Nội dung | Số lượng |
|----------|----------|
| Tổng số từ cần viết | ~216,000 từ |
| Số hình ảnh cần thu thập | ~180 hình |
| Số sơ đồ cần tạo | ~50 diagrams |

---

## 10. Next Steps

1. **Tạo thư mục assets:** `docs/assets/images/`
2. **Download sample images** từ các nguồn đã liệt kê
3. **Tạo implementation plan** chi tiết cho từng file
4. **Bắt đầu với dataset docs** (ít thay đổi nhất)

---

## Sources

- [xView Dataset Official](https://xviewdataset.org/)
- [Dataset Ninja xView](https://datasetninja.com/xview)
- [xView2 Challenge](https://xview2.org/)
- [xView3 IUU](https://iuu.xview.us/)
- [xBD Paper (CVPR 2019)](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf)
- [xView3-SAR Paper (NeurIPS 2022)](https://arxiv.org/abs/2206.00897)
- [Ultralytics xView Docs](https://docs.ultralytics.com/datasets/detect/xview/)

---

*Báo cáo tạo: 2024-12-19*
