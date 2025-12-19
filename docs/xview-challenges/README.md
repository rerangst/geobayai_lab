# Chuỗi Thử Thách xView: Tài Liệu Toàn Diện

## Tổng Quan

Tài liệu này cung cấp nghiên cứu toàn diện về Chuỗi Thử Thách xView - ba cuộc thi thị giác máy tính ảnh vệ tinh lớn do Defense Innovation Unit (DIU) tổ chức.

| Thử Thách | Năm | Trọng Tâm | Kích Thước Dataset | Số Người Đăng Ký |
|-----------|------|-------|--------------|-------------|
| [xView1](#xview1-phát-hiện-đối-tượng) | 2018 | Phát Hiện Đối Tượng | 1M+ đối tượng | 2,000+ |
| [xView2](#xview2-đánh-giá-thiệt-hại-công-trình) | 2019 | Đánh Giá Thiệt Hại Công Trình | 850K công trình | 2,000+ |
| [xView3](#xview3-phát-hiện-hàng-hải) | 2021-22 | Phát Hiện Tàu Thủy | 243K đối tượng | 1,900 |

---

## Cấu Trúc Tài Liệu

```
docs/xview-challenges/
├── README.md                          # Tệp mục lục này
├── xview1/
│   ├── dataset-xview1-detection.md    # Thông số kỹ thuật Dataset
│   ├── winner-1st-place-reduced-focal-loss.md
│   ├── winner-2nd-place-university-adelaide.md
│   ├── winner-3rd-place-university-south-florida.md
│   ├── winner-4th-place-studio-mapp.md
│   └── winner-5th-place-cmu-sei.md
├── xview2/
│   ├── dataset-xview2-xbd-building-damage.md
│   ├── winner-1st-place-siamese-unet.md
│   ├── winner-2nd-place-selim-sefidov.md
│   ├── winner-3rd-place-eugene-khvedchenya.md
│   ├── winner-4th-place-z-zheng.md
│   └── winner-5th-place-dual-hrnet.md
└── xview3/
    ├── dataset-xview3-sar-maritime.md
    ├── winner-1st-place-circlenet-bloodaxe.md
    ├── winner-2nd-place-selim-sefidov.md
    ├── winner-3rd-place-tumenn.md
    ├── winner-4th-place-ai2-skylight.md
    └── winner-5th-place-kohei.md
```

---

## xView1: Phát Hiện Đối Tượng

### Tóm Tắt Dataset

| Thuộc Tính | Giá Trị |
|-----------|-------|
| **Nhiệm Vụ** | Phát hiện đối tượng đa lớp (60 lớp) |
| **Ảnh** | WorldView-3 (0.3m GSD) |
| **Đối Tượng** | 1,000,000+ thực thể |
| **Phạm Vi** | 1,400 km² |

### Top 5 Người Chiến Thắng

| Hạng | Đội/Tác Giả | Đổi Mới Chính |
|------|-------------|----------------|
| 1 | Nikolay Sergievskiy | Reduced Focal Loss |
| 2 | Victor Stamatescu (U. Adelaide) | - |
| 3 | Sudeep Sarkar (U. South Florida) | - |
| 4 | Leonardo Dal Zovo (Studio Mapp) | - |
| 5 | CMU SEI Team | SSD variant + dual-CNN |

### Tài Liệu

- [Thông Số Kỹ Thuật Dataset](xview1/dataset-xview1-detection.md)
- [Giải Pháp Hạng 1](xview1/winner-1st-place-reduced-focal-loss.md)
- [Giải Pháp Hạng 2](xview1/winner-2nd-place-university-adelaide.md)
- [Giải Pháp Hạng 3](xview1/winner-3rd-place-university-south-florida.md)
- [Giải Pháp Hạng 4](xview1/winner-4th-place-studio-mapp.md)
- [Giải Pháp Hạng 5](xview1/winner-5th-place-cmu-sei.md)

---

## xView2: Đánh Giá Thiệt Hại Công Trình

### Tóm Tắt Dataset (xBD)

| Thuộc Tính | Giá Trị |
|-----------|-------|
| **Nhiệm Vụ** | Định vị công trình + phân loại thiệt hại |
| **Ảnh** | Maxar Open Data (<0.8m GSD) |
| **Công Trình** | 850,736 đa giác |
| **Thảm Họa** | 19 sự kiện |
| **Mức Độ Thiệt Hại** | 4 (Không thiệt hại, Nhỏ, Lớn, Phá hủy) |

### Top 5 Người Chiến Thắng

| Hạng | Đội/Tác Giả | Đổi Mới Chính |
|------|-------------|----------------|
| 1 | Anonymous | Siamese UNet (tốt hơn 266% so với baseline) |
| 2 | Selim Sefidov | DPN92/DenseNet + FocalLossWithDice |
| 3 | Eugene Khvedchenya | Ensemble + Pseudo-labeling |
| 4 | Z-Zheng | - |
| 5 | SI Analytics | Dual-HRNet |

### Tài Liệu

- [Thông Số Kỹ Thuật Dataset](xview2/dataset-xview2-xbd-building-damage.md)
- [Giải Pháp Hạng 1](xview2/winner-1st-place-siamese-unet.md)
- [Giải Pháp Hạng 2](xview2/winner-2nd-place-selim-sefidov.md)
- [Giải Pháp Hạng 3](xview2/winner-3rd-place-eugene-khvedchenya.md)
- [Giải Pháp Hạng 4](xview2/winner-4th-place-z-zheng.md)
- [Giải Pháp Hạng 5](xview2/winner-5th-place-dual-hrnet.md)

---

## xView3: Phát Hiện Tàu Thủy

### Tóm Tắt Dataset (xView3-SAR)

| Thuộc Tính | Giá Trị |
|-----------|-------|
| **Nhiệm Vụ** | Phát hiện tàu + phân loại (đánh cá/không đánh cá) |
| **Ảnh** | Sentinel-1 SAR (5-40m GSD) |
| **Đối Tượng** | 243,018 đã xác minh |
| **Phạm Vi** | 43.2 triệu km² |
| **Tổng Số Pixel** | 1,400 gigapixel |

### Top 5 Người Chiến Thắng

| Hạng | Đội/Tác Giả | Đổi Mới Chính |
|------|-------------|----------------|
| 1 | Eugene Khvedchenya | CircleNet (3× baseline) |
| 2 | Selim Sefidov | UNet multi-task segmentation |
| 3 | Tumenn | HRNet + heatmap detection |
| 4 | AI2 Skylight | Self-training strategy (giải thưởng +$50K US) |
| 5 | Kohei (smly) | HRNet + GSHHG shoreline data |

### Tài Liệu

- [Thông Số Kỹ Thuật Dataset](xview3/dataset-xview3-sar-maritime.md)
- [Giải Pháp Hạng 1](xview3/winner-1st-place-circlenet-bloodaxe.md)
- [Giải Pháp Hạng 2](xview3/winner-2nd-place-selim-sefidov.md)
- [Giải Pháp Hạng 3](xview3/winner-3rd-place-tumenn.md)
- [Giải Pháp Hạng 4](xview3/winner-4th-place-ai2-skylight.md)
- [Giải Pháp Hạng 5](xview3/winner-5th-place-kohei.md)

---

## Mô Hình Chung Qua Các Cuộc Thi

### Mô Hình Kiến Trúc Chiến Thắng

| Mô Hình | xView1 | xView2 | xView3 |
|---------|--------|--------|--------|
| **Encoder-Decoder** | FPN | UNet | UNet/CircleNet |
| **Backbone** | ResNet-50 | ResNet/DenseNet/EfficientNet | EfficientNet/HRNet |
| **Pre/Post Processing** | Siamese | Siamese | Multi-task |
| **Key Loss** | Reduced Focal Loss | Dice + Focal + CE | Reduced Focal + Entropy |

### Kỹ Thuật Chính

1. **Data Augmentation:** Tăng cường dữ liệu mạnh mẽ là thiết yếu
2. **Class Balancing:** Trọng số loss, oversampling
3. **Ensemble:** Nhiều mô hình, TTA
4. **Transfer Learning:** Pre-training ImageNet
5. **Post-Processing:** NMS, phép toán hình thái học

---

## Khả Năng Tạo Dataset

### So Sánh Chi Phí

| Loại Dataset | Chi Phí Ảnh | Chi Phí Gán Nhãn | Tổng Ước Tính |
|--------------|--------------|-----------------|----------------|
| Giống xView1 | $200K-500K | $100K-300K | **$300K-800K** |
| Giống xView2 | $50K (Open Data) | $400K-1M | **$450K-1M** |
| Giống xView3 | Miễn phí (Sentinel-1) | $75K-250K | **$75K-250K** |

### Xếp Hạng Khả Thi

1. **Khả Thi Nhất:** Giống xView3 (ảnh SAR miễn phí, ground truth AIS)
2. **Trung Bình:** Giống xView2 (dữ liệu mở sau thảm họa)
3. **Ít Khả Thi Nhất:** Giống xView1 (ảnh thương mại đắt tiền)

---

## Tài Nguyên Bên Ngoài

### Trang Web Chính Thức

- [xView Dataset (xView1)](https://xviewdataset.org/)
- [xView2 Challenge](https://xview2.org/)
- [xView3 Challenge](https://iuu.xview.us/)
- [DIU xView Series](https://www.diu.mil/ai-xview-challenge)

### Bài Báo

- [xView: Objects in Context (arXiv:1802.07856)](https://arxiv.org/abs/1802.07856)
- [Creating xBD Dataset (CVPR 2019)](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf)
- [xView3-SAR (NeurIPS 2022)](https://arxiv.org/abs/2206.00897)
- [Reduced Focal Loss (arXiv:1903.01347)](https://arxiv.org/abs/1903.01347)

### Kho GitHub

- [DIUx-xView Organization](https://github.com/DIUx-xView)
- [xView2 Hạng 1](https://github.com/DIUx-xView/xView2_first_place)
- [xView3 Hạng 1](https://github.com/BloodAxe/xView3-The-First-Place-Solution)

---

## Báo Cáo Brainstorm

Để xem tóm tắt nghiên cứu ban đầu và phân tích khả thi, xem:
- [Báo Cáo Brainstorm](../../plans/reports/brainstorm-20251218-xview-dataset-research.md)

---

*Tài liệu được tạo: 2024-12-18*
*Tổng số tài liệu: 18 (3 dataset + 15 giải pháp chiến thắng)*
