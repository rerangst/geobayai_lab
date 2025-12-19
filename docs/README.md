# Nghiên cứu Ứng dụng Deep Learning trong Viễn thám

Tài liệu nghiên cứu về ứng dụng Convolutional Neural Network và Deep Learning trong phân tích ảnh viễn thám, bao gồm lý thuyết nền tảng và case studies từ xView Challenge Series.

---

## Mục lục

### Chương 1: Giới thiệu
- [1.1. Tổng quan về CNN và Deep Learning](./chuong-01-gioi-thieu/muc-01-tong-quan/01-gioi-thieu-cnn-deep-learning.md)

### Chương 2: Cơ sở lý thuyết

**2.1. Kiến trúc CNN**
- [2.1.1. Kiến trúc cơ bản](./chuong-02-co-so-ly-thuyet/muc-01-kien-truc-cnn/01-kien-truc-co-ban.md)
- [2.1.2. Backbone Networks](./chuong-02-co-so-ly-thuyet/muc-01-kien-truc-cnn/02-backbone-networks.md)

**2.2. Phương pháp xử lý ảnh**
- [2.2.1. Phân loại ảnh](./chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/01-phan-loai-anh.md)
- [2.2.2. Phát hiện đối tượng](./chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/02-phat-hien-doi-tuong.md)
- [2.2.3. Phân đoạn ngữ nghĩa](./chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/03-phan-doan-ngu-nghia.md)
- [2.2.4. Instance Segmentation](./chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/04-instance-segmentation.md)

### Chương 3: Phát hiện tàu biển

- [3.1. Đặc điểm bài toán](./chuong-03-phat-hien-tau-bien/muc-01-dac-diem-bai-toan/01-dac-diem.md)
- [3.2. Các mô hình](./chuong-03-phat-hien-tau-bien/muc-02-mo-hinh/01-cac-mo-hinh.md)
- [3.3. Quy trình pipeline](./chuong-03-phat-hien-tau-bien/muc-03-quy-trinh/01-pipeline.md)
- [3.4. Bộ dữ liệu](./chuong-03-phat-hien-tau-bien/muc-04-bo-du-lieu/01-datasets.md)

### Chương 4: Phát hiện dầu loang

- [4.1. Đặc điểm bài toán](./chuong-04-phat-hien-dau-loang/muc-01-dac-diem-bai-toan/01-dac-diem.md)
- [4.2. Các mô hình](./chuong-04-phat-hien-dau-loang/muc-02-mo-hinh/01-cac-mo-hinh.md)
- [4.3. Quy trình pipeline](./chuong-04-phat-hien-dau-loang/muc-03-quy-trinh/01-pipeline.md)
- [4.4. Bộ dữ liệu](./chuong-04-phat-hien-dau-loang/muc-04-bo-du-lieu/01-datasets.md)

### Chương 5: TorchGeo

- [5.1. Tổng quan](./chuong-05-torchgeo/muc-01-tong-quan/01-tong-quan.md)
- [5.2. Classification Models](./chuong-05-torchgeo/muc-02-classification/01-classification-models.md)
- [5.3. Segmentation Models](./chuong-05-torchgeo/muc-03-segmentation/01-segmentation-models.md)
- [5.4. Change Detection Models](./chuong-05-torchgeo/muc-04-change-detection/01-change-detection-models.md)
- [5.5. Pre-trained Weights](./chuong-05-torchgeo/muc-05-pretrained-weights/01-pretrained-weights.md)

### Chương 6: xView Challenges

**6.1. xView1 - Object Detection**
- [6.1.1. Dataset](./chuong-06-xview-challenges/muc-01-xview1-object-detection/01-dataset.md)
- [6.1.2. Giải nhất - Reduced Focal Loss](./chuong-06-xview-challenges/muc-01-xview1-object-detection/02-giai-nhat.md)
- [6.1.3. Giải nhì - University of Adelaide](./chuong-06-xview-challenges/muc-01-xview1-object-detection/03-giai-nhi.md)
- [6.1.4. Giải ba - University of South Florida](./chuong-06-xview-challenges/muc-01-xview1-object-detection/04-giai-ba.md)
- [6.1.5. Giải tư - Studio Mapp](./chuong-06-xview-challenges/muc-01-xview1-object-detection/05-giai-tu.md)
- [6.1.6. Giải năm - CMU SEI](./chuong-06-xview-challenges/muc-01-xview1-object-detection/06-giai-nam.md)

**6.2. xView2 - Building Damage Assessment**
- [6.2.1. Dataset (xBD)](./chuong-06-xview-challenges/muc-02-xview2-building-damage/01-dataset.md)
- [6.2.2. Giải nhất - Siamese U-Net](./chuong-06-xview-challenges/muc-02-xview2-building-damage/02-giai-nhat.md)
- [6.2.3. Giải nhì - Selim Sefidov](./chuong-06-xview-challenges/muc-02-xview2-building-damage/03-giai-nhi.md)
- [6.2.4. Giải ba - Eugene Khvedchenya](./chuong-06-xview-challenges/muc-02-xview2-building-damage/04-giai-ba.md)
- [6.2.5. Giải tư - Z-Zheng](./chuong-06-xview-challenges/muc-02-xview2-building-damage/05-giai-tu.md)
- [6.2.6. Giải năm - Dual-HRNet](./chuong-06-xview-challenges/muc-02-xview2-building-damage/06-giai-nam.md)

**6.3. xView3 - Maritime Detection (SAR)**
- [6.3.1. Dataset](./chuong-06-xview-challenges/muc-03-xview3-maritime/01-dataset.md)
- [6.3.2. Giải nhất - CircleNet](./chuong-06-xview-challenges/muc-03-xview3-maritime/02-giai-nhat.md)
- [6.3.3. Giải nhì - Selim Sefidov](./chuong-06-xview-challenges/muc-03-xview3-maritime/03-giai-nhi.md)
- [6.3.4. Giải ba - Tumenn](./chuong-06-xview-challenges/muc-03-xview3-maritime/04-giai-ba.md)
- [6.3.5. Giải tư - AI2 Skylight](./chuong-06-xview-challenges/muc-03-xview3-maritime/05-giai-tu.md)
- [6.3.6. Giải năm - Kohei](./chuong-06-xview-challenges/muc-03-xview3-maritime/06-giai-nam.md)

### Chương 7: Kết luận
- [7.1. Tổng kết và hướng phát triển](./chuong-07-ket-luan/muc-01-tong-ket/01-ket-luan.md)

---

## Thống kê

| Thuộc tính | Giá trị |
|------------|---------|
| **Tổng số chương** | 7 |
| **Tổng số files** | 39 |
| **Ngôn ngữ** | Tiếng Việt (giữ thuật ngữ tiếng Anh) |

---

## Build DOCX

```bash
./scripts/build-unified-docx.sh
```

Output: `output/thesis-remote-sensing.docx`

---

## Tài liệu tham khảo

- [TorchGeo Documentation](https://torchgeo.readthedocs.io/)
- [xView Dataset](https://xviewdataset.org/)
- [Copernicus Data Space](https://dataspace.copernicus.eu/)
