# Ứng dụng Deep Learning trong Viễn thám: Phát hiện Tàu biển và Vết dầu loang

## Giới thiệu

Báo cáo này trình bày tổng quan về ứng dụng mạng Convolutional Neural Network (CNN) và các phương pháp Deep Learning trong lĩnh vực viễn thám, tập trung vào hai bài toán quan trọng: **phát hiện tàu biển (Ship Detection)** và **nhận dạng vết dầu loang (Oil Spill Detection)** từ ảnh vệ tinh.

---

## Mục lục

### Chương 1: Giới thiệu
- [Giới thiệu về CNN và Deep Learning trong Viễn thám](./01-introduction/gioi-thieu-cnn-deep-learning.md)

### Chương 2: Kiến trúc CNN
- [Kiến trúc CNN cơ bản](./02-cnn-fundamentals/kien-truc-cnn-co-ban.md)
- [Backbone Networks: ResNet, VGG, EfficientNet](./02-cnn-fundamentals/backbone-networks-resnet-vgg-efficientnet.md)

### Chương 3: Phương pháp CNN với ảnh vệ tinh
- [Phân loại ảnh (Classification)](./03-cnn-satellite-methods/phan-loai-anh-classification.md)
- [Phát hiện đối tượng (Object Detection)](./03-cnn-satellite-methods/phat-hien-doi-tuong-object-detection.md)
- [Phân đoạn ngữ nghĩa (Semantic Segmentation)](./03-cnn-satellite-methods/phan-doan-ngu-nghia-segmentation.md)
- [Instance Segmentation](./03-cnn-satellite-methods/instance-segmentation.md)

### Chương 4: Phát hiện Tàu biển (Ship Detection)
- [Đặc điểm bài toán Ship Detection](./04-ship-detection/dac-diem-bai-toan-ship-detection.md)
- [Các model phát hiện tàu](./04-ship-detection/cac-model-phat-hien-tau.md)
- [Quy trình Ship Detection Pipeline](./04-ship-detection/quy-trinh-ship-detection-pipeline.md)
- [Datasets Ship Detection](./04-ship-detection/datasets-ship-detection.md)

### Chương 5: Phát hiện Vết dầu loang (Oil Spill Detection)
- [Đặc điểm bài toán Oil Spill](./05-oil-spill-detection/dac-diem-bai-toan-oil-spill.md)
- [Các model phát hiện dầu loang](./05-oil-spill-detection/cac-model-phat-hien-dau-loang.md)
- [Quy trình Oil Spill Pipeline](./05-oil-spill-detection/quy-trinh-oil-spill-pipeline.md)
- [Datasets Oil Spill Detection](./05-oil-spill-detection/datasets-oil-spill-detection.md)

### Chương 6: TorchGeo Models
- [Tổng quan TorchGeo](./06-torchgeo-models/tong-quan-torchgeo.md)
- [Classification Models](./06-torchgeo-models/classification-models.md)
- [Segmentation Models](./06-torchgeo-models/segmentation-models.md)
- [Change Detection Models](./06-torchgeo-models/change-detection-models.md)
- [Pretrained Weights theo Sensor](./06-torchgeo-models/pretrained-weights-sensors.md)

### Chương 7: Kết luận
- [Kết luận và Hướng phát triển](./07-conclusion/ket-luan-va-huong-phat-trien.md)

---

## Thông tin

| Thuộc tính | Giá trị |
|------------|---------|
| **Ngôn ngữ** | Tiếng Việt (giữ nguyên thuật ngữ tiếng Anh) |
| **Tổng số chương** | 7 chương |
| **Tổng số file** | 21 files |
| **Phạm vi** | CNN/Deep Learning cho Ship Detection và Oil Spill Detection |

---

## Tài liệu tham khảo chính

- TorchGeo Documentation: https://torchgeo.readthedocs.io/
- xView Challenge Series: https://xviewdataset.org/
- Copernicus Data Space: https://dataspace.copernicus.eu/
