# Phase 2: Di chuyển CNN Remote Sensing Docs

**Parent:** [plan.md](./plan.md) | **Status:** Pending

## Overview
Di chuyển 21 files từ docs/cnn-remote-sensing/ sang cấu trúc mới.

## File Mapping

### Chương 1: Giới thiệu
| Old Path | New Path |
|----------|----------|
| 01-introduction/gioi-thieu-cnn-deep-learning.md | chuong-01-gioi-thieu/muc-01-tong-quan/01-gioi-thieu-cnn-deep-learning.md |

### Chương 2: Cơ sở lý thuyết
| Old Path | New Path |
|----------|----------|
| 02-cnn-fundamentals/kien-truc-cnn-co-ban.md | chuong-02-co-so-ly-thuyet/muc-01-kien-truc-cnn/01-kien-truc-co-ban.md |
| 02-cnn-fundamentals/backbone-networks-resnet-vgg-efficientnet.md | chuong-02-co-so-ly-thuyet/muc-01-kien-truc-cnn/02-backbone-networks.md |
| 03-cnn-satellite-methods/phan-loai-anh-classification.md | chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/01-phan-loai-anh.md |
| 03-cnn-satellite-methods/phat-hien-doi-tuong-object-detection.md | chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/02-phat-hien-doi-tuong.md |
| 03-cnn-satellite-methods/phan-doan-ngu-nghia-segmentation.md | chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/03-phan-doan-ngu-nghia.md |
| 03-cnn-satellite-methods/instance-segmentation.md | chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/04-instance-segmentation.md |

### Chương 3: Phát hiện tàu biển
| Old Path | New Path |
|----------|----------|
| 04-ship-detection/dac-diem-bai-toan-ship-detection.md | chuong-03-phat-hien-tau-bien/muc-01-dac-diem-bai-toan/01-dac-diem.md |
| 04-ship-detection/cac-model-phat-hien-tau.md | chuong-03-phat-hien-tau-bien/muc-02-mo-hinh/01-cac-mo-hinh.md |
| 04-ship-detection/quy-trinh-ship-detection-pipeline.md | chuong-03-phat-hien-tau-bien/muc-03-quy-trinh/01-pipeline.md |
| 04-ship-detection/datasets-ship-detection.md | chuong-03-phat-hien-tau-bien/muc-04-bo-du-lieu/01-datasets.md |

### Chương 4: Phát hiện dầu loang
| Old Path | New Path |
|----------|----------|
| 05-oil-spill-detection/dac-diem-bai-toan-oil-spill.md | chuong-04-phat-hien-dau-loang/muc-01-dac-diem-bai-toan/01-dac-diem.md |
| 05-oil-spill-detection/cac-model-phat-hien-dau-loang.md | chuong-04-phat-hien-dau-loang/muc-02-mo-hinh/01-cac-mo-hinh.md |
| 05-oil-spill-detection/quy-trinh-oil-spill-pipeline.md | chuong-04-phat-hien-dau-loang/muc-03-quy-trinh/01-pipeline.md |
| 05-oil-spill-detection/datasets-oil-spill-detection.md | chuong-04-phat-hien-dau-loang/muc-04-bo-du-lieu/01-datasets.md |

### Chương 5: TorchGeo
| Old Path | New Path |
|----------|----------|
| 06-torchgeo-models/tong-quan-torchgeo.md | chuong-05-torchgeo/muc-01-tong-quan/01-tong-quan.md |
| 06-torchgeo-models/classification-models.md | chuong-05-torchgeo/muc-02-classification/01-classification-models.md |
| 06-torchgeo-models/segmentation-models.md | chuong-05-torchgeo/muc-03-segmentation/01-segmentation-models.md |
| 06-torchgeo-models/change-detection-models.md | chuong-05-torchgeo/muc-04-change-detection/01-change-detection-models.md |
| 06-torchgeo-models/pretrained-weights-sensors.md | chuong-05-torchgeo/muc-05-pretrained-weights/01-pretrained-weights.md |

### Chương 7: Kết luận
| Old Path | New Path |
|----------|----------|
| 07-conclusion/ket-luan-va-huong-phat-trien.md | chuong-07-ket-luan/muc-01-tong-ket/01-ket-luan.md |

## Commands
```bash
# Copy (không delete để backup)
cp docs/cnn-remote-sensing/01-introduction/gioi-thieu-cnn-deep-learning.md \
   docs/chuong-01-gioi-thieu/muc-01-tong-quan/01-gioi-thieu-cnn-deep-learning.md
# ... (repeat for all 21 files)
```

## Todo
- [ ] Copy tất cả 21 files theo mapping
- [ ] Verify files đã copy đúng
- [ ] Xóa folder cnn-remote-sensing/ cũ (sau khi verify)

## Success Criteria
- 21 files được di chuyển đúng vị trí
- Nội dung không thay đổi
