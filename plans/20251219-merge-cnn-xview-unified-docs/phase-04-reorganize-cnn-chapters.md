# Phase 4: Reorganize CNN Chapters (3, 4, 8, 9, 10)

## Objective
Move CNN fundamental + method chapters to new structure with renumbering.

## Source -> Target Mapping

### Chapter 3: CNN Fundamentals (from CNN 02)
```bash
cp docs/cnn-remote-sensing/02-cnn-fundamentals/kien-truc-cnn-co-ban.md \
   docs/unified-remote-sensing/03-cnn-fundamentals/01-kien-truc-cnn-co-ban.md

cp docs/cnn-remote-sensing/02-cnn-fundamentals/backbone-networks-resnet-vgg-efficientnet.md \
   docs/unified-remote-sensing/03-cnn-fundamentals/02-backbone-networks.md
```

### Chapter 4: CNN Satellite Methods (from CNN 03)
```bash
cp docs/cnn-remote-sensing/03-cnn-satellite-methods/phan-loai-anh-classification.md \
   docs/unified-remote-sensing/04-cnn-satellite-methods/01-classification.md

cp docs/cnn-remote-sensing/03-cnn-satellite-methods/phat-hien-doi-tuong-object-detection.md \
   docs/unified-remote-sensing/04-cnn-satellite-methods/02-object-detection.md

cp docs/cnn-remote-sensing/03-cnn-satellite-methods/phan-doan-ngu-nghia-segmentation.md \
   docs/unified-remote-sensing/04-cnn-satellite-methods/03-semantic-segmentation.md

cp docs/cnn-remote-sensing/03-cnn-satellite-methods/instance-segmentation.md \
   docs/unified-remote-sensing/04-cnn-satellite-methods/04-instance-segmentation.md
```

### Chapter 8: Oil Spill Detection (from CNN 05)
```bash
cp docs/cnn-remote-sensing/05-oil-spill-detection/dac-diem-bai-toan-oil-spill.md \
   docs/unified-remote-sensing/08-oil-spill-detection/01-dac-diem-bai-toan.md

cp docs/cnn-remote-sensing/05-oil-spill-detection/cac-model-phat-hien-dau-loang.md \
   docs/unified-remote-sensing/08-oil-spill-detection/02-cac-model.md

cp docs/cnn-remote-sensing/05-oil-spill-detection/quy-trinh-oil-spill-pipeline.md \
   docs/unified-remote-sensing/08-oil-spill-detection/03-pipeline.md

cp docs/cnn-remote-sensing/05-oil-spill-detection/datasets-oil-spill-detection.md \
   docs/unified-remote-sensing/08-oil-spill-detection/04-datasets.md
```

### Chapter 9: TorchGeo Models (from CNN 06)
```bash
cp docs/cnn-remote-sensing/06-torchgeo-models/tong-quan-torchgeo.md \
   docs/unified-remote-sensing/09-torchgeo-models/01-tong-quan.md

cp docs/cnn-remote-sensing/06-torchgeo-models/classification-models.md \
   docs/unified-remote-sensing/09-torchgeo-models/02-classification-models.md

cp docs/cnn-remote-sensing/06-torchgeo-models/segmentation-models.md \
   docs/unified-remote-sensing/09-torchgeo-models/03-segmentation-models.md

cp docs/cnn-remote-sensing/06-torchgeo-models/change-detection-models.md \
   docs/unified-remote-sensing/09-torchgeo-models/04-change-detection-models.md

cp docs/cnn-remote-sensing/06-torchgeo-models/pretrained-weights-sensors.md \
   docs/unified-remote-sensing/09-torchgeo-models/05-pretrained-weights.md
```

### Chapter 10: Conclusion (from CNN 07)
```bash
cp docs/cnn-remote-sensing/07-conclusion/ket-luan-va-huong-phat-trien.md \
   docs/unified-remote-sensing/10-ket-luan/ket-luan-va-huong-phat-trien.md
```

## Create Chapter READMEs

### Chapter 3 README
```markdown
# Chuong 3: Kien truc CNN co ban

## Noi dung
1. [Kien truc CNN co ban](./01-kien-truc-cnn-co-ban.md)
2. [Backbone Networks: ResNet, VGG, EfficientNet](./02-backbone-networks.md)
```

### Chapter 4 README
```markdown
# Chuong 4: Phuong phap CNN voi anh ve tinh

## Noi dung
1. [Phan loai anh (Classification)](./01-classification.md)
2. [Phat hien doi tuong (Object Detection)](./02-object-detection.md)
3. [Phan doan ngu nghia (Semantic Segmentation)](./03-semantic-segmentation.md)
4. [Instance Segmentation](./04-instance-segmentation.md)
```

### Chapter 8 README
```markdown
# Chuong 8: Phat hien vet dau loang (Oil Spill Detection)

## Noi dung
1. [Dac diem bai toan](./01-dac-diem-bai-toan.md)
2. [Cac model phat hien dau loang](./02-cac-model.md)
3. [Pipeline](./03-pipeline.md)
4. [Datasets](./04-datasets.md)
```

### Chapter 9 README
```markdown
# Chuong 9: TorchGeo Models

## Noi dung
1. [Tong quan TorchGeo](./01-tong-quan.md)
2. [Classification Models](./02-classification-models.md)
3. [Segmentation Models](./03-segmentation-models.md)
4. [Change Detection Models](./04-change-detection-models.md)
5. [Pretrained Weights theo Sensor](./05-pretrained-weights.md)
```

### Chapter 10 README
```markdown
# Chuong 10: Ket luan

## Noi dung
- [Ket luan va huong phat trien](./ket-luan-va-huong-phat-trien.md)
```

## Verification
```bash
# Count files per chapter
for ch in 03 04 08 09 10; do
  echo "Chapter $ch: $(ls docs/unified-remote-sensing/${ch}-*/|wc -l) files"
done
# Expected: 3, 5, 5, 6, 2 files respectively
```

## Git Commit
```bash
git add docs/unified-remote-sensing/03-cnn-fundamentals/
git add docs/unified-remote-sensing/04-cnn-satellite-methods/
git add docs/unified-remote-sensing/08-oil-spill-detection/
git add docs/unified-remote-sensing/09-torchgeo-models/
git add docs/unified-remote-sensing/10-ket-luan/
git commit -m "docs: reorganize CNN chapters 3, 4, 8, 9, 10"
```
