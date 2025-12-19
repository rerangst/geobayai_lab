#!/bin/bash
# Build unified DOCX from thesis documentation
# Output: output/thesis-remote-sensing.docx

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Building Thesis DOCX...${NC}"

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
}

check_command pandoc

mkdir -p output
DATE=$(date +%Y-%m-%d)

echo -e "${YELLOW}Generating thesis-remote-sensing.docx (39 files)...${NC}"

pandoc \
    research/chuong-01-gioi-thieu/muc-01-tong-quan/01-gioi-thieu-cnn-deep-learning.md \
    research/chuong-02-co-so-ly-thuyet/muc-01-kien-truc-cnn/01-kien-truc-co-ban.md \
    research/chuong-02-co-so-ly-thuyet/muc-01-kien-truc-cnn/02-backbone-networks.md \
    research/chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/01-phan-loai-anh.md \
    research/chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/02-phat-hien-doi-tuong.md \
    research/chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/03-phan-doan-ngu-nghia.md \
    research/chuong-02-co-so-ly-thuyet/muc-02-phuong-phap-xu-ly-anh/04-instance-segmentation.md \
    research/chuong-03-phat-hien-tau-bien/muc-01-dac-diem-bai-toan/01-dac-diem.md \
    research/chuong-03-phat-hien-tau-bien/muc-02-mo-hinh/01-cac-mo-hinh.md \
    research/chuong-03-phat-hien-tau-bien/muc-03-quy-trinh/01-pipeline.md \
    research/chuong-03-phat-hien-tau-bien/muc-04-bo-du-lieu/01-datasets.md \
    research/chuong-04-phat-hien-dau-loang/muc-01-dac-diem-bai-toan/01-dac-diem.md \
    research/chuong-04-phat-hien-dau-loang/muc-02-mo-hinh/01-cac-mo-hinh.md \
    research/chuong-04-phat-hien-dau-loang/muc-03-quy-trinh/01-pipeline.md \
    research/chuong-04-phat-hien-dau-loang/muc-04-bo-du-lieu/01-datasets.md \
    research/chuong-05-torchgeo/muc-01-tong-quan/01-tong-quan.md \
    research/chuong-05-torchgeo/muc-02-classification/01-classification-models.md \
    research/chuong-05-torchgeo/muc-03-segmentation/01-segmentation-models.md \
    research/chuong-05-torchgeo/muc-04-change-detection/01-change-detection-models.md \
    research/chuong-05-torchgeo/muc-05-pretrained-weights/01-pretrained-weights.md \
    research/chuong-06-xview-challenges/muc-01-xview1-object-detection/01-dataset.md \
    research/chuong-06-xview-challenges/muc-01-xview1-object-detection/02-giai-nhat.md \
    research/chuong-06-xview-challenges/muc-01-xview1-object-detection/03-giai-nhi.md \
    research/chuong-06-xview-challenges/muc-01-xview1-object-detection/04-giai-ba.md \
    research/chuong-06-xview-challenges/muc-01-xview1-object-detection/05-giai-tu.md \
    research/chuong-06-xview-challenges/muc-01-xview1-object-detection/06-giai-nam.md \
    research/chuong-06-xview-challenges/muc-02-xview2-building-damage/01-dataset.md \
    research/chuong-06-xview-challenges/muc-02-xview2-building-damage/02-giai-nhat.md \
    research/chuong-06-xview-challenges/muc-02-xview2-building-damage/03-giai-nhi.md \
    research/chuong-06-xview-challenges/muc-02-xview2-building-damage/04-giai-ba.md \
    research/chuong-06-xview-challenges/muc-02-xview2-building-damage/05-giai-tu.md \
    research/chuong-06-xview-challenges/muc-02-xview2-building-damage/06-giai-nam.md \
    research/chuong-06-xview-challenges/muc-03-xview3-maritime/01-dataset.md \
    research/chuong-06-xview-challenges/muc-03-xview3-maritime/02-giai-nhat.md \
    research/chuong-06-xview-challenges/muc-03-xview3-maritime/03-giai-nhi.md \
    research/chuong-06-xview-challenges/muc-03-xview3-maritime/04-giai-ba.md \
    research/chuong-06-xview-challenges/muc-03-xview3-maritime/05-giai-tu.md \
    research/chuong-06-xview-challenges/muc-03-xview3-maritime/06-giai-nam.md \
    research/chuong-07-ket-luan/muc-01-tong-ket/01-ket-luan.md \
    -o output/thesis-remote-sensing.docx \
    --toc \
    --toc-depth=3 \
    --metadata title="Nghiên cứu Ứng dụng Deep Learning trong Viễn thám" \
    --metadata author="Research Team" \
    --metadata date="$DATE" \
    --standalone

FILE_SIZE=$(du -h output/thesis-remote-sensing.docx | cut -f1)
echo -e "${GREEN}Done! Output: output/thesis-remote-sensing.docx ($FILE_SIZE)${NC}"
