#!/bin/bash
# Build DOCX from CNN Remote Sensing Markdown files
# Requires: pandoc, mermaid-filter (npm install -g mermaid-filter)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building DOCX from CNN Remote Sensing Markdown files...${NC}"

# Check dependencies
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
}

check_command pandoc

# Create output directory
mkdir -p output

# Get current date
DATE=$(date +%Y-%m-%d)

# Build combined DOCX
echo -e "${YELLOW}Generating cnn-remote-sensing.docx...${NC}"

# Check if mermaid-filter is available
MERMAID_FLAG=""
if command -v mermaid-filter &> /dev/null; then
    MERMAID_FLAG="-F mermaid-filter"
fi

pandoc \
    docs/cnn-remote-sensing/README.md \
    docs/cnn-remote-sensing/01-introduction/gioi-thieu-cnn-deep-learning.md \
    docs/cnn-remote-sensing/02-cnn-fundamentals/kien-truc-cnn-co-ban.md \
    docs/cnn-remote-sensing/02-cnn-fundamentals/backbone-networks-resnet-vgg-efficientnet.md \
    docs/cnn-remote-sensing/03-cnn-satellite-methods/phan-loai-anh-classification.md \
    docs/cnn-remote-sensing/03-cnn-satellite-methods/phat-hien-doi-tuong-object-detection.md \
    docs/cnn-remote-sensing/03-cnn-satellite-methods/phan-doan-ngu-nghia-segmentation.md \
    docs/cnn-remote-sensing/03-cnn-satellite-methods/instance-segmentation.md \
    docs/cnn-remote-sensing/04-ship-detection/dac-diem-bai-toan-ship-detection.md \
    docs/cnn-remote-sensing/04-ship-detection/cac-model-phat-hien-tau.md \
    docs/cnn-remote-sensing/04-ship-detection/quy-trinh-ship-detection-pipeline.md \
    docs/cnn-remote-sensing/04-ship-detection/datasets-ship-detection.md \
    docs/cnn-remote-sensing/05-oil-spill-detection/dac-diem-bai-toan-oil-spill.md \
    docs/cnn-remote-sensing/05-oil-spill-detection/cac-model-phat-hien-dau-loang.md \
    docs/cnn-remote-sensing/05-oil-spill-detection/quy-trinh-oil-spill-pipeline.md \
    docs/cnn-remote-sensing/05-oil-spill-detection/datasets-oil-spill-detection.md \
    docs/cnn-remote-sensing/06-torchgeo-models/tong-quan-torchgeo.md \
    docs/cnn-remote-sensing/06-torchgeo-models/classification-models.md \
    docs/cnn-remote-sensing/06-torchgeo-models/segmentation-models.md \
    docs/cnn-remote-sensing/06-torchgeo-models/change-detection-models.md \
    docs/cnn-remote-sensing/06-torchgeo-models/pretrained-weights-sensors.md \
    docs/cnn-remote-sensing/07-conclusion/ket-luan-va-huong-phat-trien.md \
    -o output/cnn-remote-sensing.docx \
    --toc \
    --toc-depth=3 \
    $MERMAID_FLAG \
    --metadata title="Ứng dụng Deep Learning trong Viễn thám: Phát hiện Tàu biển và Vết dầu loang" \
    --metadata author="Research Team" \
    --metadata date="$DATE" \
    --standalone

echo -e "${GREEN}Done! Output: output/cnn-remote-sensing.docx${NC}"
