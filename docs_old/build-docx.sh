#!/bin/bash
# Build DOCX from Markdown files
# Requires: pandoc, mermaid-filter (npm install -g mermaid-filter)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building DOCX from Markdown files...${NC}"

# Check dependencies
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
}

check_command pandoc
check_command mermaid-filter

# Create output directory
mkdir -p output

# Get current date
DATE=$(date +%Y-%m-%d)

# Build combined DOCX
echo -e "${YELLOW}Generating xview-research.docx...${NC}"

pandoc \
    docs/xview-challenges/README.md \
    docs/xview-challenges/xview1/dataset-xview1-detection.md \
    docs/xview-challenges/xview1/winner-1st-place-reduced-focal-loss.md \
    docs/xview-challenges/xview1/winner-2nd-place-university-adelaide.md \
    docs/xview-challenges/xview1/winner-3rd-place-university-south-florida.md \
    docs/xview-challenges/xview1/winner-4th-place-studio-mapp.md \
    docs/xview-challenges/xview1/winner-5th-place-cmu-sei.md \
    docs/xview-challenges/xview2/dataset-xview2-xbd-building-damage.md \
    docs/xview-challenges/xview2/winner-1st-place-siamese-unet.md \
    docs/xview-challenges/xview2/winner-2nd-place-selim-sefidov.md \
    docs/xview-challenges/xview2/winner-3rd-place-eugene-khvedchenya.md \
    docs/xview-challenges/xview2/winner-4th-place-z-zheng.md \
    docs/xview-challenges/xview2/winner-5th-place-dual-hrnet.md \
    docs/xview-challenges/xview3/dataset-xview3-sar-maritime.md \
    docs/xview-challenges/xview3/winner-1st-place-circlenet-bloodaxe.md \
    docs/xview-challenges/xview3/winner-2nd-place-selim-sefidov.md \
    docs/xview-challenges/xview3/winner-3rd-place-tumenn.md \
    docs/xview-challenges/xview3/winner-4th-place-ai2-skylight.md \
    docs/xview-challenges/xview3/winner-5th-place-kohei.md \
    -o output/xview-research.docx \
    --reference-doc=templates/reference.docx \
    --toc \
    --toc-depth=3 \
    -F mermaid-filter \
    --metadata title="Nghiên cứu xView Challenge Series" \
    --metadata author="Research Team" \
    --metadata date="$DATE" \
    --standalone

echo -e "${GREEN}Done! Output: output/xview-research.docx${NC}"
