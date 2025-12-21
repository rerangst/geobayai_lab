#!/usr/bin/env python3
"""Extract figures from TorchGeo-related PDF papers."""

import fitz  # PyMuPDF
import os
from pathlib import Path

# Papers directory
PAPERS_DIR = Path(__file__).parent
OUTPUT_DIR = Path("/home/tchatb/sen_doc/docs/assets/images/chuong-05-torchgeo/papers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Key papers to extract figures from (with specific pages for main architecture figures)
PAPERS_CONFIG = {
    # Backbone models
    "resnet_1512.03385.pdf": {"name": "resnet", "pages": [1, 2, 3, 4]},
    "vit_2010.11929.pdf": {"name": "vit", "pages": [0, 1, 2, 3]},
    "swin-transformer_2103.14030.pdf": {"name": "swin", "pages": [0, 1, 2, 3]},
    "efficientnet_1905.11946.pdf": {"name": "efficientnet", "pages": [0, 1, 2, 3]},

    # Segmentation models
    "unet_1505.04597.pdf": {"name": "unet", "pages": [0, 1, 2]},
    "deeplabv3plus_1802.02611.pdf": {"name": "deeplabv3plus", "pages": [0, 1, 2, 3]},
    "fpn_1612.03144.pdf": {"name": "fpn", "pages": [0, 1, 2]},
    "pspnet_1612.01105.pdf": {"name": "pspnet", "pages": [0, 1, 2, 3]},
    "hrnet_1904.04514.pdf": {"name": "hrnet", "pages": [0, 1, 2]},

    # Self-supervised models
    "moco-v2_2003.04297.pdf": {"name": "moco-v2", "pages": [0, 1]},
    "dino_2104.14294.pdf": {"name": "dino", "pages": [0, 1, 2, 3]},
    "mae_2111.06377.pdf": {"name": "mae", "pages": [0, 1, 2]},
    "ssl4eo_2211.07044.pdf": {"name": "ssl4eo", "pages": [0, 1, 2, 3]},
    "satmae_2207.08051.pdf": {"name": "satmae", "pages": [0, 1, 2, 3]},

    # Change detection
    "fc-siam_1810.08462.pdf": {"name": "fc-siam", "pages": [0, 1, 2, 3]},
    "bit-transformer_2103.00208.pdf": {"name": "bit-transformer", "pages": [0, 1, 2, 3]},
    "stanet_2007.03078.pdf": {"name": "stanet", "pages": [0, 1, 2]},

    # Foundation models
    "torchgeo_2111.08872.pdf": {"name": "torchgeo", "pages": [0, 1, 2, 3, 4]},
    "prithvi_2310.18660.pdf": {"name": "prithvi", "pages": [0, 1, 2, 3, 4]},
}


def extract_images_from_pdf(pdf_path: Path, config: dict) -> list:
    """Extract images from specified pages of a PDF."""
    extracted = []
    name = config["name"]
    pages = config.get("pages", [])

    try:
        doc = fitz.open(pdf_path)

        for page_num in pages:
            if page_num >= len(doc):
                continue

            page = doc[page_num]
            image_list = page.get_images()

            for img_idx, img in enumerate(image_list):
                xref = img[0]

                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Skip very small images (likely icons)
                    if len(image_bytes) < 5000:
                        continue

                    # Save image
                    filename = f"{name}_page{page_num+1}_fig{img_idx+1}.{image_ext}"
                    output_path = OUTPUT_DIR / filename

                    with open(output_path, "wb") as f:
                        f.write(image_bytes)

                    extracted.append(filename)
                    print(f"  ✓ Extracted: {filename}")

                except Exception as e:
                    print(f"  ✗ Error extracting image {img_idx} from page {page_num}: {e}")

        doc.close()

    except Exception as e:
        print(f"✗ Error processing {pdf_path.name}: {e}")

    return extracted


def render_page_as_image(pdf_path: Path, config: dict) -> list:
    """Render PDF pages as images (for papers with embedded graphics)."""
    rendered = []
    name = config["name"]
    pages = config.get("pages", [0, 1, 2])

    try:
        doc = fitz.open(pdf_path)

        for page_num in pages:
            if page_num >= len(doc):
                continue

            page = doc[page_num]
            # Render at 150 DPI for good quality
            mat = fitz.Matrix(150/72, 150/72)
            pix = page.get_pixmap(matrix=mat)

            filename = f"{name}_page{page_num+1}.png"
            output_path = OUTPUT_DIR / filename
            pix.save(output_path)

            rendered.append(filename)
            print(f"  ✓ Rendered: {filename}")

        doc.close()

    except Exception as e:
        print(f"✗ Error rendering {pdf_path.name}: {e}")

    return rendered


def main():
    print(f"Extracting figures from papers...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    all_extracted = {}

    for pdf_name, config in PAPERS_CONFIG.items():
        pdf_path = PAPERS_DIR / pdf_name

        if not pdf_path.exists():
            print(f"✗ Not found: {pdf_name}")
            continue

        print(f"\n📄 Processing: {pdf_name}")

        # First try to extract embedded images
        extracted = extract_images_from_pdf(pdf_path, config)

        # If no images extracted, render pages as images
        if not extracted:
            print(f"  No embedded images, rendering pages...")
            extracted = render_page_as_image(pdf_path, config)

        all_extracted[config["name"]] = extracted

    # Summary
    print(f"\n{'='*50}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*50}")

    total = 0
    for name, files in all_extracted.items():
        if files:
            print(f"{name}: {len(files)} files")
            total += len(files)

    print(f"\nTotal extracted: {total} images")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
