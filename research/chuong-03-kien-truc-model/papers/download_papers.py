#!/usr/bin/env python3
"""
Download papers referenced in Chapter 5 (TorchGeo) from arXiv.
"""

import os
import urllib.request
import time

# Papers referenced in Chapter 5
PAPERS = {
    # Core TorchGeo
    "torchgeo": "2111.08872",

    # Architectures - Classification/Backbone
    "resnet": "1512.03385",
    "vit": "2010.11929",
    "swin-transformer": "2103.14030",
    "efficientnet": "1905.11946",

    # Architectures - Segmentation
    "unet": "1505.04597",
    "deeplabv3plus": "1802.02611",
    "fpn": "1612.03144",
    "pspnet": "1612.01105",
    "hrnet": "1904.04514",

    # Self-supervised Learning
    "moco-v2": "2003.04297",
    "dino": "2104.14294",
    "mae": "2111.06377",
    "ssl4eo": "2211.07044",
    "satmae": "2207.08051",

    # Datasets
    "eurosat": "1709.00029",
    "bigearthnet": "1902.06148",
    "landcover-ai": "2005.02264",
    "oscd": "1810.08452",
    "levir-cd": "2012.03588",
    "xview2": "1911.09296",

    # Change Detection Models
    "fc-siam": "1810.08462",
    "bit-transformer": "2103.00208",
    "stanet": "2007.03078",

    # Additional Foundation Models
    "prithvi": "2310.18660",  # IBM/NASA Prithvi
    "gassl": "2011.09980",  # Geography-aware SSL
}


def download_paper(name, arxiv_id, output_dir):
    """Download paper PDF from arXiv."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    filename = f"{name}_{arxiv_id}.pdf"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"[SKIP] {filename} already exists")
        return True

    try:
        print(f"[DOWN] {name}: {url}")
        urllib.request.urlretrieve(url, filepath)
        print(f"[DONE] {filename}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Output directory: {output_dir}")
    print(f"Total papers: {len(PAPERS)}")
    print("-" * 50)

    success = 0
    failed = []

    for name, arxiv_id in PAPERS.items():
        if download_paper(name, arxiv_id, output_dir):
            success += 1
        else:
            failed.append(name)
        time.sleep(1)  # Be nice to arXiv servers

    print("-" * 50)
    print(f"Downloaded: {success}/{len(PAPERS)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
