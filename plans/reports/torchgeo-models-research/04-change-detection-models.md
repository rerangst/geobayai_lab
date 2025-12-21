# Change Detection Models Research Report

**Date:** 2025-12-20
**Papers Analyzed:** FC-Siam, BIT-Transformer, STANet
**Focus:** Bi-temporal remote sensing change detection architectures, benchmark performance, and comparison strategies

---

## 1. FC-Siam (Fully Convolutional Siamese Networks) - 1810.08462

### Key Innovation
End-to-end fully convolutional Siamese architectures for change detection without pre-training. Two variants exploit bi-temporal relationships: concatenation (FC-Siam-conc) and difference (FC-Siam-diff) of skip connections.

### Architecture for Bi-Temporal Comparison
- **FC-EF:** Single encoder-decoder path, concatenates two inputs at beginning (early fusion)
- **FC-Siam-conc:** Dual encoder branches (shared weights), concatenates skip connections during decoding
- **FC-Siam-diff:** Dual encoder branches, concatenates absolute difference of skip connections (emphasizes changes)

All use U-Net-inspired encoder-decoder with 4 max-pool/4 upsample layers, skip connections critical for spatial preservation.

### Benchmark Performance
| Dataset | Baseline | Best F1 | Method |
|---------|----------|---------|--------|
| OSCD (13ch multispectral) | Siam: 33.85 | **48.86%** | FC-Siam-diff |
| OSCD (3ch RGB) | Siam: 37.69 | **57.92%** | FC-Siam-diff |
| Air Change Szada/1 | DSCN: 47.9 | **52.66%** | FC-Siam-diff |
| Air Change Tiszadob/3 | DSCN: 86.7 | **93.40%** | FC-EF |

**Speed:** <0.1s inference per image (500x faster than patch-based methods)

### Comparison Strategies
- **Early Fusion:** Concatenate inputs before encoding
- **Late/Skip Fusion:** Independent encoders, fuse feature differences at decoder
- **Explicit Difference:** Absolute difference of skip connections guides attention to actual changes

---

## 2. BIT-Transformer (Binary Image Transformer) - 2103.00208

### Key Innovation
Token-based space-time context modeling using transformers. Converts dense pixel features into compact semantic tokens, models long-range context in token-space, refines features back to pixel-space. 3x fewer FLOPs/parameters than convolutional baselines while achieving higher F1-scores.

### Architecture for Bi-Temporal Comparison
1. **CNN Backbone (ResNet):** Extract features X₁, X₂ from Input1, Input2
2. **Siamese Semantic Tokenizer:** Pool X₁, X₂ → compact token sets T₁, T₂ (learned spatial attention maps)
3. **Transformer Encoder:** Model context in T₁⊕T₂ → context-rich tokens Tₙₑw (MSA + MLP with positional embedding)
4. **Siamese Transformer Decoder:** Project Tₙₑw back to pixel-space → refined features Xₙₑw₁, Xₙₑw₂ (cross-attention: pixels=queries, tokens=keys)
5. **Prediction Head:** Compute Feature Difference Images (FDI), shallow CNN → change map

### Benchmark Performance
| Dataset | BIT F1 | Previous SOTA |
|---------|--------|--------------|
| LEVIR-CD | **89.31%** | STANet: 87.26 |
| WHU-CD | **83.98%** | STANet: 82.32 |
| DSIFN-CD | **69.26%** | STANet: 64.56 |

**Efficiency:** BIT_S4 (3x lower FLOPs/params) > Base_S5 (fully convolutional). BIT_S3 (lighter) > other attention methods.

### Comparison Strategies
- **Token-based vs. Pixel-based:** Models global relations in compact token-space (not dense pixel relations)
- **Feature Differencing:** Refined features enable subtle change detection
- **Multi-scale Context:** Positional embeddings and transformer depth control context modeling

---

## 3. STANet (Spatial-Temporal Attention Network) - 2007.03078

### Key Innovation
CD self-attention mechanism explicitly exploiting spatial-temporal relationships between bi-temporal images. Two attention module variants: Basic Attention Module (BAM) and Pyramid Attention Module (PAM). Addresses illumination variations and misregistration errors. Introduces LEVIR-CD dataset (637 image pairs, 1024×1024, building changes).

### Architecture for Bi-Temporal Comparison
- **Backbone:** Siamese FCN (often ResNet-18 feature extractor)
- **BAM (Basic):** Global spatial-temporal self-attention computing weighted sums across all spatio-temporal positions
- **PAM (Pyramid):** Multi-scale BAM within pyramid structure (handles objects of various sizes)
- **CD Self-Attention:** Captures pixel dependencies across time and space; generates discriminative features robust to variations

### Benchmark Performance
| Dataset | Method | F1 Score |
|---------|--------|----------|
| LEVIR-CD | STANet-Base | ~86% |
| LEVIR-CD | STANet-BAM | ~87.8% (+1.8%) |
| LEVIR-CD | STANet-PAM | **~89.4%** (+1.6% over BAM) |

Real-world application: After training on combined datasets, F1 improved from 21% → **69%**, Recall 13% → **57%** (showing transfer learning benefit).

### Comparison Strategies
- **Spatial-Temporal Self-Attention:** Explicit modeling of dependencies between pixels at different times/positions
- **Multi-scale Context:** PAM pyramid captures changes at multiple resolutions
- **Discriminative Features:** Batch-balanced contrastive loss prioritizes positive sample detection (higher recall, lower precision trade-off)

---

## Architecture Comparison Matrix

| Aspect | FC-Siam | BIT-Transformer | STANet |
|--------|---------|-----------------|--------|
| **Core Design** | Convolutional Siamese | Transformer tokens | Siamese + Self-attention |
| **Feature Fusion** | Skip connection diff/concat | Token-space context | Spatial-temporal attention |
| **Scale Handling** | Single resolution | Implicit (token abstraction) | Multi-scale (PAM pyramid) |
| **Computational Cost** | Baseline | 3x lower | Moderate |
| **Best F1 (OSCD/LEVIR)** | 48.86% (OSCD) | 89.31% (LEVIR) | 89.4% (LEVIR) |
| **Innovation Focus** | Comparison strategy | Efficiency + transformers | Robustness to illumination |

---

## Key Findings

1. **Feature Differencing Dominates:** All three use feature differencing (comparing learned representations) rather than raw pixel differences
2. **Transformer Trend:** BIT shows transformers effective for CD, 3x efficiency gain over convolutions
3. **Attention Mechanisms Critical:** Both BIT (tokens) and STANet (spatial-temporal) outperform pure convolutions
4. **Benchmark Progression:**
   - OSCD (2018): FC-Siam peaks at 48.86%
   - LEVIR-CD (2020): STANet/BIT reach 89%+ (more challenging dataset or better methods)
5. **Speed-Accuracy Trade-off:** FC-Siam fastest (0.1s), but BIT/STANet achieve better accuracy with reasonable speed

---

## Unresolved Questions

- What is specific inference time for BIT-Transformer and STANet on standard hardware?
- How do these models transfer across spectral domains (RGB → multispectral → SAR)?
- Are there ensemble or hybrid approaches combining all three strategies documented?
- What is performance on very recent datasets (DSIFN-CD, LEVIR-CD+ >1000 pairs)?
