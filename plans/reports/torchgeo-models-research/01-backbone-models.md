# Deep Learning Backbones for Remote Sensing in TorchGeo

Research date: 2025-12-20 | Focus: Architecture innovations, benchmarks, remote sensing relevance

## 1. ResNet (Deep Residual Learning, 2015)

**Paper**: arXiv:1512.03385 | Authors: He et al.

### Key Innovation
- **Residual Learning Framework**: Solves vanishing gradient problem in very deep networks by learning residual functions F(x) instead of raw functions H(x)
- **Skip Connections**: Enable direct gradient flow and identity mapping bypass, allowing training of networks up to 1000 layers
- **Bottleneck Blocks**: 3-layer blocks (1x1 → 3x3 → 1x1 conv) for efficiency

### Core Architecture
- Building block: x → Conv3×3 → ReLU → Conv3×3 → (Add skip) → ReLU
- Variants: ResNet-18/34 (64→512 filters), ResNet-50/101/152 (64→2048 filters)
- Parameter counts: ResNet-50 (25.6M), ResNet-101 (44.7M), ResNet-152 (60.4M)

### Benchmark Results
| Model | Top-1 Accuracy | Top-5 Accuracy | FLOPs |
|-------|---|---|---|
| ResNet-50 | 80%+ | - | 3.8B |
| ResNet-152 | 77% (1-crop) / 78.6% (10-crop) | 93.3% / 94.3% | - |
| Ensemble | 96.43% (ILSVRC 2015 winner) | - | - |

### Remote Sensing Relevance
- Standard baseline for land-use classification, object detection
- Strong transfer learning on satellite datasets
- Comparable to ViT on many remote sensing benchmarks
- Efficient training and deployment for edge devices

---

## 2. Vision Transformer (ViT, 2020)

**Paper**: arXiv:2010.11929 | Authors: Dosovitskiy et al. | ICLR 2021

### Key Innovation
- **Patch Embedding**: Divide images into 16×16 patches, process as token sequences
- **Pure Transformer**: Direct application of self-attention (no inductive CNN biases)
- **Position Embedding**: Learned positional encodings for spatial information
- **CLS Token**: Classification token aggregates global image features

### Core Architecture
- Image → Patch Embedding (Patch projection) → Positional Embedding → Transformer Encoder Blocks
- Standard transformer blocks with multi-head self-attention, MLP, LayerNorm
- Requires large-scale pretraining (ImageNet-21k: 14M images, 21,843 classes)

### Benchmark Results
| Model | Pretraining | Resolution | Top-1 Accuracy | Top-5 Accuracy |
|-------|---|---|---|---|
| ViT-B/16 | ImageNet-21k | 224² | 81.2% | - |
| ViT-L/16 | ImageNet-21k | 224² | 82.7% | - |
| ViT-H/14 | JFT-300M | 336² | 88.55% | - |
| ViT-L/16 (finetuned) | ImageNet-21k | 384² | 85%+ | - |

### Remote Sensing Relevance
- **Global Context**: Self-attention captures long-range dependencies (ideal for large satellite scenes)
- **Scalability**: Benefits from massive pretraining on heterogeneous data
- **Performance Gap**: Underperforms ResNet when trained from scratch on ImageNet
- **Hybrid Potential**: ResV2ViT dual-stream achieves 99.91% precision on RSI-CB256 dataset
- **Data Efficiency**: Requires less computational resources than CNNs for pretraining at scale

---

## 3. Swin Transformer (Shifted Window Attention, 2021)

**Paper**: arXiv:2103.14030 | Authors: Liu et al. | ICCV 2021 (Marr Prize)

### Key Innovation
- **Shifted Window Attention**: Local windows with shifted attention for cross-window connections
- **Linear Complexity**: O(N) computation vs ViT's O(N²), enabling dense predictions
- **Hierarchical Architecture**: Progressive feature downsampling (4→8→16→32 spatial reduction) mirrors CNN designs
- **Stage-based Design**: 4 stages with increasing channel dimensions

### Core Architecture
- Shifted Window blocks: Partition image into local windows → Self-attention → Merging
- Hierarchical feature pyramid: Enables object detection and segmentation tasks
- Each stage doubles spatial resolution reduction while expanding channels

### Benchmark Results
| Model | Resolution | Parameters | FLOPs | Top-1 Accuracy |
|-------|---|---|---|---|
| Swin-T | 224² | 29M | 4.5G | 81.3% |
| Swin-S | 224² | 50M | 8.7G | 83.0% |
| Swin-B | 224² | 88M | 15.4G | 83.4% |
| Swin-B (384²) | 384² | 88M | 47.5G | 84.4% |
| Swin-V2-B | ImageNet-V2 | - | - | 84.0% |

**Detection**: 58.7 box AP on COCO | **Segmentation**: 53.5 mIoU on ADE20K

### Remote Sensing Relevance
- **Efficiency**: Linear complexity suitable for high-resolution satellite imagery (often 512×512+)
- **Multi-scale Features**: Hierarchical design enables object detection and change detection
- **Outperforms ViT**: Better speed-accuracy tradeoff vs vanilla transformers
- **Dense Prediction**: Architecture supports pixel-level segmentation tasks

---

## 4. EfficientNet (Compound Scaling, 2019)

**Paper**: arXiv:1905.11946 | Authors: Tan & Le

### Key Innovation
- **Compound Scaling Method**: Balanced scaling of depth, width, and resolution via coefficient α/β/γ
- **Neural Architecture Search**: Auto-designed baseline EfficientNet-B0 optimized for accuracy-efficiency
- **Scaling Principle**: All dimensions scale uniformly rather than independently

### Core Architecture
- MobileNet V2 base with squeeze-and-excitation modules
- Scaling formula: Depth × α^φ, Width × β^φ, Resolution × γ^φ
- Family B0-B7: Progressive complexity from 5.3M to 66M parameters

### Benchmark Results
| Model | Resolution | Parameters | FLOPs | Top-1 Accuracy | Inference Speed |
|-------|---|---|---|---|---|
| EfficientNet-B3 | 300² | 12M | 1.8G | 81.6% | - |
| EfficientNet-B4 | 380² | 19M | 4.2G | 82.9% | - |
| EfficientNet-B5 | 456² | 30M | 9.9G | 83.6% | - |
| EfficientNet-B7 | 600² | 66M | 37.0G | 84.3% | 6.1x faster |
| EfficientNet-B7 (vs best ConvNet) | - | **8.4x smaller** | **6.1x faster** | SOTA | - |

### Remote Sensing Relevance
- **Mobile Deployment**: 8.4x model compression suitable for edge devices, drones
- **Transfer Learning**: Order of magnitude fewer parameters with strong performance on CIFAR-100, Flowers
- **Efficiency**: B3-B4 range optimal for operational satellite processing pipelines
- **Practical Systems**: Preferred for real-time applications (agriculture monitoring, disaster response)

---

## Comparative Summary for Remote Sensing

| Backbone | Depth | Scalability | Local Features | Global Context | Efficiency | Best Use Case |
|----------|-------|---|---|---|---|---|
| **ResNet** | Fixed 50-152 | Linear | Strong | Moderate | Good | Baseline, transfer learning |
| **ViT** | Fixed | Excellent with pretrain | Weak | Excellent | Moderate | Large-scale scene understanding |
| **Swin** | Fixed 4-stage | Excellent | Strong | Excellent | Excellent | Multi-task, high-res imagery |
| **EfficientNet** | B0-B7 | Excellent | Strong | Moderate | Excellent | Edge/mobile, real-time |

## Key Findings for TorchGeo

1. **No silver bullet**: Model choice depends on (a) dataset size, (b) resolution, (c) deployment constraints
2. **Hybrid approaches**: ResV2ViT dual-stream achieves 99.91% on remote sensing classification
3. **Efficiency matters**: EfficientNet B3-B4 critical for operational pipelines (8.4x compression vs B7)
4. **Swin dominance**: Hierarchical design + linear complexity best for dense predictions and high-res data
5. **Pretrain dependency**: ViT/Swin require ImageNet-21k scale for competitive performance; ResNet works from ImageNet

---

## Unresolved Questions

- What is optimal resolution/patch size for different satellite sensor types (optical vs SAR)?
- How do these backbones perform on imbalanced remote sensing datasets (common in practice)?
- Quantitative comparison of domain-specific pretrained models (e.g., PRITHVI, SatMAE) vs generic ImageNet pretraining?

## References

- [Vision Transformer Hugging Face](https://huggingface.co/google/vit-base-patch16-224)
- [Swin Transformer Official GitHub](https://github.com/microsoft/Swin-Transformer)
- [ResNet Detailed Guide](https://cv-tricks.com/keras/understand-implement-resnets/)
- [Vision Transformers for Remote Sensing](https://www.mdpi.com/2072-4292/13/3/516)
- [Dual-Stream Transformers for Remote Sensing Analysis](https://www.mdpi.com/2313-433X/11/5/156)
- [Deep Learning Methods Review for RS Image Classification](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00772-x)
