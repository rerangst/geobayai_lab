# TorchGeo Backbone Models Research

Comprehensive analysis of 4 foundational deep learning backbones for remote sensing applications.

## Files

- **01-backbone-models.md** (163 lines): Complete technical analysis of ResNet, ViT, Swin Transformer, EfficientNet
  - Key innovations and architectural details
  - ImageNet benchmark results (accuracy, parameters, FLOPs)
  - Remote sensing relevance and use cases
  - Comparative analysis table
  - Unresolved research questions

## Research Summary

### Analyzed Papers

1. **ResNet** (arXiv:1512.03385, 2015)
   - Innovation: Residual learning with skip connections
   - Peak: ResNet-152 (77-78.6% ImageNet)
   - RS Strength: Reliable baseline, strong transfer learning

2. **Vision Transformer** (arXiv:2010.11929, 2020)
   - Innovation: Patch embedding + self-attention, no convolutions
   - Peak: ViT-H/14 (88.55% ImageNet with JFT-300M pretraining)
   - RS Strength: Global context, but requires massive pretraining

3. **Swin Transformer** (arXiv:2103.14030, 2021)
   - Innovation: Shifted window attention (O(N) complexity), hierarchical stages
   - Peak: Swin-B (83.4% ImageNet, 58.7 AP on COCO detection)
   - RS Strength: Efficient multi-scale, supports dense predictions

4. **EfficientNet** (arXiv:1905.11946, 2019)
   - Innovation: Compound scaling (balanced depth/width/resolution)
   - Peak: EfficientNet-B7 (84.3% ImageNet, 8.4x smaller & 6.1x faster than competitors)
   - RS Strength: Edge deployment, operational efficiency

## Key Findings

- **Efficiency Leader**: EfficientNet B3-B4 optimal for real-time satellite processing
- **Multi-scale Champion**: Swin Transformer best for high-resolution and dense predictions
- **Hybrid Potential**: ResV2ViT achieves 99.91% on RS classification (dual-stream CNN+Transformer)
- **Pretrain Critical**: Transformers need ImageNet-21k scale; ResNet works with standard ImageNet
- **No Universal Winner**: Selection depends on dataset size, resolution, and deployment constraints

## Data Sources

- arXiv papers (direct PDF analysis)
- Hugging Face model cards
- GitHub official repositories
- MDPI remote sensing journals
- SpringerLink academic publications

## Next Steps

1. Domain-specific pretraining analysis (PRITHVI, SatMAE, SSL4EO)
2. Quantitative RS dataset evaluation (EuroSAT, UC Merced, NWPU)
3. Fine-tuning best practices for satellite data
4. Hybrid architecture design recommendations
