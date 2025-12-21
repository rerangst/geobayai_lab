# Self-Supervised Learning Models for Remote Sensing in TorchGeo

**Research Date:** 2025-12-20
**Focus:** Analyzing six foundational SSL papers for Earth Observation pre-training

---

## 1. MoCo v2 (Momentum Contrast)

**Key Innovation:** Replaces single linear projection with MLP head; incorporates stronger data augmentation similar to SimCLR while maintaining MoCo's memory-efficient momentum encoder architecture.

**Training Methodology:**
- Dynamic queue with momentum encoder decouples batch size from negative samples
- 800 epochs pre-training on ImageNet
- MLP projection head + stronger augmentations

**Transfer Learning Performance:**
- ImageNet linear eval: 71.1% accuracy (vs SimCLR 69.3%)
- State-of-the-art on small batch sizes (256)
- Requires only typical 8-GPU setups (accessibility advantage)

**Relevance to RS:** Generic contrastive learning; excellent for unlabeled RS datasets but lacks domain awareness. Foundation for specialized RS variants (GASSL builds on MoCo-v2).

---

## 2. DINO (Self-Distillation with No Labels)

**Key Innovation:** Student-teacher framework with self-distillation. No fixed dictionary/memory bank. Produces rich visual features including implicit semantic segmentation without manual labels.

**Training Methodology:**
- Multi-crop strategy: 2 global views + multiple local views
- Student processes all crops; teacher processes only globals
- Dynamic momentum update; centering & sharpening prevent collapse
- Works effectively with Vision Transformers

**Transfer Learning Performance:**
- Strong k-NN classification without fine-tuning
- Excellent feature transferability to downstream tasks
- Learned implicit object boundaries useful for segmentation

**Relevance to RS:** DINO-MC variant addresses varying object sizes in RS imagery. SatDINO extends with GSD (Ground Sample Distance) encoding and adaptive view sampling for satellite-specific multi-scale handling. Highly competitive performance with reduced compute.

---

## 3. MAE (Masked Autoencoders)

**Key Innovation:** Asymmetric encoder-decoder for masked image reconstruction. Encoder processes only visible patches; lightweight decoder reconstructs from latent representation. High masking ratio (~75%) proven effective.

**Training Methodology:**
- Vision Transformer backbone
- Random patch masking (75%)
- Efficient encoder (visible patches only)
- Reconstruction loss on masked pixels

**Transfer Learning Performance:**
- Strong benchmark results on ImageNet
- Effective transfer to downstream vision tasks
- Reduces reliance on labeled data

**Relevance to RS:** Well-suited for unlabeled RS data abundance. Enables learning robust features for land cover classification, object detection, change detection. Efficient training for large-scale datasets. SatMAE extends with multi-spectral & temporal adaptation.

---

## 4. SSL4EO-S12 (Self-Supervised Learning for Earth Observation)

**Key Innovation:** Large-scale multimodal, multi-temporal Sentinel dataset for SSL. Dataset, code, and pre-trained models publicly available. Proven four SSL methods (MoCo-v2, DINO, MAE, data2vec) on RS data.

**Training Methodology:**
- Global ESA Sentinel-1 and Sentinel-2 corpus (unlabeled)
- Supports multi-modal (SAR + optical) learning
- Multi-seasonal coverage enables temporal learning
- Systematic benchmarking of four SSL approaches

**Downstream Performance:**
- Approaches/exceeds supervised ImageNet baseline
- Superior to medium-sized labeled EO datasets
- Benchmark dataset for EO community

**Relevance to RS:** Purpose-built for Earth Observation. Provides standardized benchmark and pre-training corpus. Demonstrates that generic SSL methods (MoCo, MAE, DINO) transfer effectively to RS domain. Critical resource for reproducible EO SSL research.

---

## 5. SatMAE (Satellite-Specific Masked Autoencoder)

**Key Innovation:** Domain-adapted MAE for temporal + multi-spectral satellite imagery. Temporal embeddings; independent masking across time. Multi-spectral bands with distinct positional encodings.

**Training Methodology:**
- Pre-trained on multi-spectral Sentinel data (fMoW-Sentinel dataset)
- Temporal embedding for time-series modeling
- Adaptive masking strategy across temporal dimension
- Group spectral bands with spectral-aware positional encoding

**Transfer Learning Performance:**
- Land cover classification: up to 14% improvement over baselines
- Supervised learning: up to 7% improvement over SOTA
- Semantic segmentation: significant gains on standard benchmarks
- Superior to generic MAE on satellite tasks

**Relevance to RS:** Explicitly handles temporal + multi-spectral constraints of satellite data. Outperforms generic MAE; addresses key RS characteristics (temporal dynamics, spectral information). Represents domain-optimized approach to masked autoencoding.

---

## 6. GASSL (Geography-Aware Self-Supervised Learning)

**Key Innovation:** Exploits spatio-temporal structure of RS data via geo-location classification pre-text task + temporal positive pairs. Bridges gap between contrastive and supervised learning on remote sensing.

**Training Methodology:**
- Base: MoCo-v2 contrastive framework
- Geo-location clustering: Images clustered by coordinates; model predicts coarse geo-label
- Temporal positive pairs: Same location, different times (vs synthetic augmentations)
- Combined objective: TemporalInfoNCE loss + geo-location classification loss

**Transfer Learning Performance:**
- fMoW classification: ~8% improvement over MoCo-v2
- Can outperform supervised on temporal classification (~2%)
- xView object detection: ~2% AP improvement
- SpaceNet segmentation: +2.94% IoU over supervised baseline
- NAIP land cover: +3.77% over supervised learning

**Relevance to RS:** Purpose-designed for RS domain. Leverages unique spatio-temporal structure unavailable in generic datasets. Temporal pairs from real multi-temporal data more effective than augmentations. Geo-location task highly relevant for EO. Outperforms both generic SSL and domain-agnostic supervised learning across multiple benchmark tasks.

---

## Comparative Summary

| Model | Type | Domain-Adapted | Temporal | Multi-Spectral | Key Strength |
|-------|------|---|---|---|---|
| MoCo v2 | Contrastive | No | No | No | Memory-efficient; foundation |
| DINO | Self-distill | No | No | No | Implicit semantics; k-NN effective |
| MAE | Reconstruction | No | No | No | Scalable; high masking ratio |
| SSL4EO | Dataset/Bench | Yes | Yes | Yes | Standardized EO benchmark |
| SatMAE | Reconstruction | Yes | Yes | Yes | Spectral encoding; temporal modeling |
| GASSL | Contrastive | Yes | Yes | No | Geo-aware; spatio-temporal structure |

---

## Recommendations for TorchGeo Integration

1. **Foundation Models:** Implement MoCo v2 + MAE as generic baselines (lower compute, reproducible)
2. **Domain-Specific:** Prioritize SatMAE (temporal + spectral) or GASSL (geo-aware + temporal)
3. **Pre-trained Weights:** Leverage SSL4EO-S12 dataset for Sentinel data pre-training
4. **Benchmark:** Use SSL4EO methodology to validate new methods against standard suite
5. **Hybrid Approach:** Consider combining GASSL geo-awareness with SatMAE temporal/spectral modeling

---

## Unresolved Questions

- What is optimal masking ratio for multi-spectral Sentinel data in MAE variants?
- How does geo-location clustering scale to global high-resolution datasets?
- Can temporal positive pairs be automatically discovered without explicit geo-coordinate metadata?
- Which SSL method provides best transfer to non-Sentinel sensors (Planet, Maxar)?
