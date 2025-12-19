# Chương 4: Các Model Deep Learning cho Oil Spill Detection

## 5.9. Tổng quan về Approaches

Các phương pháp deep learning cho oil spill detection có thể được phân loại theo nhiều tiêu chí. Theo kiến trúc, có encoder-decoder networks, fully convolutional networks, và attention-based models. Theo task formulation, có binary segmentation, multi-class segmentation, và detection-based approaches. Theo input, có single-polarization, multi-polarization, và multi-sensor fusion approaches.

Khác với ship detection nơi object detection (bounding box) là approach chính, oil spill detection thường formulated như segmentation problem do nature của oil spill là extended irregular region thay vì compact object. Tuy nhiên, một số approaches sử dụng detection để find candidate regions trước khi segment.

## 5.10. Encoder-Decoder Architectures

### 5.10.1. U-Net cho Oil Spill Detection

U-Net là architecture phổ biến nhất cho oil spill segmentation, với cấu trúc symmetric encoder-decoder và skip connections.

**Encoder Path:**
Encoder sử dụng repeated blocks của convolution layers theo sau bởi max pooling để extract hierarchical features. Mỗi block giảm spatial resolution đi một nửa trong khi tăng số channels. Typical configuration có 4-5 pooling operations, tạo feature maps ở multiple scales.

Cho oil spill detection từ SAR, encoder được adapted để handle single-channel hoặc dual-polarization input:
- Single-channel (VV hoặc VH): Input 1×H×W
- Dual-polarization: Input 2×H×W (VV và VH stacked)
- Multi-look combinations: Input có thể include intensity và phase information

**Decoder Path:**
Decoder progressively upsamples feature maps và combines với corresponding encoder features qua skip connections. Transposed convolutions hoặc bilinear upsampling được sử dụng. Skip connections preserve fine-grained spatial information quan trọng cho accurate boundary delineation.

**Modifications cho SAR Oil Spill:**

1. **Attention Gates:** Thêm attention mechanisms vào skip connections để focus vào relevant regions. Attention U-Net variant được áp dụng thành công cho oil spill detection, helping model focus vào dark regions while ignoring irrelevant background.

2. **Deep Supervision:** Thêm auxiliary outputs ở multiple decoder levels, forcing intermediate layers to produce meaningful segmentation. Deep supervision improves gradient flow và helps với class imbalance.

3. **Residual Connections:** Thay basic convolution blocks bằng residual blocks (inspired by ResNet) cho better gradient flow và deeper networks.

4. **Dilated Convolutions:** Sử dụng dilated convolutions trong bottleneck để increase receptive field without losing resolution, helpful cho capturing large oil spill extents.

**Kết quả điển hình:**
Trên các datasets chuẩn, U-Net variants đạt IoU 70-85% tùy dataset và configuration. Attention U-Net và residual U-Net thường outperform vanilla U-Net.

### 5.10.2. SegNet

SegNet là một encoder-decoder architecture với đặc điểm sử dụng max pooling indices trong decoder thay vì full feature maps trong skip connections.

**Encoder:**
Tương tự VGG network, với 13 convolutional layers grouped into 5 blocks với max pooling. Trong quá trình pooling, indices của max values được lưu lại.

**Decoder:**
Sử dụng stored pooling indices để upsample, sau đó apply convolutions. Approach này giảm memory usage so với U-Net (chỉ store indices thay vì full feature maps) nhưng có thể lose một số fine-grained information.

**Cho Oil Spill Detection:**
SegNet được sử dụng trong một số early works về oil spill detection từ SAR. Performance thường thấp hơn U-Net variants do loss of information trong pooling indices approach.

### 5.10.3. DeepLabV3+ cho Oil Spill Detection

DeepLabV3+ với atrous spatial pyramid pooling (ASPP) module đã được áp dụng cho oil spill detection với kết quả tốt.

**Architecture Overview:**
- **Encoder:** Backbone network (ResNet, Xception) với output stride 8 hoặc 16
- **ASPP Module:** Multiple atrous convolutions với different dilation rates để capture multi-scale context
- **Decoder:** Simple decoder với skip connection từ low-level features

**Ưu điểm cho Oil Spill:**
- Multi-scale feature aggregation phù hợp cho variable oil spill sizes
- Large receptive field captures context for look-alike discrimination
- State-of-the-art encoder (ResNet, Xception) provides strong features

**Modifications:**
- Adapted first convolutional layer cho single/dual-channel SAR input
- Modified output channels (2 cho binary, hoặc nhiều hơn cho multi-class)
- Fine-tuned atrous rates cho typical oil spill scales

### 5.10.4. PSPNet (Pyramid Scene Parsing Network)

PSPNet sử dụng pyramid pooling module để aggregate global context information.

**Pyramid Pooling Module:**
Áp dụng pooling ở multiple scales (1×1, 2×2, 3×3, 6×6), sau đó upsample và concatenate với original features. Điều này cho model global context về toàn bộ scene.

**Cho Oil Spill Detection:**
Global context đặc biệt hữu ích cho oil spill detection:
- Hiểu overall sea state (wind conditions)
- Recognize patterns indicating look-alikes (linear features, periodic patterns)
- Distinguish isolated spill từ widespread phenomena

## 5.11. Feature Pyramid Networks

### 5.11.1. FPN-based Segmentation

Feature Pyramid Network được sử dụng không chỉ cho detection mà còn cho segmentation trong oil spill applications.

**Architecture:**
- Bottom-up pathway: Backbone network tạo multi-scale features
- Top-down pathway: Upsample và merge với bottom-up features
- Lateral connections: 1×1 convolutions cho channel adaptation
- Prediction heads: Segment prediction ở mỗi pyramid level

**Multi-scale Benefits:**
Oil spills có kích thước rất diverse (từ vài trăm meters đến hàng chục kilometers). FPN cho phép model detect và segment ở multiple scales effectively.

### 5.11.2. PANet (Path Aggregation Network)

PANet extends FPN với additional bottom-up path để strengthen feature hierarchy.

**Ý tưởng:**
Sau top-down path của FPN, thêm một bottom-up path nữa để information flow both ways. Điều này creates stronger multi-scale feature representation.

**Application:**
PANet-based segmentation đã được áp dụng cho ocean remote sensing applications including oil spill detection, với improvements over basic FPN.

## 5.12. Attention-based Models

### 5.12.1. Attention U-Net

Attention U-Net thêm attention gates vào skip connections của U-Net.

**Attention Gate:**
Cho mỗi skip connection, attention gate compute attention coefficients dựa trên:
- Features từ encoder (cao resolution, local information)
- Features từ decoder (thấp resolution, semantic information)

Output là weighted combination focusing vào relevant regions. Trong oil spill context, attention helps model focus vào dark regions (potential spills) while suppressing bright regions (normal sea surface).

**Results:**
Studies show Attention U-Net consistently outperforms vanilla U-Net cho oil spill detection, với improvements particularly noticeable for complex scenes with look-alikes.

### 5.12.2. Self-Attention và Transformers

Self-attention mechanisms và Vision Transformers (ViT) đang được áp dụng cho oil spill detection.

**Self-Attention in CNNs:**
Thêm self-attention layers vào CNN architectures (như CBAM, SE-Net, Non-local blocks) để capture long-range dependencies. Oil spills có thể extend over large areas, và self-attention helps capture correlations across distant parts of the image.

**Vision Transformers:**
ViT-based architectures như Swin Transformer đã được áp dụng cho oil spill segmentation:
- **Swin-UNet:** Combines Swin Transformer với U-Net style encoder-decoder
- **SegFormer:** Efficient transformer-based segmentation

**Trade-offs:**
- Transformers capture global context better than CNNs
- Require more training data và compute resources
- Performance gains may be marginal for well-tuned CNNs on small datasets

### 5.12.3. Dual-Attention Networks

Dual-attention networks sử dụng cả position attention và channel attention để capture comprehensive contextual information.

**Position Attention:**
Capture long-range spatial dependencies - quan trọng cho understanding oil spill shape và extent.

**Channel Attention:**
Capture inter-channel relationships - hữu ích cho multi-polarization SAR data where different polarizations capture different information.

## 5.13. Multi-scale và Multi-resolution Approaches

### 5.13.1. HRNet (High-Resolution Network)

HRNet maintains high-resolution representations throughout the network thay vì downsample rồi upsample.

**Architecture:**
- Multiple parallel branches ở different resolutions
- Repeated multi-resolution fusions
- All branches contribute to final prediction

**Benefits cho Oil Spill:**
- Preserve fine-grained boundary information
- Better segmentation của thin oil slicks và complex shapes
- Reduced information loss compared to encoder-decoder approaches

### 5.13.2. Cascade Architectures

Cascade approaches process image ở multiple resolutions sequentially:

1. Coarse detection ở low resolution để find candidate regions
2. Refined segmentation ở high resolution cho candidate regions

Approach này efficient cho large SAR scenes where oil spill coverage is typically small percentage of total area.

## 5.14. Multi-task và Auxiliary Learning

### 5.14.1. Oil Spill + Look-alike Classification

Multi-task learning joint training cho:
- Oil spill segmentation (primary task)
- Look-alike classification (auxiliary task)

**Architecture:**
Shared encoder với multiple heads:
- Segmentation head: Dense prediction của oil spill mask
- Classification head: Global prediction của look-alike presence/type

**Benefits:**
- Shared features cho related tasks improve efficiency
- Classification task provides additional supervision signal
- May help learn discriminative features for look-alike rejection

### 5.14.2. Segmentation + Confidence Estimation

Auxiliary task estimating uncertainty/confidence:
- Primary output: Segmentation mask
- Auxiliary output: Per-pixel confidence scores

High confidence regions can be trusted; low confidence regions may need operator review.

### 5.14.3. Joint Ship và Oil Spill Detection

Joint model cho both ship và oil spill detection:
- Ships appear bright trong SAR (opposite of oil)
- Shared features for understanding sea surface
- May help identify oil spill sources

## 5.15. Approaches cho Look-alike Discrimination

### 5.15.1. Contextual Feature Extraction

Models designed to explicitly extract contextual features cho look-alike discrimination:

**Wind Information Integration:**
- Input wind speed/direction as additional channels
- Model learns wind-dependent appearance variations
- Low wind regions less likely to be oil spill

**Temporal Context:**
- Optical flow hoặc difference images từ multiple acquisitions
- Oil spills persist và drift; meteorological look-alikes dissipate

**Geographic Context:**
- Distance to coastline, shipping lanes, platforms
- Historical spill locations
- Bathymetry (natural seeps at certain depths)

### 5.15.2. Two-stage Approaches

Stage 1 - Detection: Find all dark regions (potential spills và look-alikes)
Stage 2 - Classification: Classify each detected region as oil or look-alike

**Stage 1 Models:**
Simple threshold-based detection hoặc trained detector focusing on recall (không miss actual spills).

**Stage 2 Models:**
CNN classifier với features extracted từ detected regions:
- Shape features (elongation, complexity, fractal dimension)
- Intensity features (contrast, homogeneity)
- Texture features (GLCM, wavelets)
- Contextual features (proximity to ships, wind conditions)

### 5.15.3. Ensemble Approaches

Combine multiple models with different characteristics:
- Different architectures (U-Net, DeepLab, SegFormer)
- Different input (VV, VH, VV+VH)
- Different training strategies (different augmentations, loss functions)

Ensemble output: Vote hoặc average of individual model predictions. Typically reduces false positives while maintaining detection rate.

## 5.16. Loss Functions cho Oil Spill Detection

### 5.16.1. Binary Cross-Entropy (BCE)

Standard pixel-wise loss cho binary segmentation:
BCE = -[y×log(p) + (1-y)×log(1-p)]

với y là ground truth (0 hoặc 1) và p là predicted probability.

**Vấn đề cho Oil Spill:**
Class imbalance (mostly background pixels) makes BCE suboptimal - model có thể achieve low loss by predicting all background.

### 5.16.2. Weighted Cross-Entropy

Address class imbalance bằng weighting:
WCE = -[w₁×y×log(p) + w₀×(1-y)×log(1-p)]

với w₁ >> w₀ để increase importance của positive (oil) class.

Weight ratio có thể được set based on class proportions hoặc tuned as hyperparameter.

### 5.16.3. Focal Loss

Focal Loss reduces weight của easy examples, focusing training on hard examples:
FL = -α×(1-p)^γ × log(p) cho positive class
FL = -(1-α)×p^γ × log(1-p) cho negative class

γ (focusing parameter, typically 2) controls how much to down-weight easy examples.

Focal Loss đặc biệt hiệu quả cho oil spill detection với severe class imbalance.

### 5.16.4. Dice Loss

Directly optimize Dice coefficient (F1 score):
Dice Loss = 1 - (2×|P∩G| + smooth) / (|P| + |G| + smooth)

với P là predicted mask, G là ground truth mask, smooth là small constant để avoid division by zero.

**Ưu điểm:**
- Không affected by class imbalance (optimizes overlap directly)
- Better cho segmentation metrics

**Nhược điểm:**
- Gradient magnitude giảm khi prediction improves
- May need combination với BCE cho stable training

### 5.16.5. IoU Loss (Jaccard Loss)

Similar concept to Dice, directly optimize IoU:
IoU Loss = 1 - |P∩G| / |P∪G|

### 5.16.6. Combined Loss Functions

Common practice là combine multiple losses:
L_total = λ₁×BCE + λ₂×Dice + λ₃×Focal

với λ weights tuned cho optimal performance. Combined losses provide benefits of each:
- BCE cho stable gradients
- Dice cho overlap optimization
- Focal cho handling class imbalance

### 5.16.7. Boundary-aware Losses

Special losses focusing on boundary accuracy:
- **Boundary Loss:** Weight higher cho pixels near boundaries
- **Hausdorff Distance Loss:** Minimize maximum boundary deviation
- **Active Contour Loss:** Model contour explicitly

Accurate boundaries quan trọng cho oil spill extent estimation.

## 5.17. Polarimetric SAR Features

### 5.17.1. Dual-Pol và Quad-Pol Data

Sentinel-1 provides dual-polarization (VV + VH) data. Some SAR satellites provide quad-polarization (HH, HV, VH, VV).

**Using Multiple Polarizations:**
Different polarizations capture different scattering mechanisms:
- VV: Vertical-Vertical, sensitive to surface roughness
- VH: Vertical-Horizontal, sensitive to volume scattering
- HH: Horizontal-Horizontal, similar to VV but different geometry
- HV: Horizontal-Vertical, similar to VH

Oil spill affects polarizations differently, providing additional discrimination power.

### 5.17.2. Polarimetric Features cho Deep Learning

**Multi-channel Input:**
Stack polarizations as multiple channels:
- 2-channel input: VV, VH
- 4-channel input: HH, HV, VH, VV
- With derived features: VV, VH, VV/VH ratio, etc.

**Polarimetric Decomposition:**
Compute polarimetric decomposition features (Entropy, Alpha, Anisotropy for dual-pol; Pauli, Freeman-Durden for quad-pol) và use as additional input channels.

**Learning Polarimetric Features:**
Let network learn optimal combination của polarimetric information through training, rather than hand-crafting features.

### 5.17.3. Pseudo-color Composite

Common practice để visualize và possibly use as input:
- R: VV
- G: VH
- B: VV/VH ratio

Network có thể process RGB-like input using standard pre-trained backbones.

## 5.18. So sánh và Lựa chọn Model

### 5.18.1. Bảng So sánh Performance

| Model | Typical IoU | Speed | Memory | Notes |
|-------|-------------|-------|--------|-------|
| U-Net | 70-75% | Fast | Medium | Simple, effective baseline |
| Attention U-Net | 75-80% | Medium | Medium | Better focus on relevant regions |
| DeepLabV3+ | 75-82% | Medium | High | Strong multi-scale capabilities |
| HRNet | 78-85% | Slow | High | Best boundary preservation |
| Swin-UNet | 80-85% | Slow | Very High | Best global context, needs data |

*Note: Performance varies significantly với dataset, preprocessing, and training configuration.*

### 5.18.2. Recommendations theo Use Case

**Real-time Monitoring:**
- U-Net hoặc lightweight variants (EfficientNet encoder)
- Trade some accuracy for speed
- Single-scale inference

**High Accuracy Research:**
- HRNet hoặc ensemble của multiple architectures
- Multi-scale inference với TTA
- Careful post-processing

**Limited Training Data:**
- U-Net với strong augmentation
- Pre-trained encoder (từ natural images hoặc remote sensing)
- Avoid large models prone to overfitting

**Multi-class (Oil + Look-alikes):**
- DeepLabV3+ hoặc similar với sufficient capacity
- Consider two-stage approach
- Carefully balanced loss functions

### 5.18.3. Implementation Considerations

**Frameworks:**
- PyTorch: Flexible, research-oriented
- TensorFlow/Keras: Good deployment options
- Segmentation Models (Python library): Pre-built architectures

**Pre-trained Weights:**
- ImageNet pre-training helps even cho single-channel SAR
- Remote sensing pre-training (từ TorchGeo hoặc similar) may be better
- Self-supervised pre-training on unlabeled SAR data is emerging approach

**Training Practices:**
- Use combined loss (BCE + Dice typically)
- Strong augmentation (rotation, flip, scale, noise)
- Learning rate scheduling (cosine, step decay)
- Early stopping based on validation IoU

