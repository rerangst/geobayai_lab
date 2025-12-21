# Chương 2: Backbone Networks: VGG, ResNet, EfficientNet và Swin Transformer

## 2.10. Khái niệm Backbone Network

Trong các bài toán thị giác máy tính hiện đại như phát hiện đối tượng (object detection), phân đoạn ngữ nghĩa (semantic segmentation), và instance segmentation, thuật ngữ "backbone network" được sử dụng để chỉ phần mạng nơ-ron chịu trách nhiệm trích xuất đặc trưng từ ảnh đầu vào. Backbone đóng vai trò như bộ encoder, biến đổi ảnh thô thành các feature map đa tầng chứa thông tin semantic phong phú, làm nền tảng cho các task-specific head thực hiện nhiệm vụ cuối cùng như phân loại, regression bounding box, hay tạo segmentation mask.

Việc lựa chọn backbone có ảnh hưởng quyết định đến hiệu suất của toàn bộ hệ thống. Một backbone mạnh có khả năng trích xuất các đặc trưng discriminative và robust, giúp các downstream task đạt độ chính xác cao hơn. Đồng thời, kích thước và độ phức tạp của backbone cũng ảnh hưởng trực tiếp đến tốc độ inference và yêu cầu tài nguyên phần cứng. Do đó, việc cân bằng giữa accuracy và efficiency là một trong những cân nhắc quan trọng khi thiết kế hệ thống.

Trong lĩnh vực viễn thám, các backbone được pre-train trên ImageNet thường được fine-tune cho các bài toán cụ thể. Gần đây, các thư viện như TorchGeo còn cung cấp backbone được pre-train trực tiếp trên ảnh vệ tinh từ các cảm biến như Sentinel-1, Sentinel-2, và Landsat, mang lại hiệu suất tốt hơn cho các bài toán viễn thám nhờ domain-specific knowledge.

## 2.11. VGGNet: Kiến trúc Sâu với Convolution 3×3

### 2.11.1. Nguồn gốc và Đóng góp

VGGNet được phát triển bởi nhóm Visual Geometry Group tại Đại học Oxford và đạt vị trí á quân trong cuộc thi ILSVRC 2014 (sau GoogLeNet). Mặc dù không giành chiến thắng, VGGNet có ảnh hưởng sâu rộng đến thiết kế kiến trúc CNN nhờ tính đơn giản và hiệu quả của nó. Đóng góp chính của VGGNet là chứng minh rằng độ sâu của mạng là yếu tố quan trọng quyết định hiệu suất, và việc sử dụng các bộ lọc nhỏ 3×3 xếp chồng có thể đạt được receptive field tương đương các bộ lọc lớn hơn với ít tham số hơn.

### 2.11.2. Thiết kế Kiến trúc

VGGNet tuân theo một nguyên tắc thiết kế nhất quán: tất cả các lớp convolution đều sử dụng bộ lọc 3×3 với stride 1 và padding "same", và tất cả các lớp pooling đều là max pooling 2×2 với stride 2. Mạng được chia thành 5 khối convolution, mỗi khối kết thúc bằng một lớp max pooling. Số bộ lọc bắt đầu từ 64 ở khối đầu tiên và tăng gấp đôi sau mỗi lần pooling: 64 → 128 → 256 → 512 → 512.

Hai biến thể phổ biến nhất là VGG-16 và VGG-19, với 16 và 19 lớp có trọng số tương ứng. VGG-16 gồm 13 lớp convolution và 3 lớp fully connected, trong khi VGG-19 có thêm 3 lớp convolution. Cả hai đều kết thúc bằng 3 lớp fully connected với 4096, 4096, và 1000 nơ-ron (tương ứng 1000 lớp của ImageNet).

### 2.11.3. Ý nghĩa của Convolution 3×3

Một insight quan trọng từ VGGNet là hai lớp convolution 3×3 xếp chồng có receptive field 5×5, và ba lớp 3×3 có receptive field 7×7. Tuy nhiên, ba lớp 3×3 chỉ cần 3 × (3² × C²) = 27C² tham số, trong khi một lớp 7×7 cần 7² × C² = 49C² tham số, giảm 45%. Đồng thời, việc xen kẽ các hàm kích hoạt ReLU giữa các lớp 3×3 tăng khả năng biểu diễn phi tuyến của mạng.

### 2.11.4. Hạn chế

Mặc dù có kiến trúc đơn giản và hiệu quả, VGGNet có một số hạn chế đáng kể. Thứ nhất, ba lớp fully connected cuối cùng chứa khoảng 120 triệu tham số (chiếm 90% tổng số tham số của mạng), làm tăng nguy cơ overfitting và yêu cầu bộ nhớ lớn. Thứ hai, mạng có xu hướng bão hòa về độ chính xác khi tăng độ sâu, do vấn đề vanishing gradient khiến việc huấn luyện các mạng rất sâu trở nên khó khăn. Thứ ba, tốc độ inference chậm do số lượng tham số và phép tính lớn.

## 2.12. ResNet: Mạng Residual và Skip Connection

### 2.12.1. Vấn đề Degradation

Trước ResNet, một nghịch lý được quan sát: khi tăng độ sâu của mạng beyond một ngưỡng nhất định, accuracy không tăng mà còn giảm, ngay cả trên training set. Điều này không phải do overfitting (vì training error cũng tăng) mà do vấn đề optimization - các mạng rất sâu khó được huấn luyện để đạt được minimum tốt. Kaiming He và cộng sự gọi hiện tượng này là "degradation problem".

### 2.12.2. Residual Learning

Ý tưởng cốt lõi của ResNet là thay vì học ánh xạ trực tiếp H(x) từ input x sang output, mạng học residual function F(x) = H(x) - x. Output thực tế được tính là H(x) = F(x) + x, với x được truyền trực tiếp qua một kết nối tắt (skip connection hay shortcut). Nếu identity mapping là tối ưu, mạng chỉ cần học F(x) = 0, điều này dễ dàng hơn nhiều so với việc học H(x) = x từ đầu.

Skip connection cho phép gradient chảy trực tiếp qua nhiều lớp trong quá trình backpropagation, giải quyết vấn đề vanishing gradient và cho phép huấn luyện thành công các mạng với hàng trăm hoặc thậm chí hàng nghìn lớp. Điều này mở ra khả năng xây dựng các mạng sâu hơn đáng kể so với trước đây.

### 2.12.3. Kiến trúc ResNet

ResNet sử dụng các residual block làm đơn vị xây dựng cơ bản. Có hai loại block chính:

**Basic Block:** Sử dụng trong ResNet-18 và ResNet-34, gồm hai lớp convolution 3×3 với skip connection. Cấu trúc: Conv3×3 → BN → ReLU → Conv3×3 → BN → (+x) → ReLU.

**Bottleneck Block:** Sử dụng trong ResNet-50, ResNet-101, và ResNet-152, gồm ba lớp convolution theo cấu trúc 1×1 → 3×3 → 1×1. Lớp 1×1 đầu tiên giảm số kênh (bottleneck), lớp 3×3 thực hiện convolution chính, và lớp 1×1 cuối khôi phục số kênh. Thiết kế này giảm số lượng tham số và chi phí tính toán trong khi duy trì hoặc tăng khả năng biểu diễn.

Khi kích thước không gian thay đổi (do strided convolution), skip connection sử dụng convolution 1×1 với stride tương ứng để match dimension. Tương tự, khi số kênh thay đổi, convolution 1×1 được sử dụng để projection.

### 2.12.4. Các Biến thể ResNet

Từ thiết kế gốc, nhiều biến thể và cải tiến của ResNet đã được phát triển:

**ResNet-V2:** Thay đổi thứ tự các thành phần trong residual block thành BN → ReLU → Conv (pre-activation), cải thiện khả năng huấn luyện và kết quả cuối cùng.

**ResNeXt:** Giới thiệu "cardinality" như một chiều mới bên cạnh depth và width. Mỗi residual block được chia thành nhiều nhánh song song với cấu trúc giống nhau, sau đó aggregate. ResNeXt-101 (32×4d) đạt kết quả tốt hơn ResNet-101 với số tham số tương đương.

**SE-ResNet:** Tích hợp Squeeze-and-Excitation (SE) module, học cách recalibrate channel-wise feature responses bằng cách mô hình hóa interdependencies giữa các kênh.

**ResNeSt:** Kết hợp ý tưởng từ ResNeXt và SE-Net với split-attention mechanism, đạt state-of-the-art trên nhiều benchmark.

## 2.13. EfficientNet: Compound Scaling

### 2.13.1. Vấn đề Scaling

Trước EfficientNet, việc tăng hiệu suất CNN thường được thực hiện bằng cách tăng một trong ba chiều: độ sâu (thêm lớp), độ rộng (thêm kênh), hoặc độ phân giải (tăng kích thước ảnh đầu vào). Tuy nhiên, việc scale theo từng chiều riêng lẻ thường đạt điểm bão hòa nhanh chóng. Ví dụ, ResNet-1000 không tốt hơn đáng kể so với ResNet-152 mặc dù sâu hơn nhiều.

### 2.13.2. Compound Scaling

Mingxing Tan và Quoc V. Le đề xuất phương pháp compound scaling, cân bằng đồng thời cả ba chiều depth, width, và resolution theo một tỷ lệ cố định. Họ sử dụng neural architecture search (NAS) để tìm baseline network (EfficientNet-B0) tối ưu, sau đó scale up theo công thức:

- Depth: d = α^φ
- Width: w = β^φ
- Resolution: r = γ^φ

với ràng buộc α × β² × γ² ≈ 2 để đảm bảo FLOPS tăng xấp xỉ 2^φ. Các hệ số α, β, γ được tìm bằng grid search trên baseline. Compound coefficient φ được điều chỉnh để tạo ra các biến thể từ B0 đến B7 với quy mô tăng dần.

### 2.13.3. Kiến trúc MBConv

EfficientNet sử dụng Mobile Inverted Bottleneck Convolution (MBConv) làm building block chính, kế thừa từ MobileNetV2. MBConv có cấu trúc ngược với bottleneck truyền thống: mở rộng số kênh ở đầu, thực hiện depthwise separable convolution ở giữa, rồi thu hẹp ở cuối. Cấu trúc: Conv1×1 (expand) → Depthwise Conv3×3/5×5 → SE → Conv1×1 (project).

Depthwise separable convolution chia convolution thông thường thành hai bước: depthwise convolution (một filter riêng cho mỗi kênh) và pointwise convolution (1×1 để mix kênh). Điều này giảm đáng kể số lượng tham số và FLOPs. Squeeze-and-Excitation (SE) module được tích hợp để tăng khả năng biểu diễn.

### 2.13.4. Hiệu quả của EfficientNet

EfficientNet đạt được sự cân bằng ấn tượng giữa accuracy và efficiency. EfficientNet-B0 đạt 77.3% top-1 accuracy trên ImageNet với chỉ 5.3M tham số, so với ResNet-50 đạt 76% với 26M tham số. EfficientNet-B7 đạt 84.3% accuracy - state-of-the-art vào thời điểm công bố - với 66M tham số, ít hơn nhiều so với các phương pháp khác đạt accuracy tương đương.

Đối với các bài toán viễn thám, EfficientNet là lựa chọn hấp dẫn do khả năng xử lý ảnh có độ phân giải cao (compound scaling bao gồm resolution) và efficiency cho phép triển khai trên các thiết bị edge hoặc xử lý khối lượng ảnh lớn.

## 2.14. Vision Transformer (ViT): Transformer cho Computer Vision

### 2.14.1. Từ NLP sang Vision

Transformer architecture, ban đầu được phát triển cho xử lý ngôn ngữ tự nhiên (NLP), đạt thành công vượt bậc với các mô hình như BERT và GPT. Vision Transformer (ViT), được đề xuất bởi Dosovitskiy và cộng sự năm 2020, là nỗ lực đầu tiên áp dụng kiến trúc Transformer thuần túy (không dùng convolution) cho image classification và đạt kết quả cạnh tranh với các CNN state-of-the-art.

### 2.14.2. Kiến trúc ViT

**Patch Embedding:** Thay vì xử lý từng pixel, ViT chia ảnh đầu vào H×W×C thành các patch không chồng lấp, mỗi patch có kích thước P×P (thường P=16 hoặc 32). Mỗi patch được flatten thành vector và ánh xạ qua linear projection thành embedding vector có chiều D. Với ảnh 224×224 và patch size 16, ta có (224/16)² = 196 patches.

**Position Embedding:** Không giống CNN có inductive bias về cấu trúc không gian, Transformer cần được cung cấp thông tin vị trí tường minh. ViT thêm learnable position embedding vào mỗi patch embedding để mã hóa vị trí 2D của patch trong ảnh gốc.

**Transformer Encoder:** Chuỗi patch embeddings (cộng với một [CLS] token đặc biệt ở đầu) được đưa qua L layers của Transformer encoder. Mỗi layer gồm: Multi-Head Self-Attention (MSA) cho phép mỗi patch "attend" tới tất cả các patches khác, và Feed-Forward Network (FFN) áp dụng MLP riêng biệt cho từng patch. Layer Normalization và residual connections được sử dụng.

**Classification Head:** Output của [CLS] token từ layer cuối cùng được đưa qua một MLP head để tạo ra class probabilities.

### 2.14.3. Ưu và Nhược điểm của ViT

**Ưu điểm:**
- **Long-range dependencies:** Self-attention cho phép mô hình hóa mối quan hệ giữa các vùng xa nhau trong ảnh ngay từ layer đầu tiên, khác với CNN cần xếp chồng nhiều layers để mở rộng receptive field.
- **Scaling tốt với dữ liệu:** Khi được pre-train trên datasets cực lớn (hàng trăm triệu ảnh như JFT-300M), ViT vượt trội hơn CNN. Transformer architecture có khả năng scaling tốt hơn với cả model size và data size.
- **Transfer learning hiệu quả:** Pre-trained ViT weights chuyển giao tốt sang các downstream tasks.

**Nhược điểm:**
- **Cần dữ liệu lớn:** Không có inductive bias như CNN (locality, translation invariance), ViT cần lượng lớn dữ liệu training. Khi train từ đầu trên ImageNet-1K, ViT không tốt bằng ResNet. Chỉ khi pre-train trên datasets lớn hơn nhiều, ViT mới phát huy ưu thế.
- **Độ phức tạp O(n²):** Self-attention có độ phức tạp bậc hai theo số patches, không hiệu quả cho ảnh độ phân giải cao hoặc dense prediction tasks.
- **Thiếu multi-scale features:** ViT output single-scale feature map, không phù hợp cho object detection và segmentation yêu cầu multi-scale representations.

## 2.15. Swin Transformer: Hierarchical Vision Transformer

### 2.15.1. Khắc phục Hạn chế của ViT

Swin Transformer (Shifted Window Transformer), được đề xuất bởi Liu và cộng sự năm 2021, giải quyết các hạn chế của ViT để trở thành backbone general-purpose cho nhiều vision tasks. Hai cải tiến chính là hierarchical architecture và window-based self-attention.

### 2.15.2. Hierarchical Feature Maps

Giống như CNN, Swin Transformer tạo ra feature map phân cấp với độ phân giải giảm dần. Ảnh đầu vào được chia thành các patch không chồng lấp (thường 4×4 pixel mỗi patch), sau đó qua các stage với patch merging giảm resolution 2× và tăng số kênh 2× tại mỗi stage. Kết quả là feature maps với tỷ lệ 1/4, 1/8, 1/16, 1/32 so với ảnh gốc - tương tự như output của các stage trong ResNet.

### 2.15.3. Window-based Self-Attention

Thay vì tính global self-attention trên toàn bộ ảnh (như ViT), Swin Transformer chia feature map thành các window không chồng lấp (thường 7×7 patch mỗi window) và tính self-attention trong từng window. Điều này giảm độ phức tạp từ O(n²) xuống O(n × M²) với M là kích thước window, cho phép xử lý ảnh độ phân giải cao.

Để tạo kết nối giữa các window, Swin Transformer sử dụng shifted window partitioning: trong các lớp xen kẽ, các window được dịch chuyển (window_size/2, window_size/2) pixel. Điều này cho phép thông tin được trao đổi giữa các window liền kề qua các lớp liên tiếp, mở rộng receptive field mà không tăng chi phí tính toán.

### 2.15.4. Ứng dụng trong Viễn thám

Swin Transformer và các biến thể như Swin-V2 đã chứng minh hiệu quả vượt trội trong nhiều bài toán viễn thám. Khả năng mô hình hóa long-range dependencies thông qua self-attention đặc biệt hữu ích cho các bài toán yêu cầu ngữ cảnh rộng như phân loại scene, change detection, và segmentation các đối tượng lớn. TorchGeo cung cấp Swin Transformer pre-trained trên ảnh NAIP, cho phép transfer learning hiệu quả cho các bài toán viễn thám.

## 2.16. Self-Supervised Pre-training

### 2.16.1. Contrastive Learning: MoCo

Momentum Contrast (MoCo) là phương pháp self-supervised learning sử dụng contrastive loss. Ý tưởng cốt lõi là học representations bằng cách maximize agreement giữa các augmented views khác nhau của cùng một ảnh (positive pairs) trong khi minimize agreement giữa các ảnh khác nhau (negative pairs). MoCo v2 và v3 đã được áp dụng thành công cho pre-training backbone trên ảnh vệ tinh.

### 2.16.2. Masked Image Modeling: MAE

Masked Autoencoder (MAE) áp dụng ý tưởng masked language modeling (từ BERT) cho vision. Một tỷ lệ lớn patches (75%) được mask ngẫu nhiên, và mô hình học cách reconstruct các patches bị mask từ các patches visible. MAE đặc biệt hiệu quả cho ViT và đã được TorchGeo sử dụng để pre-train trên Sentinel-2 và Landsat.

**SatMAE** là biến thể của MAE được thiết kế đặc biệt cho ảnh vệ tinh đa phổ, xử lý hiệu quả việc pre-train trên 13 kênh của Sentinel-2.

### 2.16.3. SSL4EO: Self-Supervised Learning for Earth Observation

SSL4EO là framework pre-training tổng hợp cho Earth Observation, kết hợp các kỹ thuật self-supervised learning hiện đại với đặc thù của ảnh vệ tinh. Framework cung cấp pre-trained weights cho nhiều backbones trên datasets lớn như Million-AID, giúp cải thiện đáng kể performance cho các downstream tasks viễn thám với limited labeled data.

## 2.17. So sánh Backbone: CNN vs Transformer

### 2.17.1. Bảng So sánh Performance

| Backbone | Params | Top-1 Acc | FLOPs | Đặc điểm chính |
|----------|--------|-----------|-------|----------------|
| VGG-16 | 138M | 71.5% | 15.5G | Đơn giản, dễ hiểu, nhiều params |
| ResNet-50 | 26M | 76.1% | 4.1G | Skip connection, robust, phổ biến |
| ResNet-101 | 45M | 77.4% | 7.9G | Deeper ResNet, accuracy cao hơn |
| EfficientNet-B0 | 5.3M | 77.3% | 0.4G | Nhỏ gọn, hiệu quả |
| EfficientNet-B4 | 19M | 82.9% | 4.2G | Cân bằng accuracy/efficiency |
| ViT-B/16 | 86M | 77.9%* | 17.6G | Transformer, cần dữ liệu lớn (*khi pre-train trên ImageNet-21K) |
| Swin-T | 29M | 81.3% | 4.5G | Hierarchical Transformer, efficient |
| Swin-B | 88M | 83.5% | 15.4G | Larger Swin, SOTA accuracy |

### 2.17.2. So sánh CNN và Transformer cho Viễn thám

| Khía cạnh | CNN (ResNet, EfficientNet) | Transformer (ViT, Swin) |
|-----------|---------------------------|-------------------------|
| **Inductive bias** | Locality, translation invariance | Minimal (học từ data) |
| **Yêu cầu dữ liệu** | Thấp - tốt với limited data | Cao - cần pre-training hoặc large dataset |
| **Receptive field** | Tăng dần qua layers | Global từ đầu (ViT) hoặc hierarchical (Swin) |
| **Multi-scale** | Tự nhiên với pooling/stride | Cần thiết kế đặc biệt (Swin) |
| **Long-range dependencies** | Khó (cần deep network) | Dễ (self-attention) |
| **Hiệu quả tính toán** | Cao, phù hợp edge devices | Trung bình đến thấp (ViT O(n²)) |
| **Transfer learning** | ImageNet → RS: tốt | ImageNet → RS: tốt, SSL (MAE) → RS: rất tốt |
| **Ảnh SAR** | ResNet pre-trained on Sentinel-1 | ViT/Swin với SatMAE |
| **Ảnh đa phổ** | Conv đầu mở rộng cho C kênh | Patch embedding tự nhiên hỗ trợ C kênh |

### 2.17.3. Khuyến nghị cho Viễn thám

**Dùng CNN (ResNet, EfficientNet) khi:**
- Dữ liệu training hạn chế (< 10K ảnh)
- Cần tốc độ inference nhanh hoặc deploy trên edge devices
- Task yêu cầu multi-scale features rõ ràng (object detection, segmentation)
- Có pre-trained weights chất lượng trên domain tương tự (ResNet-50 on Sentinel-1)

**Dùng Transformer (ViT, Swin) khi:**
- Có lượng lớn dữ liệu hoặc sử dụng self-supervised pre-training (MAE, MoCo)
- Task cần model long-range spatial dependencies (scene classification, change detection trên vùng rộng)
- Có tài nguyên tính toán đủ mạnh
- Muốn sử dụng state-of-the-art pre-trained weights từ SatMAE hoặc SSL4EO

**Khuyến nghị cụ thể theo bài toán:**

*Phát hiện tàu biển (Ship Detection):*
- Thời gian thực: EfficientNet-B0/B1 + YOLO head
- Accuracy cao: Swin-T + Faster R-CNN hoặc ResNet-101 + Cascade R-CNN
- Cân bằng: ResNet-50 + RetinaNet

*Phát hiện dầu loang (Oil Spill Segmentation):*
- Binary segmentation: ResNet-50 + U-Net hoặc DeepLabV3+
- Phân biệt oil/look-alike phức tạp: Swin-T + DeepLabV3+ (benefit từ long-range context)
- Limited data: ResNet-50 pre-trained on Sentinel-1 + U-Net với extensive data augmentation

*Scene classification viễn thám:*
- ViT hoặc Swin pre-trained với SSL (MAE, SSL4EO) thường cho kết quả tốt nhất
- ResNet-50/101 vẫn competitive và nhanh hơn

**TorchGeo Pre-trained Weights:**
- ResNet-50: Sentinel-1 SAR, Sentinel-2 multispectral
- ViT: Sentinel-2 với SatMAE
- Swin-T: NAIP aerial imagery
- Prithvi: Foundation model từ IBM/NASA cho Sentinel-2
