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

## 2.14. Swin Transformer: Vision Transformer Phân cấp

### 2.14.1. Từ ViT đến Swin Transformer

Vision Transformer (ViT) năm 2020 đã chứng minh rằng kiến trúc Transformer thuần túy có thể đạt được kết quả cạnh tranh với CNN trên các bài toán thị giác máy tính. Tuy nhiên, ViT có một số hạn chế cho các bài toán dense prediction như object detection và segmentation: độ phức tạp tính toán O(n²) với số lượng token n khiến việc xử lý ảnh độ phân giải cao trở nên không khả thi, và thiếu tính phân cấp đa tỷ lệ như CNN.

Swin Transformer (Shifted Window Transformer) được đề xuất để khắc phục những hạn chế này, trở thành backbone phổ biến cho nhiều bài toán thị giác máy tính hiện đại.

### 2.14.2. Hierarchical Feature Maps

Giống như CNN, Swin Transformer tạo ra feature map phân cấp với độ phân giải giảm dần. Ảnh đầu vào được chia thành các patch không chồng lấp (thường 4×4 pixel mỗi patch), sau đó qua các stage với patch merging giảm resolution 2× và tăng số kênh 2× tại mỗi stage. Kết quả là feature maps với tỷ lệ 1/4, 1/8, 1/16, 1/32 so với ảnh gốc - tương tự như output của các stage trong ResNet.

### 2.14.3. Window-based Self-Attention

Thay vì tính global self-attention trên toàn bộ ảnh (như ViT), Swin Transformer chia feature map thành các window không chồng lấp (thường 7×7 patch mỗi window) và tính self-attention trong từng window. Điều này giảm độ phức tạp từ O(n²) xuống O(n × M²) với M là kích thước window, cho phép xử lý ảnh độ phân giải cao.

Để tạo kết nối giữa các window, Swin Transformer sử dụng shifted window partitioning: trong các lớp xen kẽ, các window được dịch chuyển (window_size/2, window_size/2) pixel. Điều này cho phép thông tin được trao đổi giữa các window liền kề qua các lớp liên tiếp, mở rộng receptive field mà không tăng chi phí tính toán.

### 2.14.4. Ứng dụng trong Viễn thám

Swin Transformer và các biến thể như Swin-V2 đã chứng minh hiệu quả vượt trội trong nhiều bài toán viễn thám. Khả năng mô hình hóa long-range dependencies thông qua self-attention đặc biệt hữu ích cho các bài toán yêu cầu ngữ cảnh rộng như phân loại scene, change detection, và segmentation các đối tượng lớn. TorchGeo cung cấp Swin Transformer pre-trained trên ảnh NAIP, cho phép transfer learning hiệu quả cho các bài toán viễn thám.

## 2.15. So sánh và Lựa chọn Backbone

### 2.15.1. Bảng So sánh

| Backbone | Params | Top-1 Acc | FLOPs | Đặc điểm chính |
|----------|--------|-----------|-------|----------------|
| VGG-16 | 138M | 71.5% | 15.5G | Đơn giản, dễ hiểu, nhiều params |
| ResNet-50 | 26M | 76.1% | 4.1G | Skip connection, robust, phổ biến |
| ResNet-101 | 45M | 77.4% | 7.9G | Deeper ResNet, accuracy cao hơn |
| EfficientNet-B0 | 5.3M | 77.3% | 0.4G | Nhỏ gọn, hiệu quả |
| EfficientNet-B4 | 19M | 82.9% | 4.2G | Cân bằng accuracy/efficiency |
| Swin-T | 29M | 81.3% | 4.5G | Transformer, attention mechanism |
| Swin-B | 88M | 83.5% | 15.4G | Larger Swin, SOTA accuracy |

### 2.15.2. Khuyến nghị cho Viễn thám

Đối với bài toán phát hiện tàu biển (Ship Detection):
- **Thời gian thực, edge deployment:** EfficientNet-B0/B1 hoặc MobileNetV3
- **Accuracy cao, không giới hạn tài nguyên:** ResNet-101 hoặc Swin-T/Swin-B
- **Cân bằng accuracy/speed:** ResNet-50 hoặc EfficientNet-B3

Đối với bài toán phát hiện dầu loang (Oil Spill Detection):
- **Segmentation accuracy cao:** ResNet-101 làm encoder cho U-Net/DeepLabV3+
- **Multi-scale features quan trọng:** EfficientNet với FPN
- **Long-range context cần thiết:** Swin Transformer

Trong TorchGeo, các pre-trained weights cho Sentinel-1 SAR có sẵn với backbone ResNet-50 và ViT, phù hợp cho cả ship detection và oil spill detection trên ảnh SAR.
