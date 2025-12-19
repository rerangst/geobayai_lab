# Chương 2: Phân đoạn Ngữ nghĩa (Semantic Segmentation) trong Viễn thám

## 3.12. Định nghĩa Bài toán Semantic Segmentation

Semantic Segmentation là bài toán gán nhãn lớp cho từng pixel trong ảnh, tạo ra một dense prediction map có cùng kích thước không gian với ảnh đầu vào. Khác với image classification gán một nhãn cho toàn bộ ảnh, và object detection định vị đối tượng bằng bounding box, semantic segmentation cung cấp thông tin chi tiết nhất về ranh giới và hình dạng của các vùng trong ảnh. Mỗi pixel được gán vào một trong K lớp được định nghĩa trước, tạo thành một segmentation mask.

Về mặt hình thức, cho ảnh đầu vào X có kích thước H×W×C, output của semantic segmentation là một label map Y có kích thước H×W, trong đó Y[i,j] ∈ {1, 2, ..., K} là nhãn lớp của pixel tại vị trí (i,j). Trong quá trình inference, mạng thường output một tensor có kích thước H×W×K chứa xác suất của mỗi pixel thuộc mỗi lớp, và nhãn cuối cùng được xác định bằng argmax theo chiều lớp.

Một đặc điểm quan trọng của semantic segmentation là không phân biệt các instance riêng lẻ của cùng một lớp. Nếu có hai con tàu trong ảnh, cả hai đều được gán nhãn "tàu" và thuộc cùng một vùng connected component (nếu chúng chạm nhau). Việc phân biệt từng instance là nhiệm vụ của instance segmentation, sẽ được trình bày ở phần sau.

## 3.13. Ứng dụng Segmentation trong Viễn thám

### 3.13.1. Phát hiện và Phân vùng Dầu loang (Oil Spill Segmentation)

Oil spill segmentation là một trong những ứng dụng quan trọng nhất của semantic segmentation trong giám sát biển. Bài toán yêu cầu phân loại từng pixel của ảnh SAR thành các lớp: oil spill, look-alike, water, land, và đôi khi ship. Output là mask chính xác về vị trí, hình dạng, và diện tích của vết dầu loang, thông tin quan trọng cho việc ước tính thiệt hại và lập kế hoạch ứng phó.

Ưu điểm của segmentation so với detection (bounding box) cho oil spill là khả năng capture hình dạng bất định của vết dầu. Dầu loang thường có hình dạng phức tạp, kéo dài theo hướng gió và dòng chảy, không phù hợp với bounding box hình chữ nhật. Segmentation mask cho phép tính toán chính xác diện tích và volume ước tính của dầu tràn.

Các mô hình như U-Net, DeepLabV3+, và các biến thể với attention mechanism đã đạt được IoU trên 90% cho oil spill segmentation trên các benchmark datasets.

### 3.13.2. Lập bản đồ Ngập lụt (Flood Mapping)

Flood mapping là ứng dụng quan trọng trong quản lý thiên tai, sử dụng ảnh SAR để phát hiện và phân vùng các khu vực bị ngập. SAR được ưu tiên do khả năng chụp xuyên mây, đặc biệt quan trọng trong các sự kiện thời tiết cực đoan khi ảnh quang học không khả dụng.

Bài toán thường được formulate như binary segmentation (flooded vs non-flooded) hoặc multi-class segmentation (permanent water, flood water, land, urban). Change detection giữa ảnh pre-event và post-event cũng được sử dụng để xác định vùng mới bị ngập.

### 3.13.3. Trích xuất Building Footprint

Building footprint extraction là bài toán phân vùng các tòa nhà từ ảnh vệ tinh hoặc ảnh máy bay độ phân giải cao. Output là binary mask (building vs background) hoặc multi-class mask phân biệt các loại công trình. Ứng dụng bao gồm: cập nhật bản đồ đô thị, ước tính dân số, đánh giá thiệt hại sau thiên tai, và quy hoạch đô thị.

Thách thức đặc thù của building segmentation bao gồm: sự đa dạng về kích thước và hình dạng công trình, shadows của các tòa nhà cao tầng, occlusion bởi cây xanh, và sự nhầm lẫn với các bề mặt tương tự như bãi đỗ xe và sân thể thao.

### 3.13.4. Phân loại Lớp phủ Mặt đất (Land Cover Mapping)

Land cover segmentation là phiên bản pixel-level của land cover classification, gán mỗi pixel vào một lớp lớp phủ mặt đất. Các lớp phổ biến bao gồm: rừng, đất nông nghiệp, đô thị, mặt nước, đất trống, và các phân lớp chi tiết hơn. Kết quả là bản đồ land cover chi tiết phục vụ cho nhiều ứng dụng về môi trường và quy hoạch.

Ảnh đa phổ từ Sentinel-2 với 13 kênh cung cấp thông tin phong phú cho land cover segmentation. Các chỉ số phổ như NDVI (Normalized Difference Vegetation Index) có thể được tính và sử dụng như input bổ sung hoặc feature engineering.

## 3.14. Kiến trúc Encoder-Decoder

### 3.14.1. Nguyên lý Chung

Hầu hết các kiến trúc semantic segmentation hiện đại tuân theo paradigm encoder-decoder. Encoder (thường là một backbone như ResNet) nhận ảnh đầu vào và tạo ra feature map có độ phân giải thấp nhưng giàu thông tin semantic. Decoder nhận feature map này và dần dần upscale về kích thước gốc, tạo ra dense prediction.

Quá trình encoding giảm độ phân giải không gian qua các lớp pooling hoặc strided convolution (thường giảm 16× hoặc 32× so với ảnh gốc) nhưng tăng số kênh để capture thông tin semantic phong phú. Quá trình decoding sử dụng upsampling (bilinear interpolation, transposed convolution, hoặc pixel shuffle) để khôi phục độ phân giải không gian.

Một thách thức quan trọng là thông tin spatial chi tiết (như biên đối tượng) bị mất trong quá trình encoding. Các kiến trúc khác nhau có các cơ chế khác nhau để bảo toàn và khôi phục thông tin này, điển hình là skip connections.

### 3.14.2. U-Net

U-Net là kiến trúc encoder-decoder kinh điển, ban đầu được thiết kế cho biomedical image segmentation nhưng đã trở thành baseline phổ biến cho nhiều bài toán segmentation, bao gồm viễn thám.

**Kiến trúc:** U-Net có cấu trúc đối xứng hình chữ U với encoder bên trái và decoder bên phải. Encoder gồm các block convolution theo sau bởi max pooling 2×2, giảm resolution 2× tại mỗi level (thường 4-5 levels). Decoder gồm các block upsampling (transposed convolution 2×2) theo sau bởi convolution, tăng resolution 2× tại mỗi level.

**Skip Connections:** Đặc trưng quan trọng nhất của U-Net là skip connections nối trực tiếp từ encoder sang decoder tại mỗi level có cùng resolution. Feature map từ encoder được concatenate với feature map từ decoder trước khi đưa vào các lớp convolution tiếp theo. Skip connections cho phép decoder truy cập trực tiếp vào các low-level features (như cạnh và texture) từ encoder, cải thiện đáng kể độ chính xác của biên đối tượng trong output.

**Ưu điểm cho viễn thám:** U-Net đặc biệt phù hợp với các bài toán viễn thám có ít dữ liệu training do kiến trúc hiệu quả và khả năng học tốt với limited data. Data augmentation extensive được áp dụng trong paper gốc, một practice vẫn quan trọng cho viễn thám.

### 3.14.3. DeepLabV3+

DeepLabV3+ là state-of-the-art semantic segmentation model được phát triển bởi Google, kết hợp Atrous Spatial Pyramid Pooling (ASPP) với encoder-decoder structure.

**Atrous Convolution (Dilated Convolution):** Thay vì convolution thông thường, atrous convolution chèn "lỗ hổng" (zeros) giữa các weights của kernel, hiệu quả tăng receptive field mà không tăng số lượng parameters hoặc giảm resolution. Atrous convolution với rate r tương đương với việc upscale kernel từ k×k lên (k + (k-1)×(r-1))×(...).

**ASPP Module:** ASPP áp dụng song song nhiều atrous convolution với các rate khác nhau (ví dụ 1, 6, 12, 18) trên cùng feature map, sau đó concatenate kết quả. Điều này cho phép capture multi-scale context, quan trọng cho việc segmentation các đối tượng có kích thước đa dạng. Một branch pooling toàn cục cũng được thêm vào để capture image-level context.

**Encoder-Decoder với ASPP:** DeepLabV3+ kết hợp ASPP với decoder module. Low-level features từ encoder (thường từ output của stage sớm trong backbone) được concatenate với upsampled output của ASPP, sau đó đi qua các lớp convolution và upsampling cuối cùng để tạo ra full-resolution prediction.

**Backbone:** DeepLabV3+ thường sử dụng modified ResNet hoặc Xception làm backbone. Các lớp pooling cuối được thay bằng atrous convolution để duy trì resolution cao hơn trong encoder (output stride 8 hoặc 16 thay vì 32).

### 3.14.4. FPN cho Segmentation

Feature Pyramid Network (FPN), ban đầu được thiết kế cho object detection, cũng được áp dụng hiệu quả cho semantic segmentation. Kiến trúc xây dựng multi-scale feature maps với cả low-level detail và high-level semantics tại mọi scale.

**Panoptic FPN:** Là extension của FPN cho segmentation, thêm một lightweight semantic segmentation branch lên trên FPN features. Branch này upsample và merge predictions từ tất cả các levels của pyramid.

**FPN trong TorchGeo:** TorchGeo cung cấp FPN-based segmentation models với various backbones, phù hợp cho các bài toán viễn thám đòi hỏi xử lý đối tượng đa tỷ lệ.

### 3.14.5. Các Kiến trúc Khác

**PSPNet (Pyramid Scene Parsing Network):** Sử dụng Pyramid Pooling Module với 4 scale khác nhau (1×1, 2×2, 3×3, 6×6 pooling) để aggregate global context trước khi upsampling.

**HRNet (High-Resolution Network):** Duy trì high-resolution representations throughout network thay vì giảm rồi khôi phục resolution như encoder-decoder truyền thống. Các parallel branches với resolutions khác nhau liên tục trao đổi thông tin.

**SegFormer:** Kết hợp hierarchical Transformer encoder với lightweight MLP decoder, đạt state-of-the-art trên nhiều benchmarks.

## 3.15. Loss Function cho Segmentation

### 3.15.1. Pixel-wise Cross-Entropy

Áp dụng Cross-Entropy loss cho từng pixel độc lập:

L = -(1/N) × Σᵢ Σⱼ Σₖ yᵢⱼₖ × log(pᵢⱼₖ)

trong đó N là tổng số pixels, yᵢⱼₖ là ground truth (1 nếu pixel (i,j) thuộc lớp k, 0 nếu không), và pᵢⱼₖ là predicted probability.

**Class-weighted Cross-Entropy:** Để xử lý class imbalance (ví dụ background nhiều hơn nhiều so với oil spill), weights được gán cho mỗi lớp, thường inversely proportional với class frequency:

L = -(1/N) × Σᵢ Σⱼ Σₖ wₖ × yᵢⱼₖ × log(pᵢⱼₖ)

### 3.15.2. Dice Loss

Dice Loss dựa trên Dice coefficient (hay F1-Score), đo overlap giữa prediction và ground truth:

Dice = 2 × |P ∩ G| / (|P| + |G|)
Dice Loss = 1 - Dice

Đối với soft prediction (probabilities thay vì hard labels):

Dice = 2 × Σ pᵢ × gᵢ / (Σ pᵢ + Σ gᵢ)

Dice Loss ít bị ảnh hưởng bởi class imbalance hơn Cross-Entropy do nó đo ratio thay vì absolute numbers. Dice Loss thường được sử dụng cho binary segmentation hoặc tính riêng cho mỗi lớp trong multi-class setting.

### 3.15.3. Focal Loss cho Segmentation

Áp dụng Focal Loss cho từng pixel để down-weight easy pixels và focus vào hard pixels:

FL = -(1/N) × Σᵢ Σⱼ αc × (1 - pᵢⱼc)^γ × log(pᵢⱼc)

trong đó c là ground truth class của pixel (i,j). Focal Loss đặc biệt hữu ích khi có nhiều easy background pixels.

### 3.15.4. Combined Loss

Nhiều paper sử dụng combination của multiple losses:

L = λ₁ × LCE + λ₂ × LDice + λ₃ × LFocal

với các weights λ được tune như hyperparameters. Combination thường cho kết quả tốt hơn single loss do mỗi loss có điểm mạnh riêng.

### 3.15.5. Boundary Loss

Để cải thiện segmentation của object boundaries, boundary-aware losses được sử dụng:

**Binary Cross-Entropy on Boundaries:** Tăng weight cho pixels gần boundary của ground truth.

**Active Contour Loss:** Dựa trên active contour model, encourage smooth boundaries.

## 3.16. Metrics Đánh giá Segmentation

### 3.16.1. Pixel Accuracy

Pixel Accuracy là tỷ lệ pixels được phân loại đúng:

Pixel Accuracy = Σᵢ nᵢᵢ / Σᵢ Σⱼ nᵢⱼ

trong đó nᵢⱼ là số pixels thuộc lớp i được dự đoán là lớp j. Đây là metric đơn giản nhưng bị dominate bởi các lớp lớn (như background).

### 3.16.2. Intersection over Union (IoU) và Mean IoU

IoU cho mỗi lớp k:

IoU_k = nₖₖ / (Σⱼ nₖⱼ + Σᵢ nᵢₖ - nₖₖ)

= TP / (TP + FP + FN)

Mean IoU (mIoU) là trung bình IoU trên tất cả các lớp:

mIoU = (1/K) × Σₖ IoU_k

mIoU là metric tiêu chuẩn nhất cho semantic segmentation, coi tất cả các lớp có tầm quan trọng như nhau bất kể kích thước.

### 3.16.3. Dice Coefficient (F1-Score)

Dice coefficient tương đương F1-Score cho binary segmentation:

Dice = 2TP / (2TP + FP + FN) = 2 × |P ∩ G| / (|P| + |G|)

Dice và IoU có mối quan hệ: Dice = 2×IoU / (1 + IoU), nên ranking bởi Dice và IoU thường tương đồng.

### 3.16.4. Boundary Metrics

Để đánh giá chất lượng segmentation tại boundaries:

**Boundary F1 (BF1):** Precision và Recall được tính chỉ cho pixels gần boundary (trong một khoảng cách threshold).

**Hausdorff Distance:** Đo khoảng cách lớn nhất từ điểm trên boundary prediction đến boundary ground truth.
