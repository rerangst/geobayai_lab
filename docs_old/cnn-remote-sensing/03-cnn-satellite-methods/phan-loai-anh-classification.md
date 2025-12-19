# Phân loại Ảnh (Image Classification) trong Viễn thám

## 3.1. Định nghĩa Bài toán Classification

Image Classification hay phân loại ảnh là bài toán cơ bản và quan trọng nhất trong thị giác máy tính, đóng vai trò nền tảng cho nhiều bài toán phức tạp hơn như object detection và semantic segmentation. Trong bài toán classification, mục tiêu là gán một hoặc nhiều nhãn lớp (class label) cho toàn bộ ảnh đầu vào hoặc một vùng ảnh (patch/tile). Đây là bài toán học có giám sát (supervised learning), trong đó mô hình được huấn luyện trên tập dữ liệu đã được gán nhãn để học cách phân biệt các lớp khác nhau.

Về mặt hình thức, bài toán classification có thể được phát biểu như sau: cho ảnh đầu vào X có kích thước H×W×C (chiều cao × chiều rộng × số kênh), tìm hàm f: X → Y ánh xạ ảnh sang một trong K lớp được định nghĩa trước Y = {1, 2, ..., K}. Trong thực tế, output của mô hình thường là một vector xác suất p = (p₁, p₂, ..., pₖ) với pᵢ là xác suất ảnh thuộc lớp i và Σpᵢ = 1. Nhãn dự đoán cuối cùng thường được chọn là lớp có xác suất cao nhất: ŷ = argmax(pᵢ).

Bài toán classification được chia thành hai loại chính dựa trên số lượng nhãn có thể gán cho mỗi ảnh. Trong single-label classification (phân loại đơn nhãn), mỗi ảnh chỉ thuộc một lớp duy nhất, và các lớp là mutually exclusive. Ví dụ, trong phân loại scene viễn thám, một patch ảnh có thể được phân loại là "cảng biển", "khu công nghiệp", hoặc "vùng nông nghiệp", nhưng không thể đồng thời thuộc nhiều lớp. Trong multi-label classification (phân loại đa nhãn), mỗi ảnh có thể thuộc nhiều lớp đồng thời. Ví dụ, một ảnh vệ tinh có thể đồng thời chứa "tòa nhà", "cây xanh", và "đường giao thông".

## 3.2. Ứng dụng Classification trong Viễn thám

### 3.2.1. Phân loại Lớp phủ Mặt đất (Land Cover Classification)

Land cover classification là một trong những ứng dụng phổ biến nhất của classification trong viễn thám, nhằm phân loại từng pixel hoặc vùng ảnh vào các lớp lớp phủ mặt đất như rừng, nước, đô thị, đất nông nghiệp, và đất trống. Thông tin land cover có vai trò quan trọng trong quy hoạch đô thị, quản lý tài nguyên thiên nhiên, giám sát môi trường, và nghiên cứu biến đổi khí hậu.

Với cách tiếp cận deep learning, ảnh vệ tinh được chia thành các patch/tile có kích thước cố định (ví dụ 64×64 hoặc 256×256 pixel), và mỗi patch được phân loại vào một lớp land cover dominant. Các backbone như ResNet và EfficientNet được sử dụng rộng rãi cho bài toán này. Đặc biệt, việc sử dụng ảnh đa phổ (multispectral) từ các vệ tinh như Sentinel-2 với 13 kênh phổ mang lại nhiều thông tin hữu ích cho việc phân biệt các loại thực vật và bề mặt khác nhau.

### 3.2.2. Phân loại Cảnh quan (Scene Classification)

Scene classification trong viễn thám hướng đến việc phân loại toàn bộ ảnh hoặc một tile lớn vào các danh mục cảnh quan có ý nghĩa semantic như "sân bay", "bến cảng", "khu dân cư", "khu công nghiệp", "vùng nông nghiệp", "rừng", "sa mạc", v.v. Khác với land cover classification tập trung vào đặc điểm vật lý của bề mặt, scene classification cố gắng nắm bắt chức năng và ngữ cảnh tổng thể của vùng được chụp.

Bài toán scene classification đặt ra thách thức lớn hơn do cần mô hình hóa các mối quan hệ không gian phức tạp giữa các đối tượng trong scene. Một "sân bay" không chỉ đơn thuần là sự hiện diện của đường băng mà còn bao gồm nhà ga, bãi đỗ máy bay, đường lăn, và bố cục không gian đặc trưng giữa chúng. Các kiến trúc CNN với Global Average Pooling và các cơ chế attention đã chứng minh hiệu quả trong việc học các đặc trưng discriminative cho scene classification.

Các dataset benchmark phổ biến cho scene classification viễn thám bao gồm UC Merced Land Use (21 lớp, 2,100 ảnh), AID (30 lớp, 10,000 ảnh), NWPU-RESISC45 (45 lớp, 31,500 ảnh), và EuroSAT (10 lớp, 27,000 ảnh từ Sentinel-2). Các mô hình state-of-the-art đạt accuracy trên 95% trên nhiều benchmark này.

### 3.2.3. Phân loại Loại Tàu (Vessel Type Classification)

Trong bài toán ship detection, sau khi phát hiện và crop vùng chứa tàu, bước tiếp theo thường là phân loại loại tàu. Các lớp phổ biến bao gồm: tàu container, tàu dầu (oil tanker), tàu hàng rời (bulk carrier), tàu đánh cá (fishing vessel), tàu khách/du lịch (passenger/cruise ship), tàu kéo (tugboat), tàu quân sự, và giàn khoan dầu (offshore platform).

Việc phân loại loại tàu có ý nghĩa quan trọng trong giám sát hàng hải và phát hiện hoạt động đánh bắt bất hợp pháp (IUU fishing). Thông qua việc kết hợp phân loại từ ảnh vệ tinh với dữ liệu AIS (Automatic Identification System), có thể phát hiện các tàu "dark" - tàu tắt hệ thống nhận dạng để tránh bị theo dõi, thường liên quan đến hoạt động bất hợp pháp.

Classification tàu biển gặp thách thức từ sự đa dạng về góc nhìn (tàu có thể xuất hiện theo nhiều hướng), điều kiện thời tiết và ánh sáng khác nhau, độ phân giải ảnh hạn chế (đặc biệt với ảnh SAR), và sự tương đồng giữa một số loại tàu. Data augmentation với rotation, flip, và color jittering là các kỹ thuật quan trọng để tăng tính robust của mô hình.

### 3.2.4. Phân loại Dầu và Look-alike (Oil vs Look-alike Classification)

Trong bài toán oil spill detection, một thách thức quan trọng là phân biệt vết dầu thực sự với các hiện tượng trông giống dầu trên ảnh SAR, gọi chung là "look-alikes". Các look-alike bao gồm: vùng biển yên lặng do gió thấp, vết tàu đi qua, bloom tảo, film sinh học tự nhiên, và các hiện tượng oceanographic khác. Tất cả đều tạo ra các vùng tối (dark patch) trên ảnh SAR tương tự như dầu.

Bài toán classification oil vs look-alike thường được thực hiện như bước post-processing sau khi đã phát hiện các dark patch trên ảnh SAR. Input là patch ảnh chứa dark spot đã được crop, và output là xác suất đó là oil spill hay look-alike. Các đặc trưng quan trọng bao gồm: hình dạng và biên của dark patch (dầu thường có biên mềm mại hơn), kết cấu bên trong, contrast với vùng xung quanh, và thông tin ngữ cảnh như vị trí tuyến hàng hải và khoảng cách đến bờ.

## 3.3. Kiến trúc CNN cho Classification

### 3.3.1. Cấu trúc Chung

Một mạng CNN điển hình cho image classification bao gồm hai phần chính: feature extractor (backbone) và classifier head. Feature extractor bao gồm các lớp convolution, pooling, và normalization để biến đổi ảnh đầu vào thành các feature map chứa thông tin semantic. Classifier head nhận feature map cuối cùng và tạo ra vector xác suất cho các lớp.

Trong các kiến trúc truyền thống như VGG và AlexNet, classifier head bao gồm một hoặc nhiều lớp fully connected. Feature map cuối cùng được flatten thành vector một chiều, sau đó đưa qua các lớp fully connected với hàm kích hoạt ReLU và Dropout, kết thúc bằng lớp fully connected với số nơ-ron bằng số lớp và hàm kích hoạt Softmax.

Trong các kiến trúc hiện đại như ResNet và EfficientNet, Global Average Pooling (GAP) thay thế cho flatten và các lớp fully connected trung gian. GAP tính trung bình trên toàn bộ feature map cho mỗi kênh, tạo ra vector có số chiều bằng số kênh của feature map cuối. Sau đó, một lớp fully connected duy nhất (hoặc convolution 1×1) ánh xạ vector này sang vector output. Cách tiếp cận này giảm số lượng tham số, giảm overfitting, và cho phép mạng xử lý ảnh có kích thước bất kỳ.

### 3.3.2. Transfer Learning và Fine-tuning

Transfer learning là kỹ thuật quan trọng trong classification viễn thám, cho phép tận dụng các mô hình đã được pre-train trên các dataset lớn như ImageNet. Thay vì huấn luyện mạng từ đầu, backbone được khởi tạo với pre-trained weights, sau đó fine-tune trên dataset mục tiêu. Điều này đặc biệt hữu ích khi dataset viễn thám có kích thước nhỏ, không đủ để huấn luyện mạng sâu từ đầu.

Có nhiều chiến lược fine-tuning khác nhau. Trong feature extraction, toàn bộ backbone được đóng băng (freeze weights), chỉ huấn luyện classifier head mới. Cách này nhanh và ổn định nhưng có thể không tối ưu nếu domain viễn thám khác biệt nhiều so với ImageNet. Trong fine-tuning toàn bộ mạng, tất cả các lớp được cập nhật với learning rate thấp để tránh phá hủy pre-trained features. Một chiến lược phổ biến là sử dụng learning rate thấp hơn cho các lớp đầu (layer-wise learning rate decay).

TorchGeo cung cấp các pre-trained weights đặc biệt cho ảnh vệ tinh, huấn luyện trên Sentinel-1, Sentinel-2, và Landsat. Các weights này thường cho kết quả tốt hơn ImageNet weights cho các bài toán viễn thám do domain similarity.

### 3.3.3. Xử lý Ảnh Đa kênh (Multi-channel Input)

Ảnh vệ tinh thường có nhiều kênh hơn ảnh RGB thông thường. Sentinel-2 có 13 kênh phổ, Landsat có 7-11 kênh, và ảnh SAR từ Sentinel-1 có 2 kênh phân cực (VV và VH). Để xử lý input đa kênh, lớp convolution đầu tiên của mạng cần được điều chỉnh.

Nếu sử dụng ImageNet pre-trained weights (3 kênh RGB), có hai cách tiếp cận chính. Cách thứ nhất là chọn 3 kênh phù hợp nhất từ ảnh đa phổ (ví dụ Red, Green, Blue bands) và sử dụng trực tiếp weights. Cách thứ hai là mở rộng lớp convolution đầu tiên để nhận input C kênh: weights cho 3 kênh RGB được giữ nguyên, và weights cho các kênh mới được khởi tạo ngẫu nhiên hoặc sao chép từ kênh tương tự.

Đối với ảnh SAR 2 kênh (VV, VH), một số nghiên cứu tạo kênh thứ 3 bằng tỷ lệ VV/VH hoặc sử dụng VV cho cả 3 kênh RGB. TorchGeo cung cấp các mô hình được huấn luyện trực tiếp trên ảnh 2 kênh Sentinel-1, tránh cần các workaround này.

## 3.4. Loss Function cho Classification

### 3.4.1. Cross-Entropy Loss

Cross-Entropy Loss là hàm loss tiêu chuẩn cho bài toán classification đa lớp. Với ground truth one-hot encoding y = (y₁, y₂, ..., yₖ) và output xác suất p = (p₁, p₂, ..., pₖ), Cross-Entropy được định nghĩa:

L = -Σᵢ yᵢ × log(pᵢ)

Do y là one-hot vector với chỉ một phần tử bằng 1 (tại vị trí của lớp đúng c), công thức đơn giản thành:

L = -log(pₓ)

Cross-Entropy phạt mạnh các dự đoán confident nhưng sai (pₓ thấp cho lớp đúng), và thưởng cho các dự đoán confident đúng. Gradient của Cross-Entropy kết hợp với Softmax có dạng đẹp: ∂L/∂zₓ = pₓ - yₓ, thuận tiện cho tính toán backpropagation.

### 3.4.2. Focal Loss

Focal Loss được đề xuất bởi Lin và cộng sự (2017) để giải quyết vấn đề class imbalance nghiêm trọng trong object detection, và sau đó được áp dụng rộng rãi cho classification với dữ liệu không cân bằng. Ý tưởng là giảm trọng số của các mẫu dễ (easy examples) để mạng tập trung vào các mẫu khó (hard examples).

Focal Loss được định nghĩa:

FL = -αₓ × (1 - pₓ)^γ × log(pₓ)

trong đó γ ≥ 0 là focusing parameter (thường γ = 2) và αₓ là class-balancing weight. Khi pₓ cao (mẫu dễ, dự đoán tốt), (1 - pₓ)^γ gần 0, giảm đáng kể contribution của mẫu đó vào loss. Khi pₓ thấp (mẫu khó, dự đoán sai), (1 - pₓ)^γ gần 1, giữ nguyên loss.

Focal Loss đặc biệt hữu ích trong viễn thám khi các lớp không cân bằng, ví dụ số lượng patch "biển" nhiều hơn nhiều so với "tàu" hay "dầu loang".

### 3.4.3. Label Smoothing

Label Smoothing là kỹ thuật regularization thay đổi ground truth từ hard labels (one-hot) sang soft labels. Thay vì y = (0, 0, 1, 0, 0) cho lớp 3, label smoothing với ε = 0.1 tạo ra y = (0.02, 0.02, 0.92, 0.02, 0.02). Điều này ngăn mạng trở nên quá confident và cải thiện generalization.

Label smoothing được định nghĩa:

y'ᵢ = yᵢ × (1 - ε) + ε/K

trong đó K là số lớp và ε thường được đặt từ 0.1 đến 0.2.

## 3.5. Metrics Đánh giá Classification

### 3.5.1. Accuracy, Precision, Recall, F1-Score

**Accuracy** là tỷ lệ dự đoán đúng trên tổng số mẫu: Accuracy = (TP + TN) / (TP + TN + FP + FN). Accuracy là metric đơn giản và trực quan, nhưng có thể gây hiểu nhầm khi dữ liệu không cân bằng. Ví dụ, nếu 95% mẫu thuộc lớp A, mô hình luôn dự đoán A sẽ đạt 95% accuracy mà không học được gì.

**Precision** (độ chính xác) cho một lớp là tỷ lệ dự đoán đúng trong số các mẫu được dự đoán là lớp đó: Precision = TP / (TP + FP). Precision cao nghĩa là ít false positive, quan trọng khi chi phí của false positive cao.

**Recall** (độ nhạy, sensitivity) là tỷ lệ dự đoán đúng trong số các mẫu thực sự thuộc lớp đó: Recall = TP / (TP + FN). Recall cao nghĩa là ít false negative, quan trọng khi không muốn bỏ sót.

**F1-Score** là trung bình điều hòa của Precision và Recall: F1 = 2 × Precision × Recall / (Precision + Recall). F1-Score cân bằng giữa Precision và Recall, hữu ích khi cần một metric duy nhất đánh giá cả hai khía cạnh.

Đối với bài toán đa lớp, có nhiều cách tính trung bình các metric: Macro-average tính trung bình cộng của metric từng lớp (coi các lớp có tầm quan trọng như nhau), Micro-average tính từ tổng TP, FP, FN của tất cả các lớp (bị chi phối bởi các lớp lớn), và Weighted-average cân theo số mẫu mỗi lớp.

### 3.5.2. Confusion Matrix

Confusion Matrix là bảng thể hiện số lượng mẫu được phân loại vào từng cặp (lớp thực, lớp dự đoán). Hàng tương ứng với lớp thực, cột tương ứng với lớp dự đoán. Đường chéo chính chứa số mẫu được phân loại đúng (True Positive cho mỗi lớp), các ô ngoài đường chéo thể hiện các lỗi phân loại.

Confusion matrix cho cái nhìn chi tiết về hành vi của mô hình: lớp nào hay bị nhầm với lớp nào, lớp nào có accuracy cao/thấp. Ví dụ, trong phân loại tàu, confusion matrix có thể cho thấy "tàu hàng" và "tàu container" hay bị nhầm với nhau do hình dạng tương tự, gợi ý cần thêm đặc trưng hoặc dữ liệu để phân biệt.

### 3.5.3. ROC Curve và AUC

ROC (Receiver Operating Characteristic) Curve là đồ thị thể hiện trade-off giữa True Positive Rate (Recall) và False Positive Rate khi thay đổi ngưỡng quyết định. AUC (Area Under Curve) là diện tích dưới đường ROC, có giá trị từ 0 đến 1. AUC = 1 nghĩa là mô hình hoàn hảo, AUC = 0.5 nghĩa là mô hình random.

AUC có ưu điểm là không phụ thuộc vào ngưỡng cụ thể và không bị ảnh hưởng bởi class imbalance như accuracy. Đối với bài toán đa lớp, ROC và AUC được tính riêng cho từng lớp theo cách one-vs-rest.

## 3.6. Data Augmentation cho Classification Viễn thám

Data augmentation là kỹ thuật tăng cường dữ liệu training bằng cách áp dụng các phép biến đổi lên ảnh gốc, tạo ra các mẫu "mới" mà vẫn giữ nguyên nhãn. Augmentation giúp tăng kích thước effective của dataset, giảm overfitting, và cải thiện khả năng generalization của mô hình.

Các kỹ thuật augmentation phổ biến cho ảnh viễn thám bao gồm:

**Geometric transformations:** Random horizontal/vertical flip, rotation (0°, 90°, 180°, 270° hoặc góc tùy ý), scaling, và random crop. Rotation đặc biệt quan trọng cho ảnh vệ tinh vì không có hướng "up" cố định như ảnh tự nhiên.

**Photometric transformations:** Thay đổi brightness, contrast, saturation, và hue. Cần cẩn thận với ảnh đa phổ - chỉ áp dụng cho các kênh RGB hoặc điều chỉnh phù hợp cho từng kênh phổ.

**Noise injection:** Thêm Gaussian noise hoặc speckle noise (đặc biệt cho ảnh SAR) để tăng robustness.

**Mixup và CutMix:** Các kỹ thuật augmentation nâng cao kết hợp hai ảnh training với nhau, tạo ra các mẫu "blended" với soft labels tương ứng. Đã chứng minh hiệu quả trong việc cải thiện generalization và calibration của mô hình.

Với ảnh SAR, cần lưu ý rằng một số augmentation photometric không phù hợp do tính chất khác biệt của dữ liệu radar so với ảnh quang học. Các augmentation geometric như flip và rotation vẫn áp dụng được. Có thể thêm các augmentation đặc thù SAR như simulated speckle noise với các mức độ khác nhau.
