# Kiến trúc Convolutional Neural Network Cơ bản

## 2.1. Tổng quan về Kiến trúc CNN

Convolutional Neural Network (CNN) là một kiến trúc mạng nơ-ron được thiết kế đặc biệt để xử lý dữ liệu có cấu trúc không gian, điển hình là ảnh số. Không giống như mạng nơ-ron fully connected truyền thống xử lý đầu vào như một vector phẳng, CNN bảo toàn cấu trúc không gian của dữ liệu thông qua việc sử dụng các phép toán convolution. Điều này cho phép mạng học được các đặc trưng có ý nghĩa về mặt không gian như cạnh, góc, kết cấu, và hình dạng đối tượng.

Một mạng CNN điển hình được cấu thành từ nhiều lớp xếp chồng lên nhau, mỗi lớp thực hiện một phép biến đổi cụ thể trên dữ liệu đầu vào. Các lớp này có thể được phân loại thành ba nhóm chính: lớp convolution thực hiện trích xuất đặc trưng, lớp pooling giảm kích thước không gian và tăng tính bất biến, và lớp fully connected thực hiện phân loại hoặc hồi quy cuối cùng. Ngoài ra, các thành phần bổ sung như hàm kích hoạt (activation function), batch normalization, và dropout đóng vai trò quan trọng trong việc cải thiện khả năng học và tổng quát hóa của mạng.

Kiến trúc CNN có thể được hình dung như một pipeline xử lý phân cấp, trong đó các lớp đầu học được các đặc trưng cấp thấp như cạnh và gradient, các lớp giữa học được các đặc trưng cấp trung như kết cấu và pattern, và các lớp sâu hơn học được các đặc trưng cấp cao như bộ phận đối tượng và semantic concept. Sự phân cấp này cho phép CNN xây dựng biểu diễn phức tạp từ các thành phần đơn giản, tương tự như cách hệ thống thị giác sinh học xử lý thông tin hình ảnh.

## 2.2. Lớp Convolution (Convolutional Layer)

### 2.2.1. Nguyên lý Hoạt động

Lớp convolution là thành phần cốt lõi và đặc trưng nhất của CNN. Phép toán convolution trong ngữ cảnh xử lý ảnh được định nghĩa là phép trượt một bộ lọc (filter hay kernel) có kích thước nhỏ trên toàn bộ ảnh đầu vào, tại mỗi vị trí thực hiện phép nhân element-wise giữa bộ lọc và vùng ảnh tương ứng, sau đó cộng tất cả các tích lại để tạo thành một giá trị output. Kết quả của phép convolution là một feature map (hay activation map) thể hiện phản ứng của bộ lọc tại từng vị trí trong ảnh.

Về mặt toán học, nếu gọi đầu vào là tensor ba chiều X có kích thước H×W×C (chiều cao × chiều rộng × số kênh), và bộ lọc K có kích thước k×k×C, thì output Y tại vị trí (i,j) được tính như sau: Y[i,j] = Σ Σ Σ X[i+m, j+n, c] × K[m, n, c] + b, trong đó b là bias term. Phép tính này được lặp lại cho tất cả các vị trí hợp lệ trong ảnh để tạo thành một feature map hoàn chỉnh.

Một lớp convolution thường sử dụng nhiều bộ lọc khác nhau, mỗi bộ lọc học cách phát hiện một loại đặc trưng cụ thể. Số lượng bộ lọc quyết định số kênh của output feature map. Ví dụ, một lớp convolution với 64 bộ lọc kích thước 3×3 áp dụng lên ảnh RGB sẽ tạo ra output có 64 kênh, mỗi kênh tương ứng với một loại đặc trưng được trích xuất.

### 2.2.2. Các Tham số Quan trọng

**Kích thước bộ lọc (Kernel Size):** Quyết định receptive field cục bộ của mỗi nơ-ron. Các kích thước phổ biến bao gồm 3×3, 5×5, và 7×7. Bộ lọc 3×3 được sử dụng rộng rãi nhất vì có thể xếp chồng nhiều lớp 3×3 để đạt được receptive field tương đương bộ lọc lớn hơn, đồng thời giảm số lượng tham số và tăng khả năng biểu diễn phi tuyến thông qua việc xen kẽ các hàm kích hoạt.

**Stride:** Xác định bước nhảy của bộ lọc khi trượt qua ảnh. Stride bằng 1 nghĩa là bộ lọc di chuyển một pixel mỗi bước, tạo ra output có kích thước gần bằng input. Stride bằng 2 làm giảm kích thước không gian xuống một nửa, thường được sử dụng thay thế cho pooling trong một số kiến trúc hiện đại.

**Padding:** Kỹ thuật thêm các pixel có giá trị 0 (zero-padding) hoặc giá trị khác vào viền ảnh đầu vào. Padding "same" đảm bảo output có cùng kích thước không gian với input khi stride bằng 1. Padding "valid" không thêm pixel nào, dẫn đến output nhỏ hơn input. Zero-padding là lựa chọn phổ biến nhất, giúp bảo toàn thông tin ở biên ảnh và kiểm soát kích thước output.

**Số lượng bộ lọc (Number of Filters):** Quyết định số kênh của output và khả năng biểu diễn của lớp. Các lớp convolution đầu tiên thường có ít bộ lọc (32-64), và số lượng tăng dần theo độ sâu của mạng (128, 256, 512...) để học được các đặc trưng ngày càng phức tạp.

### 2.2.3. Chia sẻ Trọng số và Kết nối Cục bộ

Hai tính chất quan trọng làm nên hiệu quả của lớp convolution là chia sẻ trọng số (weight sharing) và kết nối cục bộ (local connectivity). Chia sẻ trọng số có nghĩa là cùng một bộ lọc được sử dụng cho tất cả các vị trí trong ảnh, giả định rằng đặc trưng hữu ích ở một vị trí cũng hữu ích ở các vị trí khác. Điều này giảm đáng kể số lượng tham số so với mạng fully connected và cho phép mạng học được các đặc trưng bất biến với vị trí.

Kết nối cục bộ có nghĩa là mỗi nơ-ron output chỉ kết nối với một vùng nhỏ của input, không phải toàn bộ. Điều này phản ánh thực tế rằng các pixel gần nhau thường có mối tương quan cao hơn các pixel ở xa, và các đặc trưng cục bộ như cạnh hay góc là thành phần cơ bản để xây dựng các đặc trưng phức tạp hơn.

## 2.3. Lớp Pooling

### 2.3.1. Mục đích và Nguyên lý

Lớp pooling thực hiện phép downsample, giảm kích thước không gian của feature map trong khi giữ lại thông tin quan trọng nhất. Pooling có ba mục đích chính: giảm số lượng tham số và chi phí tính toán cho các lớp tiếp theo, tăng receptive field của các lớp sâu hơn, và cung cấp một mức độ bất biến với các biến đổi nhỏ như dịch chuyển và xoay.

Khác với lớp convolution, pooling không có tham số học được. Thay vào đó, nó áp dụng một phép toán cố định (như lấy max hoặc average) trên từng vùng nhỏ của input. Phép toán này được thực hiện độc lập trên từng kênh, do đó số kênh output bằng số kênh input.

### 2.3.2. Max Pooling và Average Pooling

**Max Pooling** là loại pooling phổ biến nhất, chọn giá trị lớn nhất trong mỗi vùng pooling làm output. Max pooling 2×2 với stride 2 là cấu hình tiêu chuẩn, chia feature map thành các vùng 2×2 không chồng lấp và giữ lại giá trị max của mỗi vùng, giảm kích thước không gian xuống một nửa theo mỗi chiều. Max pooling có ưu điểm là giữ lại các đặc trưng nổi bật nhất (activation cao nhất) và cung cấp tính bất biến với các dịch chuyển nhỏ.

**Average Pooling** tính giá trị trung bình của tất cả các phần tử trong vùng pooling. So với max pooling, average pooling mềm mại hơn và giữ lại nhiều thông tin ngữ cảnh hơn, nhưng có thể làm mờ các đặc trưng quan trọng. Average pooling thường được sử dụng ở cuối mạng (Global Average Pooling) để thay thế cho các lớp fully connected, giúp giảm overfitting và cho phép mạng xử lý ảnh có kích thước bất kỳ.

**Global Average Pooling (GAP)** là một biến thể đặc biệt, tính trung bình trên toàn bộ feature map cho mỗi kênh, tạo ra output có kích thước 1×1×C. GAP được sử dụng rộng rãi trong các kiến trúc hiện đại như ResNet và EfficientNet, thay thế hoàn toàn các lớp fully connected trước lớp classification cuối cùng.

## 2.4. Hàm Kích hoạt (Activation Function)

### 2.4.1. Vai trò của Hàm Kích hoạt

Hàm kích hoạt đóng vai trò thiết yếu trong việc thêm tính phi tuyến vào mạng nơ-ron. Nếu không có hàm kích hoạt phi tuyến, việc xếp chồng nhiều lớp convolution hay fully connected sẽ chỉ tương đương với một phép biến đổi tuyến tính duy nhất, làm mất khả năng học các hàm phức tạp. Hàm kích hoạt được áp dụng element-wise sau mỗi lớp convolution hoặc fully connected, biến đổi output thành một không gian biểu diễn phi tuyến.

### 2.4.2. Các Hàm Kích hoạt Phổ biến

**ReLU (Rectified Linear Unit):** Định nghĩa là f(x) = max(0, x), ReLU là hàm kích hoạt được sử dụng rộng rãi nhất trong CNN hiện đại. ReLU có nhiều ưu điểm: đơn giản về mặt tính toán, giúp giảm vấn đề vanishing gradient so với sigmoid và tanh, và tạo ra sparse activation (nhiều output bằng 0). Tuy nhiên, ReLU có nhược điểm "dying ReLU" - các nơ-ron có thể "chết" vĩnh viễn nếu output luôn âm trong quá trình huấn luyện.

**Leaky ReLU và PReLU:** Để khắc phục vấn đề dying ReLU, Leaky ReLU định nghĩa f(x) = x nếu x > 0, và f(x) = αx nếu x ≤ 0, với α là một hằng số nhỏ (thường 0.01). PReLU (Parametric ReLU) mở rộng ý tưởng này bằng cách cho phép α được học từ dữ liệu.

**Sigmoid và Softmax:** Sigmoid f(x) = 1/(1 + e^(-x)) nén output vào khoảng (0,1), thường được sử dụng cho bài toán phân loại nhị phân. Softmax là phiên bản đa lớp của sigmoid, biến đổi vector output thành phân phối xác suất với tổng bằng 1, được sử dụng ở lớp cuối cùng cho bài toán phân loại đa lớp.

**GELU và Swish:** Các hàm kích hoạt hiện đại như GELU (Gaussian Error Linear Unit) và Swish (f(x) = x × sigmoid(x)) kết hợp tính chất của ReLU và các hàm mịn, được sử dụng trong các kiến trúc transformer và EfficientNet với kết quả cải thiện trên nhiều benchmark.

## 2.5. Lớp Fully Connected

Lớp fully connected (hay dense layer) kết nối mọi nơ-ron ở lớp trước với mọi nơ-ron ở lớp sau, tương tự như trong mạng nơ-ron truyền thống. Trong CNN, các lớp fully connected thường được đặt ở cuối mạng, sau các lớp convolution và pooling, để thực hiện phân loại hoặc hồi quy cuối cùng.

Input cho lớp fully connected đầu tiên là feature map được làm phẳng (flatten) thành vector một chiều. Ví dụ, nếu output của lớp pooling cuối cùng có kích thước 7×7×512, nó sẽ được flatten thành vector 25,088 chiều trước khi đưa vào lớp fully connected. Các lớp fully connected tiếp theo giảm dần số chiều, với lớp cuối cùng có số nơ-ron bằng số lớp cần phân loại.

Trong các kiến trúc CNN hiện đại, xu hướng là giảm thiểu hoặc loại bỏ hoàn toàn các lớp fully connected để giảm số lượng tham số và tránh overfitting. Global Average Pooling kết hợp với một lớp fully connected duy nhất (hay thậm chí chỉ một lớp convolution 1×1) thường đủ để thực hiện phân loại.

## 2.6. Batch Normalization

Batch Normalization (BN) là kỹ thuật quan trọng được giới thiệu năm 2015 bởi Ioffe và Szegedy, giúp tăng tốc độ huấn luyện và ổn định quá trình học của mạng nơ-ron sâu. BN chuẩn hóa activation của mỗi lớp theo mean và variance của mini-batch hiện tại, sau đó scale và shift bằng các tham số học được γ và β.

Công thức của Batch Normalization: y = γ × (x - μ) / √(σ² + ε) + β, trong đó μ và σ² là mean và variance của mini-batch, ε là hằng số nhỏ để tránh chia cho 0, và γ, β là các tham số học được. Trong quá trình inference, μ và σ² được thay thế bằng running average được tính trong quá trình training.

BN có nhiều lợi ích: cho phép sử dụng learning rate lớn hơn mà không làm mất ổn định, giảm sự phụ thuộc vào initialization, đóng vai trò như một hình thức regularization giúp giảm overfitting, và giải quyết phần nào vấn đề internal covariate shift. BN thường được đặt sau lớp convolution hoặc fully connected, trước hoặc sau hàm kích hoạt tùy theo kiến trúc cụ thể.

## 2.7. Dropout

Dropout là kỹ thuật regularization được giới thiệu bởi Hinton và cộng sự năm 2014, đặc biệt hiệu quả trong việc chống overfitting cho các mạng nơ-ron sâu. Ý tưởng cốt lõi của dropout là trong mỗi iteration huấn luyện, ngẫu nhiên "tắt" (set về 0) một tỷ lệ p các nơ-ron, buộc mạng phải học các biểu diễn robust không phụ thuộc vào bất kỳ nơ-ron cụ thể nào.

Trong quá trình training, mỗi nơ-ron được giữ lại với xác suất (1-p), và output được scale lên 1/(1-p) để đảm bảo tổng expected output không đổi. Trong quá trình inference, tất cả các nơ-ron được sử dụng mà không cần scale. Tỷ lệ dropout thường được đặt từ 0.2 đến 0.5, với các lớp fully connected thường sử dụng tỷ lệ cao hơn do số lượng tham số lớn.

Dropout có thể được hiểu như một hình thức model averaging: mỗi lần training với một subset nơ-ron khác nhau tương đương với training một mạng con khác nhau, và kết quả cuối cùng là trung bình của nhiều mạng con này. Trong các kiến trúc CNN hiện đại, dropout thường ít được sử dụng hơn do hiệu quả của Batch Normalization và các kỹ thuật regularization khác như weight decay và data augmentation.

## 2.8. Kiến trúc Tổng thể của CNN

Một mạng CNN điển hình cho bài toán phân loại ảnh có cấu trúc xen kẽ giữa các khối convolution-pooling, theo sau bởi các lớp fully connected. Mỗi khối convolution thường bao gồm một hoặc nhiều lớp convolution, mỗi lớp theo sau bởi batch normalization và hàm kích hoạt ReLU. Sau một số khối convolution, một lớp pooling được sử dụng để giảm kích thước không gian.

Khi đi sâu vào mạng, kích thước không gian của feature map giảm dần (do pooling hoặc strided convolution) trong khi số kênh tăng dần (do tăng số bộ lọc). Điều này phản ánh sự chuyển đổi từ biểu diễn không gian chi tiết ở các lớp đầu sang biểu diễn semantic trừu tượng ở các lớp sâu. Cuối cùng, feature map được flatten và đưa qua các lớp fully connected để tạo ra vector output với số chiều bằng số lớp phân loại.

Các kiến trúc hiện đại như ResNet và EfficientNet thay đổi đáng kể paradigm này bằng việc sử dụng skip connection, bottleneck block, và Global Average Pooling, sẽ được trình bày chi tiết trong phần tiếp theo.

## 2.9. Quá trình Huấn luyện CNN

### 2.9.1. Forward Propagation

Forward propagation là quá trình tính toán output của mạng từ input đầu vào. Dữ liệu ảnh được đưa qua từng lớp theo thứ tự, với output của lớp trước trở thành input của lớp sau. Tại mỗi lớp convolution, các phép tích chập được thực hiện; tại mỗi lớp pooling, phép downsample được áp dụng; và các hàm kích hoạt biến đổi output thành không gian phi tuyến. Kết quả cuối cùng là một vector xác suất cho các lớp phân loại hoặc các giá trị hồi quy.

### 2.9.2. Loss Function

Loss function đo lường sự khác biệt giữa output dự đoán của mạng và nhãn ground truth. Đối với bài toán phân loại đa lớp, Cross-Entropy Loss là lựa chọn phổ biến nhất: L = -Σ y_i × log(p_i), trong đó y_i là one-hot encoding của nhãn thực và p_i là xác suất dự đoán cho lớp i. Đối với bài toán phát hiện đối tượng và phân đoạn, các loss function phức tạp hơn được sử dụng, bao gồm classification loss, regression loss cho bounding box, và Dice loss hoặc IoU loss cho segmentation mask.

### 2.9.3. Backpropagation và Optimization

Backpropagation tính toán gradient của loss function theo các tham số của mạng thông qua chain rule. Gradient được truyền ngược từ lớp output qua từng lớp về lớp đầu tiên, cho phép cập nhật tất cả các tham số học được bao gồm trọng số convolution, bias, và các tham số của batch normalization.

Các thuật toán optimization như Stochastic Gradient Descent (SGD) với momentum, Adam, và AdamW sử dụng gradient để cập nhật tham số theo hướng giảm loss. Learning rate scheduling (như step decay, cosine annealing) và warmup là các kỹ thuật quan trọng để đạt được convergence tốt. Weight decay (L2 regularization) được sử dụng để hạn chế độ lớn của tham số và giảm overfitting.

Quá trình huấn luyện được thực hiện qua nhiều epoch, mỗi epoch duyệt qua toàn bộ tập training data. Trong mỗi epoch, data được chia thành các mini-batch, và gradient descent được thực hiện trên từng mini-batch. Validation set được sử dụng để đánh giá hiệu suất và điều chỉnh hyperparameter, trong khi test set được giữ riêng cho đánh giá cuối cùng.
