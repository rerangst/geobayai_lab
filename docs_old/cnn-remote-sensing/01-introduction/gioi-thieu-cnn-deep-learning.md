# Giới thiệu về Convolutional Neural Network và Deep Learning trong Viễn thám

## 1.1. Bối cảnh và Tầm quan trọng

Trong những thập kỷ gần đây, sự phát triển vượt bậc của công nghệ vệ tinh quan sát Trái Đất đã mở ra một kỷ nguyên mới trong lĩnh vực viễn thám. Các vệ tinh quan sát như Sentinel-1, Sentinel-2 của Cơ quan Vũ trụ Châu Âu (ESA), WorldView-3 của Maxar Technologies, hay Planet Labs với hệ thống vệ tinh nhỏ quy mô lớn, liên tục thu thập hàng petabyte dữ liệu ảnh mỗi ngày. Khối lượng dữ liệu khổng lồ này đặt ra thách thức lớn trong việc phân tích và trích xuất thông tin hữu ích một cách tự động và hiệu quả.

Trước đây, các phương pháp xử lý ảnh viễn thám truyền thống chủ yếu dựa vào việc trích xuất đặc trưng thủ công (handcrafted features) kết hợp với các bộ phân loại cổ điển như Support Vector Machine (SVM), Random Forest, hay các thuật toán threshold đơn giản. Tuy nhiên, những phương pháp này bộc lộ nhiều hạn chế khi đối mặt với sự đa dạng và phức tạp của dữ liệu ảnh vệ tinh thực tế. Các đặc trưng được thiết kế thủ công thường khó có thể tổng quát hóa tốt trên nhiều loại cảnh quan, điều kiện thời tiết, hay góc chụp khác nhau.

Sự ra đời và phát triển mạnh mẽ của Deep Learning, đặc biệt là mạng Convolutional Neural Network (CNN), đã tạo ra bước đột phá quan trọng trong việc giải quyết các bài toán xử lý ảnh viễn thám. Khác với các phương pháp truyền thống, CNN có khả năng tự động học và trích xuất các đặc trưng phân cấp từ dữ liệu thô, từ các đặc trưng cấp thấp như cạnh (edge), góc (corner), đến các đặc trưng cấp cao như hình dạng đối tượng, mối quan hệ không gian. Khả năng học biểu diễn end-to-end này cho phép CNN đạt được hiệu suất vượt trội so với các phương pháp truyền thống trong hầu hết các bài toán xử lý ảnh viễn thám.

Trong bối cảnh an ninh hàng hải và bảo vệ môi trường biển, hai bài toán quan trọng được đặc biệt quan tâm là phát hiện tàu biển (Ship Detection) và nhận dạng vết dầu loang (Oil Spill Detection). Phát hiện tàu biển đóng vai trò then chốt trong việc giám sát hoạt động hàng hải, phát hiện tàu đánh cá bất hợp pháp (Illegal, Unreported, and Unregulated fishing - IUU), và đảm bảo an ninh vùng biển. Trong khi đó, nhận dạng vết dầu loang là nhiệm vụ thiết yếu để phát hiện sớm các sự cố tràn dầu, hỗ trợ ứng phó kịp thời và giảm thiểu tác động đến môi trường sinh thái biển.

## 1.2. Convolutional Neural Network là gì?

Convolutional Neural Network, thường được viết tắt là CNN hoặc ConvNet, là một lớp kiến trúc mạng nơ-ron sâu (Deep Neural Network) được thiết kế đặc biệt để xử lý dữ liệu có cấu trúc lưới (grid-like topology), điển hình nhất là dữ liệu ảnh hai chiều. Tên gọi "Convolutional" xuất phát từ phép toán tích chập (convolution) được sử dụng làm thao tác xử lý chính trong mạng, thay thế cho phép nhân ma trận thông thường trong ít nhất một lớp của mạng.

Ý tưởng cốt lõi của CNN được lấy cảm hứng từ cơ chế hoạt động của vỏ não thị giác (visual cortex) trong não động vật có vú. Các nghiên cứu sinh học thần kinh của Hubel và Wiesel vào những năm 1960 đã chỉ ra rằng các tế bào thần kinh trong vỏ não thị giác phản ứng với các kích thích trong vùng thị giác giới hạn, gọi là receptive field. Các receptive field của các tế bào khác nhau chồng chéo lên nhau một phần để bao phủ toàn bộ trường thị giác. CNN mô phỏng cơ chế này bằng cách sử dụng các bộ lọc (filter hay kernel) có kích thước nhỏ quét qua toàn bộ ảnh đầu vào.

Điểm khác biệt quan trọng giữa CNN và mạng nơ-ron truyền thống fully connected là ba tính chất kiến trúc đặc trưng: kết nối cục bộ (local connectivity), chia sẻ trọng số (weight sharing), và pooling. Trong mạng fully connected, mỗi nơ-ron ở lớp sau được kết nối với tất cả các nơ-ron ở lớp trước, dẫn đến số lượng tham số rất lớn và dễ bị overfitting. Ngược lại, trong CNN, mỗi nơ-ron chỉ kết nối với một vùng nhỏ của lớp trước (receptive field cục bộ), và các trọng số kết nối được chia sẻ trên toàn bộ ảnh. Điều này không chỉ giảm đáng kể số lượng tham số mà còn cho phép mạng học được các đặc trưng bất biến với vị trí (translation invariant).

## 1.3. Lịch sử Phát triển của CNN

### 1.3.1. Giai đoạn Khởi đầu: LeNet (1989-1998)

Lịch sử của CNN bắt đầu từ những năm cuối thập niên 1980 với công trình tiên phong của Yann LeCun và cộng sự tại AT&T Bell Labs. Năm 1989, LeCun lần đầu tiên áp dụng thành công thuật toán backpropagation để huấn luyện mạng nơ-ron tích chập nhận dạng chữ số viết tay. Đến năm 1998, kiến trúc LeNet-5 được công bố, trở thành một trong những CNN hoàn chỉnh đầu tiên được triển khai trong thực tế để đọc mã ZIP trên thư tín tại Hoa Kỳ.

LeNet-5 có kiến trúc gồm 7 lớp (không kể đầu vào): 3 lớp convolution, 2 lớp subsampling (tương tự pooling), và 2 lớp fully connected. Mặc dù quy mô nhỏ theo tiêu chuẩn hiện đại, LeNet-5 đã thiết lập nền tảng kiến trúc cơ bản cho các CNN sau này: xen kẽ giữa các lớp convolution và pooling, sau đó kết nối với các lớp fully connected để thực hiện phân loại cuối cùng.

### 1.3.2. Thời kỳ Trầm lắng (1998-2012)

Sau thành công ban đầu của LeNet, nghiên cứu về mạng nơ-ron nói chung và CNN nói riêng trải qua giai đoạn trầm lắng kéo dài gần 15 năm. Nguyên nhân chính bao gồm: hạn chế về năng lực tính toán của phần cứng thời bấy giờ, thiếu các tập dữ liệu quy mô lớn để huấn luyện, và sự thống trị của các phương pháp học máy khác như SVM và boosting vốn cho kết quả tốt với chi phí tính toán thấp hơn.

Trong giai đoạn này, các phương pháp dựa trên đặc trưng thủ công như SIFT (Scale-Invariant Feature Transform), HOG (Histogram of Oriented Gradients), và các biến thể của chúng chiếm ưu thế trong các bài toán thị giác máy tính. Đối với xử lý ảnh viễn thám, các kỹ thuật truyền thống như phân loại dựa trên chỉ số phổ, phân tích kết cấu (texture analysis), và phân đoạn dựa trên ngưỡng vẫn là những phương pháp chính được sử dụng.

### 1.3.3. Bước Ngoặt AlexNet (2012)

Năm 2012 đánh dấu bước ngoặt lịch sử của Deep Learning khi Alex Krizhevsky, Ilya Sutskever và Geoffrey Hinton giành chiến thắng áp đảo trong cuộc thi ImageNet Large Scale Visual Recognition Challenge (ILSVRC). Mạng AlexNet của họ đạt top-5 error rate 15.3%, vượt xa đáng kể so với phương pháp xếp thứ hai với 26.2%. Khoảng cách vượt trội này đã gây chấn động cộng đồng nghiên cứu và khởi đầu kỷ nguyên Deep Learning hiện đại.

AlexNet có kiến trúc sâu hơn đáng kể so với LeNet với 8 lớp học được (5 lớp convolution và 3 lớp fully connected), sử dụng hàm kích hoạt ReLU thay cho sigmoid/tanh, áp dụng kỹ thuật Dropout để chống overfitting, và được huấn luyện trên GPU - một bước tiến quan trọng về mặt kỹ thuật. Thành công của AlexNet chứng minh rằng với đủ dữ liệu và năng lực tính toán, các mạng nơ-ron sâu có thể học được các biểu diễn đặc trưng mạnh mẽ hơn nhiều so với các đặc trưng thiết kế thủ công.

### 1.3.4. Kỷ nguyên Mạng Rất Sâu: VGGNet và ResNet (2014-2015)

Sau AlexNet, xu hướng chính trong nghiên cứu CNN là tăng độ sâu của mạng. Năm 2014, nhóm Visual Geometry Group tại Đại học Oxford giới thiệu VGGNet với kiến trúc sử dụng các bộ lọc convolution nhỏ 3×3 xếp chồng lên nhau. VGG-16 và VGG-19 với 16 và 19 lớp học được đạt kết quả ấn tượng trên ImageNet, đồng thời chứng minh rằng độ sâu của mạng là yếu tố quan trọng quyết định hiệu suất.

Tuy nhiên, việc tăng độ sâu mạng gặp phải vấn đề nghiêm trọng: gradient vanishing và gradient exploding trong quá trình backpropagation, khiến việc huấn luyện các mạng rất sâu trở nên khó khăn hoặc không thể. Năm 2015, Kaiming He và cộng sự tại Microsoft Research đề xuất kiến trúc Residual Network (ResNet) với ý tưởng đột phá: thêm các kết nối tắt (skip connection hay shortcut connection) cho phép gradient chảy trực tiếp qua nhiều lớp. Với kỹ thuật này, ResNet có thể huấn luyện thành công các mạng với hàng trăm thậm chí hàng nghìn lớp. ResNet-152 đạt top-5 error rate 3.57% trên ImageNet, lần đầu tiên vượt qua ngưỡng nhận dạng của con người (khoảng 5%).

### 1.3.5. Từ EfficientNet đến Vision Transformer (2019-Hiện tại)

Năm 2019, Mingxing Tan và Quoc V. Le từ Google Research giới thiệu EfficientNet với phương pháp compound scaling - cân bằng đồng thời ba chiều: độ sâu, độ rộng, và độ phân giải của mạng. EfficientNet đạt được độ chính xác state-of-the-art với số lượng tham số và chi phí tính toán ít hơn đáng kể so với các kiến trúc trước đó.

Bước ngoặt mới nhất trong lĩnh vực thị giác máy tính là sự xuất hiện của Vision Transformer (ViT) vào năm 2020. Lấy cảm hứng từ kiến trúc Transformer vốn thành công vượt trội trong xử lý ngôn ngữ tự nhiên, ViT chia ảnh thành các patch và xử lý như một chuỗi token. Khi được huấn luyện trên tập dữ liệu đủ lớn, ViT đạt được hoặc vượt qua hiệu suất của các CNN tốt nhất trên nhiều benchmark. Các biến thể như Swin Transformer còn kết hợp ưu điểm của cả CNN (xử lý phân cấp) và Transformer (attention toàn cục).

## 1.4. Tại sao CNN phù hợp với ảnh vệ tinh?

Ảnh vệ tinh có nhiều đặc điểm khác biệt so với ảnh tự nhiên thông thường, và CNN tỏ ra đặc biệt phù hợp để xử lý những đặc thù này.

Thứ nhất, ảnh vệ tinh thường có kích thước rất lớn, từ hàng nghìn đến hàng chục nghìn pixel mỗi chiều. Một scene Sentinel-1 điển hình có kích thước khoảng 250×250 km với độ phân giải 10m, tương đương ảnh 25,000×25,000 pixel. Kiến trúc CNN với các lớp convolution và pooling cho phép xử lý ảnh lớn một cách hiệu quả thông qua việc chia nhỏ thành các tile và trượt cửa sổ.

Thứ hai, các đối tượng trong ảnh vệ tinh xuất hiện ở nhiều tỷ lệ khác nhau. Một tàu container có thể chiếm hàng trăm pixel trong ảnh WorldView-3 (0.3m GSD) nhưng chỉ vài pixel trong ảnh Sentinel-1 (10m GSD). Các kiến trúc CNN hiện đại như Feature Pyramid Network (FPN) được thiết kế đặc biệt để xử lý vấn đề đa tỷ lệ này bằng cách kết hợp các đặc trưng từ nhiều mức độ phân giải khác nhau.

Thứ ba, ảnh vệ tinh thường có nhiều kênh phổ hơn ảnh RGB thông thường. Ảnh Sentinel-2 có 13 kênh phổ từ khả kiến đến hồng ngoại nhiệt, trong khi ảnh Sentinel-1 SAR có 2 kênh phân cực VV và VH. CNN có thể dễ dàng mở rộng để xử lý đầu vào đa kênh bằng cách điều chỉnh số kênh ở lớp convolution đầu tiên.

Thứ tư, các đối tượng trong ảnh vệ tinh có thể xuất hiện theo nhiều hướng khác nhau, không như ảnh tự nhiên thường có hướng "lên-xuống" cố định. Tính chất translation invariant của CNN, kết hợp với data augmentation xoay ảnh trong quá trình huấn luyện, giúp mạng có khả năng nhận dạng đối tượng bất kể hướng.

## 1.5. Các bài toán chính trong xử lý ảnh viễn thám

### 1.5.1. Phân loại ảnh (Image Classification)

Phân loại ảnh là bài toán gán một hoặc nhiều nhãn lớp cho toàn bộ ảnh hoặc một vùng ảnh (patch). Trong viễn thám, bài toán này thường được áp dụng cho phân loại lớp phủ mặt đất (land cover classification), phân loại cảnh quan (scene classification), hay phân loại loại tàu (vessel type classification). Output của bài toán phân loại là một vector xác suất cho các lớp, và mạng CNN điển hình cho bài toán này bao gồm các lớp convolution để trích xuất đặc trưng, theo sau bởi các lớp fully connected để thực hiện phân loại cuối cùng.

### 1.5.2. Phát hiện đối tượng (Object Detection)

Phát hiện đối tượng yêu cầu mạng không chỉ phân loại mà còn định vị vị trí của các đối tượng trong ảnh thông qua bounding box. Đây là bài toán quan trọng trong phát hiện tàu biển, phương tiện, máy bay, và các đối tượng nhân tạo khác từ ảnh vệ tinh. Các kiến trúc phổ biến bao gồm họ YOLO (You Only Look Once) cho phát hiện thời gian thực, và Faster R-CNN cho độ chính xác cao. Đối với ảnh vệ tinh, các biến thể như Rotated Faster R-CNN còn hỗ trợ oriented bounding box để xử lý các đối tượng có hướng nghiêng như tàu biển.

### 1.5.3. Phân đoạn ngữ nghĩa (Semantic Segmentation)

Phân đoạn ngữ nghĩa yêu cầu gán nhãn lớp cho từng pixel trong ảnh, tạo ra mask phân vùng chi tiết. Bài toán này đặc biệt quan trọng trong nhận dạng vết dầu loang, lập bản đồ ngập lụt, hay trích xuất vùng đô thị. Các kiến trúc encoder-decoder như U-Net và DeepLabV3+ là những lựa chọn phổ biến, với encoder trích xuất đặc trưng và decoder khôi phục độ phân giải không gian. Các kết nối skip connection giữa encoder và decoder giúp bảo toàn thông tin chi tiết biên đối tượng.

### 1.5.4. Instance Segmentation

Instance Segmentation kết hợp phát hiện đối tượng và phân đoạn ngữ nghĩa, không chỉ phân vùng theo lớp mà còn phân biệt từng instance riêng lẻ. Ví dụ, trong một cảng biển có nhiều tàu, instance segmentation sẽ tạo mask riêng biệt cho từng tàu, cho phép đếm chính xác số lượng và phân tích đặc điểm từng tàu. Mask R-CNN là kiến trúc tiêu biểu cho bài toán này.

## 1.6. Mục tiêu và Phạm vi Báo cáo

Báo cáo này tập trung vào hai bài toán ứng dụng cụ thể trong lĩnh vực giám sát biển: **phát hiện tàu biển (Ship Detection)** và **nhận dạng vết dầu loang (Oil Spill Detection)**. Đây là hai bài toán có tầm quan trọng chiến lược trong bối cảnh Việt Nam là quốc gia biển với đường bờ biển dài hơn 3,260 km và vùng đặc quyền kinh tế rộng lớn.

Phát hiện tàu biển từ ảnh vệ tinh là công cụ quan trọng để giám sát hoạt động đánh bắt cá bất hợp pháp, đảm bảo an ninh hàng hải, và quản lý giao thông đường biển. Với việc sử dụng ảnh SAR (Synthetic Aperture Radar) từ vệ tinh Sentinel-1, hệ thống có thể hoạt động liên tục 24/7, bất kể điều kiện thời tiết hay thời gian trong ngày - một ưu điểm vượt trội so với ảnh quang học.

Nhận dạng vết dầu loang đóng vai trò quan trọng trong bảo vệ môi trường biển. Việc phát hiện sớm các sự cố tràn dầu, dù từ tai nạn hay xả thải bất hợp pháp, cho phép các cơ quan chức năng ứng phó kịp thời, giảm thiểu tác động đến hệ sinh thái biển và ngành thủy sản. Ảnh SAR cũng là lựa chọn hàng đầu cho bài toán này do khả năng phát hiện vết dầu dựa trên sự thay đổi độ nhám bề mặt biển.

Báo cáo sẽ trình bày chi tiết về kiến trúc CNN cơ bản, các phương pháp áp dụng CNN cho ảnh vệ tinh, quy trình xử lý hoàn chỉnh cho từng bài toán, các model state-of-the-art, và các bộ dữ liệu chuẩn được sử dụng trong nghiên cứu và đánh giá. Đặc biệt, báo cáo sẽ tập trung phân tích các model có sẵn trong thư viện TorchGeo - một thư viện Python được thiết kế đặc biệt cho các bài toán Deep Learning trong viễn thám, với các pre-trained weights cho nhiều loại cảm biến vệ tinh khác nhau.

Thông qua việc tổng hợp và phân tích các nghiên cứu, phương pháp, và công cụ hiện đại, báo cáo nhằm cung cấp một cái nhìn toàn diện về ứng dụng Deep Learning trong viễn thám biển, đồng thời đề xuất các hướng tiếp cận phù hợp cho việc triển khai thực tế tại Việt Nam.
