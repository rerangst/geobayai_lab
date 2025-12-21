# Chương 6: Ứng dụng Phát hiện Dầu loang

## Giới thiệu

Tiếp nối bài toán object detection ở **Chương 5**, chương này trình bày ứng dụng semantic segmentation vào phát hiện dầu loang trên ảnh SAR. Kỹ thuật segmentation từ **Chương 3** và phương pháp damage assessment từ xView2 (**Chương 4**) sẽ được áp dụng vào bối cảnh giám sát môi trường biển.

Bài toán phát hiện dầu loang có đặc điểm riêng so với phát hiện tàu biển: thay vì object detection với bounding box, đây là semantic segmentation task để xác định vùng dầu loang với ranh giới không đều. Việc áp dụng các kiến trúc encoder-decoder như U-Net, DeepLabV3+ (đã trình bày tại **Mục 3.3**) vào bối cảnh giám sát môi trường biển Việt Nam tạo nên ứng dụng thực tế có giá trị cao.

## 6.1. Tổng quan về Oil Spill Detection

Oil spill detection từ ảnh vệ tinh là một ứng dụng quan trọng của remote sensing, đóng vai trò thiết yếu trong bảo vệ môi trường biển và quản lý sự cố tràn dầu. Mỗi năm, hàng triệu tấn dầu được vận chuyển qua các tuyến đường biển toàn cầu, và sự cố tràn dầu từ các nguồn khác nhau (tai nạn tàu, rò rỉ đường ống, xả thải bất hợp pháp, rò rỉ tự nhiên) gây ra những thiệt hại nghiêm trọng cho hệ sinh thái biển và vùng ven biển.

Việc phát hiện nhanh chóng và chính xác các vết dầu loang là yếu tố quyết định trong việc ứng phó hiệu quả với sự cố. Thời gian từ khi xảy ra sự cố đến khi phát hiện càng ngắn, khả năng kiểm soát và khắc phục hậu quả càng cao. Satellite imagery, đặc biệt là Synthetic Aperture Radar (SAR), cung cấp khả năng giám sát diện rộng, liên tục, và không phụ thuộc vào điều kiện thời tiết hay ánh sáng.

Deep learning đã mang lại bước tiến đột phá trong oil spill detection, vượt qua các phương pháp truyền thống dựa trên threshold và rule-based approaches. Các mô hình CNN có khả năng học các đặc trưng phức tạp từ dữ liệu, phân biệt hiệu quả giữa oil spill thực sự và các look-alikes, và xử lý đa dạng điều kiện môi trường.

## 6.2. Tại sao SAR là Sensor Chính cho Oil Spill Detection

### 6.2.1. Nguyên lý Phát hiện Dầu bằng SAR

**Tham khảo lý thuyết SAR:** Nguyên lý cơ bản về SAR polarimetry và backscatter đã được trình bày tại **Chương 2.1**. Phần này tập trung vào ứng dụng cụ thể cho phát hiện dầu loang.

SAR phát hiện oil spill dựa trên nguyên lý damping effect - dầu làm giảm độ gồ ghề (roughness) của bề mặt biển, dẫn đến backscatter thấp hơn so với nước biển xung quanh. Trên SAR imagery, oil spill xuất hiện như các vùng tối (dark spots) trên nền biển sáng hơn.

**Cơ chế vật lý:**
Bề mặt biển có roughness do sóng ngắn (capillary waves và short gravity waves). Radar backscatter phụ thuộc vào roughness này - bề mặt gồ ghề hơn cho backscatter cao hơn (sáng hơn trên ảnh). Dầu tạo một lớp màng mỏng trên mặt biển, làm giảm sức căng bề mặt và suppress các sóng ngắn. Kết quả là vùng có dầu có roughness thấp hơn, backscatter thấp hơn, và xuất hiện tối trên ảnh SAR.

**Yếu tố ảnh hưởng đến khả năng phát hiện:**
- **Loại dầu:** Dầu nặng (crude oil, heavy fuel oil) tạo hiệu ứng damping mạnh hơn dầu nhẹ (diesel, gasoline). Dầu thực vật cũng gây damping nhưng yếu hơn dầu khoáng.
- **Độ dày lớp dầu:** Lớp dầu dày hơn thường cho contrast cao hơn, nhưng mối quan hệ không tuyến tính và phụ thuộc vào nhiều yếu tố.
- **Thời gian từ khi tràn:** Dầu "tươi" thường dễ phát hiện hơn. Theo thời gian, dầu bị phong hóa (weathering), bay hơi các thành phần nhẹ, và có thể emulsify với nước, làm thay đổi tín hiệu radar.
- **Điều kiện gió:** Tốc độ gió 3-10 m/s là lý tưởng. Gió yếu (<3 m/s) không tạo đủ roughness cho nước biển, khiến cả vùng có và không có dầu đều tối. Gió mạnh (>10-12 m/s) có thể phá vỡ lớp dầu và tạo sóng lớn che khuất hiệu ứng damping.

### 6.2.2. Ưu điểm của SAR so với Optical Sensors

**Bối cảnh Việt Nam:** Vùng biển Việt Nam có điều kiện thời tiết nhiệt đới với mây che phủ cao, đặc biệt trong mùa mưa. SAR là lựa chọn phù hợp cho giám sát liên tục trong điều kiện này.

**1. Hoạt động mọi điều kiện thời tiết:**
SAR là active sensor, phát sóng radar riêng và thu tín hiệu phản xạ. Không như optical sensors phụ thuộc vào ánh sáng mặt trời và bị cản bởi mây, SAR có thể hoạt động ngày đêm, xuyên qua mây, mưa nhẹ, và sương mù. Đây là ưu điểm quan trọng vì nhiều sự cố tràn dầu xảy ra trong điều kiện thời tiết xấu.

**2. Phát hiện dầu trên mặt biển:**
Nguyên lý damping của SAR đặc biệt phù hợp cho việc phát hiện chất lỏng làm thay đổi roughness bề mặt. Optical sensors phát hiện dầu qua sự khác biệt về phản xạ quang học, nhưng hiệu ứng này yếu hơn và dễ bị nhiễu bởi sun glint, whitecaps, và biến đổi màu nước.

**3. Coverage rộng:**
Các vệ tinh SAR như Sentinel-1 có swath width lên đến 400 km (Extra Wide Swath mode), cho phép giám sát diện tích lớn trong mỗi lần chụp.

**4. Revisit time hợp lý:**
Với constellation Sentinel-1 (trước khi mất Sentinel-1B), revisit time toàn cầu là 6 ngày, đủ cho giám sát routine. Các vệ tinh thương mại (ICEYE, Capella) có thể cung cấp revisit nhanh hơn cho các vùng quan tâm.

### 6.2.3. Hạn chế và Thách thức của SAR

**1. Look-alikes:**
Đây là thách thức lớn nhất của oil spill detection từ SAR. Nhiều hiện tượng tự nhiên tạo ra dark spots tương tự oil spill:
- **Natural oil seeps:** Dầu rò rỉ tự nhiên từ đáy biển, xảy ra ở nhiều vùng biển
- **Biogenic films:** Chất hữu cơ từ sinh vật biển (phytoplankton, algae) có thể tạo màng trên mặt biển
- **Low wind areas:** Vùng gió yếu cục bộ tạo dark spots do thiếu roughness
- **Rain cells:** Mưa cục bộ làm smooth bề mặt biển tạm thời
- **Internal waves:** Sóng nội tạo patterns trên bề mặt
- **Current shear zones:** Ranh giới các dòng hải lưu
- **Upwelling zones:** Vùng nước trồi có thể có màng sinh học

Phân biệt oil spill thực sự và look-alikes là nhiệm vụ khó, đòi hỏi expert knowledge hoặc các mô hình machine learning phức tạp.

**2. Speckle noise:**
Như đã thảo luận trong phần ship detection, SAR imagery có speckle noise inherent. Speckle làm khó việc delineate chính xác ranh giới oil spill và có thể tạo false dark spots.

**3. Không thể xác định loại và độ dày dầu:**
SAR chỉ cho biết có "something" làm giảm roughness, không thể xác định cụ thể loại dầu hay độ dày lớp dầu - thông tin quan trọng cho việc ứng phó.

**4. Phụ thuộc vào điều kiện gió:**
Như đã đề cập, wind speed ngoài dải 3-10 m/s làm giảm khả năng phát hiện. Điều này có thể được mitigate bằng cách sử dụng ancillary wind data.

## 6.3. Optical Imagery cho Oil Spill Detection

### 6.3.1. Khả năng Phát hiện

Mặc dù SAR là sensor chính, optical imagery vẫn có vai trò trong oil spill detection và đặc biệt hữu ích cho một số ứng dụng.

**Nguyên lý phát hiện:**
Dầu thay đổi tính chất quang học của bề mặt nước:
- **Reflectance:** Dầu thường có reflectance cao hơn nước ở một số bước sóng, đặc biệt trong near-infrared
- **Color:** Dầu dày có thể cho màu nâu, đen, hoặc bóng (rainbow sheen cho lớp mỏng)
- **Texture:** Vùng dầu có thể có texture khác biệt

**Ưu điểm:**
- **Thông tin về loại dầu:** Multispectral/hyperspectral có thể giúp phân loại loại dầu dựa trên spectral signature
- **Ước tính độ dày:** Có thể ước tính relative thickness từ color/intensity
- **Dễ giải thích:** Optical imagery trực quan hơn cho non-experts

### 6.3.2. Hạn chế của Optical Sensors

**1. Phụ thuộc thời tiết và ánh sáng:**
- Yêu cầu daylight (không hoạt động ban đêm)
- Mây che phủ block hoàn toàn signal
- Sun glint tạo bright spots có thể mask hoặc confuse với oil

**2. Khả năng phát hiện hạn chế:**
- Lớp dầu mỏng khó phát hiện (không đủ contrast)
- Emulsified oil (oil-water mixture) khó phân biệt với nước
- Background variation (chlorophyll, sediment) gây nhiễu

**3. Resolution vs coverage trade-off:**
- High-resolution optical (WorldView, Pléiades) có swath nhỏ
- Wide-swath optical (Sentinel-2, Landsat) có resolution thấp hơn

### 6.3.3. Multi-sensor Approaches

Kết hợp SAR và optical cung cấp nhiều lợi ích:
- SAR cho detection trong mọi điều kiện thời tiết
- Optical cho verification và thông tin bổ sung về loại/độ dày dầu
- Cross-validation giảm false positives
- Temporal coverage tốt hơn khi combine nhiều sources

Trong thực tế, hệ thống giám sát oil spill thường sử dụng SAR làm primary detection sensor và optical làm secondary/verification.

## 6.4. Định dạng Bài toán Machine Learning

### 6.4.1. Binary Segmentation

Định dạng phổ biến nhất cho oil spill detection là binary semantic segmentation: phân loại mỗi pixel thành một trong hai lớp (oil spill hoặc background/water).

**Input:** SAR image (single channel sigma0, hoặc multi-channel với các polarizations)

**Output:** Binary mask có cùng spatial dimensions với input, mỗi pixel có giá trị 0 (không có dầu) hoặc 1 (có dầu)

**Ưu điểm:**
- Đơn giản và intuitive
- Phù hợp với mục tiêu chính: xác định vùng có dầu
- Nhiều architectures và frameworks hỗ trợ

**Hạn chế:**
- Không phân biệt các vết dầu riêng lẻ (không có instance information)
- Không phân loại look-alikes (chỉ binary oil/no-oil)

### 6.4.2. Multi-class Segmentation

Mở rộng từ binary segmentation, multi-class segmentation phân loại pixels thành nhiều lớp:
- Oil spill
- Look-alike (natural phenomena mimicking oil)
- Sea/Water (clean water)
- Ship (nếu có)
- Land (nếu có trong image)

**Ưu điểm:**
- Giảm false positives bằng cách explicitly model look-alikes
- Cung cấp nhiều thông tin hơn cho operators

**Hạn chế:**
- Cần training data được annotated cho tất cả các lớp
- Look-alike là category rất diverse, khó model hết

### 6.4.3. Detection + Segmentation

Một số approaches kết hợp object detection và segmentation:
1. **Detection phase:** Tìm các candidate regions (dark spots)
2. **Classification phase:** Classify mỗi candidate là oil spill hay look-alike
3. **Segmentation phase:** Segment chính xác extent của oil spill

Approach này phù hợp khi cần:
- Confidence score cho mỗi detection
- Attribute thêm thông tin (size, shape metrics)
- Prioritize alerts cho operators

### 6.4.4. Instance Segmentation

Instance segmentation cho oil spill detection cho phép:
- Phân biệt các vết dầu riêng lẻ (multiple spills từ different sources)
- Tracking từng vết dầu qua thời gian
- Analyze characteristics của từng spill

Trong thực tế, multiple oil spills trong cùng một image không quá phổ biến, nhưng khi xảy ra (ví dụ major shipping accident với multiple vessels), instance-level information rất hữu ích.

## 6.5. Thách thức Đặc thù của Oil Spill Detection

### 6.5.1. Class Imbalance

Oil spill là rare event - trong hầu hết SAR images, không có oil spill. Ngay cả khi có, vùng dầu thường chiếm tỷ lệ rất nhỏ của image. Class imbalance này đặt ra thách thức cho training:

**Vấn đề:**
- Model có thể predict "no oil" cho mọi pixel và vẫn đạt accuracy cao
- Rare positive class bị overwhelm bởi negative class
- Gradient updates dominated bởi negative samples

**Giải pháp:**
- **Focal Loss:** Giảm weight cho easy (negative) samples, focus vào hard (positive) samples
- **Dice Loss / IoU Loss:** Optimize trực tiếp overlap metric thay vì per-pixel accuracy
- **Class weighting:** Tăng weight cho positive class trong loss function
- **Oversampling:** Sample nhiều hơn từ images có oil spill
- **Synthetic data augmentation:** Generate thêm positive samples

### 6.5.2. Look-alike Discrimination

Như đã thảo luận, phân biệt oil spill và look-alikes là thách thức lớn nhất. Các features có thể giúp phân biệt:

**Shape features:**
- Oil spill thường có irregular shape, theo hướng gió/dòng chảy
- Look-alikes từ wind có thể có linear patterns
- Internal waves có periodic patterns

**Texture features:**
- Oil spill có thể có variation trong intensity (thick vs thin regions)
- Look-alikes có texture đồng nhất hơn

**Context features:**
- Proximity to shipping lanes, platforms, pipelines
- Wind direction vs. spill orientation
- Temporal persistence (oil persists longer than weather-related look-alikes)

**Intensity features:**
- Contrast ratio với surrounding water
- Absolute backscatter values (oil thường darker)

Deep learning models có thể learn các features này implicitly, nhưng việc encode domain knowledge vào architecture hoặc loss function có thể cải thiện performance.

### 6.5.3. Variability của Oil Spill Appearance

Oil spill appearance trong SAR imagery rất diverse:

**Theo loại dầu:**
- Crude oil: Very dark, high contrast
- Refined products (diesel, gasoline): Lighter, lower contrast
- Vegetable oils: Similar to mineral oils nhưng weathering khác

**Theo weathering state:**
- Fresh spill: High contrast, well-defined edges
- Weathered spill: Lower contrast, diffuse edges
- Emulsified: Can appear lighter, patchy

**Theo wind conditions:**
- Low wind: Large, coherent dark patch
- Moderate wind: Elongated in wind direction, may have streaks
- High wind: Broken into smaller patches, reduced detectability

**Theo SAR acquisition parameters:**
- Incidence angle: Lower angles give higher contrast
- Polarization: HH often gives better contrast than VV
- Resolution: Higher resolution shows more detail but more speckle

### 6.5.4. Annotation Challenges

Creating high-quality ground truth for oil spill detection là particularly challenging:

**Verification difficulty:**
- Ground truth often requires confirmation từ multiple sources (aerial survey, in-situ sampling, AIS data)
- Không phải lúc nào cũng có confirmation
- Some "look-alikes" may actually be undocumented spills

**Temporal mismatch:**
- Oil spill là dynamic phenomenon - nó di chuyển, lan rộng, và biến đổi theo thời gian
- SAR acquisition time may not match verification time
- Drift and diffusion can significantly change extent

**Subjective boundaries:**
- Oil spill edges often không sharp và clear
- Annotators có thể disagree về exact boundary
- Thin sheen vs. thick oil regions khó phân biệt

**Limited positive samples:**
- Major spills (ví dụ Deepwater Horizon, Prestige) có extensive documentation
- Minor spills và illegal discharges thường không được document
- Synthetic/simulated data có thể không capture all real-world variations

### 6.5.5. Generalization across Regions

Model trained trên data từ một region có thể không generalize tốt sang region khác:

**Environmental differences:**
- Sea state, typical wind patterns
- Water temperature, salinity
- Biological activity (amount of biogenic films)
- Typical look-alike types

**Sensor differences:**
- Different SAR satellites có different characteristics
- Same satellite nhưng different modes (resolution, polarization)
- Processing level và calibration differences

**Oil type differences:**
- Different regions have different dominant oil types (crude vs refined, heavy vs light)
- Local shipping patterns affect spill characteristics

Transfer learning và domain adaptation techniques có thể giúp address generalization issues.

## 6.6. Metrics Đánh giá

### 6.6.1. Pixel-level Metrics

**IoU (Intersection over Union):**
Metric phổ biến nhất cho segmentation, đo overlap giữa predicted mask và ground truth:
IoU = TP / (TP + FP + FN)
trong đó TP = true positives, FP = false positives, FN = false negatives.

IoU range từ 0 (không overlap) đến 1 (perfect overlap). Cho oil spill detection, IoU > 0.5 thường được coi là acceptable detection.

**Dice Coefficient (F1 Score):**
Dice = 2TP / (2TP + FP + FN)
Dice có mối quan hệ với IoU: Dice = 2×IoU / (1 + IoU). Dice thường cho giá trị cao hơn IoU cho cùng prediction.

**Precision và Recall:**
- Precision = TP / (TP + FP): Tỷ lệ positive predictions là correct
- Recall = TP / (TP + FN): Tỷ lệ actual positives được detect

Trong oil spill detection, recall thường được prioritize (không bỏ sót spill quan trọng hơn là có một số false alarms).

### 6.6.2. Object-level Metrics

Đánh giá ở object level (mỗi vết dầu là một object):

**Detection Rate:**
Số oil spills được detect / Tổng số oil spills trong ground truth

**False Alarm Rate:**
Số false positive detections / (Số detections hoặc diện tích scan)

**Area Error:**
|Predicted Area - True Area| / True Area

### 6.6.3. Operational Metrics

Trong operational context, các metrics khác cũng quan trọng:

**Detection Latency:**
Thời gian từ khi acquire image đến khi có detection result.

**Alert Quality:**
Tỷ lệ alerts dẫn đến verified oil spills (sau khi operator review).

**Coverage:**
Diện tích biển được monitor per unit time.

## 6.7. So sánh với Ship Detection

### 6.7.1. Điểm Tương đồng

| Aspect | Ship Detection | Oil Spill Detection |
|--------|----------------|---------------------|
| Primary sensor | SAR | SAR |
| Challenge | Small objects, clutter | Look-alikes, variability |
| Data imbalance | Có (nhiều background) | Có (rất ít positive) |
| Need for context | Maritime traffic patterns | Wind, current data |

### 6.7.2. Điểm Khác biệt

| Aspect | Ship Detection | Oil Spill Detection |
|--------|----------------|---------------------|
| Object appearance | Bright point/small object | Dark extended region |
| Task type | Detection (bbox) | Segmentation (mask) |
| Shape | Relatively uniform | Highly variable |
| Temporal behavior | Moving objects | Drifting, spreading regions |
| Ground truth | AIS correlation possible | Verification more difficult |

### 6.7.3. Potential Synergies

Ship và oil spill detection có thể được kết hợp:
- Ship detection để identify potential sources
- Oil spill detection để identify consequences
- Joint model có thể share features
- Ship trails (wake) có thể confuse với narrow oil slicks

## 6.8. Các Hệ thống Giám sát Thực tế

### 6.8.1. CleanSeaNet (European Maritime Safety Agency)

CleanSeaNet là hệ thống giám sát oil spill của EMSA cho vùng biển châu Âu:
- Sử dụng SAR imagery từ Sentinel-1 và commercial satellites
- Near real-time detection và alerting
- Covers European waters và neighboring regions
- Combines automatic detection với operator verification
- Đã hoạt động từ 2007

### 6.8.2. NOAA và US Coast Guard

Hệ thống giám sát của Mỹ:
- Kết hợp satellite (SAR, optical) với aerial surveillance
- Integrated với AIS data
- Focus on US waters và areas of interest

### 6.8.3. Commercial Solutions

Nhiều công ty cung cấp commercial oil spill monitoring:
- **SkyTruth:** NGO sử dụng satellite data cho environmental monitoring
- **Orbital Insight:** AI-powered analytics including oil spill detection
- **Windward:** Maritime intelligence including pollution monitoring

Các hệ thống này thường kết hợp:
- Multiple satellite data sources
- Machine learning detection algorithms
- Human-in-the-loop verification
- Integration với maritime traffic data
- Alert và reporting systems


## Kết chương

Chương này đã trình bày ứng dụng các kỹ thuật deep learning cho phát hiện dầu loang trên ảnh SAR. Khác với phát hiện tàu biển (**Chương 5**) sử dụng object detection, bài toán này yêu cầu semantic segmentation để xác định vùng dầu loang với ranh giới không đều. Các kiến trúc encoder-decoder như U-Net, DeepLabV3+ (**Chương 3.3**) được điều chỉnh cho dữ liệu SAR single-channel và multi-polarization.

Thách thức chính là phân biệt oil spill thực sự với look-alikes (hiện tượng tự nhiên tạo dark spots tương tự). Multi-temporal analysis và integration với ancillary data (wind, AIS, currents) giúp cải thiện độ chính xác. Quy trình end-to-end từ data acquisition đến alerting cho phép giám sát môi trường biển near real-time.

**Ứng dụng tại Việt Nam:** Vùng biển Việt Nam với điều kiện thời tiết nhiệt đới và mây che phủ cao đặc biệt phù hợp cho giám sát bằng SAR. Hệ thống tự động phát hiện dầu loang có thể hỗ trợ cơ quan chức năng trong việc giám sát các tuyến đường vận chuyển dầu, phát hiện sự cố và xả thải bất hợp pháp, bảo vệ hệ sinh thái biển và vùng ven biển.

Hai chương ứng dụng (phát hiện tàu và dầu loang) đã minh họa cách áp dụng các kiến trúc deep learning vào bài toán viễn thám thực tế. **Chương 7** sẽ tổng kết các nội dung chính và đề xuất hướng phát triển trong tương lai.
