# Quy trình Ship Detection Pipeline

## 4.12. Tổng quan Pipeline

Ship detection pipeline là quy trình hoàn chỉnh từ thu thập dữ liệu vệ tinh đến output cuối cùng có thể sử dụng cho các ứng dụng downstream. Pipeline điển hình bao gồm các giai đoạn: Data Acquisition, Preprocessing, Detection, Post-processing, và Integration. Mỗi giai đoạn có vai trò quan trọng và ảnh hưởng đến chất lượng kết quả cuối cùng.

Thiết kế pipeline cần cân nhắc nhiều yếu tố: loại ảnh (SAR/optical), yêu cầu về độ chính xác, tốc độ xử lý, tài nguyên tính toán, và mục đích sử dụng cuối cùng. Pipeline cho hệ thống giám sát real-time sẽ khác với pipeline cho phân tích historical archives.

## 4.13. Data Acquisition

### 4.13.1. SAR Data Sources

**Sentinel-1 (ESA):**
- **Đặc điểm:** C-band SAR, dual polarization (VV+VH), GSD 10m (IW mode)
- **Coverage:** Global, 6-day revisit với constellation
- **Access:** Miễn phí qua Copernicus Data Space
- **Products:** GRD (Ground Range Detected), SLC (Single Look Complex)
- **Kích thước:** ~1GB mỗi scene, coverage ~250×250 km

**TerraSAR-X / TanDEM-X (DLR):**
- **Đặc điểm:** X-band, GSD 1-3m (Spotlight mode)
- **Ưu điểm:** Resolution cao hơn Sentinel-1
- **Hạn chế:** Commercial, coverage hạn chế

**ICEYE:**
- **Đặc điểm:** X-band, GSD 0.25-1m
- **Ưu điểm:** Tasking flexibility, very high resolution
- **Hạn chế:** Commercial pricing

**Capella Space:**
- **Đặc điểm:** X-band, GSD 0.5m
- **Ưu điểm:** High resolution SAR
- **Hạn chế:** Commercial

### 4.13.2. Optical Data Sources

**WorldView-3 (Maxar):**
- **Đặc điểm:** GSD 0.31m panchromatic, 1.24m multispectral
- **Ưu điểm:** Highest resolution commercial optical
- **Hạn chế:** Cloud obstruction, commercial pricing

**Planet (SkySat, SuperDove):**
- **Đặc điểm:** GSD 0.5-3m, daily global coverage
- **Ưu điểm:** High revisit frequency
- **Hạn chế:** Lower resolution, commercial

**Sentinel-2 (ESA):**
- **Đặc điểm:** GSD 10m (RGB), 13 bands multispectral
- **Ưu điểm:** Free, good spectral information
- **Hạn chế:** Resolution không đủ cho small ships

### 4.13.3. Data Download và Management

**Automated Download:**
Sử dụng APIs và tools để tự động download data theo schedule hoặc trigger:
- `sentinelsat` Python package cho Sentinel data
- `eodag` cho multiple data sources
- AWS/Google Cloud open data programs

**Data Organization:**
Cấu trúc thư mục rõ ràng theo sensor, date, và region:
```
data/
├── sentinel1/
│   ├── 2024/
│   │   ├── 01/
│   │   │   ├── S1A_IW_GRDH_20240101_...
│   │   │   └── ...
```

**Metadata Management:**
Lưu trữ metadata (acquisition time, bounds, polarization) trong database để query và tracking.

## 4.14. Preprocessing

### 4.14.1. SAR Preprocessing Pipeline

SAR data yêu cầu nhiều bước preprocessing trước khi sử dụng cho detection:

**1. Apply Orbit File:**
Cập nhật thông tin quỹ đạo chính xác để cải thiện geolocation accuracy.

**2. Thermal Noise Removal:**
Loại bỏ thermal noise artifacts, đặc biệt ở các vùng azimuth của ảnh.

**3. Calibration:**
Chuyển đổi digital numbers sang radiometric values:
- Sigma0 (σ°): Backscatter coefficient, phổ biến nhất cho ship detection
- Beta0 (β°): Radar brightness
- Gamma0 (γ°): Terrain-flattened

**4. Speckle Filtering:**
Giảm speckle noise trong khi giữ edge information:
- Lee Filter: Adaptive filter phổ biến
- Gamma-MAP: Maximum a posteriori estimation
- Frost Filter: Exponential model
- Non-local means: Preserve fine details

Trade-off: Aggressive filtering giảm noise nhưng có thể blur small ships.

**5. Terrain Correction:**
Chuyển đổi từ slant range sang ground range geometry:
- Range Doppler Terrain Correction với DEM
- Output: Geocoded product với projection chuẩn (UTM, WGS84)

**6. Land Masking:**
Tạo mask để loại bỏ land areas:
- Sử dụng global land-water masks (GSHHG, OpenStreetMap)
- Hoặc automatic water detection từ SAR intensity

**Tools phổ biến:**
- SNAP (ESA Sentinel Application Platform)
- GDAL cho raster operations
- Python packages: `snappy`, `rasterio`, `pyrosar`

### 4.14.2. Optical Preprocessing Pipeline

**1. Atmospheric Correction:**
Loại bỏ ảnh hưởng của khí quyển:
- Convert DN to TOA (Top of Atmosphere) reflectance
- Convert TOA to BOA (Bottom of Atmosphere) nếu cần
- Tools: Sen2Cor cho Sentinel-2, FLAASH cho commercial imagery

**2. Pan-sharpening (nếu cần):**
Merge high-resolution panchromatic với lower-resolution multispectral.

**3. Cloud Masking:**
Detect và mask vùng mây:
- Sentinel-2 Cloud Mask (SCL band)
- Fmask algorithm
- Deep learning-based cloud detection

**4. Sun Glint Correction:**
Giảm phản xạ ánh sáng từ mặt biển nếu cần.

### 4.14.3. Tiling Strategy

Ảnh vệ tinh lớn cần được chia thành tiles để xử lý:

**Tile Size:**
- Phổ biến: 512×512, 640×640, 1024×1024 pixels
- Larger tiles: Nhiều context nhưng cần nhiều memory
- Smaller tiles: Faster processing nhưng có thể cắt objects

**Overlap:**
- Overlap 10-25% để tránh miss objects ở boundaries
- Objects detected ở overlap region cần merge trong post-processing

**Stride:**
- Stride = Tile Size - Overlap
- Ví dụ: Tile 1024, Overlap 256 → Stride 768

**Dynamic Tiling:**
Một số systems sử dụng adaptive tiling dựa trên content complexity.

## 4.15. Data Augmentation

### 4.15.1. Geometric Augmentations

**Random Horizontal/Vertical Flip:**
Ships có thể xuất hiện theo mọi hướng, flip không thay đổi semantic.

**Random Rotation:**
- 90°, 180°, 270° rotations
- Arbitrary angle rotation (với padding hoặc crop)
- Đặc biệt quan trọng cho ship orientation diversity

**Random Scale:**
- Scale 0.8× đến 1.2× để simulate distance variation
- Cần adjust annotations tương ứng

**Random Crop:**
- Crop random regions từ larger tiles
- Giúp model học detect ships ở mọi vị trí

### 4.15.2. Photometric Augmentations (Optical)

**Brightness/Contrast:**
- Random adjustment ±10-20%
- Simulate different lighting conditions

**Color Jittering:**
- Slight variations trong RGB channels
- Không áp dụng cho SAR

**Gaussian Noise:**
- Add random noise để tăng robustness

### 4.15.3. SAR-specific Augmentations

**Speckle Noise Simulation:**
Thêm simulated speckle với various intensities để tăng robustness với speckle.

**Intensity Scaling:**
Random scaling của intensity values (trong reasonable range).

### 4.15.4. Advanced Augmentations

**Mosaic:**
Combine 4 images thành 1, mỗi góc từ image khác nhau. Tăng variety và context.

**MixUp:**
Blend 2 images với weighted average, blend labels tương ứng. Regularization hiệu quả.

**CutMix:**
Cut patch từ image này paste vào image khác, adjust labels. Combine benefits của cutout và mixup.

**Copy-Paste:**
Copy ships từ image này paste vào image khác. Tăng số lượng ships trong training, đặc biệt hữu ích cho rare ship types.

## 4.16. Model Training

### 4.16.1. Transfer Learning Strategy

**From ImageNet:**
- Load backbone pre-trained trên ImageNet
- Modify first conv layer nếu số kênh khác (SAR có 2 kênh)
- Fine-tune entire network hoặc freeze early layers

**From Satellite Pretrained:**
- TorchGeo cung cấp weights trained trên satellite data
- Sentinel-1 pretrained weights cho SAR ship detection
- Thường tốt hơn ImageNet pretrained cho viễn thám

**Layer-wise Learning Rate:**
- Lower learning rate cho early layers (general features)
- Higher learning rate cho later layers và new heads

### 4.16.2. Loss Functions

**Classification Loss:**
- Cross-Entropy hoặc Focal Loss
- Focal Loss preferred do class imbalance (nhiều background)

**Box Regression Loss:**
- Smooth L1 Loss (traditional)
- IoU-based losses (GIoU, DIoU, CIoU): Directly optimize IoU
- CIoU phổ biến: considers overlap, center distance, và aspect ratio

**Angle Loss (cho Oriented Detection):**
- Smooth L1 trên angle
- Circular Smooth Label (CSL)
- Gaussian Wasserstein Distance (GWD)

**Total Loss:**
```
L = λ_cls × L_cls + λ_box × L_box + λ_angle × L_angle
```
với λ weights tunable.

### 4.16.3. Optimizer và Scheduler

**Optimizers phổ biến:**
- SGD với momentum (0.9): Stable, proven
- Adam / AdamW: Faster convergence, need tuning
- AdamW preferred cho transformers và modern CNNs

**Learning Rate Schedulers:**
- Step Decay: Reduce LR at fixed epochs
- Cosine Annealing: Smooth decay following cosine
- OneCycleLR: Increase then decrease, good for fine-tuning

**Warmup:**
- Gradually increase LR từ small value trong first epochs
- Helps stability khi training từ scratch hoặc fine-tuning

### 4.16.4. Training Configuration

**Batch Size:**
- Larger batch: More stable gradients, need more memory
- Typical: 8-32 tùy GPU memory và image size

**Epochs:**
- Typical: 50-300 epochs tùy dataset size
- Early stopping dựa trên validation metrics

**Image Size:**
- Larger images: Better for small ships, more memory
- Common: 640, 800, 1024, 1280

## 4.17. Inference Pipeline

### 4.17.1. Sliding Window

Cho large satellite scenes:
1. Chia scene thành overlapping tiles
2. Run detection trên mỗi tile
3. Merge detections, handle overlaps

**Overlap Handling:**
- Detections trong overlap regions appear multiple times
- NMS across tiles để remove duplicates
- Hoặc weighted box fusion

### 4.17.2. Test Time Augmentation (TTA)

Apply augmentations lúc inference và aggregate predictions:
- Horizontal flip
- Vertical flip
- 90° rotations
- Scale variations

Merge predictions bằng NMS hoặc weighted box fusion. TTA tăng accuracy với cost là inference time.

### 4.17.3. Confidence Threshold

Filter detections với confidence score thấp:
- Typical threshold: 0.3-0.5
- Lower: More detections, more false positives
- Higher: Fewer detections, may miss some ships

Tune threshold dựa trên precision-recall requirements của application.

## 4.18. Post-processing

### 4.18.1. Non-Maximum Suppression (NMS)

Loại bỏ duplicate detections:
1. Sort detections by confidence
2. Keep highest confidence
3. Remove all detections với IoU > threshold với kept detection
4. Repeat cho remaining detections

**IoU Threshold:**
- Typical: 0.5-0.7
- Lower: More aggressive suppression
- Higher: Keep more overlapping detections (for dense ships)

### 4.18.2. Soft-NMS

Thay vì remove, reduce confidence của overlapping detections:
- Linear: score = score × (1 - IoU)
- Gaussian: score = score × exp(-IoU²/σ)

Better cho dense scenes với nhiều overlapping ships.

### 4.18.3. Weighted Box Fusion

Alternative to NMS, merge overlapping boxes:
- Combine boxes với weighted average based on confidence
- Produce single fused box
- Often better than NMS cho multi-model ensembles

### 4.18.4. Filtering

Additional filters based on domain knowledge:
- **Size filtering:** Remove detections too small hoặc too large cho reasonable ships
- **Location filtering:** Remove detections on land (với land mask)
- **Aspect ratio filtering:** Remove boxes với unusual aspect ratios
- **Confidence filtering:** Adjust thresholds per region (port vs open sea)

## 4.19. Integration và Output

### 4.19.1. AIS Correlation

Match detections với AIS (Automatic Identification System) data:
- Cross-reference detected ships với AIS positions
- Identify "dark ships" (detected but no AIS)
- Validate detections với known vessels

**Challenges:**
- AIS data có latency
- Position accuracy của cả SAR và AIS
- Temporal mismatch giữa image acquisition và AIS timestamp

### 4.19.2. Tracking

Cho sequential imagery hoặc video:
- Track ships across frames
- Assign consistent IDs
- Algorithms: SORT, DeepSORT, ByteTrack

### 4.19.3. Output Formats

**GeoJSON:**
Standard format cho geospatial features:
```json
{
  "type": "Feature",
  "geometry": {
    "type": "Point",
    "coordinates": [longitude, latitude]
  },
  "properties": {
    "confidence": 0.95,
    "class": "cargo_ship",
    "bbox": [x, y, w, h, angle]
  }
}
```

**Shapefile:**
Compatible với GIS software.

**COCO Format:**
Cho ML evaluation và benchmarking.

### 4.19.4. Visualization và Reporting

- Overlay detections trên base map
- Generate statistics (counts, density maps)
- Alert systems cho specific conditions
- Dashboard integration
