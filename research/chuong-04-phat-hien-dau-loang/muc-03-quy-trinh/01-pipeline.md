# Chương 4: Quy trình Oil Spill Detection Pipeline

## 5.19. Tổng quan Pipeline

Oil spill detection pipeline là quy trình hoàn chỉnh từ thu thập dữ liệu SAR đến cảnh báo và báo cáo cuối cùng cho các operators và cơ quan chức năng. Pipeline điển hình bao gồm các giai đoạn: Data Acquisition, Preprocessing, Detection/Segmentation, Post-processing, Verification, và Alerting. Mỗi giai đoạn đóng vai trò quan trọng trong việc đảm bảo detection chính xác và timely response.

Thiết kế pipeline cần cân nhắc các yếu tố: yêu cầu về latency (near real-time vs offline analysis), độ chính xác cần thiết, tỷ lệ false alarm chấp nhận được, tài nguyên tính toán, và integration với hệ thống downstream (cơ quan quản lý, đội ứng phó). Pipeline cho hệ thống giám sát operational sẽ khác với pipeline cho nghiên cứu retrospective.

## 5.20. Data Acquisition

### 5.20.1. SAR Data Sources

**Sentinel-1 (ESA):**
- **Characteristics:** C-band SAR, dual polarization (VV+VH), GSD 10m (IW mode), 20m×22m (EW mode)
- **Coverage:** Global, systematic acquisition plan
- **Access:** Miễn phí qua Copernicus Data Space, Sentinel Hub
- **Products:** GRD (Ground Range Detected) phổ biến nhất cho oil spill detection
- **Timeliness:** Data available within hours of acquisition

**RADARSAT Constellation Mission (MDA):**
- **Characteristics:** C-band, multiple polarization modes, resolution 1-50m depending on mode
- **Advantages:** Tasking capability cho specific areas
- **Access:** Commercial, hoặc qua agreements với Canadian Space Agency

**ICEYE:**
- **Characteristics:** X-band, very high resolution (0.25-1m)
- **Advantages:** High resolution cho detailed spill mapping, frequent revisit
- **Access:** Commercial

**TerraSAR-X / TanDEM-X:**
- **Characteristics:** X-band, high resolution
- **Advantages:** Proven performance cho oil spill detection
- **Access:** Commercial/research agreements

### 5.20.2. Ancillary Data

**Wind Data:**
Crucial cho oil spill detection quality:
- **ECMWF ERA5:** Reanalysis data, global coverage, ~0.25° resolution
- **NCEP GFS:** Forecast data, global coverage
- **Sentinel-1 derived:** Wind speed/direction estimated từ SAR image itself
- **In-situ:** Buoys, ships (limited spatial coverage)

Wind information used to:
- Assess detection reliability (wind 3-10 m/s optimal)
- Understand look-alike likelihood (low wind areas)
- Model oil drift trajectory

**AIS (Automatic Identification System):**
- Track vessel positions
- Identify potential spill sources
- Cross-reference detected ships với spill locations

**Bathymetry:**
- Identify areas của natural seeps
- Understand oceanographic context

**Ocean Current Data:**
- Model oil drift
- Predict future spill extent

### 5.20.3. Data Download Automation

**Sentinel-1 Access:**
```
Copernicus Data Space API
Sentinel Hub API
Google Earth Engine (GEE)
Amazon Web Services (AWS) Open Data
```

Automated download workflow:
1. Define area of interest (AOI) và time range
2. Query available products
3. Filter by mode, orbit direction, coverage
4. Download selected products
5. Trigger preprocessing pipeline

**Scheduling:**
- Continuous monitoring: Download new products as they become available
- Event-driven: Download specific products when alert received
- Historical analysis: Bulk download past products

## 5.21. SAR Preprocessing

### 5.21.1. Standard Processing Chain

SAR preprocessing cho oil spill detection follows similar steps như cho ship detection, với một số considerations đặc thù.

**1. Apply Orbit File:**
Cập nhật precise orbit information cho accurate geolocation. Essential cho combining với ancillary data.

**2. Thermal Noise Removal:**
Remove thermal noise artifacts, đặc biệt quan trọng cho oil spill detection vì:
- Noise có thể mask subtle dark features
- Edge effects có thể create false dark regions
- Improves signal-to-noise ratio cho weak oil signals

**3. Radiometric Calibration:**
Convert to sigma naught (σ°):
- Standard radiometric quantity cho surface scattering
- Enables comparison across different images/acquisitions
- Required cho quantitative analysis

**4. Speckle Filtering:**
Critical choice cho oil spill detection:
- **Lee Filter:** Common choice, moderate smoothing
- **Lee Sigma:** Better edge preservation
- **Gamma-MAP:** Good balance
- **IDAN (Intensity-Driven Adaptive-Neighborhood):** Adaptive to local statistics

**Trade-off considerations:**
- Too little filtering: Speckle noise creates false dark spots
- Too much filtering: Blurs oil spill boundaries, may smooth out thin slicks
- Recommendation: Moderate filtering với edge preservation, hoặc process multiple filter strengths và compare

**5. Land Masking:**
Essential step - oil spills only occur on water:
- Use high-resolution coastline data (GSHHG, OpenStreetMap)
- Buffer coastline to avoid edge effects
- Account cho inaccuracies in geolocation
- Consider tidal variations cho coastal areas

**6. Terrain Correction:**
Geocoding to standard map projection:
- Range-Doppler Terrain Correction với DEM
- Output projection: UTM hoặc WGS84 lat/lon
- Resampling: Bilinear hoặc cubic

### 5.21.2. Enhanced Preprocessing cho Oil Spill

**Incidence Angle Normalization:**
Backscatter varies với incidence angle. Normalizing to reference angle cho consistent appearance across image và between images.

**Adaptive Thresholding:**
Pre-compute statistics cho adaptive threshold detection:
- Mean và standard deviation của water areas
- Local statistics trong sliding window
- Helps với subsequent detection step

**Multi-temporal Compositing:**
Khi có multiple acquisitions:
- Create temporal median/mean để reduce transient phenomena
- Highlight persistent features (oil vs temporary look-alikes)
- Requires careful co-registration

### 5.21.3. Tiling Strategy

Cho large SAR scenes (typical Sentinel-1 IW product: ~30,000 × 20,000 pixels):

**Fixed Grid Tiling:**
- Divide scene into fixed-size tiles (e.g., 512×512, 1024×1024)
- Overlap 10-25% để avoid missing features ở boundaries
- Process each tile independently
- Merge results in post-processing

**Content-aware Tiling:**
- First pass: Coarse detection của candidate regions
- Second pass: Tile around candidates at higher resolution
- More efficient cho sparse detections (oil spill is rare)

**Considerations:**
- Large tiles: More context, higher memory requirement
- Small tiles: Less context, faster processing
- Oil spills có thể extend across multiple tiles - merging essential

## 5.22. Detection/Segmentation

### 5.22.1. Two-phase Approach

Nhiều operational systems sử dụng two-phase approach:

**Phase 1 - Candidate Detection:**
Quick screening để find all potential dark spots:
- Adaptive threshold below local mean
- Morphological operations để clean up
- Connected component analysis
- High recall, accept false positives

**Phase 2 - Classification/Segmentation:**
Detailed analysis của each candidate:
- Deep learning segmentation model
- Feature extraction và classification
- Look-alike discrimination
- High precision, reject false positives

### 5.22.2. End-to-End Deep Learning

Alternative approach - single deep learning model cho entire segmentation:

**Input:** Preprocessed SAR tile (single or multi-channel)

**Output:** Segmentation mask (binary or multi-class)

**Processing:**
- Feed tile through trained network
- Get probability map
- Apply threshold to get binary mask
- Optional: Multiple thresholds cho confidence levels

### 5.22.3. Multi-scale Processing

Process same image ở multiple scales:
- Large scale (downsampled): Detect large spills, global context
- Original scale: Standard detection
- Fine scale (overlapping patches): Detect small/thin features

Merge predictions từ different scales:
- Union: Include all detections (higher recall)
- Intersection: Only consistent detections (higher precision)
- Weighted average: Based on confidence

### 5.22.4. Confidence Estimation

Output confidence scores cho predictions:
- Model output probability (softmax/sigmoid)
- Monte Carlo dropout: Multiple forward passes với dropout
- Ensemble variance: Disagreement between ensemble members

Confidence helps prioritize alerts và guide operator attention.

## 5.23. Post-processing

### 5.23.1. Morphological Operations

Clean up segmentation output:

**Opening (Erosion followed by Dilation):**
- Remove small isolated false positives
- Smooth boundaries

**Closing (Dilation followed by Erosion):**
- Fill small holes within detected regions
- Connect nearby components

**Size filtering:**
- Remove detections below minimum size threshold
- Oil spills have minimum detectable size depending on resolution

### 5.23.2. Connected Component Analysis

Identify và analyze individual detections:
- Label connected regions
- Compute area của each region
- Compute shape metrics (perimeter, elongation, solidity, circularity)
- Use shape metrics for additional filtering (oil spills have characteristic shapes)

### 5.23.3. Merging Tiles

Reconstruct full-scene results từ tile-based processing:

**Overlap Handling:**
- For overlapping regions, average predictions
- Or use maximum prediction (more conservative for detection)
- Or use learned fusion

**Stitching:**
- Place tile predictions in global coordinate system
- Handle edge effects (predictions near tile boundaries may be less reliable)
- Verify connected components spanning multiple tiles are properly joined

### 5.23.4. Look-alike Filtering

Apply additional filters based on domain knowledge:

**Wind-based Filtering:**
- If local wind speed < 3 m/s, mark as low confidence (possible low-wind look-alike)
- If wind > 10 m/s, detection less reliable overall

**Proximity Filtering:**
- High probability if near shipping lane, platform, or known seep location
- Lower probability if in open ocean far from potential sources

**Shape-based Filtering:**
- Very circular shapes less likely oil (biogenic slicks often more circular)
- Very linear, narrow shapes may be ship wakes, not oil

**Temporal Filtering:**
- If same feature detected in multiple acquisitions: More likely persistent oil or seep
- If feature only in one acquisition: Possible transient look-alike

## 5.24. Verification và Fusion

### 5.24.1. Multi-source Verification

Cross-reference SAR detections với other data:

**AIS Correlation:**
- Check for vessels near detected spill
- Identify potential source vessel
- Flag "dark ships" (SAR detected, no AIS) as suspicious

**Optical Imagery:**
- If available, check optical image của same area/time
- Confirm presence của oil signature in optical
- Note: Optical may not always confirm (thin oil, cloud cover)

**Previous Detections:**
- Compare với historical detection database
- Identify known seep locations vs new spills
- Track evolution của ongoing spills

### 5.24.2. Confidence Scoring

Combine multiple factors into overall confidence score:

| Factor | Weight | Example |
|--------|--------|---------|
| Model confidence | 0.3 | Probability từ DL model |
| Wind conditions | 0.2 | Optimal wind → high score |
| Shape metrics | 0.2 | Oil-like shape → high score |
| Source proximity | 0.15 | Near shipping lane → higher |
| Historical context | 0.15 | Known seep → different handling |

Final confidence score guides operator attention và alert priority.

### 5.24.3. Human-in-the-loop

Operational systems typically include human verification:

**Operator Review:**
- Examine high-confidence detections
- Compare với multiple information sources
- Make final determination (confirmed oil, probable, possible, likely look-alike)

**Feedback Loop:**
- Operator decisions recorded
- Used để improve model through retraining
- Builds ground truth database

## 5.25. Alerting và Reporting

### 5.25.1. Alert Generation

Automatic alert generation based on detection results:

**Alert Content:**
- Location (lat/lon coordinates)
- Detection time và satellite acquisition time
- Estimated area/extent
- Confidence level
- Nearby vessels từ AIS
- Wind conditions
- Image chip/thumbnail

**Alert Levels:**
- Level 1 (Low): Possible detection, needs review
- Level 2 (Medium): Probable oil spill, operator review required
- Level 3 (High): Confirmed oil spill, immediate action may be needed

### 5.25.2. Output Formats

**GeoJSON:**
Standard format cho geospatial features, easily integrated với GIS systems:
```json
{
  "type": "Feature",
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[lon1, lat1], [lon2, lat2], ...]]
  },
  "properties": {
    "detection_time": "2024-01-15T10:30:00Z",
    "satellite": "Sentinel-1A",
    "confidence": 0.87,
    "area_km2": 12.5,
    "alert_level": 2
  }
}
```

**Shapefile:**
Compatible với traditional GIS software (ArcGIS, QGIS).

**KML/KMZ:**
Cho visualization trong Google Earth.

**PDF Report:**
Formatted report cho non-technical stakeholders.

### 5.25.3. Integration với Response Systems

**Maritime Authorities:**
- Automatic notification to coast guard, environmental agencies
- Integration với existing monitoring systems (CleanSeaNet, NOAA)

**Response Teams:**
- Alert dispatch to cleanup teams
- Trajectory prediction cho planning response

**Legal/Enforcement:**
- Evidence package for potential prosecution
- Chain of custody considerations

## 5.26. Trajectory Modeling

### 5.26.1. Oil Drift Prediction

Once oil spill detected, predict future trajectory:

**Factors Affecting Drift:**
- Surface currents (3-4% of current speed)
- Wind (3-4% of wind speed, với some deflection angle)
- Waves
- Oil properties (density, viscosity)

**Models:**
- NOAA GNOME (General NOAA Operational Modeling Environment)
- OSCAR (Oil Spill Contingency And Response)
- OpenDrift (open-source)

### 5.26.2. Integration với Detection Pipeline

Trajectory prediction helps:
- Forecast where oil will be in coming hours/days
- Guide response vessel deployment
- Identify threatened coastal areas
- Plan future satellite acquisitions to track spill

## 5.27. Performance Monitoring

### 5.27.1. System Metrics

Track operational performance:

**Detection Metrics:**
- Number of alerts per day/week
- Confirmed oil spills detected
- Missed detections (if ground truth available)
- False positive rate

**Latency Metrics:**
- Time từ satellite acquisition to data available
- Time từ data available to detection complete
- Time từ detection to alert sent
- Total end-to-end latency

**Coverage Metrics:**
- Area monitored per day
- Gap areas (not covered by recent acquisitions)
- Revisit frequency cho priority areas

### 5.27.2. Model Performance Tracking

Monitor model performance over time:
- Accuracy metrics on new ground truth
- Drift detection (performance degradation)
- Seasonal variations
- Performance by region

### 5.27.3. Continuous Improvement

Feedback-driven improvement:
- Incorporate operator feedback
- Retrain models periodically với new data
- A/B testing of model updates
- Adjust thresholds based on operational requirements

## 5.28. Scalability Considerations

### 5.28.1. Processing Large Volumes

Handling global monitoring:
- Sentinel-1 produces ~3 TB/day của data
- Need efficient processing pipeline
- Cloud computing enables scalability

**Strategies:**
- Prioritize high-risk areas (shipping lanes, platforms)
- Use quick screening to reduce processing load
- Parallel processing across tiles/images
- Queue management cho processing jobs

### 5.28.2. Cloud Deployment

Cloud platforms cho oil spill detection:

**Google Earth Engine:**
- Direct access to Sentinel-1 archive
- Processing at scale
- Good cho research và prototyping

**AWS/Azure/GCP:**
- Custom pipeline deployment
- Integration với ML services
- Scalable compute resources

**Copernicus Data Space:**
- Direct access to Copernicus data
- Processing close to data
- European infrastructure

### 5.28.3. Edge Processing

Cho near real-time applications:
- Process on board satellite (future capability)
- Process at ground station immediately after downlink
- Reduce latency to minutes instead of hours

## 5.29. Example End-to-End Pipeline

### 5.29.1. Workflow Description

1. **Data Ingestion:**
   - Monitor Sentinel-1 data availability
   - Download new products for AOI
   - Store in processing queue

2. **Preprocessing:**
   - Apply orbit file
   - Thermal noise removal
   - Radiometric calibration
   - Speckle filtering (Lee Sigma)
   - Land masking
   - Geocoding to UTM

3. **Tiling:**
   - Divide scene into 1024×1024 tiles với 128 pixel overlap
   - Generate tile metadata (coordinates, statistics)

4. **Detection:**
   - Run U-Net model on each tile
   - Output probability maps
   - Apply threshold (0.5 for binary, multiple for confidence levels)

5. **Post-processing:**
   - Morphological cleaning
   - Merge tiles
   - Connected component analysis
   - Size and shape filtering

6. **Verification:**
   - Query AIS for nearby vessels
   - Query wind data for conditions
   - Compute confidence scores

7. **Alerting:**
   - Generate GeoJSON output
   - Create alert notifications
   - Send to operators/systems

8. **Archiving:**
   - Store results in database
   - Index cho future queries
   - Link to source data

### 5.29.2. Latency Budget

| Stage | Target Time |
|-------|-------------|
| Data availability after acquisition | 1-3 hours |
| Download và preprocessing | 10-30 minutes |
| Detection processing | 5-15 minutes |
| Post-processing và verification | 5-10 minutes |
| Alert generation | 1-2 minutes |
| **Total end-to-end** | **1.5-4 hours** |

Near real-time systems aim cho total latency < 1 hour after acquisition.

