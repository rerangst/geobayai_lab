# Chương 6: Datasets cho Oil Spill Detection

## 6.30. Tổng quan về Oil Spill Datasets

Datasets đóng vai trò quan trọng trong việc training và benchmarking các models oil spill detection. Tuy nhiên, so với các domains khác của remote sensing như land cover classification hay ship detection, oil spill datasets có số lượng hạn chế hơn và thường có kích thước nhỏ hơn. Điều này xuất phát từ một số nguyên nhân: oil spill là rare event, ground truth verification khó khăn, và concerns về data sharing từ các tổ chức.

Các datasets có thể được phân loại theo nhiều tiêu chí: nguồn gốc (academic, competition, operational), loại annotation (bounding box, segmentation mask, point), số lượng classes (binary vs multi-class với look-alikes), và độ phức tạp của scenarios (controlled vs diverse real-world).

## 6.31. Datasets Chính cho SAR Oil Spill Detection

### 6.31.1. Oil Spill Detection Dataset (Kaggle/Zenodo)

Dataset phổ biến nhất cho SAR oil spill detection research, available trên Kaggle và Zenodo.

| Thuộc tính | Giá trị |
|------------|---------|
| **Nguồn vệ tinh** | Sentinel-1 |
| **Số lượng images** | ~1,000+ patches |
| **Kích thước patch** | Varied (typically 256×256 to 512×512) |
| **Annotation** | Pixel-level segmentation mask |
| **Classes** | Binary (oil spill vs background) hoặc Multi-class |

**Đặc điểm:**
- Curated patches từ Sentinel-1 imagery
- Mix của confirmed oil spills và look-alikes
- Multiple versions/variations available
- Commonly used benchmark cho research papers

**Phiên bản đáng chú ý:**

**1. Oil Spill SAR Dataset (Kaggle):**
- Approximately 1,112 images
- 5 classes: Oil Spill, Look-alike, Ship, Land, Sea
- Balanced training samples

**2. Extended versions:**
- Additional samples từ various research groups
- May include temporal sequences
- Regional variations (Mediterranean, Gulf of Mexico, etc.)

**Download:**
```
Kaggle: https://www.kaggle.com/datasets/
Zenodo: https://zenodo.org/search?q=oil%20spill%20sar
```

### 6.31.2. SOS Dataset (Sea Oil Spill)

Dataset được tạo từ nghiên cứu về oil spill detection.

| Thuộc tính | Giá trị |
|------------|---------|
| **Nguồn** | Sentinel-1, ENVISAT ASAR |
| **Coverage** | Mediterranean Sea, European waters |
| **Annotation** | Segmentation masks |
| **Classes** | Oil spill, Look-alike, Sea |

**Đặc điểm:**
- Carefully annotated với expert verification
- Includes look-alike samples cho discrimination training
- Multiple polarization data available cho some samples

### 6.31.3. NOAA Oil Spill Datasets

NOAA (National Oceanic and Atmospheric Administration) maintains data từ various oil spill events.

**Deepwater Horizon (2010):**
- Extensive SAR coverage từ multiple satellites
- One of largest documented spills
- Used in many research studies
- Temporal sequences showing spill evolution

**Other documented spills:**
- Various smaller spills in US waters
- Mix of SAR và optical data
- Often requires data request

**Access:**
```
NOAA NCEI: https://www.ncei.noaa.gov/
ERMA: https://erma.noaa.gov/
```

### 6.31.4. CleanSeaNet Data

European Maritime Safety Agency (EMSA) CleanSeaNet system has accumulated extensive detection data.

| Thuộc tính | Giá trị |
|------------|---------|
| **Coverage** | European waters |
| **Period** | 2007 - present |
| **Volume** | Thousands of detections per year |
| **Verification** | Many with ground/aerial verification |

**Access Limitations:**
- Not publicly available as dataset
- May be accessible for research through agreements
- Individual verified cases can be found in publications

### 6.31.5. Research Group Datasets

Various research groups have created custom datasets:

**Mediterranean Oil Spill Dataset:**
- Focus on Mediterranean region
- Includes natural seeps và anthropogenic spills
- Often used in European research projects

**Asian Seas Dataset:**
- Coverage of shipping lanes in Asia
- May include different oil types

**Gulf of Mexico Dataset:**
- Natural seeps common in this region
- Good cho seep vs spill discrimination
- Often combined với Deepwater Horizon data

## 6.32. Multi-class và Look-alike Datasets

### 6.32.1. Oil Spill Look-alike Dataset

Dedicated datasets cho look-alike discrimination:

| Classes | Description |
|---------|-------------|
| Oil Spill | Confirmed petroleum spills |
| Biogenic Slick | Natural organic films |
| Low Wind | Areas of reduced wind speed |
| Rain Cell | Rain-affected areas |
| Current Boundary | Ocean current shear zones |
| Ship Wake | Vessel trails |

**Importance:**
- Critical cho reducing false alarms
- Represents real operational challenge
- Enables training discriminative models

### 6.32.2. Extended Annotation Datasets

Some datasets provide additional annotations:

**Attributes:**
- Spill source (ship, platform, unknown)
- Estimated area
- Confidence level
- Weather conditions
- Oil type (if known)

**Temporal Information:**
- Acquisition time
- Spill age estimate
- Trajectory data

## 6.33. Competition Datasets

### 6.33.1. DARPA Challenges

DARPA has sponsored challenges related to maritime domain awareness including oil spill detection components:

**Characteristics:**
- Large-scale, realistic scenarios
- Multi-modal data (SAR, optical, AIS)
- Rigorous evaluation protocols
- Often restricted access post-competition

### 6.33.2. ESA/EMSA Challenges

European Space Agency và EMSA have sponsored oil spill detection challenges:

**Characteristics:**
- Focus on European waters
- Sentinel-1 data
- Operational relevance
- Integration với CleanSeaNet

### 6.33.3. Academic Competitions

Various academic institutions have organized oil spill detection competitions:

- IEEE GRSS Data Fusion Contest (some years include maritime)
- Regional competitions (Mediterranean, Baltic, etc.)
- Student challenges

## 6.34. Synthetic và Augmented Datasets

### 6.34.1. Simulated Oil Spill Data

Due to limited real oil spill samples, synthetic data generation is important:

**Physics-based Simulation:**
- Model SAR backscatter physics
- Simulate oil damping effect
- Add realistic speckle noise
- Generate various oil spill shapes và extents

**GAN-based Generation:**
- Train GAN on limited real samples
- Generate additional realistic samples
- Augment training data

### 6.34.2. Augmentation Strategies

Standard augmentation cho oil spill datasets:

**Geometric:**
- Rotation (any angle - oil spills have no preferred orientation)
- Horizontal/vertical flip
- Scale variation (simulate different spill sizes)
- Elastic deformation (simulate shape variations)

**Intensity:**
- Brightness adjustment
- Contrast variation
- Simulated speckle noise variation

**Domain-specific:**
- Copy-paste oil patches to different backgrounds
- Blend oil masks với various sea textures
- Simulate different wind conditions

### 6.34.3. Transfer Learning Datasets

Pre-training trên related datasets can improve oil spill detection:

**General SAR Datasets:**
- SpaceNet (urban SAR)
- BigEarthNet (Sentinel data)
- SEN12MS (SAR-optical pairs)

**Marine Remote Sensing:**
- Ship detection datasets (SSDD, HRSID)
- Sea ice datasets
- Ocean feature datasets

**Natural Image:**
- ImageNet (general pre-training)
- Places365 (scene understanding)

## 6.35. Optical Oil Spill Datasets

### 6.35.1. Satellite Optical Data

For optical oil spill detection:

| Source | Resolution | Availability |
|--------|------------|--------------|
| Sentinel-2 | 10m | Free, global |
| Landsat 8/9 | 30m | Free, global |
| WorldView | 0.3-1m | Commercial |
| Planet | 3-5m | Commercial |

**Challenges:**
- Cloud cover limits usability
- Fewer annotated samples than SAR
- Sun glint complicates detection

### 6.35.2. Aerial/Drone Imagery

Higher resolution data from airborne platforms:

- NOAA aerial surveys (US oil spills)
- Coast Guard imagery
- Research flight data

**Advantages:**
- Very high resolution
- Closer to ground truth
- Often used for verification

**Limitations:**
- Limited spatial coverage
- Event-specific (only major spills)
- Access restrictions

### 6.35.3. Multi-sensor Fusion Datasets

Datasets combining SAR và optical:

**Paired Observations:**
- Same spill captured by SAR và optical
- Enables sensor fusion research
- Rare due to timing requirements

**Complementary Data:**
- SAR detection + optical verification
- Multi-temporal sequences

## 6.36. Dataset Creation Guidelines

### 6.36.1. Data Collection

**Source Selection:**
- Prioritize confirmed oil spill events
- Include diverse geographic regions
- Balance different oil types và conditions
- Include look-alike examples

**Image Selection:**
- Optimal wind conditions (3-10 m/s)
- Good image quality (low noise)
- Clear oil signature visible
- Minimal land contamination

### 6.36.2. Annotation Guidelines

**Segmentation Mask Annotation:**

**Boundary Delineation:**
- Follow visible oil boundary
- Include thin sheen if visible
- Be consistent về what's included
- Document uncertainty in difficult cases

**Class Assignment:**
- Oil Spill: Confirmed petroleum
- Look-alike: Natural phenomena
- Unknown: Cannot determine với certainty

**Quality Control:**
- Multiple annotators for verification
- Inter-annotator agreement metrics
- Expert review for difficult cases
- Revision based on additional information

### 6.36.3. Metadata Collection

Essential metadata cho each sample:

| Field | Description |
|-------|-------------|
| Satellite | Sentinel-1A, TerraSAR-X, etc. |
| Acquisition Time | UTC timestamp |
| Processing Level | GRD, SLC, etc. |
| Polarization | VV, VH, etc. |
| Incidence Angle | Degrees |
| Wind Speed/Direction | From external source |
| Location | Lat/lon center hoặc bbox |
| Spill Source | Ship, platform, unknown |
| Verification | How confirmed |

### 6.36.4. Data Splits

Recommended splitting strategy:

**Geographic Split:**
- Training: Region A
- Validation: Region B
- Test: Region C

Prevents spatial autocorrelation leakage.

**Temporal Split:**
- Training: Before date X
- Validation/Test: After date X

Ensures generalization to future events.

**Random Split (baseline):**
- 70% training, 15% validation, 15% test
- Stratified by class và source

## 6.37. Dataset Challenges và Limitations

### 6.37.1. Limited Positive Samples

**Problem:**
- Oil spills are rare events
- Most images contain no oil
- Limited number of confirmed spills

**Mitigations:**
- Data augmentation
- Synthetic data generation
- Transfer learning
- Few-shot learning techniques

### 6.37.2. Annotation Quality

**Challenges:**
- Difficult to verify ground truth
- Expert disagreement on boundaries
- Temporal mismatch với verification
- Some "look-alikes" may be undocumented spills

**Recommendations:**
- Use multiple annotators
- Document confidence levels
- Separate certain từ uncertain samples
- Continuous refinement với new information

### 6.37.3. Geographic và Temporal Bias

**Issues:**
- Major documented spills dominate (Deepwater Horizon, Prestige)
- Certain regions over-represented
- Seasonal/temporal patterns may be captured

**Mitigation:**
- Diverse data collection
- Careful splitting strategies
- Explicit domain adaptation
- Multi-region validation

### 6.37.4. Access Restrictions

**Limitations:**
- Operational data often restricted
- Commercial satellite data expensive
- Some datasets require agreements
- Privacy/security concerns

**Recommendations:**
- Leverage open data (Sentinel-1)
- Collaborate với agencies
- Publish và share research data
- Use public archives

## 6.38. So sánh Datasets

### 6.38.1. Bảng So sánh Tổng hợp

| Dataset | Images | Classes | Annotation | Resolution | Access |
|---------|--------|---------|------------|------------|--------|
| Kaggle Oil Spill | ~1,000 | 5 | Segmentation | 10m | Public |
| SOS Dataset | ~500 | 3 | Segmentation | 10-20m | Request |
| Deepwater Horizon | ~100 scenes | 2 | Segmentation | Varied | Public/Request |
| CleanSeaNet | Thousands | Multi | Segmentation | 10-20m | Restricted |

### 6.38.2. Recommendations theo Use Case

**Research và Benchmarking:**
- Kaggle Oil Spill Dataset: Standard benchmark
- Combine với augmentation
- Report consistent metrics cho comparison

**Operational Development:**
- CleanSeaNet data (if accessible)
- Regional datasets for specific deployment
- Include extensive look-alike samples

**Transfer Learning:**
- Pre-train on larger SAR datasets
- Fine-tune on oil spill specific data
- Consider semi-supervised approaches

**Multi-sensor Research:**
- Collect paired SAR-optical data
- Leverage verified events với multi-source coverage
- Deepwater Horizon comprehensive coverage

## 6.39. Accessing và Using Datasets

### 6.39.1. Public Dataset Access

**Kaggle:**
```
1. Create Kaggle account
2. Search "oil spill SAR"
3. Download dataset
4. Review license/terms
```

**Zenodo:**
```
1. Search Zenodo for "oil spill detection"
2. Filter by data type
3. Check DOI for citations
4. Download with attribution
```

**GitHub:**
```
Many research papers publish data on GitHub
Search: "oil spill detection dataset"
Check paper for usage terms
```

### 6.39.2. Data Request Process

For restricted datasets:
1. Identify data custodian (agency, research group)
2. Prepare research proposal/justification
3. Submit formal data request
4. Sign data use agreement
5. Receive access credentials
6. Download và use per agreement

### 6.39.3. Creating Own Dataset

Steps cho custom dataset:

1. **Define Scope:**
   - Geographic region
   - Time period
   - Target phenomena

2. **Collect SAR Data:**
   - Sentinel-1 từ Copernicus
   - Filter by coverage, mode, orbit

3. **Identify Oil Spill Events:**
   - News reports
   - Incident databases
   - Agency alerts

4. **Download Matching Imagery:**
   - Temporal alignment với events
   - Appropriate spatial coverage

5. **Annotate:**
   - Define annotation protocol
   - Use annotation tools (QGIS, CVAT)
   - Quality control process

6. **Document:**
   - Metadata for each sample
   - Annotation guidelines used
   - Known limitations

## 6.40. Future Directions

### 6.40.1. Larger Benchmark Datasets

Need for:
- Larger, more diverse datasets
- Standardized evaluation protocols
- Annual benchmarking challenges
- Multi-regional coverage

### 6.40.2. Automatic Labeling

Leveraging:
- Operational detection systems
- Multi-source verification
- Active learning for annotation
- Citizen science contributions

### 6.40.3. Multi-modal Integration

Future datasets including:
- SAR + optical pairs
- In-situ measurements
- AIS vessel data
- Meteorological data
- Ocean current data

### 6.40.4. Time Series Datasets

Tracking oil spill evolution:
- Multiple acquisitions của same spill
- Drift và spreading dynamics
- Weathering progression
- Response intervention effects

These temporal datasets would enable:
- Spill trajectory prediction
- Volume estimation
- Response effectiveness assessment

