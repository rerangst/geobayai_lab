# xView2/xBD Image Reference Catalog

Reference documentation for image types, formats, and visual characteristics in the xBD dataset.

---

## Image Types in xBD

### Type 1: Pre-Disaster Satellite Image
**Filename Pattern**: `{disaster_name}_{image_id}_pre.png`

**Specifications**:
- Format: PNG (lossless compression)
- Dimensions: 1024×1024 pixels
- Color: RGB (24-bit, 8 bits per channel)
- Color Space: sRGB
- Data Type: uint8 (0-255 range per channel)
- File Size: Typically 200-400 KB per image

**Content**: Satellite imagery of region before disaster event occurred

**Example Filenames**:
- `guatemala-volcano_00000000_pre.png`
- `hurricane-harvey_00001234_pre.png`
- `palu-tsunami_00005678_pre.png`

**Visual Characteristics**:
- Clear view of buildings, streets, vegetation
- No destruction visible
- Realistic satellite perspective (off-nadir angles 15-30 degrees)
- Variable sun elevation (30-60 degrees)
- Natural shadows and perspective distortion

---

### Type 2: Post-Disaster Satellite Image
**Filename Pattern**: `{disaster_name}_{image_id}_post.png`

**Specifications**: Identical to pre-disaster image
- Format: PNG (lossless)
- Dimensions: 1024×1024 pixels
- Color: RGB (24-bit)
- Same location as corresponding pre-disaster image
- Captured within days of disaster (same year, typically)

**Content**: Satellite imagery of same region after disaster struck

**Visual Differences from Pre-Image**:
- Visible building destruction in affected areas
- Color/contrast changes due to rubble, ash, water
- Damaged vegetation and infrastructure
- Smoke or dust if recent capture
- Shadow changes due to different time of day/season

**Example Timeline**:
- Hurricane Harvey: Pre (2017-08-25) → Post (2017-08-31) [6 days]
- Palu Tsunami: Pre (2018-09-28) → Post (2018-10-01) [3 days]

---

### Type 3: Building Polygon Annotations
**Filename Pattern**: `{disaster_name}_{image_id}_buildings.json`

**Format**: GeoJSON FeatureCollection

**Structure**:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "uid": "unique_polygon_id",
        "damage": 0,
        "class": "building"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [[x1, y1], [x2, y2], [x3, y3], ...]
        ]
      }
    },
    ... (more buildings)
  ]
}
```

**Key Fields**:
- `uid`: Unique identifier for polygon within dataset
- `damage`: Ordinal damage level (0-3)
- `class`: Object type ("building", sometimes "water", "smoke")
- `coordinates`: Pixel-space polygon vertices (not lat/lon)

**Polygon Characteristics**:
- Closed ring (first point = last point)
- Typically 5-40 vertices per building
- Coordinate system: Image pixel coordinates (0-1023 range)
- Origin: Top-left corner (standard image convention)

---

## Damage Classification Visualization

### 4-Level Damage Scale with Visual Examples

#### Level 0: No Damage (GREEN)
- **Definition**: Building undisturbed, no structural damage
- **Visual Indicators**:
  - Intact roof and walls
  - Clear building outline
  - Color similar to pre-disaster image
  - No visible cracks or deformation
  - Surrounding area unaffected
- **Annotation Value**: 0
- **Hex Color**: #00AA00 (RGB: 0, 170, 0)
- **Dataset Count**: 313,033 polygons (84%)
- **Examples**: Most buildings away from disaster epicenter
- **Challenge**: Easiest class, but severely imbalanced

#### Level 1: Minor Damage (BLUE)
- **Definition**: Partial damage but structure standing
- **Visual Indicators**:
  - Some roof elements missing or discolored
  - Partial wall cracks visible
  - Building envelope partially compromised
  - Water surrounding (flooding scenario)
  - Isolated damage (not widespread)
  - Building silhouette recognizable
- **Annotation Value**: 1
- **Hex Color**: #0000FF (RGB: 0, 0, 255)
- **Dataset Count**: 36,860 polygons (5%)
- **Examples**:
  - Partially burned houses in wildfires
  - Buildings with roof damage from wind
  - Water-damaged buildings near rivers
- **Challenge**: Subtle visual differences from no-damage
- **Confusion Rate**: Often misclassified as no-damage or major-damage

#### Level 2: Major Damage (ORANGE)
- **Definition**: Significant structural damage, partial collapse
- **Visual Indicators**:
  - Partial wall or roof collapse
  - Debris visible inside structure
  - Multiple sections compromised
  - Building outline significantly altered
  - Large visible gaps in structure
  - Color distinctly different from pre-disaster
- **Annotation Value**: 2
- **Hex Color**: #FF8800 (RGB: 255, 136, 0)
- **Dataset Count**: 29,904 polygons (4%)
- **Examples**:
  - Partially collapsed earthquake buildings
  - Heavily burned structures in wildfires
  - Flood-damaged multi-story buildings
- **Challenge**: Ambiguous boundary between minor and major
- **Confusion Rate**: High confusion with both minor and destroyed classes

#### Level 3: Destroyed (RED)
- **Definition**: Total collapse or complete destruction
- **Visual Indicators**:
  - Complete structural collapse
  - Only rubble/foundation visible
  - Building footprint barely recognizable
  - Debris scattered around perimeter
  - Major color/contrast change
  - Structure reduced to ground level
- **Annotation Value**: 3
- **Hex Color**: #FF0000 (RGB: 255, 0, 0)
- **Dataset Count**: 31,560 polygons (4%)
- **Examples**:
  - Completely collapsed earthquake buildings
  - Fully burned out structures
  - Tsunami/flood destroyed homes
- **Challenge**: Generally easier to identify than minor/major distinction
- **Clarity**: Highest inter-rater agreement (typically >90%)

---

## Visualization Color Mapping

### Standard Colormaps Used in xBD Community

**Damage Pixel Values** (for output predictions):
```
0 → Black/Background  (no building)
1 → Green             (no damage)
2 → Blue              (minor damage)
3 → Orange            (major damage)
4 → Red               (destroyed)
```

**Visualization Code** (Python/OpenCV):
```python
import cv2
import numpy as np

# Create colormap
colormap = {
    0: [0, 0, 0],         # Black - background
    1: [0, 170, 0],       # Green - no damage
    2: [0, 0, 255],       # Blue - minor damage
    3: [0, 136, 255],     # Orange - major damage
    4: [0, 0, 255]        # Red - destroyed
}

# Apply to damage map
damage_map = cv2.imread('prediction_damage.png', cv2.IMREAD_GRAYSCALE)
colored = np.zeros((damage_map.shape[0], damage_map.shape[1], 3), dtype=np.uint8)

for pixel_value, rgb_color in colormap.items():
    mask = damage_map == pixel_value
    colored[mask] = rgb_color

cv2.imwrite('damage_colored.png', colored)
```

---

## Spatial Distribution of Damage

### Geographic Patterns Across Disasters

#### High-Impact Events (>30% affected buildings)
- **Santa Rosa Wildfire (2017)**: Fire spreads across urban area
- **Palu Tsunami (2018)**: Concentrated damage along coast
- **Mexico Earthquake (2017)**: Widespread damage in Mexico City

#### Moderate-Impact Events (5-30% affected)
- **Hurricane Harvey (2017)**: Scattered damage across Houston
- **Lower Puna Eruption (2018)**: Lava flow corridors

#### Low-Impact Events (<5% affected)
- **Nepal Flooding (2017)**: Scattered riverside damage
- **India Monsoon**: Concentrated in specific regions

### Damage Density Distribution
```
Pre-Disaster:
  All buildings = 100% (full inventory)

Post-Disaster:
  No Damage: 70-90% (depends on disaster)
  Minor Damage: 2-15% (depends on disaster type)
  Major Damage: 2-8% (depends on impact zone)
  Destroyed: 2-5% (epicenter/high-intensity areas)
```

---

## Image Quality & Metadata

### Resolution Specifications
- **GSD (Ground Sampling Distance)**: 0.3-0.8 meters
  - Higher resolution = more detail in smaller buildings
  - Lower resolution = faster processing, coarser features

- **Off-Nadir Angle**: 15-45 degrees (realistic satellite acquisition)
  - Affects shadow direction and length
  - Influences perspective distortion
  - Varies by satellite pass geometry

- **Sun Elevation**: 25-65 degrees (varies by season/location)
  - Low elevation = long shadows, harder to interpret
  - High elevation = minimal shadows, clearer damage
  - Affects overall image brightness

### Sensor Information
- **Primary Sensors**: DigitalGlobe WorldView-1, WorldView-2, WorldView-3
- **Spectral Bands**: RGB (3-band, visible spectrum only)
- **Dynamic Range**: 11-12 bits (displayed as 8-bit)
- **Radiometric Processing**: Pansharpened where applicable

### Image Registration & Alignment
- **Pre/Post Alignment**: Images georeferenced to same UTM zone
- **Registration Accuracy**: Sub-pixel (usually <1 pixel RMS error)
- **Resampling Method**: Bilinear or cubic for coregistration
- **Deformation**: Some areas may have georeferencing artifacts due to terrain

---

## Annotation Quality Metrics

### Coverage Statistics
- **Building Annotation Rate**: ~99% of visible buildings
- **Missed Buildings**: <1% (very small or occluded structures)
- **False Positives**: <0.5% (misclassified vegetation/structures)

### Damage Label Reliability
- **No Damage**: 95%+ inter-rater agreement (clear and consistent)
- **Minor vs. Major**: 70-80% inter-rater agreement (subjective boundary)
- **Destroyed**: 90%+ inter-rater agreement (obvious destruction)
- **Overall Kappa**: Not officially reported (estimated 0.75-0.85)

### Annotation Challenges Documented
1. **Visual Ambiguity**: Minor and major damage hard to distinguish
2. **Timing Bias**: Early post-disaster captures may underestimate damage
3. **Environmental Obscuration**: Smoke, dust, water obscure damage
4. **Building Density**: Hard to separate adjacent buildings in dense areas
5. **Sensor Variation**: Different satellites have different noise patterns

---

## Disaster Type Characteristics

### Earthquake/Tsunami Damage Pattern
- **Spatial Pattern**: Concentric from epicenter; coastal strip for tsunami
- **Damage Type**: Structural collapse (vertical walls fail)
- **Visual Signature**: Building footprints barely visible; rubble piles
- **Primary Classes**: Destroyed and major damage concentrated
- **Examples**: Mexico City (2017), Palu (2018)

### Wildfire Damage Pattern
- **Spatial Pattern**: Along fire perimeter; scattered based on wind
- **Damage Type**: Thermal destruction; combustible materials burn
- **Visual Signature**: Charred areas; melted metal; ground scorching
- **Primary Classes**: Destroyed common; some minor partially burned
- **Examples**: Santa Rosa (2017), Socal (2017)

### Flooding Damage Pattern
- **Spatial Pattern**: Along river corridors; low-lying areas
- **Damage Type**: Water damage; structural undermining; debris impact
- **Visual Signature**: Water stains; mud; debris piles; foundation erosion
- **Primary Classes**: Minor damage more common than destruction
- **Examples**: Midwest (2019), Nepal (2017)

### Wind/Hurricane Damage Pattern
- **Spatial Pattern**: Radial from storm track; highest near eye
- **Damage Type**: Roof and envelope damage; some structural failure
- **Visual Signature**: Torn roofs; debris scatter; some intact structures nearby
- **Primary Classes**: Mix of minor and major; less destruction than earthquake
- **Examples**: Hurricane Harvey (2017), Hurricane Michael (2018)

### Volcanic Eruption Damage Pattern
- **Spatial Pattern**: Along lava flows; ashfall zones; pyroclastic zones
- **Damage Type**: Thermal damage; burial under lava/ash; structural damage
- **Visual Signature**: Lava-covered buildings; ash accumulation; charring
- **Primary Classes**: Complete destruction in lava path; minor in ashfall zones
- **Examples**: Lower Puna (2018), Guatemala volcano (2018)

---

## Data Format Reference

### PNG Image Format
```
PNG Header: 89 50 4E 47 ... (magic bytes)
IHDR Chunk: Image width=1024, height=1024, bit depth=8, color type=2 (RGB)
IDAT Chunk: Compressed pixel data
IEND Chunk: End marker

Binary Layout:
- Bytes 0-7: PNG signature
- Bytes 8-24: IHDR chunk (width, height, bit depth, color type, etc.)
- Bytes 25+: Compressed image data (DEFLATE algorithm)
```

### JSON Polygon Format
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "uid": "damage_00001",
        "damage": 2,
        "class": "building"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [x1, y1], [x2, y2], [x3, y3], ..., [x1, y1]
          ]
        ]
      }
    }
  ]
}
```

### CSV Metadata Format
```csv
ImageID,DisasterType,Region,PreDate,PostDate,OffNadir,SunElevation,Sensor,GSD
disaster_00000000,earthquake,Mexico,2017-09-19,2017-09-21,28.5,45.2,WorldView-3,0.31
disaster_00000001,wind,Texas,2017-08-25,2017-08-31,15.3,52.1,DigitalGlobe,0.50
```

---

## Class Imbalance Challenge

### Imbalance Factors
- **Global Imbalance**: 313,033 no-damage vs. 91,424 other classes = **3.4:1 ratio**
- **Severe Imbalance**: 313,033 no-damage vs. 31,560 destroyed = **9.9:1 ratio**
- **Practical Impact**: Models trained on raw data heavily bias toward "no damage"

### Mitigation Strategies Used in Literature
1. **Class Weighting**: Loss function weights inversely to class frequency
2. **Oversampling**: Duplicate damaged building examples
3. **Undersampling**: Sample subset of no-damage buildings
4. **Focal Loss**: Focus on hard examples (misclassified buildings)
5. **Stratified Sampling**: Ensure balanced class distribution in batches

### Performance Variance by Class
- **No Damage**: High F1 (0.80+), but misleading due to prevalence
- **Minor/Major**: Low F1 (0.40-0.60), high confusion between classes
- **Destroyed**: Moderate-high F1 (0.60-0.75), clearer boundary

---

## Visual Quality Factors

### Signal-to-Noise Ratio
- **Building Boundaries**: Typically sharp and clear (high SNR)
- **Damage Indicators**: Moderate SNR (especially minor damage)
- **Atmospheric Effects**: Low SNR (clouds, haze, dust)

### Artifacts & Limitations
- **Compression Artifacts**: PNG lossless, but JPEG versions may have artifacts
- **Georeferencing Errors**: Sub-pixel misalignment in dense areas
- **Spectral Ambiguity**: RGB-only (no NIR, thermal bands for damage detection)
- **Temporal Gaps**: Pre/post may be weeks apart (vegetation regrowth masks damage)
- **Geometric Distortion**: Off-nadir angles create relief displacement

### Enhancement Techniques
- **Histogram Equalization**: Improves visibility in dark/bright regions
- **Contrast Stretching**: Normalizes image brightness
- **Unsharp Masking**: Enhances edge sharpness
- **False Color Composites**: Synthetic bands improve interpretation (if multi-band available)

---

## Performance Baselines

### Localization (Building Detection)
- **Best Reported F1**: 0.80 (CMU SEI baseline)
- **Best Reported IoU**: 0.66 (same baseline)
- **Main Challenge**: Small buildings, dense urban areas, occluded structures

### Damage Classification
- **Best Reported F1 (weighted)**: 0.71 (IBM approach)
- **F1 by Class**:
  - No Damage: 0.81
  - Minor Damage: 0.42 (high confusion)
  - Major Damage: 0.48 (high confusion)
  - Destroyed: 0.65 (more separable)

### Combined Metric
- **Weighted F1**: Average of localization and classification
- **Typical Combined F1**: 0.60-0.72 across top challenge submissions

---

## Summary: When to Use xBD Images

**Use xBD When You Need**:
- ✓ Large-scale building damage training data
- ✓ Pre/post disaster temporal pairs
- ✓ Multi-disaster representation (19 events)
- ✓ High-resolution satellite imagery (~0.3-0.8m GSD)
- ✓ Standardized damage classification
- ✓ Real-world humanitarian/disaster response application
- ✓ Multi-country geographic diversity

**Limitations to Consider**:
- ⚠️ Severely imbalanced classes (84% no-damage)
- ⚠️ RGB-only (no multispectral/thermal bands)
- ⚠️ Small dataset compared to other CV benchmarks (11k pairs vs. millions)
- ⚠️ Subjective damage boundaries (minor vs. major ambiguous)
- ⚠️ Realistic but challenging acquisition angles (not nadir)
- ⚠️ Sparse annotation (buildings only, not road/vegetation)

---

## References

- Main Paper: https://arxiv.org/abs/1911.09296
- Dataset Portal: https://xview2.org/
- Imagery Source: https://www.digitalglobe.com/ecosystem/open-data
- TorchGeo: https://torchgeo.readthedocs.io/
