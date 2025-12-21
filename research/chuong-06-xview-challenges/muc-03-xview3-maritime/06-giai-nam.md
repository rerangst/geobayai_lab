# 6.3.6 Giải Pháp Hạng Năm xView3: Geographic Context và GSHHG

## Lời Dẫn

Giải pháp hạng năm của Kohei (smly) - một Kaggle Master người Nhật - nổi bật với việc tận dụng thông tin ngữ cảnh địa lý. Trong khi các giải pháp khác chủ yếu tập trung vào đặc trưng hình ảnh SAR, Kohei nhận ra rằng vị trí tàu so với bờ biển, độ sâu nước, và các yếu tố địa lý khác cung cấp thông tin quan trọng cho phân loại. Giải pháp sử dụng GSHHG (Global Self-consistent Hierarchical High-resolution Geography) để tính khoảng cách đến bờ, kết hợp với HRNet backbone - một lựa chọn thống nhất với giải pháp hạng ba nhưng được tối ưu khác biệt.

| Thuộc tính | Giá trị |
|-----------|---------|
| **Xếp hạng** | 5/1,900+ đội |
| **Tác giả** | Kohei (smly) |
| **Đóng góp chính** | Geographic context, GSHHG distance-to-shore |
| **Backbone** | HRNet |
| **Hardware** | 2× RTX 3080 |

---

## 1. Insight: Geographic Context Matters

### 1.1 Quan Sát Then Chốt

```mermaid
flowchart TB
    subgraph OBSERVATIONS["Quan Sát Từ Dữ Liệu"]
        O1["Tàu đánh cá thường<br/>gần bờ hoặc ngư trường"]
        O2["Tàu container thường<br/>theo tuyến vận tải"]
        O3["Infrastructure cố định<br/>ở vùng nước nông"]
    end

    subgraph INSIGHT["Insight"]
        I["Vị trí địa lý cung cấp<br/>prior cho classification"]
    end
```

### 1.2 Các Yếu Tố Địa Lý

| Yếu tố | Ảnh hưởng |
|--------|-----------|
| **Khoảng cách bờ** | Fishing vessels gần bờ hơn |
| **Độ sâu nước** | Oil rigs ở vùng nông |
| **Vùng biển** | Mật độ fishing khác nhau |
| **Tuyến hàng hải** | Container ships theo lane |

---

## 2. GSHHG Database

### 2.1 Global Self-consistent Hierarchical High-resolution Geography

GSHHG là database địa lý độ phân giải cao, cung cấp:

```mermaid
flowchart TB
    subgraph GSHHG["GSHHG Database"]
        COAST["Coastlines<br/>Đường bờ biển"]
        RIVERS["Rivers<br/>Sông ngòi"]
        BORDERS["Borders<br/>Biên giới"]
        LAKES["Lakes<br/>Hồ"]
    end

    subgraph USE["Sử Dụng"]
        DIST["Distance-to-shore<br/>Tính khoảng cách"]
    end

    COAST --> DIST
```

### 2.2 Tính Distance-to-Shore

Với mỗi pixel trong ảnh SAR, tính khoảng cách đến bờ biển gần nhất:

| Khoảng cách | Ý nghĩa |
|-------------|---------|
| < 10 km | Coastal waters |
| 10-50 km | Nearshore |
| 50-200 km | Offshore |
| > 200 km | Deep sea |

---

## 3. Kiến Trúc Mô Hình

### 3.1 Multi-Channel Input

```mermaid
flowchart TB
    subgraph INPUT["Đầu Vào"]
        VH["VH Polarization"]
        VV["VV Polarization"]
        BATH["Bathymetry<br/>Độ sâu"]
        WIND_S["Wind Speed"]
        WIND_D["Wind Direction"]
        DIST_SHORE["Distance-to-Shore<br/>GSHHG"]
    end

    subgraph CONCAT["Concatenation"]
        TENSOR["6-Channel<br/>Input Tensor"]
    end

    subgraph MODEL["HRNet"]
        HRNET["Multi-resolution<br/>Processing"]
    end

    INPUT --> CONCAT --> MODEL
```

### 3.2 HRNet Configuration

| Component | Specification |
|-----------|---------------|
| **Backbone** | HRNet-W48 |
| **Input channels** | 6 |
| **Resolution branches** | 4 |
| **Output heads** | 4 (det, vessel, fishing, length) |

---

## 4. Training Strategy

### 4.1 Tile-Based Processing

```mermaid
flowchart LR
    subgraph SCENE["Full Scene"]
        FULL["10,000 × 10,000<br/>pixels"]
    end

    subgraph TILES["Tiling"]
        T1["Tile 1<br/>512×512"]
        T2["Tile 2<br/>512×512"]
        TN["..."]
    end

    subgraph OVERLAP["Overlap"]
        OV["128px overlap<br/>Handle edge effects"]
    end

    FULL --> TILES --> OV
```

### 4.2 Augmentation Strategy

```mermaid
flowchart TB
    subgraph AUGS["Augmentations"]
        subgraph GEOMETRIC["Geometric"]
            G1["Flip H/V"]
            G2["Rotation 90°"]
            G3["Random scale<br/>(0.8-1.2)"]
        end

        subgraph INTENSITY["Intensity"]
            I1["SAR normalization"]
            I2["Channel-wise scaling"]
        end
    end
```

### 4.3 Loss Functions

| Task | Loss | Weight |
|------|------|--------|
| Detection | Focal Loss | 1.0 |
| is_vessel | Weighted BCE | 0.5 |
| is_fishing | Weighted BCE | 0.5 |
| vessel_length | Smooth L1 | 0.3 |

Class weights được áp dụng do imbalance (90% non-vessel).

---

## 5. Hiệu Quả của Geographic Context

### 5.1 Ablation Study

| Configuration | Detection F1 | Classification Acc |
|---------------|--------------|-------------------|
| SAR only (VH+VV) | 0.72 | 0.78 |
| + Bathymetry | 0.74 | 0.80 |
| + Wind info | 0.75 | 0.80 |
| **+ GSHHG dist** | **0.78** | **0.82** |

### 5.2 Đóng Góp Distance-to-Shore

```mermaid
flowchart LR
    subgraph IMPROVE["Cải Thiện"]
        BASE["Baseline<br/>F1: 0.75"]
        FINAL["+ GSHHG<br/>F1: 0.78"]
    end

    BASE -->|"+4%"| FINAL
```

Distance-to-shore đóng góp 4% F1 improvement, cho thấy giá trị của geographic context.

---

## 6. Tarfile Streaming

### 6.1 Vấn Đề Storage

Dataset xView3 rất lớn (~1.4 TB), gây khó khăn cho storage và data loading.

### 6.2 Giải Pháp

```mermaid
flowchart LR
    subgraph STORAGE["Storage Strategy"]
        TAR["Data in .tar files"]
        STREAM["Stream processing<br/>Không extract toàn bộ"]
        RAM["Load on-demand<br/>Memory efficient"]
    end

    TAR --> STREAM --> RAM
```

### 6.3 Lợi Ích

| Khía cạnh | Standard | Tarfile Streaming |
|-----------|----------|-------------------|
| **Disk usage** | Extract + Raw | Raw only |
| **I/O overhead** | High (many small files) | Low (sequential reads) |
| **Setup time** | Long (extraction) | Minimal |

---

## 7. Inference Pipeline

### 7.1 Full Scene Processing

```mermaid
flowchart TB
    subgraph INFERENCE["Inference Pipeline"]
        LOAD["Load full scene"]
        TILE["Split into tiles<br/>with overlap"]
        PRED["Predict each tile"]
        MERGE["Merge predictions<br/>Handle overlaps"]
        NMS["Apply NMS"]
        OUT["Final detections"]
    end

    LOAD --> TILE --> PRED --> MERGE --> NMS --> OUT
```

### 7.2 Handling Overlap Regions

| Strategy | Overlap pixels | Merge method |
|----------|---------------|--------------|
| **Kohei's approach** | 128 | Max pooling |
| Alternative | 256 | Weighted average |

---

## 8. So Sánh Với Các Giải Pháp Khác

### 8.1 Feature Comparison

| Feature | Hạng 1-4 | **Hạng 5** |
|---------|----------|------------|
| SAR channels | ✅ | ✅ |
| Bathymetry | ✅ | ✅ |
| Wind data | ✅ | ✅ |
| **GSHHG dist** | ❌ | **✅** |
| Geographic context | Minimal | **Emphasized** |

### 8.2 Architecture Comparison

| Khía cạnh | Hạng 3 (Tumenn) | **Hạng 5 (Kohei)** |
|-----------|-----------------|---------------------|
| **Backbone** | HRNet | HRNet |
| **Pipeline** | Dual-model | Single model |
| **Extra features** | Standard | **GSHHG distance** |
| **Focus** | Task separation | Geographic context |

Cả hai đều dùng HRNet nhưng với emphasis khác nhau.

---

## 9. Bài Học Rút Ra

### 9.1 Về Domain Knowledge

1. **Context matters**: Không chỉ pixels, mà còn vị trí địa lý

2. **External data**: GSHHG là external database nhưng hoàn toàn hợp lệ và hữu ích

3. **Prior information**: Geographic priors bổ sung cho learned features

### 9.2 Về Implementation

1. **Efficient data loading**: Tarfile streaming cho large datasets

2. **Multi-channel input**: Combine nhiều nguồn thông tin

3. **HRNet for small objects**: Consistent finding across solutions

### 9.3 Cho Nghiên Cứu Tương Lai

1. **More geographic features**: Shipping lanes, EEZ boundaries, protected areas

2. **Temporal context**: Vessel tracks, historical patterns

3. **Multi-modal fusion**: SAR + AIS + optical

---

## 10. Tổng Kết Các Giải Pháp xView3

### 10.1 So Sánh Top 5

| Hạng | Đóng góp chính | Score |
|------|----------------|-------|
| 1 | CircleNet, Stride-2 output | 0.617 |
| 2 | Segmentation paradigm, Data quality | 0.604 |
| 3 | Dual-model pipeline, HRNet | 0.598 |
| 4 | Self-training, Semi-supervised | 0.591 |
| **5** | **Geographic context, GSHHG** | **0.585** |

### 10.2 Common Themes

```mermaid
flowchart TB
    subgraph COMMON["Điểm Chung"]
        C1["Multi-channel input<br/>(VH, VV, bath, wind)"]
        C2["Dense prediction<br/>(heatmap-based)"]
        C3["Multi-task learning<br/>(4 outputs)"]
        C4["Ensemble/averaging"]
    end
```

### 10.3 Unique Contributions

| Giải pháp | Đóng góp độc đáo |
|-----------|------------------|
| Hạng 1 | Stride-2 for small objects |
| Hạng 2 | Quality > Quantity data |
| Hạng 3 | Detection-Classification separation |
| Hạng 4 | Semi-supervised learning |
| **Hạng 5** | **Geographic context** |

---

## Tài Liệu Tham Khảo

1. Wessel, P., & Smith, W. H. F. (1996). A global, self-consistent, hierarchical, high-resolution shoreline database. JGR.

2. Wang, J., et al. (2020). Deep High-Resolution Representation Learning for Visual Recognition. IEEE PAMI.

3. Paolo, F. S., et al. (2022). xView3-SAR: Detecting Dark Fishing Activity Using SAR Imagery. NeurIPS.

---

*Kết thúc Mục 6.3 về xView3 Challenge. Chương 6 đã trình bày ba cuộc thi lớn trong chuỗi xView, từ object detection (xView1), đến building damage assessment (xView2), và cuối cùng là maritime surveillance (xView3). Mỗi cuộc thi đều đặt ra những thách thức kỹ thuật riêng biệt và thúc đẩy sự phát triển của các phương pháp deep learning cho viễn thám.*
