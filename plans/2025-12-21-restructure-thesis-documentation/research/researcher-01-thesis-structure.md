# Thesis Structure Reorganization Research

**Report Date:** 2025-12-21
**Researcher:** Academic Documentation Analyst
**Status:** Comprehensive Analysis

## Executive Summary

Proposed structure implements **Bloom's Taxonomy progression** (knowledge → application) with TorchGeo as conceptual bridge between theory and practical remote sensing tasks. This enables readers to build contextual understanding before encountering domain-specific applications.

---

## 1. Recommended Chapter Organization

### Current Structure (7 chapters, problem-based)
- Sequential application studies (ship → oil spill) with separate theory/library chapters
- **Issue:** Theory scattered; TorchGeo isolated; applications appear disconnected

### Proposed Structure (7 chapters, knowledge-building)
| Ch | Title | Content | Rationale |
|----|-------|---------|-----------|
| 1 | Giới thiệu | CNN/Deep Learning fundamentals + remote sensing context | Foundation layer |
| 2 | Cơ sở lý thuyết | CNN architectures + image processing theory for remote sensing | Theory consolidation |
| 3 | TorchGeo Models | Classification, segmentation, detection, change detection models | Unified framework |
| 4 | xView Challenges | Datasets + winning solutions across 3 competitions | Practical validation |
| 5 | Ship Detection | Theory application + case study (object detection pipeline) | Task-specific application |
| 6 | Oil Spill Detection | Theory application + case study (semantic segmentation pipeline) | Task-specific application |
| 7 | Kết luận | Summary, research gaps, future directions | Synthesis |

---

## 2. Content Flow & Dependencies

### Knowledge Progression Path
```
Ch1: Concepts → Ch2: Theory → Ch3: Tools/Architectures
                          ↓
                    Ch4: Datasets/Solutions
                          ↓
                    Ch5&6: Task-Specific Applications
                          ↓
                    Ch7: Integration & Future Work
```

### Critical Dependencies
- **Ch2 ← Ch1:** Remote sensing context prerequisite for understanding CNN theory applications
- **Ch3 ← Ch1, Ch2:** TorchGeo models require CNN/architecture knowledge; introduces unified analysis framework
- **Ch4 ← Ch2, Ch3:** xView challenges leverage both theory and model architectures; demonstrates real-world validation
- **Ch5 ← Ch1-4:** Ship detection assumes full theory understanding; applies object detection (covered in Ch3)
- **Ch6 ← Ch1-4:** Oil spill detection assumes full theory; applies semantic segmentation (covered in Ch3)

### Vietnamese Academic Coherence
- **论文逻辑:** Thesis logic follows "从理论到应用" (theory-to-application) progression
- **知识铺垫:** Each chapter builds prerequisite knowledge for next
- **实践验证:** Applications (Ch5-6) grounded in established frameworks (Ch3), not standalone case studies

---

## 3. Key Structural Insights

### Why TorchGeo as Central Framework (Ch3)
1. **Unifies model presentation:** Object detection, segmentation, classification under single library lens
2. **Bridges gap:** Connects theoretical architectures (ResNet, UNet, etc.) to practical remote sensing implementation
3. **Enables cross-reference:** Ch5-6 applications reference consistent TorchGeo model terminology
4. **Reduces duplication:** Single source for architecture descriptions vs. repeated in each application

### Why Consolidate Applications (Ch5-6)
- **Current:** Separate chapters for ship/oil detection with redundant theory sections
- **Proposed:** Single theory-application binding per chapter; clear task specialization
  - Ch5: Object detection pipeline (bounding boxes, confidence scores)
  - Ch6: Semantic segmentation pipeline (pixel-level classification)

### Why Merge xView Earlier (Ch4)
- **Motivates learning:** Concrete datasets/solutions validate theory from Ch1-3
- **Positions challenges:** Competitions as benchmarks for model comparison, not isolated achievements
- **Enables reference:** Winning solutions documented before application chapters reference them

---

## 4. Vietnamese Academic Writing Considerations

### Terminology Consistency
- Establish unified terminology in Ch1-2 glossary:
  - CNN/RNN/Transformer (keep English)
  - Phát hiện đối tượng (object detection) vs. Phát hiện tàu (ship detection) distinction clear
  - TorchGeo as proper noun (not translated)

### Sectioning for Academic Rigor
- Each application chapter (5-6) includes:
  1. Bối cảnh bài toán (Problem context)
  2. Phương pháp tiếp cận (Methodology w/ TorchGeo model references)
  3. Kết quả và so sánh (Results vs. xView benchmarks)
  4. Hạn chế và hướng mở rộng (Limitations & future work)

### Cross-References
- Ch5-6 cite Ch3 model architectures explicitly: "Như mô tả trong Chương 3..."
- Avoid repeating architectural details; use references instead (DRY principle)

---

## 5. Migration Mapping (Current → Proposed)

| Current Ch | Current Content | → | Proposed Location |
|------------|-----------------|---|------------------|
| Ch1 | CNN intro | → | Ch1 (expanded) |
| Ch2 | CNN theory | → | Ch2 (consolidated) |
| Ch2 | Image processing | → | Ch2 (consolidated) |
| Ch3 | Ship detection | → | Ch5 (theory + case study) |
| Ch4 | Oil spill detection | → | Ch6 (theory + case study) |
| Ch5 | TorchGeo overview | → | Ch3 (model architectures) |
| Ch6 | xView challenges | → | Ch4 (datasets + solutions) |
| Ch7 | Conclusion | → | Ch7 (enhanced synthesis) |

---

## 6. Implementation Benefits

✓ **Pedagogical:** Readers follow cognitive progression (foundation → specialized knowledge)
✓ **DRY:** Single source for theory/models; applications reference rather than duplicate
✓ **Academic rigor:** Explicit dependencies between chapters; clear knowledge scaffolding
✓ **Vietnamese coherence:** Thesis "四要素" (four-part structure): Intro → Theory → Tools → Application → Conclusion
✓ **Searchability:** Related content (all xView solutions) grouped in single chapter

---

## Unresolved Questions

1. Should Ch5-6 include new experiments/datasets or strictly analyze existing xView results?
2. Does TorchGeo Ch3 need subsection on installation/environment setup, or defer to appendix?
3. Any existing figures/diagrams that require new arrangement due to chapter reordering?
