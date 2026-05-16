# 📋 Engineering Spec — Data Cleaning & Categorization

**Project:** Solar Panel Inspection AI  
**Phase:** Data Cleaning  
**Assigned to:** Codex (engineer)  
**Tech Lead:** Claude  
**Priority:** P0  
**Prerequisite:** EDA notebook ✅ + `unified_annotations.csv`

---

## 🎯 Goal

Clean, dedupe, และ categorize all 4 datasets เพื่อเตรียม sample สำหรับ SAM mask conversion ขั้นถัดไป
Output ต้องเป็น manifest CSV ที่บอกชัดว่าแต่ละภาพ:
1. อยู่ category ไหน (A/B1/B2/C)
2. ต้องการ SAM processing แบบไหน
3. Final class counts ก่อนเข้า training

---

## 📦 Input

- `/content/datasets/unified_annotations.csv` (จาก EDA)
- `/content/datasets/unified_images.csv` (จาก EDA)
- Original datasets ใน Google Drive

---

## 🔒 Locked Decisions

```yaml
deduplication:
  method: perceptual hash (pHash)
  threshold: conservative (exact duplicates only, hamming distance = 0)
  priority_for_keep:
    1: polygon dataset > bbox dataset
    2: dataset with more annotations on the image
    3: smaller dataset (preserve diversity)

categorization:
  strategy: Smart Hybrid (Option C)
  categories:
    A: Polygon dataset + defects nested in panel polygons
    B1: Polygon dataset + orphan defects + panel visible in image
    B2: Polygon dataset + orphan defects + panel NOT clearly visible (EXCLUDE)
    C: Bbox dataset (IA-Cobotics + bird_dust_leaf)

snow_class: dropped (already in EDA)
```

---

## 📊 Required Pipeline

### Step 1: Deduplication

**Input:** All images across 4 datasets  
**Method:** Perceptual hash matching  
**Threshold:** Hamming distance = 0 (exact duplicates only — conservative)

```python
def deduplicate_images(images_df):
    """
    1. Calculate pHash for each image
    2. Group by identical pHash
    3. For each group with >1 image:
       a. Priority 1: keep polygon dataset over bbox
       b. Priority 2: keep image with more annotations
       c. Priority 3: keep image from smaller dataset (preserve diversity)
       d. Mark others as duplicate (don't physically delete files)
    4. Output: dedup_manifest with columns:
       - original_image_id
       - kept (bool)
       - dropped_reason (if kept=False)
       - duplicate_group_id (if part of group)
    """
```

**Save:** `/content/datasets/cleaned/dedup_manifest.csv`

**Visualizations required:**
- Histogram: duplicate group sizes
- Bar chart: drops per dataset
- Table: dataset image counts before/after dedup
- Sample side-by-side: 5 duplicate groups visualized

---

### Step 2: Panel Hierarchy Verification (Category A vs B)

**Apply only to polygon datasets** (o6dwf, b1cmz) after dedup

```python
def categorize_polygon_image(image, defect_polygons, panel_polygons):
    """
    if no panel_polygons and no defect_polygons:
        return 'EMPTY'  # skip
    
    if has panel_polygons:
        check nested rate
        if all defects nested:
            return 'A'
        elif partially nested:
            return 'A_partial'  # treat as A, but flag
    
    if no panel_polygons but has defect_polygons:
        return 'B_candidate'  # need B1/B2 split next
    """
```

**Save:** category column added to manifest

---

### Step 3: B1 vs B2 Split (Panel Visibility Heuristic)

**For Category B candidates only**

ใช้ heuristic 2 ทาง — combine ผลทั้งคู่:

```python
def is_panel_visible(image):
    """
    Heuristic 1: Color analysis
       - Detect blue-ish pixels (typical PV panel color)
       - Range: HSV hue 200-240, saturation > 40
       - If blue pixel ratio > 25% → likely has panel
    
    Heuristic 2: Defect coverage
       - Calculate total defect bbox area / image area
       - If defects cover > 50% of image → likely close-up of defect (no panel context)
       - If defects cover < 30% → likely has panel context
    
    Combined decision:
       - B1 (keep): blue_ratio > 25% AND defect_coverage < 50%
       - B2 (exclude): otherwise
    """
```

**Save:** B1/B2 split into manifest + sample visualization (10 examples each)

⚠️ **Critical:** Show 20 sample images (10 B1, 10 B2) ใน notebook เพื่อ validate heuristic ด้วยสายตา

---

### Step 4: Final Category Assignment

```python
# After all steps:
Category A:    polygon + nested defects (use as-is)
Category B1:   polygon + orphan but panel visible (needs SAM panel mask)
Category B2:   polygon + orphan + no panel visible (EXCLUDE)
Category C:    bbox dataset (needs SAM panel mask + bbox→mask conversion)
```

Add columns to manifest:
- `final_category`: A | B1 | B2 | C | EMPTY | DUPLICATE
- `needs_sam_panel`: bool
- `needs_sam_defect_conversion`: bool
- `kept_in_dataset`: bool (final flag for usage)

---

### Step 5: Final Class Counts ⭐ (User's request)

หลัง cleaning + categorization สรุป **final counts** ละเอียด:

#### Table 5.1: Image counts per category
```
Category | Source           | Images | Notes
A        | o6dwf            | ?      |
A        | b1cmz            | ?      |
B1       | o6dwf            | ?      | needs SAM panel
B1       | b1cmz            | ?      | needs SAM panel
B2       | o6dwf            | ?      | EXCLUDED
B2       | b1cmz            | ?      | EXCLUDED
C        | IA-Cobotics      | ?      | needs SAM (panel + defects)
C        | bird_dust_leaf   | ?      | needs SAM (panel + defects)
DUPLICATE| (any)            | ?      | EXCLUDED
EMPTY    | (any)            | ?      | EXCLUDED
─────────────────────────────────────
TOTAL kept (A+B1+C)         | ?      |
TOTAL excluded              | ?      |
```

#### Table 5.2: Final annotation counts per unified class
```
Class           | A     | B1    | C     | TOTAL kept | from EDA (before clean)
panel_clean     | ?     | -     | ?     | ?          | 12,399
panel_defective | ?     | -     | ?     | ?          | 3,766
dust            | ?     | ?     | ?     | ?          | 6,635
bird_drop       | ?     | ?     | ?     | ?          | 6,316
physical_damage | ?     | ?     | ?     | ?          | 3,255
leaf            | ?     | ?     | ?     | ?          | 2,666
─────────────────────────────────────────────────────
TOTAL                                   | ?          | 35,037
```

**Notes:**
- C category panel_clean / panel_defective จะมาจาก SAM ยังนับใน column "from SAM later"
- B1 category panel_clean / panel_defective จะมาจาก SAM ด้วย

#### Table 5.3: Image counts per defect class
```
Class           | Images containing at least 1 annotation
dust            | ?
bird_drop       | ?
physical_damage | ?
leaf            | ?
panel_clean     | ?
panel_defective | ?
```

#### Table 5.4: Class imbalance assessment
```
Defect class imbalance after cleaning:
  - Highest: ? (?)
  - Lowest:  ? (?)
  - Imbalance ratio: ?

Compared to EDA: improved / worse / same

Recommendation:
  [ ] Oversampling needed for: ?
  [ ] Augmentation strategy: ?
```

---

### Step 6: Re-split Train/Val/Test

หลัง dedup + categorization:

```python
def stratified_split(manifest, target_ratios=(0.8, 0.1, 0.1)):
    """
    Stratify by:
      - final_category (A, B1, C)
      - source_dataset
      - dominant_class
    
    Constraints:
      - No duplicate group across splits (use duplicate_group_id)
      - Preserve class balance per split
      - Aim for 80/10/10 train/val/test
    """
```

Output split counts:
```
Split | Images | dust | bird_drop | physical_damage | leaf | panel_clean | panel_defective
train | ?      | ?    | ?         | ?               | ?    | ?           | ?
val   | ?      | ?    | ?         | ?               | ?    | ?           | ?
test  | ?      | ?    | ?         | ?               | ?    | ?           | ?
```

---

### Step 7: Final Manifest Output

Save manifest ที่ใช้ต่อใน SAM step:

`/content/datasets/cleaned/final_manifest.csv`

Columns:
```
image_id, original_path, source_dataset, original_split,
phash, duplicate_group_id, is_duplicate,
final_category (A/B1/B2/C/EMPTY/DUPLICATE),
panel_visible_score, defect_coverage_score,
needs_sam_panel, needs_sam_defect_conversion,
final_split (train/val/test),
kept_in_dataset (bool),
exclusion_reason (if not kept)
```

Plus per-class annotation manifest:

`/content/datasets/cleaned/final_annotations.csv`

Columns:
```
image_id, annotation_id, class_unified, class_original,
format (bbox/polygon), coords_normalized,
area_normalized, source_dataset
```

---

## 🛠️ Technical Requirements

### Dependencies
```bash
pip install imagehash pillow pandas numpy matplotlib opencv-python tqdm
```

### Helper functions
```python
def calculate_phash(image_path): pass
def find_duplicate_groups(images_df): pass
def detect_blue_pixels(image): pass
def calculate_defect_coverage(image_id, annotations_df): pass
def stratified_split(manifest, ratios): pass
```

### Code quality
- Cache pHash calculations (don't re-compute)
- Use multiprocessing for pHash (3000+ images)
- Save intermediate results so notebook is resumable
- Progress bars with tqdm

---

## 📋 Acceptance Criteria

- [ ] Step 1: Dedup manifest with reasoning per drop
- [ ] Step 2: Polygon images categorized A vs B_candidate
- [ ] Step 3: B1/B2 split with 20 sample images shown for visual validation
- [ ] Step 4: All 4 datasets have final_category assigned
- [ ] **Step 5: All 4 summary tables (5.1, 5.2, 5.3, 5.4) clearly displayed** ⭐
- [ ] Step 6: Stratified split done with no leakage
- [ ] Step 7: 2 manifest CSV files saved
- [ ] Notebook < 50 cells
- [ ] Resumable (can re-run cells without re-computing pHash)

---

## 📤 Deliverables

1. **Notebook:** `data_cleaning_4_datasets.ipynb`
2. **CSVs:**
   - `cleaned/dedup_manifest.csv`
   - `cleaned/final_manifest.csv`
   - `cleaned/final_annotations.csv`
3. **Plots:** saved to `cleaned/plots/`
4. **Summary report (ASCII box):** showing final counts
5. **Sample visualizations:**
   - 5 dedup groups (side-by-side)
   - 10 B1 samples (panel visible)
   - 10 B2 samples (excluded — close-up)
   - Per-category 6 random samples

---

## 🎯 Success Metrics

Project Owner สามารถตอบคำถามต่อไปนี้จาก notebook:
1. หลัง cleaning เหลือกี่ภาพ?
2. แต่ละ class เหลือกี่ annotations?
3. ภาพไหนต้อง SAM panel mask? (count)
4. ภาพไหนต้อง SAM defect mask conversion? (count)
5. Class imbalance หลัง cleaning ดีขึ้นหรือแย่ลง?
6. Train/val/test split balanced หรือไม่?
7. มีภาพซ้ำที่ leak across splits หรือไม่?

---

## ⚠️ Risks & Watch Points

| Risk | Mitigation |
|---|---|
| pHash computation slow (3000+ images) | multiprocessing + caching |
| Heuristic Step 3 misclassifies | Manual visualization of 20 samples for validation |
| Loss too many images in dedup | Conservative threshold (Hamming = 0) |
| Class imbalance worsens after cleaning | Report in Table 5.4 + recommend oversampling |
| Split leakage from duplicate groups | Use duplicate_group_id in split logic |
