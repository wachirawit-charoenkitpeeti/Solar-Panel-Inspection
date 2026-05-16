# 📋 Engineering Spec — Data Cleaning v2 (Revised Dedup Strategy)

**Project:** Solar Panel Inspection AI  
**Phase:** Data Cleaning — Revision  
**Assigned to:** Codex (engineer)  
**Tech Lead:** Claude  
**Priority:** P0  
**Version:** v2 — replaces previous cleaning notebook  
**Prerequisite:** `unified_annotations.csv` + `unified_images.csv` จาก EDA ✅

---

## 🎯 What Changed from v1

v1 ใช้ pHash (perceptual hash) ซึ่ง treat Roboflow augmented images (flip/rotate/brightness) ว่าเป็น duplicates ทำให้:
- o6dwf เสีย 956/1,723 ภาพ (55%)
- dust annotations หายไป 74%
- รวมหาย 1,231 ภาพจาก 3,246

v2 เปลี่ยนเป็น **MD5 exact hash** — ลบเฉพาะ pixel-identical จริงๆ เท่านั้น augmented variants ไม่โดนลบ

---

## 🔒 Locked Decisions (ไม่เปลี่ยน)

```yaml
categorization:
  A:         polygon + defects nested in panel polygons
  A_partial: polygon + partially nested (keep, flag only)
  B1:        polygon + orphan defects + panel visible → SAM panel mask
  B2:        polygon + orphan defects + panel NOT visible → EXCLUDE
  C:         bbox datasets (IA-Cobotics + bird_dust_leaf)

snow_class: dropped
b2_heuristic:
  blue_pixel_ratio_threshold: 0.25
  defect_coverage_threshold:  0.50
  rule: B1 if blue_ratio > 0.25 AND defect_coverage < 0.50, else B2

dedup_priority (ถ้ายังเจอ exact duplicates):
  1: polygon > bbox
  2: more annotations
  3: smaller dataset
```

---

## 📊 Pipeline Steps

### Step 1: MD5 Exact Deduplication

**เปลี่ยนจาก pHash → MD5**

```python
import hashlib

def get_md5(image_path):
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
```

Logic:
- คำนวณ MD5 ทุกภาพใน 4 datasets
- Group ภาพที่มี MD5 เหมือนกัน (pixel identical เท่านั้น)
- ใน group ที่มี >1 ภาพ → keep ตาม priority เดิม (polygon > bbox > more anns > smaller dataset)
- Mark ภาพที่ drop ว่า `dropped_reason = exact_duplicate`

**Output table ที่ต้องแสดง:**

```
Dataset            | Before | After | Dropped
IA-Cobotics        |    302 |     ? |       ?
bird_dust_leaf     |    500 |     ? |       ?
solar-panel-b1cmz  |    721 |     ? |       ?
solar-panel-o6dwf  | 1,723  |     ? |       ?
TOTAL              | 3,246  |     ? |       ?
```

⚠️ **คาดว่า drop จะน้อยกว่า v1 มาก** เพราะ augmented variants ไม่โดนลบแล้ว

---

### Step 2: Panel Hierarchy Categorization (เหมือน v1)

Apply เฉพาะ polygon datasets (o6dwf, b1cmz) หลัง dedup

```python
def categorize_polygon_image(image_id, annotations_df):
    panel_polygons = annotations_df[
        (annotations_df['image_id'] == image_id) &
        (annotations_df['class_unified'].isin(['panel_clean', 'panel_defective']))
    ]
    defect_polygons = annotations_df[
        (annotations_df['image_id'] == image_id) &
        (~annotations_df['class_unified'].isin(['panel_clean', 'panel_defective']))
    ]

    if len(panel_polygons) == 0 and len(defect_polygons) == 0:
        return 'EMPTY'
    if len(panel_polygons) > 0:
        nested_rate = check_nesting_rate(defect_polygons, panel_polygons)
        if nested_rate == 1.0:
            return 'A'
        elif nested_rate > 0:
            return 'A_partial'
    if len(panel_polygons) == 0 and len(defect_polygons) > 0:
        return 'B_candidate'
```

---

### Step 3: B1 vs B2 Split

**Heuristic เหมือน v1 — แต่ต้องแสดง visual samples สำหรับ validation (ดู Step 5)**

```python
def classify_b_candidate(image_path, image_id, annotations_df):
    blue_ratio = detect_blue_pixels(image_path)
    defect_cov = calculate_defect_coverage(image_id, annotations_df)

    if blue_ratio > 0.25 and defect_cov < 0.50:
        return 'B1'
    else:
        return 'B2'

def detect_blue_pixels(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100, 40, 40), (140, 255, 255))
    return mask.sum() / (mask.shape[0] * mask.shape[1] * 255)
```

---

### Step 4: Final Category Assignment + Class Counts

เหมือน v1 — assign `final_category` ทุกภาพ และสร้าง **4 summary tables**:

#### Table 4.1: Image counts per category
```
Category   | Source           | Images | Notes
A          | solar-panel-b1cmz|      ? |
A          | solar-panel-o6dwf|      ? |
A_partial  | solar-panel-b1cmz|      ? |
A_partial  | solar-panel-o6dwf|      ? |
B1         | solar-panel-b1cmz|      ? | needs SAM panel
B1         | solar-panel-o6dwf|      ? | needs SAM panel
B2         | solar-panel-b1cmz|      ? | EXCLUDED
B2         | solar-panel-o6dwf|      ? | EXCLUDED
C          | IA-Cobotics      |      ? | needs SAM panel + defect
C          | bird_dust_leaf   |      ? | needs SAM panel + defect
DUPLICATE  | (all)            |      ? | EXCLUDED (MD5 exact only)
EMPTY      | (all)            |      ? | EXCLUDED
─────────────────────────────────────────────────
TOTAL kept (A + A_partial + B1 + C) |  ? |
TOTAL excluded                       |  ? |
```

#### Table 4.2: Annotation counts per class after cleaning
```
Class           | A+A_partial | B1 | C  | TOTAL | Before cleaning (EDA)
panel_clean     |           ? |  - |  0 |     ? |        12,399
panel_defective |           ? |  - |  0 |     ? |         3,766
dust            |           ? |  ? |  ? |     ? |         6,635  ← key metric
bird_drop       |           ? |  ? |  ? |     ? |         6,316
physical_damage |           ? |  ? |  ? |     ? |         3,255
leaf            |           ? |  ? |  ? |     ? |         2,666
─────────────────────────────────────────────────────
TOTAL           |             |    |    |     ? |        35,037
```

**Note:** B1 + C panel masks will be SAM-generated (show 0 + note)

#### Table 4.3: Images per class
```
Class           | Images with at least 1 annotation
dust            | ?
bird_drop       | ?
physical_damage | ?
leaf            | ?
panel_clean     | ?
panel_defective | ?
```

#### Table 4.4: Class imbalance after cleaning
```
Defect classes only (dust / bird_drop / physical_damage / leaf):
  Largest:          ? (? annotations)
  Smallest:         ? (? annotations)
  Imbalance ratio:  ?  (was 2.49 before cleaning)

Compare to EDA: improved / worse / same?
Recommendation: oversampling for ?
```

---

### Step 5: ⭐ B1/B2 Visual Validation (New in v2)

**นี่คือ step ใหม่ที่สำคัญที่สุดใน v2**

เพื่อให้ Tech Lead ตรวจสอบว่า heuristic ทำงานถูกต้อง ต้องแสดง visual grid:

#### 5a. B1 samples — 15 ภาพ (3 rows × 5 cols)
- สุ่มจาก B1 category
- แต่ละภาพต้องแสดง:
  - ภาพจริง
  - Defect annotations overlay (polygon หรือ bbox ก็ได้)
  - Title: `blue_ratio=X.XX | defect_cov=X.XX`
- เป้าหมาย: ภาพ B1 ควรเห็น panel ชัด มี panel structure อยู่

#### 5b. B2 samples — 15 ภาพ (3 rows × 5 cols)
- สุ่มจาก B2 category
- Format เดียวกับ 5a
- เป้าหมาย: ภาพ B2 ควรเป็น close-up defect หรือไม่มี panel ชัด

#### 5c. Edge cases — 6 ภาพ
- ภาพที่ blue_ratio ใกล้ threshold 0.25 มากที่สุด (±0.03)
- แสดงเพื่อดูว่า threshold เหมาะสมไหม

⚠️ **หลังรัน Codex ต้อง:** paste visual grid กลับมาให้ Tech Lead ดูก่อน confirm B2 exclusion

---

### Step 6: Stratified Train/Val/Test Split (เหมือน v1)

```python
def stratified_split(manifest, target_ratios=(0.8, 0.1, 0.1)):
    # Stratify by: final_category + source_dataset + dominant_class
    # Constraint: no MD5 duplicate group across splits (แต่ตอนนี้ groups เล็กลงมาก)
    # Aim for 80/10/10
```

Output:
```
Split | Images | dust | bird_drop | physical_damage | leaf
train |      ? |    ? |         ? |               ? |    ?
val   |      ? |    ? |         ? |               ? |    ?
test  |      ? |    ? |         ? |               ? |    ?
```

---

### Step 7: Final Manifest Output (เหมือน v1)

Save 2 files:

`/content/datasets/cleaned_v2/final_manifest.csv`
```
image_id, source_dataset, md5_hash, is_exact_duplicate,
final_category, blue_ratio, defect_coverage,
needs_sam_panel, needs_sam_defect_conversion,
final_split, kept_in_dataset, exclusion_reason
```

`/content/datasets/cleaned_v2/final_annotations.csv`
```
image_id, annotation_id, class_unified, class_original,
format, coords_normalized, area_normalized, source_dataset
```

---

## 🛠️ Dependencies

```bash
pip install imagehash pillow pandas numpy matplotlib opencv-python tqdm shapely
```

⚠️ `imagehash` ยังต้อง install เพราะใช้ใน EDA แต่ใน cleaning v2 **ไม่ต้องใช้ pHash แล้ว** ใช้แค่ `hashlib.md5` (built-in Python)

---

## 📋 Acceptance Criteria

- [ ] Step 1: MD5 dedup table แสดง before/after per dataset
- [ ] Step 2–3: Categorization A/A_partial/B1/B2/C ครบ
- [ ] Step 4: Tables 4.1–4.4 ครบ — โดยเฉพาะ dust count เทียบกับ EDA
- [ ] **Step 5: Visual grids 5a (15 B1) + 5b (15 B2) + 5c (6 edge cases) ต้องแสดงผล** ⭐
- [ ] Step 6: Split done, no leakage
- [ ] Step 7: 2 CSV files saved to `cleaned_v2/`
- [ ] Notebook < 40 cells
- [ ] Code มี error handling + reproducible (random seed = 42)

---

## 📤 Deliverables

1. **Notebook:** `data_cleaning_v2.ipynb`
2. **CSVs:** `cleaned_v2/final_manifest.csv` + `cleaned_v2/final_annotations.csv`
3. **Visual grids** (Step 5) embedded in notebook output
4. **ASCII summary box** at end of notebook:

```
╔══════════════════════════════════════════════════════════╗
║  Cleaning v2 Summary                                     ║
╠══════════════════════════════════════════════════════════╣
║  Dedup method:  MD5 exact                                ║
║  Dropped (dup): ?  (was 1,231 in v1)                     ║
║                                                          ║
║  Final kept:    ?  images                                ║
║  Final annots:  ?  annotations                           ║
║                                                          ║
║  dust (before): 6,635  →  after: ?  (drop ?%)           ║
║  (was 74% drop in v1 — should improve significantly)     ║
║                                                          ║
║  B2 excluded:   ?  images                                ║
║  SAM needed:    ?  panel masks + ? defect conversions    ║
╚══════════════════════════════════════════════════════════╝
```

---

## 🎯 Key Metric to Watch

**dust annotation retention rate** คือตัวชี้วัดหลักว่า v2 ดีขึ้นจาก v1 แค่ไหน

| Version | dust annotations | drop rate |
|---------|-----------------|-----------|
| EDA     | 6,635           | baseline  |
| v1      | 1,718           | −74%      |
| v2      | ?               | should be < −30% |

ถ้า dust drop ยังเกิน 50% หลัง MD5 → แปลว่ามีภาพซ้ำจริงๆ ใน source data ไม่ใช่แค่ Roboflow augmentation
