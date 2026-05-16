# 📋 Engineering Spec — EDA Notebook (4 Datasets)

**Project:** Solar Panel Inspection AI  
**Phase:** Exploratory Data Analysis  
**Assigned to:** Codex (engineer)  
**Tech Lead:** Claude  
**Priority:** P0  
**Prerequisite:** All 4 datasets verified ✅

---

## 🎯 Goal

ทำ EDA ครบทุกมิติของ 4 datasets เพื่อ:
1. เข้าใจ data distribution & quality ก่อน cleaning/merging
2. Identify edge cases และ data issues
3. Document insights ที่ใช้ใน proposal (ตอบเกณฑ์ AI Builders ข้อ 4 — 20pt)
4. ตัดสินใจ data strategy จาก insights จริง

---

## 📦 Input Datasets

ทั้ง 4 datasets อยู่ใน Google Drive (assume mounted):
```
Dataset 1: IA-Cobotics Soiling_DS_v2_14  (302 images, bbox)
Dataset 2: bird_dust_leaf                 (500 images, bbox)
Dataset 3: solar-panel-o6dwf              (1,723 images, polygon)
Dataset 4: solar-panel-b1cmz              (721 images, polygon)
```

**Class mapping (locked):**
```python
CLASS_MAPPING = {
    'IA-Cobotics': {
        'Dust': 'dust',
        'bird-droppings': 'bird_drop',
    },
    'bird_dust_leaf': {
        'bird': 'bird_drop',
        'dust': 'dust',
        'leaf': 'leaf',
    },
    'solar-panel-o6dwf': {
        'Non-Defective': 'panel_clean',       # ⚠️ updated: panel mask
        'Defective': 'panel_defective',       # ⚠️ updated: panel mask
        'Dusty': 'dust',
        'Bird-drop': 'bird_drop',
        'Physical-Damage': 'physical_damage',
        'Electrical-Damage': 'physical_damage',
        'Snow': '_SKIP_',                     # decided: drop
    },
    'solar-panel-b1cmz': {
        'Non-Defective': 'panel_clean',
        'Defective': 'panel_defective',
        'Dusty': 'dust',
        'Bird-drop': 'bird_drop',
        'Physical-Damage': 'physical_damage',
        'Electrical-Damage': 'physical_damage',
    },
}
```

---

## 📊 Required EDA Sections

### Section 1: Setup & Class Mapping Application

**Cells:**
1. Imports + path setup
2. Apply class mapping to all 4 datasets → create unified annotation dataframe
3. Print summary of mapping (how many annotations dropped for `_SKIP_`)

**Output:**
- DataFrame columns: `dataset, image_path, label_path, split, class_orig, class_unified, bbox_or_polygon, area_normalized, ...`
- Save to: `/content/datasets/unified_annotations.csv`

---

### Section 2: Image-Level Analysis

**2.1 Image counts per dataset/split**
```
                  | train | valid | test | TOTAL
IA-Cobotics       | ?     | ?     | ?    | 302
bird_dust_leaf    | ?     | ?     | ?    | 500
solar-panel-o6dwf | ?     | ?     | ?    | 1723
solar-panel-b1cmz | ?     | ?     | ?    | 721
TOTAL             | ?     | ?     | ?    | 3246
```

**2.2 Image resolution distribution**
- Histogram width/height per dataset
- Most common resolution (should be 640×640)
- Aspect ratio analysis

**2.3 Images with NO annotations**
- Count empty-label images per dataset
- Are they intentional (negative samples)?

**2.4 Images per class** (unified taxonomy)
- ภาพแต่ละใบมีกี่ class?
- กี่ภาพมี 1 class? 2 classes? 3+?

---

### Section 3: Annotation-Level Analysis

**3.1 Annotation count per unified class**
```
class_unified      | annotations | images | datasets contributing
panel_clean        | ?           | ?      | o6dwf, b1cmz
panel_defective    | ?           | ?      | o6dwf, b1cmz
dust               | ?           | ?      | all 4
bird_drop          | ?           | ?      | all 4
physical_damage    | ?           | ?      | o6dwf, b1cmz
leaf               | ?           | ?      | bird_dust_leaf
```
+ visualization: stacked bar chart per class showing source breakdown

**3.2 Class imbalance metrics**
- Imbalance ratio = max_count / min_count
- Per-class proportion
- Recommendation: which classes need oversampling?

**3.3 Annotation size analysis** (normalized 0-1)
- Width/height distribution per class
- Area distribution per class
- Per-class boxplot
- **Hypothesis check:** ใบไม้ใหญ่กว่า dust ไหม? bird_drop เล็กๆ?

**3.4 Annotations per image distribution**
- Histogram: x = #annotations, y = #images
- Per-dataset comparison
- Identify outliers (image มี >50 annotations)

---

### Section 4: Format-Specific Analysis

**4.1 Bbox datasets (IA-Cobotics + bird_dust_leaf)**
- Aspect ratio distribution per class
- Bbox area histogram
- Cluster small/medium/large bboxes

**4.2 Polygon datasets (o6dwf + b1cmz)**
- Number of vertices per polygon (histogram)
- Polygon complexity (convex vs concave estimation)
- Area distribution per class

**4.3 Panel hierarchy analysis** (CRITICAL — for o6dwf + b1cmz)

**Goal:** ยืนยัน hypothesis ว่า defect polygons อยู่ภายใน panel polygons

For each image:
- หา panel_clean + panel_defective polygons
- หา defect polygons (dust, bird_drop, physical_damage)
- ตรวจว่า defect polygon centroid อยู่ใน panel polygon ไหม?
- คำนวณ % ของ defects ที่อยู่ใน panel correctly

**Output:**
- Number of "orphan" defects (อยู่นอก panel) per image
- Number of panels with vs without defects inside

---

### Section 5: Visual Inspection — Edge Cases

**5.1 Per-class galleries**
- 9 images per unified class (random sample)
- Annotations overlaid (bbox or polygon)
- For panel_defective: show with defect polygons inside

**5.2 Hard cases**
- Most annotations per image (top 6)
- Largest annotation areas (top 6)
- Smallest annotation areas (top 6 — possibly noise?)

**5.3 Multi-class images**
- ภาพที่มี class >= 3 (top 6)
- Useful for understanding complex scenes

**5.4 Cross-dataset visual comparison**
- 1 sample from each dataset side-by-side
- Same class (e.g., dust) — compare annotation style across datasets

---

### Section 6: Data Quality Concerns

**6.1 Duplicate detection**
- Check image filenames across datasets (might be duplicates)
- Optional: perceptual hash comparison (use `imagehash` library)

**6.2 Anomalies**
- Annotations outside image bounds
- Zero-area annotations
- Negative coordinates
- Polygons with <3 vertices

**6.3 Class label inconsistencies**
- Same class but different naming (e.g., "Dust" vs "dust" vs "Dusty")
- Document in mapping cell

**6.4 Domain gap visualization**
- Compare image styles: aerial vs close-up vs angled
- Hypothesis: bbox datasets may have different perspective than polygon datasets

---

### Section 7: Recommendations & Decision Points

Final cell ที่ output:

**7.1 Data Quality Report**
```
✅ Good news:
   - [list positive findings]
   
⚠️ Issues found:
   - [list issues]
   
🎯 Recommendations:
   1. [actionable next step]
   2. ...
```

**7.2 Cleaning Strategy Proposal**
ตามที่ EDA พบ propose strategy สำหรับ:
- Class balancing approach
- Outlier handling
- Duplicate handling
- Train/val/test re-split strategy (if needed)

**7.3 Mask Conversion Decision**
สำหรับ 802 bbox images → suggest based on EDA:
- ถ้า bbox tight & clean → SAM should work well
- ถ้า bbox sloppy → manual annotation needed
- Estimate effort

---

## 🛠️ Technical Requirements

### Dependencies
```bash
pip install pandas numpy matplotlib seaborn pillow pyyaml shapely imagehash
```

### Helper functions ต้องมี
```python
def parse_yolo_label(label_path, format='auto'):
    """Parse YOLO label file → list of (class_id, coords)"""
    # auto-detect bbox (5 cols) vs polygon (>5 cols)
    
def polygon_area(coords, image_w, image_h):
    """Calculate polygon area from YOLO normalized coords"""
    # use shapely.Polygon
    
def is_point_in_polygon(point, polygon_coords):
    """Check if (x, y) inside polygon"""
    # use shapely
    
def load_unified_annotations(class_mapping):
    """Load all 4 datasets → unified pandas DataFrame"""
```

### Code quality
- Use pandas for tabular analysis (don't reinvent the wheel)
- Use seaborn for prettier plots
- Save important plots as PNG to `/content/datasets/eda_outputs/`
- Cache parsed annotations (avoid re-parsing every cell)

---

## 📋 Acceptance Criteria

- [ ] Section 1-7 ครบทุกข้อ
- [ ] DataFrame `unified_annotations.csv` ถูกสร้าง
- [ ] ทุก visualization render ได้ + readable
- [ ] Panel hierarchy analysis (Section 4.3) ตอบคำถามได้ว่า defects อยู่ใน panels จริงไหม
- [ ] Section 7 มี actionable recommendations (ไม่ใช่ generic)
- [ ] Code มี error handling + reproducible (random seed)
- [ ] Notebook น้อยกว่า 50 cells

---

## 📤 Deliverables

1. **Notebook:** `eda_4_datasets.ipynb`
2. **CSV:** `unified_annotations.csv` (all annotations with unified class labels)
3. **Plots:** saved to `/content/datasets/eda_outputs/`
4. **Final recommendation** ที่ Project Owner ใช้ตัดสินใจ next step ได้

---

## 🎯 Success Metrics

EDA ดีเมื่อ Project Owner สามารถตอบคำถามต่อไปนี้จาก notebook:
1. Total unified data after merge + drop = ?
2. Class with most/fewest data = ?
3. ภาพไหนต้อง clean เร่งด่วน?
4. Panel hierarchy hypothesis ถูกหรือไม่?
5. SAM conversion น่าจะทำงานได้ดีไหม?
6. Train/val/test split ตอนนี้ balanced หรือไม่?
