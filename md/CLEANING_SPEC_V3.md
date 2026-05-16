# 📋 Engineering Spec — Data Cleaning v3 (Final)

**Project:** Solar Panel Inspection AI  
**Phase:** Data Cleaning — Final Version  
**Assigned to:** Codex (engineer)  
**Tech Lead:** Claude  
**Priority:** P0  
**Version:** v3 — patches v2, then saves to Drive permanently  
**Prerequisite:** `data_cleaning_v2.ipynb` ✅ + `skipped_annotations.csv` จาก EDA ✅

---

## 🎯 What Changed from v2

| Issue | v2 | v3 |
|---|---|---|
| Snow | drop annotation only | **exclude image ทั้งรูป** |
| B2 rule | `blue_ratio > 0.25 AND defect_coverage < 0.50` | **`blue_ratio > 0.25` only** |
| Output | local `/content/` only | **save to Google Drive ถาวร** |

---

## 🔒 Locked Decisions (ไม่เปลี่ยน)

```yaml
dedup:           MD5 exact hash (จาก v2 — ไม่ต้อง rerun)
dedup_priority:  polygon > bbox > more annotations > smaller dataset
categories:      A / A_partial / B1 / B2 / C / EMPTY / DUPLICATE / SNOW_EXCLUDED
split:           80 / 10 / 10 stratified, no leakage
```

---

## 📊 Pipeline Steps

### Step 0: Load v2 Outputs (อย่า rerun dedup หรือ categorization ใหม่)

```python
import pandas as pd

manifest    = pd.read_csv('/content/datasets/cleaned_v2/final_manifest.csv')
annotations = pd.read_csv('/content/datasets/cleaned_v2/final_annotations.csv')
skipped     = pd.read_csv('/content/datasets/skipped_annotations.csv')

print(f"Loaded manifest:    {len(manifest):,} images")
print(f"Loaded annotations: {len(annotations):,} annotations")
print(f"Loaded skipped:     {len(skipped):,} skipped annotations")
```

---

### Step 1: SNOW_EXCLUDED — Exclude Images Containing Snow

**เหตุผล:** ถ้าปล่อยภาพ snow ไว้แต่ลบแค่ annotation model จะเห็น "panel ที่มีหิมะ" แล้วเรียนรู้ว่านั่นคือ `panel_clean` — label noise ที่อันตราย

```python
# หา image_ids ที่มี Snow annotation อย่างน้อย 1 อัน
snow_image_ids = skipped[
    skipped['class_orig'].str.lower() == 'snow'
]['image_id'].unique()

print(f"Images containing Snow: {len(snow_image_ids)}")

# Mark ใน manifest
manifest['snow_image'] = manifest['image_id'].isin(snow_image_ids)

# Override final_category และ kept_in_dataset
manifest.loc[manifest['snow_image'], 'final_category']    = 'SNOW_EXCLUDED'
manifest.loc[manifest['snow_image'], 'kept_in_dataset']   = False
manifest.loc[manifest['snow_image'], 'exclusion_reason']  = 'image_contains_snow'

# Drop snow images จาก annotations ด้วย
annotations = annotations[~annotations['image_id'].isin(snow_image_ids)]

# Report
snow_by_dataset = manifest[manifest['snow_image']].groupby('source_dataset').size()
print("\nSnow images excluded per dataset:")
print(snow_by_dataset.to_string())
```

**Output table ที่ต้องแสดง:**
```
Dataset            | Snow images excluded
IA-Cobotics        | (expected: 0)
bird_dust_leaf     | (expected: 0)
solar-panel-b1cmz  | (expected: 0)
solar-panel-o6dwf  | ?
TOTAL              | ?
```

---

### Step 2: Revise B2 Rule — Blue Ratio Only

**เหตุผลที่เปลี่ยน:** `defect_coverage` สูงไม่ได้แปลว่าไม่เห็น panel — แปลแค่ว่าแผงมีฝุ่นเต็ม ซึ่งคือ training example ที่มีประโยชน์มาก โดยเฉพาะ class `dust`

**Rule ใหม่:**
```python
# Rule เดิม (v2) — ผิด:
# B1 = blue_ratio > 0.25 AND defect_coverage < 0.50
# B2 = otherwise

# Rule ใหม่ (v3) — ถูก:
# B1 = blue_ratio > 0.25   → panel visible → SAM ทำได้
# B2 = blue_ratio <= 0.25  → panel ไม่ชัด → SAM ทำไม่ได้

def reclassify_b_candidates(manifest):
    # เฉพาะ rows ที่เป็น B1 หรือ B2 ใน v2
    b_mask = manifest['final_category'].isin(['B1', 'B2'])
    
    manifest.loc[b_mask & (manifest['blue_ratio'] > 0.25),  'final_category'] = 'B1'
    manifest.loc[b_mask & (manifest['blue_ratio'] <= 0.25), 'final_category'] = 'B2'
    
    # Update kept_in_dataset
    manifest.loc[manifest['final_category'] == 'B1', 'kept_in_dataset'] = True
    manifest.loc[manifest['final_category'] == 'B2', 'kept_in_dataset'] = False
    manifest.loc[manifest['final_category'] == 'B2', 'exclusion_reason'] = 'no_panel_visible'
    
    return manifest

manifest = reclassify_b_candidates(manifest)
```

**Output table ที่ต้องแสดง:**
```
Category | v2 count | v3 count | delta
B1       |      160 |        ? |     ?   ← คาดว่าเพิ่ม
B2       |      345 |        ? |     ?   ← คาดว่าลด
```

---

### Step 3: Final Category Summary

```python
kept_categories = ['A', 'A_partial', 'B1', 'C']

final_kept = manifest[manifest['kept_in_dataset'] == True]
final_excl = manifest[manifest['kept_in_dataset'] == False]

print(f"Total kept:     {len(final_kept):,}")
print(f"Total excluded: {len(final_excl):,}")
```

**Table 3.1: Image counts per category**
```
Category       | Source           | Images | Notes
A              | solar-panel-b1cmz|      ? |
A              | solar-panel-o6dwf|      ? |
A_partial      | solar-panel-b1cmz|      ? |
A_partial      | solar-panel-o6dwf|      ? |
B1             | solar-panel-b1cmz|      ? | needs SAM panel
B1             | solar-panel-o6dwf|      ? | needs SAM panel
B2             | solar-panel-b1cmz|      ? | EXCLUDED
B2             | solar-panel-o6dwf|      ? | EXCLUDED
C              | IA-Cobotics      |      ? | needs SAM panel + defect
C              | bird_dust_leaf   |      ? | needs SAM panel + defect
SNOW_EXCLUDED  | solar-panel-o6dwf|      ? | EXCLUDED
DUPLICATE      | (all)            |    418 | EXCLUDED (MD5)
EMPTY          |                  |     14 | EXCLUDED
─────────────────────────────────────────────────
TOTAL kept (A + A_partial + B1 + C) |  ? |
TOTAL excluded                       |  ? |
```

---

### Step 4: Annotation Counts After v3

```python
kept_ids = manifest[manifest['kept_in_dataset'] == True]['image_id']
final_annotations = annotations[annotations['image_id'].isin(kept_ids)]
```

**Table 4.1: Annotations per class**
```
Class           | A+A_partial | B1 | C  | TOTAL v3 | v2    | EDA
panel_clean     |           ? |  0 |  0 |        ? | 10835 | 12399
panel_defective |           ? |  0 |  0 |        ? |  3150 |  3766
dust            |           ? |  ? |  ? |        ? |  2466 |  6635
bird_drop       |           ? |  ? |  ? |        ? |  5321 |  6316
physical_damage |           ? |  ? |  ? |        ? |  2671 |  3255
leaf            |           ? |  ? |  ? |        ? |  2666 |  2666
──────────────────────────────────────────────────────────────────
TOTAL           |             |    |    |        ? | 27109 | 35037
```

**Note:** B1 + C panel_clean / panel_defective = 0 ตอนนี้ จะ generate ด้วย SAM ใน phase ถัดไป

**Table 4.2: Class imbalance**
```
Defect classes (dust / bird_drop / physical_damage / leaf):
  Largest:         ? (?)
  Smallest:        ? (?)
  Imbalance ratio: ?   (v2 = 2.16, EDA = 2.49)
  
Recommendation: oversample ?
```

---

### Step 5: Visual Validation — B2 Rule Check ⭐

เพื่อยืนยันว่า rule ใหม่ทำงานถูกต้อง

#### 5a. B1 samples — 15 ภาพ (3 rows × 5 cols)
- สุ่มจาก B1 category หลัง reclassify
- Title แต่ละภาพ: `blue_ratio=X.XX`
- ✅ ควรเห็น panel ชัด

#### 5b. B2 samples — 15 ภาพ (3 rows × 5 cols)
- สุ่มจาก B2 category หลัง reclassify
- Title แต่ละภาพ: `blue_ratio=X.XX`
- ✅ ควรเป็น close-up หรือ ไม่เห็น panel

#### 5c. "Rescued" samples — ภาพที่ถูก B2 ใน v2 แต่กลายเป็น B1 ใน v3 — 10 ภาพ
```python
rescued = manifest[
    (manifest['final_category'] == 'B1') &
    (manifest['v2_category'] == 'B2')   # ถ้า track v2 category ไว้
]
```
- ✅ ควรเห็นว่าเป็น "dusty full-panel" ที่ควร keep จริงๆ
- นี่คือหลักฐานว่า rule เปลี่ยนถูกต้อง

#### 5d. Snow excluded samples — 6 ภาพ
- สุ่มจาก SNOW_EXCLUDED
- ยืนยันว่า exclude ถูกภาพ

---

### Step 6: Re-split Train/Val/Test

```python
from sklearn.model_selection import train_test_split

def stratified_split(manifest, ratios=(0.8, 0.1, 0.1), seed=42):
    kept = manifest[manifest['kept_in_dataset'] == True].copy()
    
    # Stratify by: final_category + source_dataset + dominant_class
    kept['strat_key'] = (
        kept['final_category'] + '_' +
        kept['source_dataset'] + '_' +
        kept['dominant_class'].fillna('none')
    )
    
    train, temp = train_test_split(kept, test_size=0.2,
                                   stratify=kept['strat_key'],
                                   random_state=seed)
    val, test   = train_test_split(temp, test_size=0.5,
                                   stratify=temp['strat_key'],
                                   random_state=seed)
    
    manifest.loc[train.index, 'final_split'] = 'train'
    manifest.loc[val.index,   'final_split'] = 'val'
    manifest.loc[test.index,  'final_split'] = 'test'
    
    return manifest
```

**Output:**
```
Split | Images | dust | bird_drop | physical_damage | leaf
train |      ? |    ? |         ? |               ? |    ?
val   |      ? |    ? |         ? |               ? |    ?
test  |      ? |    ? |         ? |               ? |    ?

Duplicate leakage across splits: 0 groups
```

---

### Step 7: Save to Google Drive (Permanent)

```python
import shutil, os

DRIVE_OUTPUT = '/content/drive/MyDrive/ai builders/dataset/cleaned_v3/'
os.makedirs(DRIVE_OUTPUT, exist_ok=True)

# Save manifests
manifest.to_csv(f'{DRIVE_OUTPUT}final_manifest.csv', index=False)
final_annotations.to_csv(f'{DRIVE_OUTPUT}final_annotations.csv', index=False)

# Save summary
summary = {
    'version': 'v3',
    'total_kept_images': int(manifest['kept_in_dataset'].sum()),
    'total_kept_annotations': len(final_annotations),
    'snow_excluded': int((manifest['final_category'] == 'SNOW_EXCLUDED').sum()),
    'b2_excluded': int((manifest['final_category'] == 'B2').sum()),
    'dedup_dropped': int((manifest['final_category'] == 'DUPLICATE').sum()),
    'dust_annotations': int((final_annotations['class_unified'] == 'dust').sum()),
    'imbalance_ratio': round(
        final_annotations['class_unified'].value_counts().iloc[0] /
        final_annotations[
            final_annotations['class_unified'].isin(
                ['dust','bird_drop','physical_damage','leaf']
            )
        ]['class_unified'].value_counts().iloc[-1], 2
    ),
}

import json
with open(f'{DRIVE_OUTPUT}summary_v3.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✅ Saved to Drive:")
print(f"   {DRIVE_OUTPUT}final_manifest.csv")
print(f"   {DRIVE_OUTPUT}final_annotations.csv")
print(f"   {DRIVE_OUTPUT}summary_v3.json")
```

---

### Step 8: Final ASCII Summary Box

```python
kept      = manifest['kept_in_dataset'].sum()
excl      = (~manifest['kept_in_dataset']).sum()
snow_excl = (manifest['final_category'] == 'SNOW_EXCLUDED').sum()
b2_excl   = (manifest['final_category'] == 'B2').sum()
dup_excl  = (manifest['final_category'] == 'DUPLICATE').sum()
dust_v3   = (final_annotations['class_unified'] == 'dust').sum()

print(f"""
╔══════════════════════════════════════════════════════════╗
║  Cleaning v3 — Final Summary                             ║
╠══════════════════════════════════════════════════════════╣
║  Dedup method:    MD5 exact (same as v2)                 ║
║  Dropped (dup):   {dup_excl:<6} images  (v1 was 1,231)        ║
║                                                          ║
║  SNOW_EXCLUDED:   {snow_excl:<6} images  (new in v3)           ║
║  B2 excluded:     {b2_excl:<6} images  (rule revised)         ║
║                                                          ║
║  Final kept:      {kept:<6} images                       ║
║  Final annots:    {len(final_annotations):<6} annotations              ║
║                                                          ║
║  dust (EDA):      6,635  →  v2: 2,466  →  v3: {dust_v3:<6}  ║
║  Imbalance ratio: ? (v2 was 2.16, EDA was 2.49)          ║
║                                                          ║
║  SAM needed:                                             ║
║    panel masks:   ? images  (B1 + C)                     ║
║    defect masks:  ? images  (C only)                     ║
║                                                          ║
║  ✅ Saved to Google Drive: cleaned_v3/                   ║
╚══════════════════════════════════════════════════════════╝
""")
```

---

## 📋 Acceptance Criteria

- [ ] Step 0: Load v2 outputs สำเร็จ ไม่ rerun dedup
- [ ] Step 1: Snow images excluded ครบ + table แสดงจำนวนต่อ dataset
- [ ] Step 2: B2 reclassified ด้วย blue_ratio only + table before/after
- [ ] Step 3: Category summary table ครบทุก category รวม SNOW_EXCLUDED
- [ ] Step 4: Tables 4.1 และ 4.2 ครบ — เทียบ v2 และ EDA
- [ ] **Step 5: Visual grids 5a–5d แสดงผลใน notebook** ⭐
- [ ] Step 6: Split done, leakage = 0
- [ ] **Step 7: Files saved to Google Drive ถาวร** ⭐
- [ ] Step 8: ASCII summary box แสดงครบ
- [ ] Notebook < 30 cells (ไม่ต้อง rerun expensive steps)

---

## 📤 Deliverables

1. **Notebook:** `data_cleaning_v3.ipynb`
2. **Google Drive output:** `ai builders/dataset/cleaned_v3/`
   - `final_manifest.csv`
   - `final_annotations.csv`
   - `summary_v3.json`
3. **Visual grids** (Step 5) embedded in notebook
4. **ASCII summary box** (Step 8)

---

## ⚠️ Risks

| Risk | Mitigation |
|---|---|
| `skipped_annotations.csv` ไม่มี `image_id` column | ตรวจ column names ก่อน Step 1 |
| B_candidate ใน v2 ไม่ได้ track `blue_ratio` ไว้ | ถ้าไม่มี column ต้อง recompute สำหรับ B rows เท่านั้น |
| Drive path ต่างกัน | print path จริงก่อน save |
| Snow images มีน้อยมาก (อาจ 0 ใน some datasets) | handle gracefully ด้วย if len > 0 |
