# 📋 Engineering Spec — Apply Panel Decisions + Finalize Dataset

**Project:** Solar Panel Inspection AI  
**Phase:** 4.7 — Final Dataset Preparation  
**Assigned to:** Codex (engineer)  
**Tech Lead:** Claude  
**Priority:** P0  
**Prerequisite:** `sam_outlier_decisions.ipynb` ✅ + `outlier_panel_v2.csv` ✅

---

## 🎯 Goal

Apply panel decisions ที่ Tech Lead lock แล้ว จากนั้น finalize dataset ให้พร้อมเข้า Phase 5 (Merge + YOLO Export)

---

## 🔒 Panel Decisions (Locked)

```python
PANEL_DECISIONS = {
    'ok':        'keep',     # 881 images
    'too_large': 'keep',     # 20  images — panel ใหญ่เต็มภาพ ถูกต้อง
    'too_small': 'exclude',  # 100 images — SAM จับ busbar/damage แทน panel
    'noise':     'exclude',  #   7 images — SAM fail จับไม่ได้จริง
}
# KEEP:    901 images
# EXCLUDE: 107 images
```

**Note:** Images ที่ exclude panel mask ยังเก็บ defect annotations ไว้ได้ แค่ไม่มี panel mask = severity calculation ไม่ได้สำหรับภาพเหล่านี้

---

## 📦 Input

```python
DRIVE_BASE = '/content/drive/MyDrive/ai builders/dataset/cleaned_v3/'
SAM_OUT    = f'{DRIVE_BASE}sam_outputs/'

manifest          = pd.read_csv(f'{DRIVE_BASE}final_manifest.csv')
final_annotations = pd.read_csv(f'{DRIVE_BASE}final_annotations_v2.csv')
panel_outliers    = pd.read_csv(f'{SAM_OUT}outlier_panel_v2.csv')
sam_progress      = pd.read_csv(f'{SAM_OUT}sam_progress.csv')
```

---

## 📊 Pipeline Steps

### Step 1: Apply Panel Decisions

```python
import pandas as pd
import json

# Map flag → decision
panel_outliers['decision'] = panel_outliers['flag'].map(PANEL_DECISIONS)

# Image IDs ที่ exclude panel mask
excluded_panel_ids = set(
    panel_outliers[panel_outliers['decision'] == 'exclude']['image_id']
)
kept_panel_ids = set(
    panel_outliers[panel_outliers['decision'] == 'keep']['image_id']
)

# Images ที่ flag = ok (ไม่อยู่ใน outlier list) = keep ทั้งหมด
all_panel_image_ids = set(
    sam_progress[sam_progress['task'] == 'panel']['image_id']
)
ok_ids = all_panel_image_ids - set(panel_outliers['image_id'])

final_keep_panel = ok_ids | kept_panel_ids

print(f"Panel masks - KEEP:    {len(final_keep_panel)}")
print(f"Panel masks - EXCLUDE: {len(excluded_panel_ids)}")
print(f"Total:                 {len(final_keep_panel) + len(excluded_panel_ids)}")
```

---

### Step 2: Update Manifest

```python
# เพิ่ม column panel_mask_valid
manifest['panel_mask_valid'] = manifest['image_id'].apply(
    lambda iid: (
        True  if iid in final_keep_panel else
        False if iid in excluded_panel_ids else
        None  # ภาพที่ไม่ต้องการ panel mask (category A / A_partial)
    )
)

# ภาพ category A + A_partial ไม่ต้องการ SAM panel mask เลย set None
a_ids = set(manifest[manifest['final_category'].isin(['A', 'A_partial'])]['image_id'])
manifest.loc[manifest['image_id'].isin(a_ids), 'panel_mask_valid'] = None

# Summary
print("panel_mask_valid distribution:")
print(manifest['panel_mask_valid'].value_counts(dropna=False))
```

---

### Step 3: Final Usable Images

```python
# Image ที่ใช้ train ได้จริง = kept + (มี panel mask หรือ ไม่ต้องการ panel mask)
def is_training_ready(row):
    if not row['kept_in_dataset']:
        return False
    if row['final_category'] in ['A', 'A_partial']:
        return True   # ไม่ต้องการ SAM panel mask
    if row['final_category'] in ['B1', 'C']:
        return row['panel_mask_valid'] == True  # ต้องมี valid panel mask
    return False

manifest['training_ready'] = manifest.apply(is_training_ready, axis=1)

print(f"\nFinal training-ready images: {manifest['training_ready'].sum():,}")
print(f"Not ready:                   {(~manifest['training_ready']).sum():,}")
print()
print("Training-ready by category:")
print(manifest[manifest['training_ready']]['final_category'].value_counts())
```

---

### Step 4: Final Annotation Count

```python
# Annotations สำหรับ training-ready images เท่านั้น
ready_ids = set(manifest[manifest['training_ready']]['image_id'])

# ใช้ final_annotations_v2 (defect decisions applied แล้ว)
if 'annotation_id' not in final_annotations.columns:
    final_annotations['annotation_id'] = [
        f'ann_{i:06d}' for i in range(len(final_annotations))
    ]

training_annotations = final_annotations[
    (final_annotations['image_id'].isin(ready_ids)) &
    (final_annotations['kept_for_training'] == True)
].copy()

print(f"Final training annotations: {len(training_annotations):,}")
print()
print("Per class:")
print(training_annotations['class_unified'].value_counts())
```

**Table ที่ต้องแสดง:**
```
Class            | Annotations | vs EDA    | drop %
panel_clean      |           ? | 12,399    | ?%
panel_defective  |           ? |  3,766    | ?%
dust             |           ? |  6,635    | ?%
bird_drop        |           ? |  6,316    | ?%
physical_damage  |           ? |  3,255    | ?%
leaf             |           ? |  2,666    | ?%
─────────────────────────────────────────────────
TOTAL            |           ? | 35,037    | ?%
```

---

### Step 5: Save Final Files

```python
import os

# Save updated manifest
manifest.to_csv(f'{DRIVE_BASE}final_manifest_v2.csv', index=False)
print(f"✅ Saved: final_manifest_v2.csv ({len(manifest):,} rows)")

# Save training annotations
training_annotations.to_csv(
    f'{DRIVE_BASE}training_annotations_final.csv', index=False
)
print(f"✅ Saved: training_annotations_final.csv ({len(training_annotations):,} rows)")

# Save panel outliers with decisions filled
panel_outliers.to_csv(f'{SAM_OUT}outlier_panel_decided.csv', index=False)
print(f"✅ Saved: outlier_panel_decided.csv")
```

---

### Step 6: Final Summary Report

```python
ready = manifest['training_ready']
cats  = manifest[ready]['final_category'].value_counts()

print(f"""
╔══════════════════════════════════════════════════════════╗
║  Dataset Finalization — Ready for Phase 5                ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  PANEL DECISIONS APPLIED:                                ║
║    Kept panel masks:    {len(final_keep_panel):>4}  (ok + too_large)     ║
║    Excluded:            {len(excluded_panel_ids):>4}  (too_small + noise) ║
║                                                          ║
║  FINAL TRAINING DATASET:                                 ║
║    Total images:        {ready.sum():>4}                       ║
║      Category A:        {cats.get('A', 0):>4}                       ║
║      Category A_partial:{cats.get('A_partial', 0):>4}                       ║
║      Category B1:       {cats.get('B1', 0):>4}  (w/ SAM panel)       ║
║      Category C:        {cats.get('C', 0):>4}  (w/ SAM masks)       ║
║                                                          ║
║  Total annotations:     {len(training_annotations):>6,}                     ║
║                                                          ║
║  ✅ Files saved to Drive:                                ║
║     final_manifest_v2.csv                                ║
║     training_annotations_final.csv                       ║
║     sam_outputs/outlier_panel_decided.csv                ║
║                                                          ║
║  🚀 Ready for Phase 5: Merge + YOLO Export               ║
╚══════════════════════════════════════════════════════════╝
""")
```

---

## 📋 Acceptance Criteria

- [ ] Step 1: Panel decisions applied ครบ — KEEP 901, EXCLUDE 107
- [ ] Step 2: `manifest['panel_mask_valid']` มี True/False/None ถูกต้อง
- [ ] Step 3: `training_ready` column สะท้อน category logic ถูกต้อง
- [ ] Step 4: Annotation count table แสดงเทียบ EDA baseline
- [ ] Step 5: 3 files saved to Drive
- [ ] Step 6: Summary box แสดงครบ

---

## 📤 Deliverables

1. **Notebook:** `dataset_finalization.ipynb`
2. **Files saved to Drive:**
   - `final_manifest_v2.csv` — manifest พร้อม `training_ready` + `panel_mask_valid`
   - `training_annotations_final.csv` — annotations สุดท้ายสำหรับ training
   - `sam_outputs/outlier_panel_decided.csv` — panel decisions record

---

## 🎯 Phase 5 Preview

หลังจากนี้ Codex จะใช้ 2 files นี้เป็น input ของ Phase 5:

```python
# Phase 5 will use:
manifest    = pd.read_csv('final_manifest_v2.csv')
annotations = pd.read_csv('training_annotations_final.csv')

# Filter training-ready images
train_images = manifest[
    (manifest['training_ready'] == True) &
    (manifest['final_split'] == 'train')
]
# → Copy images + generate YOLO .txt labels
# → Output: dataset_yolo/ folder structure
```

Phase 5 จะ:
1. Copy images ไปใส่ `images/train|val|test/`
2. Convert polygon coords + SAM masks → YOLO `.txt` labels
3. สร้าง `data.yaml`
4. Export พร้อม train
