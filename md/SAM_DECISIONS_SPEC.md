# 📋 Engineering Spec — Panel QA Fix + Apply Defect Decisions

**Project:** Solar Panel Inspection AI  
**Phase:** 4.6 — Outlier Decisions  
**Assigned to:** Codex (engineer)  
**Tech Lead:** Claude  
**Priority:** P0  
**Prerequisite:** `sam_outlier_qa.ipynb` รันแล้ว ✅

---

## 🎯 Goal

2 งานในรอบเดียว:
1. **แก้ panel QA bug** — ตอนนี้ flag missing_mask 1,008/1,008 เพราะ path mismatch (ไม่ใช่ quality issue จริง)
2. **Apply defect decisions** — Tech Lead ตัดสินแล้วว่า exclude/keep อะไรบ้าง

---

## 📦 Part 1: Fix Panel QA Bug

### ปัญหา

Notebook สร้าง path เอง:
```python
panel_mask_path = f'{SAM_OUT}panel_masks/{image_id}_panel.png'  # ❌ path mismatch
```

แต่ไฟล์จริงใน Drive อาจชื่อต่างจาก expected — กลายเป็น `area=nan%, aspect=nan` ทุกแถว

### วิธีแก้

ใช้ `sam_progress.csv` เป็น source of truth สำหรับ `output_path`

```python
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

DRIVE_BASE = '/content/drive/MyDrive/ai builders/dataset/cleaned_v3/'
SAM_OUT    = f'{DRIVE_BASE}sam_outputs/'

# Load source of truth
sam_progress = pd.read_csv(f'{SAM_OUT}sam_progress.csv')
manifest     = pd.read_csv(f'{DRIVE_BASE}final_manifest.csv')

# Filter เฉพาะ panel masks ที่ done
panel_done = sam_progress[
    (sam_progress['task'] == 'panel') & 
    (sam_progress['status'] == 'done')
].copy()

print(f"Panel masks in progress: {len(panel_done)}")

# ตรวจว่า output_path ใช้ได้จริง
panel_done['file_exists'] = panel_done['output_path'].apply(
    lambda p: Path(str(p)).exists() if pd.notna(p) else False
)
print(f"Files actually exist: {panel_done['file_exists'].sum()}")
```

### Recompute Panel Stats

```python
def mask_stats(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    H, W = mask.shape
    binary = (mask > 127).astype(np.uint8)
    area_ratio = binary.sum() / (H * W)
    
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return {'area_ratio': 0.0, 'aspect_ratio': 0.0}
    bw = xs.max() - xs.min() + 1
    bh = ys.max() - ys.min() + 1
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    return {'area_ratio': float(area_ratio), 'aspect_ratio': float(aspect)}


panel_stats = []
for _, row in tqdm(panel_done.iterrows(), total=len(panel_done), desc='Panel stats'):
    if not row['file_exists']:
        continue
    s = mask_stats(row['output_path'])
    if s:
        panel_stats.append({
            'image_id':     row['image_id'],
            'mask_path':    row['output_path'],
            'area_ratio':   s['area_ratio'],
            'aspect_ratio': s['aspect_ratio'],
        })

panel_df = pd.DataFrame(panel_stats)
panel_df = panel_df.merge(
    manifest[['image_id', 'final_category', 'source_dataset', 'original_path']],
    on='image_id', how='left'
)

print(f"Panel df: {len(panel_df)} rows")
print(panel_df['area_ratio'].describe())
```

### Re-flag Panel Outliers

```python
panel_df['flag'] = 'ok'
panel_df.loc[panel_df['area_ratio'] < 0.30, 'flag'] = 'too_small'
panel_df.loc[panel_df['area_ratio'] > 0.90, 'flag'] = 'too_large'
panel_df.loc[panel_df['area_ratio'] < 0.01, 'flag'] = 'noise'

print(panel_df['flag'].value_counts())

panel_outliers = panel_df[panel_df['flag'] != 'ok'].copy()
panel_outliers['decision'] = ''  # Tech Lead จะกรอกหลังดู gallery
panel_outliers.to_csv(f'{SAM_OUT}outlier_panel_v2.csv', index=False)
print(f"✅ Saved: outlier_panel_v2.csv ({len(panel_outliers)} rows)")
```

### Re-generate Panel Outlier Gallery

```python
import matplotlib.pyplot as plt

def panel_outlier_gallery(panel_outliers, max_per_flag=12):
    for flag in panel_outliers['flag'].unique():
        subset = panel_outliers[panel_outliers['flag'] == flag].head(max_per_flag)
        if len(subset) == 0:
            continue
        
        n = len(subset)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        axes = np.array(axes).flatten()
        
        for ax, (_, row) in zip(axes, subset.iterrows()):
            img = cv2.imread(row['original_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
            
            overlay = img.copy()
            overlay[mask > 127] = [255, 50, 50]
            blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
            
            ax.imshow(blended)
            ax.set_title(
                f'{row["image_id"][:20]}\n'
                f'{row["final_category"]} | area={row["area_ratio"]*100:.1f}%',
                fontsize=9
            )
            ax.axis('off')
        for ax in axes[n:]:
            ax.axis('off')
        
        plt.suptitle(
            f'PANEL OUTLIERS: {flag} '
            f'({len(panel_outliers[panel_outliers["flag"]==flag])} total)',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        out_path = f'{SAM_OUT}outlier_panel_{flag}_v2.png'
        plt.savefig(out_path, dpi=80, bbox_inches='tight')
        plt.show()
        print(f"Saved: {out_path}")

panel_outlier_gallery(panel_outliers)
```

---

## 📦 Part 2: Apply Defect Decisions

### Decisions (Locked by Tech Lead)

```python
# Pattern: (class, flag) → decision
DEFECT_DECISIONS = {
    # KEEP — SAM ทำได้ตรงกับ object จริง
    ('leaf',      'sam_missed'):       'keep',   # 1,057 — ใบเล็ก mask ตรงกับ object จริง
    ('dust',      'sam_missed'):       'keep',   #   142 — จับ defect เล็กๆ ได้
    ('leaf',      'elongated_strip'):  'keep',   #    91 — จริงๆ คือกิ่ง/ก้านยาว
    ('bird_drop', 'sam_missed'):       'keep',   #    32 — จับ core ของ bird drop
    ('bird_drop', 'too_small'):        'keep',   #    13 — bird drop เล็กตามธรรมชาติ
    ('leaf',      'too_large'):        'keep',   #     1 — ใบใหญ่จริง
    
    # EXCLUDE — SAM artifact (frame / panel grabbing / noise)
    ('dust',      'elongated_strip'):  'exclude', # 66 — จับ frame ขอบ panel
    ('dust',      'too_large'):        'exclude', # 28 — กิน panel ทั้งใบ
    ('bird_drop', 'elongated_strip'):  'exclude', #  4 — frame artifact
    ('bird_drop', 'too_large'):        'exclude', #  1 — ไม่ใช่ bird drop ปกติ
    ('leaf',      'too_small'):        'exclude', #  1 — tiny noise
}
```

### Apply to outlier_defect.csv

```python
outlier_defect = pd.read_csv(f'{SAM_OUT}outlier_defect.csv')

def get_decision(row):
    key = (row['class_unified'], row['flag'])
    return DEFECT_DECISIONS.get(key, 'keep')  # default keep ถ้าไม่เจอ

outlier_defect['decision'] = outlier_defect.apply(get_decision, axis=1)

# Report
print("Decisions applied:")
print(outlier_defect['decision'].value_counts())
print()
print("Excluded by (class, flag):")
excluded = outlier_defect[outlier_defect['decision'] == 'exclude']
print(excluded.groupby(['class_unified', 'flag']).size())

outlier_defect.to_csv(f'{SAM_OUT}outlier_defect_decided.csv', index=False)
print(f"\n✅ Saved: outlier_defect_decided.csv")
```

### Update Annotations — Mark Excluded

```python
final_annotations = pd.read_csv(f'{DRIVE_BASE}final_annotations.csv')

# Recreate deterministic annotation_id (same logic as SAM notebook)
if 'annotation_id' not in final_annotations.columns:
    final_annotations['annotation_id'] = [
        f'ann_{i:06d}' for i in range(len(final_annotations))
    ]

# Mark excluded annotations
excluded_ann_ids = set(
    outlier_defect[outlier_defect['decision'] == 'exclude']['annotation_id']
)

final_annotations['sam_excluded'] = final_annotations['annotation_id'].isin(excluded_ann_ids)
final_annotations['kept_for_training'] = ~final_annotations['sam_excluded']

# Report
print(f"Total annotations:       {len(final_annotations):,}")
print(f"Excluded by SAM QA:      {final_annotations['sam_excluded'].sum():,}")
print(f"Kept for training:       {final_annotations['kept_for_training'].sum():,}")
print()
print("Per class after exclusion:")
kept = final_annotations[final_annotations['kept_for_training']]
print(kept['class_unified'].value_counts())

# Save
final_annotations.to_csv(f'{DRIVE_BASE}final_annotations_v2.csv', index=False)
print(f"\n✅ Saved: final_annotations_v2.csv")
```

---

## 📊 Step 3: Summary Report

```python
panel_flag_counts = panel_df['flag'].value_counts().to_dict()
defect_decisions = outlier_defect['decision'].value_counts().to_dict()

print(f"""
╔══════════════════════════════════════════════════════════╗
║  SAM Outlier QA — Decision Summary                       ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  PANEL QA (fixed):                                       ║
║    Total panel masks:     {len(panel_df):>5}                       ║
║    too_small (<30%):      {panel_flag_counts.get('too_small', 0):>5}                       ║
║    too_large (>90%):      {panel_flag_counts.get('too_large', 0):>5}                       ║
║    noise (<1%):           {panel_flag_counts.get('noise', 0):>5}                       ║
║    ok:                    {panel_flag_counts.get('ok', 0):>5}                       ║
║                                                          ║
║  DEFECT DECISIONS APPLIED:                               ║
║    Total flagged:         {len(outlier_defect):>5}                       ║
║    Keep:                  {defect_decisions.get('keep', 0):>5}                       ║
║    Exclude:               {defect_decisions.get('exclude', 0):>5}                       ║
║                                                          ║
║  FINAL ANNOTATIONS:                                      ║
║    Before:                26,705                         ║
║    Excluded:              {final_annotations['sam_excluded'].sum():>5}                       ║
║    Kept for training:     {final_annotations['kept_for_training'].sum():>5}                       ║
║                                                          ║
║  ⏳ Pending: Tech Lead review panel outlier galleries    ║
╚══════════════════════════════════════════════════════════╝
""")
```

---

## 📋 Acceptance Criteria

- [ ] **Part 1:** `panel_df` มี real stats (ไม่ใช่ NaN) ทั้ง 1,008 rows
- [ ] **Part 1:** Panel outlier flags = `ok / too_small / too_large / noise` (ไม่ใช่ `missing_mask`)
- [ ] **Part 1:** Panel outlier galleries v2 generated สำหรับทุก flag
- [ ] **Part 2:** `outlier_defect_decided.csv` มี `decision` column ครบทุก row
- [ ] **Part 2:** `final_annotations_v2.csv` มี `sam_excluded` + `kept_for_training` columns
- [ ] **Part 2:** Exclude count ตรงกับ Tech Lead decisions (~100 annotations)
- [ ] Summary report แสดงตัวเลขถูกต้อง

---

## 📤 Deliverables

1. **Notebook:** `sam_outlier_decisions.ipynb`
2. **Files saved to Drive:**
   - `sam_outputs/outlier_panel_v2.csv` (with empty decision column)
   - `sam_outputs/outlier_defect_decided.csv` (with decisions applied)
   - `sam_outputs/outlier_panel_*_v2.png` (galleries per flag)
   - `final_annotations_v2.csv` (with exclusion flags)

---

## 🎯 Next Step

หลังจาก Codex รันเสร็จ:

1. **Paste panel outlier galleries มาให้ Tech Lead** (โดยเฉพาะ `too_small` และ `too_large`)
2. Tech Lead ตัดสิน panel decisions
3. Update `outlier_panel_v2.csv` ด้วย decisions
4. ค่อยเข้า Phase 5 (Merge + YOLO Export)
