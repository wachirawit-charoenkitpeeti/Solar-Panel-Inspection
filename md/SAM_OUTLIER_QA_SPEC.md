# 📋 Engineering Spec — SAM Outlier QA (Phase 4.5)

**Project:** Solar Panel Inspection AI  
**Phase:** SAM Output Quality Review  
**Assigned to:** Codex (engineer)  
**Tech Lead:** Claude  
**Priority:** P0 (blocker for Phase 5)  
**Prerequisite:** SAM masks generated ✅ (1,008 panel + 5,197 defect masks)

---

## 🎯 Goal

Generate galleries ของ outlier masks เพื่อให้ Tech Lead/Owner ตัดสินใจ keep/exclude/review ทุกตัว ก่อน export เป็น training format

จาก QA review รอบแรกพบ 2 ปัญหาหลัก:
1. **Panel mask undersized** — บางภาพ SAM จับแค่ส่วนของ panel (เช่น 22.1%)
2. **Defect mask = frame artifact** — SAM จับขอบ panel แทน defect → mask เป็นแถบยาวๆ

---

## 🔒 Outlier Thresholds (Locked)

```python
PANEL_THRESHOLDS = {
    'too_small':  0.30,   # < 30% — suspect undersized
    'too_large':  0.90,   # > 90% — suspect background grabbing
}

DEFECT_THRESHOLDS = {
    # Per class — based on EDA median + observed outliers
    'dust':            {'max_area': 0.20, 'min_area': 0.0001},
    'bird_drop':       {'max_area': 0.03, 'min_area': 0.0001},
    'leaf':            {'max_area': 0.03, 'min_area': 0.0001},
    'physical_damage': {'max_area': 0.30, 'min_area': 0.0001},
}

# Frame-artifact detection (mask shape)
ELONGATION_THRESHOLD = 5.0   # aspect ratio of mask bbox > 5:1 = strip-like
MASK_TO_BBOX_RATIO   = 0.20  # mask < 20% of bbox = SAM missed defect
```

---

## 📦 Input

```python
DRIVE_BASE = '/content/drive/MyDrive/ai builders/dataset/cleaned_v3/'
SAM_OUT    = f'{DRIVE_BASE}sam_outputs/'

manifest         = pd.read_csv(f'{DRIVE_BASE}final_manifest.csv')
final_annotations = pd.read_csv(f'{DRIVE_BASE}final_annotations.csv')
sam_progress     = pd.read_csv(f'{SAM_OUT}sam_progress.csv')
```

---

## 📊 Step 1: Compute Mask Stats

```python
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def mask_stats(mask_path):
    """Return area_ratio, bbox_aspect_ratio."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    H, W = mask.shape
    binary = (mask > 127).astype(np.uint8)
    area_ratio = binary.sum() / (H * W)
    
    # Bbox aspect ratio
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return {'area_ratio': 0, 'aspect_ratio': 0}
    bw = xs.max() - xs.min() + 1
    bh = ys.max() - ys.min() + 1
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    
    return {'area_ratio': float(area_ratio), 'aspect_ratio': float(aspect)}
```

### 1a. Panel mask stats
```python
panel_stats = []
panel_files = list(Path(f'{SAM_OUT}panel_masks').glob('*_panel.png'))

for fp in tqdm(panel_files, desc='Panel stats'):
    image_id = fp.stem.replace('_panel', '')
    s = mask_stats(str(fp))
    if s:
        panel_stats.append({
            'image_id':     image_id,
            'mask_path':    str(fp),
            'area_ratio':   s['area_ratio'],
            'aspect_ratio': s['aspect_ratio'],
        })

panel_df = pd.DataFrame(panel_stats)
panel_df = panel_df.merge(
    manifest[['image_id', 'final_category', 'source_dataset', 'original_path']],
    on='image_id', how='left'
)
```

### 1b. Defect mask stats
```python
defect_stats = []
defect_files = list(Path(f'{SAM_OUT}defect_masks').glob('*.png'))

for fp in tqdm(defect_files, desc='Defect stats'):
    # filename format: {image_id}_{annotation_id}.png
    parts = fp.stem.rsplit('_', 1)
    image_id, ann_id = parts[0], parts[1]
    s = mask_stats(str(fp))
    if s:
        defect_stats.append({
            'image_id':      image_id,
            'annotation_id': ann_id,
            'mask_path':     str(fp),
            'mask_area':     s['area_ratio'],
            'aspect_ratio':  s['aspect_ratio'],
        })

defect_df = pd.DataFrame(defect_stats)

# Merge with annotation info (class + bbox area)
defect_df = defect_df.merge(
    final_annotations[['annotation_id', 'class_unified', 'area_normalized', 'coords_normalized']],
    on='annotation_id', how='left'
)
defect_df = defect_df.rename(columns={'area_normalized': 'bbox_area'})
defect_df['mask_to_bbox_ratio'] = defect_df['mask_area'] / defect_df['bbox_area'].clip(lower=1e-6)
```

---

## 🚩 Step 2: Flag Outliers

### 2a. Panel outliers
```python
panel_df['flag'] = 'ok'
panel_df.loc[panel_df['area_ratio'] < 0.30, 'flag'] = 'too_small'
panel_df.loc[panel_df['area_ratio'] > 0.90, 'flag'] = 'too_large'
panel_df.loc[panel_df['area_ratio'] < 0.01, 'flag'] = 'noise'

panel_outliers = panel_df[panel_df['flag'] != 'ok'].copy()
print(f"Panel outliers: {len(panel_outliers)} / {len(panel_df)}")
print(panel_outliers['flag'].value_counts())
```

### 2b. Defect outliers
```python
def flag_defect(row):
    cls = row['class_unified']
    if cls not in DEFECT_THRESHOLDS:
        return 'unknown_class'
    th = DEFECT_THRESHOLDS[cls]
    
    if row['mask_area'] < th['min_area']:
        return 'too_small'
    if row['mask_area'] > th['max_area']:
        return 'too_large'
    if row['aspect_ratio'] > ELONGATION_THRESHOLD:
        return 'elongated_strip'  # likely frame artifact
    if row['mask_to_bbox_ratio'] < MASK_TO_BBOX_RATIO:
        return 'sam_missed'        # mask much smaller than bbox
    return 'ok'

defect_df['flag'] = defect_df.apply(flag_defect, axis=1)
defect_outliers = defect_df[defect_df['flag'] != 'ok'].copy()
print(f"Defect outliers: {len(defect_outliers)} / {len(defect_df)}")
print(defect_outliers.groupby(['class_unified', 'flag']).size())
```

---

## 🖼️ Step 3: Generate Outlier Galleries

### 3a. Panel outlier gallery
```python
import matplotlib.pyplot as plt

def panel_outlier_gallery(panel_outliers, max_per_flag=12):
    """Grid by flag type. 4 rows × 5 cols per flag."""
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
            overlay[mask > 127] = [255, 50, 50]  # red overlay for outliers
            blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
            
            ax.imshow(blended)
            ax.set_title(
                f'{row["image_id"][:20]}\n'
                f'{row["final_category"]} | area={row["area_ratio"]*100:.1f}% | '
                f'aspect={row["aspect_ratio"]:.1f}',
                fontsize=9
            )
            ax.axis('off')
        for ax in axes[n:]:
            ax.axis('off')
        
        plt.suptitle(f'PANEL OUTLIERS: {flag} ({len(panel_outliers[panel_outliers["flag"]==flag])} total)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = f'{SAM_OUT}outlier_panel_{flag}.png'
        plt.savefig(out_path, dpi=80, bbox_inches='tight')
        plt.show()
        print(f"Saved: {out_path}")

panel_outlier_gallery(panel_outliers)
```

### 3b. Defect outlier gallery (per class × per flag)
```python
def defect_outlier_gallery(defect_outliers, max_per_group=10):
    """Grid by (class, flag) — show bbox + mask overlay."""
    for cls in defect_outliers['class_unified'].unique():
        for flag in defect_outliers[defect_outliers['class_unified']==cls]['flag'].unique():
            subset = defect_outliers[
                (defect_outliers['class_unified'] == cls) &
                (defect_outliers['flag'] == flag)
            ].head(max_per_group)
            
            if len(subset) == 0:
                continue
            
            n = len(subset)
            cols = 5
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
            axes = np.array(axes).flatten()
            
            for ax, (_, row) in zip(axes, subset.iterrows()):
                img_row = manifest[manifest['image_id'] == row['image_id']].iloc[0]
                img = cv2.imread(img_row['original_path'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                H, W = img.shape[:2]
                mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
                
                # Draw bbox
                coords = [float(x) for x in str(row['coords_normalized']).split(',')[:4]]
                cx, cy, bw, bh = coords
                x1, y1 = int((cx-bw/2)*W), int((cy-bh/2)*H)
                x2, y2 = int((cx+bw/2)*W), int((cy+bh/2)*H)
                
                overlay = img.copy()
                overlay[mask > 127] = [50, 255, 50]  # green mask
                blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
                cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                ax.imshow(blended)
                ax.set_title(
                    f'{cls} | {flag}\n'
                    f'mask={row["mask_area"]*100:.2f}% | '
                    f'aspect={row["aspect_ratio"]:.1f} | '
                    f'm/b={row["mask_to_bbox_ratio"]:.2f}',
                    fontsize=8
                )
                ax.axis('off')
            for ax in axes[n:]:
                ax.axis('off')
            
            plt.suptitle(f'DEFECT OUTLIERS: {cls} / {flag}',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            out_path = f'{SAM_OUT}outlier_defect_{cls}_{flag}.png'
            plt.savefig(out_path, dpi=80, bbox_inches='tight')
            plt.show()
            print(f"Saved: {out_path}")

defect_outlier_gallery(defect_outliers)
```

---

## 📋 Step 4: Save Outlier CSVs

```python
panel_outliers['decision'] = ''  # to be filled by Tech Lead
defect_outliers['decision'] = ''

panel_outliers.to_csv(f'{SAM_OUT}outlier_panel.csv', index=False)
defect_outliers.to_csv(f'{SAM_OUT}outlier_defect.csv', index=False)

print(f"✅ Saved: outlier_panel.csv ({len(panel_outliers)} rows)")
print(f"✅ Saved: outlier_defect.csv ({len(defect_outliers)} rows)")
```

**Decision column values (Tech Lead fills in later):**
- `keep` — mask ดูแล้ว ok เป็น false positive ของ filter
- `exclude` — drop annotation/image จาก training
- `regen` — re-run SAM ด้วย prompt อื่น
- `manual` — annotate ใหม่ใน Roboflow

---

## 📊 Step 5: Summary Report

```python
print(f"""
╔══════════════════════════════════════════════════════════╗
║  SAM Outlier QA Report                                   ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  PANEL MASKS ({len(panel_df)} total):                            ║
║    too_small (<30%):  {(panel_df['flag']=='too_small').sum():>4}                          ║
║    too_large (>90%):  {(panel_df['flag']=='too_large').sum():>4}                          ║
║    noise   (<1%):     {(panel_df['flag']=='noise').sum():>4}                          ║
║    ─────────────────────                                 ║
║    Total flagged:     {(panel_df['flag']!='ok').sum():>4}  ({(panel_df['flag']!='ok').mean()*100:.1f}%)               ║
║                                                          ║
║  DEFECT MASKS ({len(defect_df)} total):                          ║""")

for cls in ['dust', 'bird_drop', 'leaf', 'physical_damage']:
    cls_df = defect_df[defect_df['class_unified'] == cls]
    if len(cls_df) == 0: continue
    flagged = (cls_df['flag'] != 'ok').sum()
    print(f"║    {cls:<18} {flagged:>4} / {len(cls_df):<5} flagged             ║")

print(f"""║                                                          ║
║  Outlier galleries saved to:                             ║
║    sam_outputs/outlier_panel_*.png                       ║
║    sam_outputs/outlier_defect_*_*.png                    ║
║                                                          ║
║  CSVs for Tech Lead review:                              ║
║    sam_outputs/outlier_panel.csv                         ║
║    sam_outputs/outlier_defect.csv                        ║
╚══════════════════════════════════════════════════════════╝
""")
```

---

## 📋 Acceptance Criteria

- [ ] `panel_df` + `defect_df` คำนวณ stats สำเร็จทั้ง 6,205 masks
- [ ] Panel outlier flags: `too_small`, `too_large`, `noise`
- [ ] Defect outlier flags: `too_small`, `too_large`, `elongated_strip`, `sam_missed`
- [ ] Panel outlier galleries แสดงทุก flag type
- [ ] Defect outlier galleries แสดงทุก (class × flag) combination
- [ ] `outlier_panel.csv` + `outlier_defect.csv` saved with `decision` column (ว่าง)
- [ ] Summary report แสดงจำนวน flagged ต่อ category

---

## 📤 Deliverables

1. **Notebook:** `sam_outlier_qa.ipynb`
2. **Outlier galleries:** PNG files ใน `sam_outputs/`
3. **CSVs:** `outlier_panel.csv` + `outlier_defect.csv`
4. **Summary box** แสดงใน notebook output

---

## 🎯 Next Step After This

Tech Lead จะ:
1. ดู gallery แต่ละ flag → ตัดสินใจ pattern ของ outliers
2. Fill `decision` column ใน CSVs
3. Codex apply decisions → exclude/regen/manual review
4. ค่อย export YOLO format ใน Phase 5
