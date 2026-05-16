# 📋 Engineering Spec — Phase 5: Merge + YOLO Export

**Project:** Solar Panel Inspection AI  
**Phase:** 5 — Dataset Export to YOLO Segmentation Format  
**Assigned to:** Codex (engineer)  
**Tech Lead:** Claude  
**Priority:** P0  
**Prerequisite:** Dataset Finalization ✅ (`final_manifest_v2.csv` + `training_annotations_final.csv`)

---

## 🎯 Goal

แปลง dataset ทั้งหมดเป็น YOLO segmentation format พร้อมสำหรับ train YOLOv8-seg

**Output target:**
```
dataset_yolo/
├── images/
│   ├── train/  (~1,874 images)
│   ├── val/    (~234 images)
│   └── test/   (~235 images)
├── labels/
│   ├── train/  (.txt YOLO seg format)
│   ├── val/
│   └── test/
├── data.yaml
└── export_summary.json
```

---

## 🔒 Locked Specifications

### Class IDs (final taxonomy)
```python
CLASS_IDS = {
    'panel_clean':      0,
    'panel_defective':  1,
    'dust':             2,
    'bird_drop':        3,
    'physical_damage':  4,
    'leaf':             5,
}
```

### Annotation Sources (3 ways)
```yaml
A + A_partial: polygon coords จาก Roboflow เดิม (ใช้ตรงๆ)
B1:
  panel:  SAM mask → polygon (เฉพาะที่ panel_mask_valid = True)
  defect: polygon coords เดิม
C:
  panel:  SAM mask → polygon (เฉพาะที่ panel_mask_valid = True)
  defect: SAM mask → polygon (จาก bbox prompt)
```

### YOLO Segmentation Format
```
class_id  x1 y1 x2 y2 x3 y3 ... xn yn
```
- coordinates ต้อง normalized (0-1)
- ทุก polygon ต้องมี ≥ 3 จุด
- 1 บรรทัด = 1 instance

---

## 📦 Input

```python
DRIVE_BASE = '/content/drive/MyDrive/ai builders/dataset/cleaned_v3/'
SAM_OUT    = f'{DRIVE_BASE}sam_outputs/'
OUTPUT     = '/content/drive/MyDrive/ai builders/dataset/dataset_yolo/'

manifest    = pd.read_csv(f'{DRIVE_BASE}final_manifest_v2.csv')
annotations = pd.read_csv(f'{DRIVE_BASE}training_annotations_final.csv')

# Filter เฉพาะ training-ready
manifest = manifest[manifest['training_ready'] == True]
print(f"Training-ready images: {len(manifest):,}")
```

---

## 🛠️ Setup

```python
import os, json, shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Create folder structure
for split in ['train', 'val', 'test']:
    Path(f'{OUTPUT}images/{split}').mkdir(parents=True, exist_ok=True)
    Path(f'{OUTPUT}labels/{split}').mkdir(parents=True, exist_ok=True)
```

---

## 🔧 Helper Functions

### 1. Mask → Polygon Converter

```python
def mask_to_polygon(mask_path, min_points=3, simplify_epsilon=2.0):
    """
    Convert binary mask PNG → normalized polygon coords.
    Returns list of (x, y) normalized tuples or None if failed.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    H, W = mask.shape
    binary = (mask > 127).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Pick largest contour
    largest = max(contours, key=cv2.contourArea)
    
    # Simplify polygon (reduce point count)
    epsilon = simplify_epsilon
    simplified = cv2.approxPolyDP(largest, epsilon, closed=True)
    
    if len(simplified) < min_points:
        return None
    
    # Normalize coords
    points = []
    for p in simplified:
        x, y = p[0]
        points.append((x / W, y / H))
    
    return points
```

### 2. Polygon Parser (for existing polygon annotations)

```python
def parse_existing_polygon(coords_str):
    """
    Parse normalized polygon string from final_annotations.
    Returns list of (x, y) tuples.
    """
    try:
        nums = [float(x) for x in str(coords_str).split(',') if x.strip()]
        if len(nums) < 6 or len(nums) % 2 != 0:
            return None
        return [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
    except:
        return None
```

### 3. Bbox to Polygon (fallback)

```python
def bbox_to_polygon(coords_str):
    """
    Convert YOLO bbox (cx, cy, w, h normalized) → 4-point polygon.
    Used as fallback if SAM mask missing.
    """
    nums = [float(x) for x in str(coords_str).split(',')[:4]]
    cx, cy, w, h = nums
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
```

---

## 📊 Main Export Loop

```python
def export_image(image_row, image_annotations, split):
    """
    Process 1 image:
    1. Copy image to images/{split}/
    2. Build YOLO .txt label using polygon/SAM masks
    3. Write label to labels/{split}/
    """
    image_id = image_row['image_id']
    src_img = image_row['original_path']
    category = image_row['final_category']
    
    # Standardized output filename
    out_img = f'{OUTPUT}images/{split}/{image_id}.jpg'
    out_txt = f'{OUTPUT}labels/{split}/{image_id}.txt'
    
    # Copy image
    if not os.path.exists(out_img):
        img = cv2.imread(src_img)
        cv2.imwrite(out_img, img)
    
    # Build label lines
    lines = []
    
    for _, ann in image_annotations.iterrows():
        cls_name = ann['class_unified']
        cls_id   = CLASS_IDS.get(cls_name)
        if cls_id is None:
            continue
        
        polygon = None
        is_panel = cls_name in ['panel_clean', 'panel_defective']
        
        # === Source selection ===
        if category in ['A', 'A_partial']:
            # Always use existing polygon
            polygon = parse_existing_polygon(ann['coords_normalized'])
        
        elif category == 'B1':
            if is_panel:
                # SAM panel mask
                mask_path = f'{SAM_OUT}panel_masks/{image_id}_panel.png'
                polygon = mask_to_polygon(mask_path)
            else:
                # Existing defect polygon
                polygon = parse_existing_polygon(ann['coords_normalized'])
        
        elif category == 'C':
            if is_panel:
                mask_path = f'{SAM_OUT}panel_masks/{image_id}_panel.png'
                polygon = mask_to_polygon(mask_path)
            else:
                # SAM defect mask
                ann_id = ann['annotation_id']
                mask_path = f'{SAM_OUT}defect_masks/{image_id}_{ann_id}.png'
                polygon = mask_to_polygon(mask_path)
                # Fallback: bbox if SAM mask failed
                if polygon is None:
                    polygon = bbox_to_polygon(ann['coords_normalized'])
        
        if polygon is None or len(polygon) < 3:
            continue
        
        # Build YOLO seg line
        coords_str = ' '.join(f'{x:.6f} {y:.6f}' for x, y in polygon)
        lines.append(f'{cls_id} {coords_str}')
    
    # Write label file
    with open(out_txt, 'w') as f:
        f.write('\n'.join(lines))
    
    return len(lines)
```

---

## 🚀 Run Export

```python
export_stats = {
    'images_exported':       0,
    'annotations_exported':  0,
    'images_skipped_empty':  0,
    'per_split': {'train': 0, 'val': 0, 'test': 0},
    'per_class': {k: 0 for k in CLASS_IDS},
}

for split in ['train', 'val', 'test']:
    split_images = manifest[manifest['final_split'] == split]
    print(f"\n--- {split.upper()} ({len(split_images)} images) ---")
    
    for _, img_row in tqdm(split_images.iterrows(), total=len(split_images)):
        img_id = img_row['image_id']
        img_anns = annotations[annotations['image_id'] == img_id]
        
        if len(img_anns) == 0:
            export_stats['images_skipped_empty'] += 1
            continue
        
        # Skip if image has ONLY panel annotations (no defect)
        # in B1/C with no valid panel mask
        if img_row['final_category'] in ['B1', 'C']:
            if not img_row.get('panel_mask_valid', False):
                non_panel = img_anns[
                    ~img_anns['class_unified'].isin(['panel_clean', 'panel_defective'])
                ]
                if len(non_panel) == 0:
                    export_stats['images_skipped_empty'] += 1
                    continue
                img_anns = non_panel  # keep only defects
        
        n_lines = export_image(img_row, img_anns, split)
        if n_lines > 0:
            export_stats['images_exported'] += 1
            export_stats['annotations_exported'] += n_lines
            export_stats['per_split'][split] += 1
            
            for cls in img_anns['class_unified']:
                if cls in export_stats['per_class']:
                    export_stats['per_class'][cls] += 1

print("\n✅ Export complete")
print(json.dumps(export_stats, indent=2))
```

---

## 📝 Generate data.yaml

```python
data_yaml = f"""# Solar Panel Inspection — YOLOv8-seg dataset
# Generated from cleaned_v3 + SAM masks + QA decisions

path: {OUTPUT}
train: images/train
val:   images/val
test:  images/test

nc: 6
names:
  0: panel_clean
  1: panel_defective
  2: dust
  3: bird_drop
  4: physical_damage
  5: leaf
"""

with open(f'{OUTPUT}data.yaml', 'w') as f:
    f.write(data_yaml)
print(f"✅ Saved: data.yaml")
```

---

## 🔍 Validation Step

```python
def validate_export():
    """Sanity checks on exported dataset."""
    issues = []
    
    for split in ['train', 'val', 'test']:
        img_dir = Path(f'{OUTPUT}images/{split}')
        lbl_dir = Path(f'{OUTPUT}labels/{split}')
        
        imgs = sorted(img_dir.glob('*.jpg'))
        lbls = sorted(lbl_dir.glob('*.txt'))
        
        # Pair check
        img_stems = {f.stem for f in imgs}
        lbl_stems = {f.stem for f in lbls}
        
        missing_lbl = img_stems - lbl_stems
        missing_img = lbl_stems - img_stems
        
        if missing_lbl:
            issues.append(f"{split}: {len(missing_lbl)} images without labels")
        if missing_img:
            issues.append(f"{split}: {len(missing_img)} labels without images")
        
        # Check empty labels
        empty = sum(1 for l in lbls if l.read_text().strip() == '')
        if empty > 0:
            issues.append(f"{split}: {empty} empty label files")
        
        # Sample 5 labels — verify format
        for lbl_file in lbls[:5]:
            for line in lbl_file.read_text().strip().split('\n'):
                parts = line.split()
                if len(parts) < 7:  # class_id + at least 3 points
                    issues.append(f"Malformed: {lbl_file.name}: {line[:50]}")
                    break
                try:
                    cls = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    if cls < 0 or cls >= 6:
                        issues.append(f"Invalid class {cls} in {lbl_file.name}")
                    if any(c < 0 or c > 1 for c in coords):
                        issues.append(f"Coords out of [0,1] in {lbl_file.name}")
                except ValueError:
                    issues.append(f"Non-numeric in {lbl_file.name}")
        
        print(f"{split}: {len(imgs)} images, {len(lbls)} labels")
    
    if issues:
        print("\n⚠️ Issues found:")
        for i in issues[:20]:
            print(f"  - {i}")
    else:
        print("\n✅ All validation checks passed")
    
    return issues

issues = validate_export()
```

---

## 📊 Final Summary

```python
summary = {
    'output_path':           OUTPUT,
    'total_images':          export_stats['images_exported'],
    'total_annotations':     export_stats['annotations_exported'],
    'images_skipped_empty':  export_stats['images_skipped_empty'],
    'split_counts':          export_stats['per_split'],
    'class_image_counts':    export_stats['per_class'],
    'validation_issues':     len(issues),
}

with open(f'{OUTPUT}export_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"""
╔══════════════════════════════════════════════════════════╗
║  Phase 5 — YOLO Export Complete                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Output: dataset_yolo/                                   ║
║                                                          ║
║  Images exported:    {export_stats['images_exported']:>5}                       ║
║  Annotations:        {export_stats['annotations_exported']:>5}                       ║
║  Skipped empty:      {export_stats['images_skipped_empty']:>5}                       ║
║                                                          ║
║  Splits:                                                 ║
║    train: {export_stats['per_split']['train']:>5}                                  ║
║    val:   {export_stats['per_split']['val']:>5}                                  ║
║    test:  {export_stats['per_split']['test']:>5}                                  ║
║                                                          ║
║  Validation issues: {len(issues):>3}                              ║
║                                                          ║
║  🚀 Ready for Phase 6: Training                          ║
╚══════════════════════════════════════════════════════════╝
""")
```

---

## 📋 Acceptance Criteria

- [ ] Folder structure ครบ: images/{train,val,test} + labels/{train,val,test}
- [ ] `data.yaml` ถูกสร้าง + path ถูกต้อง
- [ ] Every image มี matching `.txt` label
- [ ] ทุก label line มี format ถูกต้อง: `class_id x1 y1 x2 y2 ... xn yn`
- [ ] ทุก coord อยู่ในช่วง [0, 1]
- [ ] ทุก class_id อยู่ในช่วง [0, 5]
- [ ] Validation issues = 0
- [ ] B1/C images ที่ไม่มี panel mask → keep defects only (ตามที่ Tech Lead lock)
- [ ] Images ที่มีแค่ panel annotation ใน B1/C (panel exclude) → skip

---

## 📤 Deliverables

1. **Notebook:** `phase5_yolo_export.ipynb`
2. **Dataset folder ใน Drive:**
   ```
   dataset_yolo/
   ├── images/{train,val,test}/
   ├── labels/{train,val,test}/
   ├── data.yaml
   └── export_summary.json
   ```
3. **Summary report** แสดงใน notebook

---

## ⚠️ Risks

| Risk | Mitigation |
|---|---|
| `mask_to_polygon` ได้ < 3 จุด | skip annotation, log warning |
| `coords_normalized` parse error | wrap in try/except, log |
| Drive write throttling (4,500+ files) | batch by split, ไม่ใช่ commit ทีละไฟล์ |
| Image filename มี special char | normalize ด้วย `image_id` แทน original filename |
| SAM mask path mismatch (B1/C) | check exists ก่อน convert, fallback to bbox |

---

## 🎯 Next Phase Preview (Phase 6)

หลัง YOLO export เสร็จ Codex จะ train YOLOv8-seg:

```python
from ultralytics import YOLO

model = YOLO('yolov8m-seg.pt')
results = model.train(
    data='/content/drive/MyDrive/ai builders/dataset/dataset_yolo/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    # weighted loss for class imbalance (dust)
)
```

Class weights สำหรับชดเชย dust drop 58.57% จะ apply ตอน training config

---

ขั้นนี้คือสะพานเชื่อมระหว่าง "data pipeline" และ "training" — งานยากผ่านมาหมดแล้วครับ
