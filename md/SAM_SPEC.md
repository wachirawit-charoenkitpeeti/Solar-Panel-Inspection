# 📋 Engineering Spec — SAM Mask Generation (Phase 4)

**Project:** Solar Panel Inspection AI  
**Phase:** SAM Mask Generation  
**Assigned to:** Codex (engineer)  
**Tech Lead:** Claude  
**Priority:** P0  
**Platform:** Google Colab (T4 GPU)  
**Prerequisite:** `cleaned_v3/` saved to Drive ✅

---

## 🎯 Goal

Generate masks ที่ขาดอยู่ใน cleaned dataset ด้วย SAM (Segment Anything Model) เพื่อให้ทุกภาพมี panel + defect masks ครบสำหรับ training Pipeline B

**Workload:**

| Task | Images | Method |
|---|---|---|
| Panel masks (B1 + C) | 1,008 | Center-point prompt SAM |
| Defect masks (C only) | 762 | Bbox-prompt SAM |
| **Total SAM operations** | **~3,000+** | (bbox conversions = multi-bbox per image) |

---

## 🔒 Locked Decisions

```yaml
sam_model:        ViT-H (highest quality, ok on T4 for batch)
sam_checkpoint:   sam_vit_h_4b8939.pth
checkpoint_freq:  every 50 images
output_format:    PNG binary masks (0 / 255)
mask_resolution:  640×640 (match source images)
qa_samples:       20 panel + 20 defect for visual review
resume:           must support resume from checkpoint
```

---

## 📦 Input

```python
# Load from Drive (saved in previous spec)
DRIVE_BASE = '/content/drive/MyDrive/ai builders/dataset/cleaned_v3/'

manifest         = pd.read_csv(f'{DRIVE_BASE}final_manifest.csv')
final_annotations = pd.read_csv(f'{DRIVE_BASE}final_annotations.csv')
```

---

## 📂 Output Structure

```
cleaned_v3/
└── sam_outputs/
    ├── panel_masks/
    │   └── {image_id}_panel.png         ← 1,008 files
    ├── defect_masks/
    │   └── {image_id}_{ann_id}.png      ← ~3,000 files
    ├── sam_progress.csv                 ← checkpoint tracker
    ├── sam_qa_panel.png                 ← 20-image grid
    ├── sam_qa_defect.png                ← 20-image grid
    └── sam_summary.json
```

---

## 🛠️ Setup

### Cell 1: Install dependencies
```bash
!pip install segment-anything-py
!pip install opencv-python-headless
```

### Cell 2: Download SAM checkpoint
```python
import os, urllib.request

CKPT_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
CKPT_PATH = '/content/sam_vit_h.pth'

if not os.path.exists(CKPT_PATH):
    print("Downloading SAM ViT-H checkpoint (~2.4 GB)...")
    urllib.request.urlretrieve(CKPT_URL, CKPT_PATH)
print(f"✅ Checkpoint ready: {CKPT_PATH}")
```

### Cell 3: Load SAM model
```python
import torch
from segment_anything import sam_model_registry, SamPredictor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

sam = sam_model_registry['vit_h'](checkpoint=CKPT_PATH)
sam.to(device=device)
predictor = SamPredictor(sam)
print("✅ SAM ViT-H loaded")
```

---

## 🔄 Checkpoint System

```python
import pandas as pd
from pathlib import Path

PROGRESS_PATH = f'{DRIVE_BASE}sam_outputs/sam_progress.csv'
Path(f'{DRIVE_BASE}sam_outputs').mkdir(exist_ok=True, parents=True)
Path(f'{DRIVE_BASE}sam_outputs/panel_masks').mkdir(exist_ok=True)
Path(f'{DRIVE_BASE}sam_outputs/defect_masks').mkdir(exist_ok=True)

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        return pd.read_csv(PROGRESS_PATH)
    return pd.DataFrame(columns=['image_id', 'task', 'status', 'output_path'])

def save_progress(progress_df):
    progress_df.to_csv(PROGRESS_PATH, index=False)

def is_done(progress_df, image_id, task):
    return ((progress_df['image_id'] == image_id) &
            (progress_df['task'] == task) &
            (progress_df['status'] == 'done')).any()
```

---

## 🎨 Phase A: Panel Mask Generation

**Target:** B1 + C categories = 1,008 images  
**Method:** Center-point prompt + filter for largest panel-like mask

### Strategy
- Click point at image center → SAM proposes 3 masks
- Pick the largest mask with reasonable aspect ratio (not whole image, not tiny)
- Filter: mask area should be 10–90% of image area

### Code

```python
import cv2
import numpy as np
from tqdm import tqdm

def generate_panel_mask(image_path):
    """
    Generate panel mask using center-point prompt.
    Returns binary mask (H×W, uint8: 0 or 255).
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    
    predictor.set_image(image)
    
    # Center point prompt
    point_coords = np.array([[W // 2, H // 2]])
    point_labels = np.array([1])  # foreground
    
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    
    # Filter: pick mask with largest area within 10-90% of image
    image_area = H * W
    best_mask = None
    best_score = -1
    
    for mask, score in zip(masks, scores):
        mask_area = mask.sum()
        ratio = mask_area / image_area
        if 0.10 <= ratio <= 0.90 and score > best_score:
            best_mask = mask
            best_score = score
    
    # Fallback: use highest-confidence mask if no filter match
    if best_mask is None:
        best_mask = masks[np.argmax(scores)]
    
    return (best_mask * 255).astype(np.uint8)


def process_panel_phase(manifest, progress_df):
    targets = manifest[
        (manifest['kept_in_dataset'] == True) &
        (manifest['final_category'].isin(['B1', 'C']))
    ]
    print(f"Panel mask phase: {len(targets)} images")
    
    for idx, (_, row) in enumerate(tqdm(targets.iterrows(), total=len(targets))):
        image_id = row['image_id']
        
        if is_done(progress_df, image_id, 'panel'):
            continue
        
        try:
            mask = generate_panel_mask(row['original_path'])
            out_path = f'{DRIVE_BASE}sam_outputs/panel_masks/{image_id}_panel.png'
            cv2.imwrite(out_path, mask)
            
            progress_df = pd.concat([progress_df, pd.DataFrame([{
                'image_id': image_id, 'task': 'panel',
                'status': 'done', 'output_path': out_path
            }])], ignore_index=True)
        except Exception as e:
            progress_df = pd.concat([progress_df, pd.DataFrame([{
                'image_id': image_id, 'task': 'panel',
                'status': f'error: {str(e)[:50]}', 'output_path': None
            }])], ignore_index=True)
        
        # Checkpoint every 50 images
        if (idx + 1) % 50 == 0:
            save_progress(progress_df)
            print(f"  Checkpoint at {idx+1}/{len(targets)}")
    
    save_progress(progress_df)
    return progress_df

progress_df = load_progress()
progress_df = process_panel_phase(manifest, progress_df)
```

---

## 🎯 Phase B: Defect Mask Conversion (bbox → mask)

**Target:** Category C only = 762 images (~3,000 bbox annotations)  
**Method:** Bbox-prompt SAM (best SAM use case)

### Code

```python
def generate_defect_mask(image_path, bbox_normalized):
    """
    Convert YOLO bbox (cx, cy, w, h normalized) → binary mask via SAM.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    
    # Convert YOLO normalized → pixel xyxy
    cx, cy, bw, bh = bbox_normalized
    x1 = int((cx - bw/2) * W)
    y1 = int((cy - bh/2) * H)
    x2 = int((cx + bw/2) * W)
    y2 = int((cy + bh/2) * H)
    
    predictor.set_image(image)
    input_box = np.array([x1, y1, x2, y2])
    
    masks, scores, _ = predictor.predict(
        box=input_box[None, :],
        multimask_output=False,
    )
    
    return (masks[0] * 255).astype(np.uint8)


def process_defect_phase(manifest, final_annotations, progress_df):
    c_images = manifest[
        (manifest['kept_in_dataset'] == True) &
        (manifest['final_category'] == 'C')
    ]['image_id']
    
    targets = final_annotations[
        (final_annotations['image_id'].isin(c_images)) &
        (final_annotations['format'] == 'bbox')
    ]
    print(f"Defect mask phase: {len(targets)} annotations from {len(c_images)} images")
    
    current_image = None
    
    for idx, (_, ann) in enumerate(tqdm(targets.iterrows(), total=len(targets))):
        image_id = ann['image_id']
        ann_id = ann['annotation_id']
        task_key = f'defect_{ann_id}'
        
        if is_done(progress_df, image_id, task_key):
            continue
        
        try:
            # Parse bbox coords (format: "cx,cy,w,h" or similar)
            coords = [float(x) for x in str(ann['coords_normalized']).split(',')]
            
            img_path = manifest[manifest['image_id'] == image_id]['original_path'].iloc[0]
            mask = generate_defect_mask(img_path, coords[:4])
            
            out_path = f'{DRIVE_BASE}sam_outputs/defect_masks/{image_id}_{ann_id}.png'
            cv2.imwrite(out_path, mask)
            
            progress_df = pd.concat([progress_df, pd.DataFrame([{
                'image_id': image_id, 'task': task_key,
                'status': 'done', 'output_path': out_path
            }])], ignore_index=True)
        except Exception as e:
            progress_df = pd.concat([progress_df, pd.DataFrame([{
                'image_id': image_id, 'task': task_key,
                'status': f'error: {str(e)[:50]}', 'output_path': None
            }])], ignore_index=True)
        
        if (idx + 1) % 100 == 0:
            save_progress(progress_df)
            print(f"  Checkpoint at {idx+1}/{len(targets)}")
    
    save_progress(progress_df)
    return progress_df

progress_df = process_defect_phase(manifest, final_annotations, progress_df)
```

---

## 🖼️ Visual QA

หลัง SAM เสร็จต้องสร้าง **2 grids ให้ Tech Lead review** ก่อน confirm

### QA Grid 1: Panel masks — 20 random samples (4 rows × 5 cols)
- แสดง original image + panel mask overlay (สีน้ำเงิน opacity 40%)
- Title: `image_id | category (B1 หรือ C) | mask area %`

### QA Grid 2: Defect masks — 20 random samples
- แสดง original image + bbox (สีแดง) + SAM mask (สีเขียว opacity 50%)
- Title: `class | bbox area % | mask area %`

```python
import matplotlib.pyplot as plt

def make_panel_qa_grid(manifest, sample_size=20):
    samples = manifest[
        (manifest['kept_in_dataset']) &
        (manifest['final_category'].isin(['B1', 'C']))
    ].sample(min(sample_size, 20), random_state=42)
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for ax, (_, row) in zip(axes.flat, samples.iterrows()):
        img = cv2.imread(row['original_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_path = f'{DRIVE_BASE}sam_outputs/panel_masks/{row["image_id"]}_panel.png'
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            overlay = img.copy()
            overlay[mask > 127] = [0, 100, 255]
            blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
            ax.imshow(blended)
            area_pct = (mask > 127).sum() / mask.size * 100
            ax.set_title(f'{row["image_id"][:15]}\n{row["final_category"]} | {area_pct:.0f}%', fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{DRIVE_BASE}sam_outputs/sam_qa_panel.png', dpi=80, bbox_inches='tight')
    plt.show()

make_panel_qa_grid(manifest)
# (similar function for defect grid — bbox + mask side by side)
```

---

## 📊 Final Summary

```python
import json

panel_done   = (progress_df['task'] == 'panel') & (progress_df['status'] == 'done')
defect_done  = (progress_df['task'].str.startswith('defect_')) & (progress_df['status'] == 'done')
panel_error  = (progress_df['task'] == 'panel') & (progress_df['status'].str.startswith('error'))
defect_error = (progress_df['task'].str.startswith('defect_')) & (progress_df['status'].str.startswith('error'))

summary = {
    'phase': 'SAM mask generation',
    'sam_model': 'ViT-H',
    'panel_masks_target': 1008,
    'panel_masks_done':   int(panel_done.sum()),
    'panel_masks_error':  int(panel_error.sum()),
    'defect_masks_done':  int(defect_done.sum()),
    'defect_masks_error': int(defect_error.sum()),
    'output_path': f'{DRIVE_BASE}sam_outputs/',
}

with open(f'{DRIVE_BASE}sam_outputs/sam_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"""
╔══════════════════════════════════════════════════════════╗
║  SAM Mask Generation — Complete                          ║
╠══════════════════════════════════════════════════════════╣
║  Panel masks:    {summary['panel_masks_done']:>4}/{summary['panel_masks_target']:>4}  ({summary['panel_masks_error']} errors)        ║
║  Defect masks:   {summary['defect_masks_done']:>4}        ({summary['defect_masks_error']} errors)             ║
║                                                          ║
║  Output: cleaned_v3/sam_outputs/                         ║
║  QA grids: sam_qa_panel.png, sam_qa_defect.png           ║
╚══════════════════════════════════════════════════════════╝
""")
```

---

## 📋 Acceptance Criteria

- [ ] SAM ViT-H downloaded + loaded on GPU
- [ ] Checkpoint system works (test by interrupting + resuming)
- [ ] Phase A: 1,008 panel masks generated (>95% success rate)
- [ ] Phase B: ~3,000 defect masks generated (>95% success rate)
- [ ] Panel QA grid (20 samples) saved + displayed in notebook
- [ ] Defect QA grid (20 samples) saved + displayed in notebook
- [ ] `sam_progress.csv` + `sam_summary.json` saved to Drive
- [ ] Error rate < 5% per phase

---

## ⚠️ Risks

| Risk | Mitigation |
|---|---|
| Colab session timeout (12hr limit) | Checkpoint every 50 → resume in new session |
| GPU OOM with ViT-H | Fallback to ViT-B (`sam_vit_b_01ec64.pth`) |
| SAM panel center prompt picks wrong region | Track in QA grid; if >20% wrong, switch to AutomaticMaskGenerator |
| Drive write throttling (many small files) | Batch writes; consider zipping masks at end |
| `coords_normalized` format mismatch | Print first 5 examples + verify before batch run |

---

## 🎯 What Tech Lead Will Review

หลัง Codex ทำเสร็จ ผมจะดู:

1. **Error rate** — < 5% ทั้งสอง phase
2. **Panel QA grid** — panel masks ครอบ panel จริงไม่ใช่พื้นหลัง
3. **Defect QA grid** — defect masks ตรงกับ bbox ที่ใช้เป็น prompt
4. **B1 vs C panel quality** — C อาจยากกว่าเพราะภาพ close-up

ถ้า QA ผ่าน → ต่อ Phase 5 (COCO export + training prep)  
ถ้า QA fail บางส่วน → ระบุภาพที่ผิด + decide manual review หรือ exclude
