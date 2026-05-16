# 📋 Engineering Spec — Save Cleaning v3 to Google Drive

**Project:** Solar Panel Inspection AI  
**Phase:** Data Cleaning — Save Final Output  
**Assigned to:** Codex (engineer)  
**Tech Lead:** Claude  
**Priority:** P0  
**Prerequisite:** `data_cleaning_v3.ipynb` รันครบแล้ว ✅  
**Runtime:** ต้องรันใน Colab session เดียวกับที่รัน v3 (หรือ re-run v3 ก่อน)

---

## 🎯 Goal

Save output ของ cleaning v3 ลง Google Drive ถาวร เพื่อให้ใช้ได้ใน session ถัดไปโดยไม่ต้อง rerun cleaning

---

## 📦 Files to Save

### Output path
```
/content/drive/MyDrive/ai builders/dataset/cleaned_v3/
```

### Files ที่ต้อง save

| File | Source (in runtime) | Description |
|---|---|---|
| `final_manifest.csv` | `manifest` DataFrame | ทุก image + category + split + flags |
| `final_annotations.csv` | `final_annotations` DataFrame | ทุก annotation หลัง cleaning |
| `summary_v3.json` | compute ใหม่ | key metrics สำหรับ reference |

---

## 🛠️ Code

### Cell 1: Mount Drive (ถ้ายังไม่ mount)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Create output folder

```python
import os

DRIVE_OUTPUT = '/content/drive/MyDrive/ai builders/dataset/cleaned_v3/'
os.makedirs(DRIVE_OUTPUT, exist_ok=True)
print(f"Output folder ready: {DRIVE_OUTPUT}")
```

### Cell 3: Save CSVs

```python
# Save manifest
manifest_path = f'{DRIVE_OUTPUT}final_manifest.csv'
manifest.to_csv(manifest_path, index=False)
print(f"✅ Saved: final_manifest.csv  ({len(manifest):,} rows)")

# Save annotations
ann_path = f'{DRIVE_OUTPUT}final_annotations.csv'
final_annotations.to_csv(ann_path, index=False)
print(f"✅ Saved: final_annotations.csv  ({len(final_annotations):,} rows)")
```

### Cell 4: Save summary JSON

```python
import json

kept_mask  = manifest['kept_in_dataset'] == True
defect_cls = ['dust', 'bird_drop', 'physical_damage', 'leaf']

defect_counts = (
    final_annotations[final_annotations['class_unified'].isin(defect_cls)]
    ['class_unified'].value_counts()
)

summary = {
    'version': 'v3',
    'dedup_method': 'MD5 exact',

    # Image counts
    'total_images_raw': 3246,
    'md5_duplicate_drops': int((manifest['final_category'] == 'DUPLICATE').sum()),
    'snow_excluded': int((manifest['final_category'] == 'SNOW_EXCLUDED').sum()),
    'b2_excluded': int((manifest['final_category'] == 'B2').sum()),
    'empty_excluded': int((manifest['final_category'] == 'EMPTY').sum()),
    'total_kept_images': int(kept_mask.sum()),

    # Category breakdown
    'category_counts': manifest['final_category'].value_counts().to_dict(),

    # Annotation counts
    'total_kept_annotations': len(final_annotations),
    'class_counts': final_annotations['class_unified'].value_counts().to_dict(),

    # Defect imbalance
    'defect_imbalance_ratio': round(
        float(defect_counts.iloc[0]) / float(defect_counts.iloc[-1]), 2
    ) if len(defect_counts) > 0 else None,

    # SAM workload
    'needs_sam_panel_masks': int(
        manifest[kept_mask & (manifest['needs_sam_panel'] == True)].shape[0]
    ),
    'needs_sam_defect_masks': int(
        manifest[kept_mask & (manifest['needs_sam_defect_conversion'] == True)].shape[0]
    ),

    # Split
    'split_counts': manifest[kept_mask]['final_split'].value_counts().to_dict(),
    'duplicate_leakage_groups': 0,
}

json_path = f'{DRIVE_OUTPUT}summary_v3.json'
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✅ Saved: summary_v3.json")
```

### Cell 5: Verify save + print confirmation

```python
# Verify all files exist
files = ['final_manifest.csv', 'final_annotations.csv', 'summary_v3.json']
all_ok = True

for fname in files:
    fpath = f'{DRIVE_OUTPUT}{fname}'
    if os.path.exists(fpath):
        size_kb = os.path.getsize(fpath) / 1024
        print(f"✅ {fname:<30} {size_kb:,.0f} KB")
    else:
        print(f"❌ MISSING: {fname}")
        all_ok = False

print()
if all_ok:
    print("🎉 All files saved to Google Drive successfully!")
    print(f"   Path: {DRIVE_OUTPUT}")
else:
    print("⚠️ Some files missing — check errors above")

# Print summary
print(f"""
╔══════════════════════════════════════════════════════════╗
║  Cleaning v3 — Saved to Drive ✅                         ║
╠══════════════════════════════════════════════════════════╣
║  Path: ai builders/dataset/cleaned_v3/                   ║
║                                                          ║
║  Images kept:       {summary['total_kept_images']:<6,}                       ║
║  Annotations kept:  {summary['total_kept_annotations']:<6,}                       ║
║                                                          ║
║  Snow excluded:     {summary['snow_excluded']:<6}  (o6dwf only)          ║
║  B2 excluded:       {summary['b2_excluded']:<6}                           ║
║  MD5 dup dropped:   {summary['md5_duplicate_drops']:<6}                           ║
║                                                          ║
║  SAM panel masks:   {summary['needs_sam_panel_masks']:<6}  images to process     ║
║  SAM defect masks:  {summary['needs_sam_defect_masks']:<6}  images to process     ║
║                                                          ║
║  Split: train={summary['split_counts'].get('train',0)} / val={summary['split_counts'].get('val',0)} / test={summary['split_counts'].get('test',0)}        ║
║  Leakage: 0 groups                                       ║
╚══════════════════════════════════════════════════════════╝
""")
```

---

## 📋 Acceptance Criteria

- [ ] Drive mounted สำเร็จ
- [ ] Folder `cleaned_v3/` สร้างใน Drive
- [ ] `final_manifest.csv` — มี rows ตรงกับ total images (3,246)
- [ ] `final_annotations.csv` — มี rows ตรงกับ kept annotations (~26,705)
- [ ] `summary_v3.json` — เปิดอ่านได้ key ครบ
- [ ] Verify cell แสดง ✅ ทุกไฟล์
- [ ] ASCII box แสดงตัวเลขถูกต้อง

---

## ⚠️ Common Issues

| ปัญหา | วิธีแก้ |
|---|---|
| `manifest` หรือ `final_annotations` ไม่มีใน memory | Re-run `data_cleaning_v3.ipynb` ทั้งหมดก่อน แล้วค่อยรัน save cells |
| Drive path ไม่มี `ai builders/` folder | ตรวจชื่อ folder ใน Drive แล้วแก้ `DRIVE_OUTPUT` ให้ตรง |
| Permission error | ตรวจว่า mount Drive ด้วย account เดียวกับที่มี folder |
| `needs_sam_panel` column ไม่มี | เปลี่ยน key ใน summary เป็น `final_category.isin(['B1','C']).sum()` แทน |
