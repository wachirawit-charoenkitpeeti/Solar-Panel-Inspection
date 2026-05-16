# 📋 Engineering Spec — Roboflow Dataset Verification

**Project:** Solar Panel Inspection AI  
**Phase:** Data Verification & EDA  
**Assigned to:** Codex (engineer)  
**Tech Lead:** Claude  
**Priority:** P0 (blocker for next phase)

---

## 🎯 Goal

Verify 3 Roboflow datasets และ generate report ใน format เดียวกับ IA-Cobotics report
เพื่อให้ project owner ตัดสินใจ:
1. Pipeline B compatibility (polygon mask vs bbox)
2. Class mapping plan
3. Total data volume after merge

---

## 📦 Datasets to Verify

### Dataset 1: bird_dust_leaf
- **URL:** https://universe.roboflow.com/panneauxphotovoltaique/bird_dust_leaf
- **Workspace:** `panneauxphotovoltaique`
- **Project:** `bird_dust_leaf`
- **Expected classes:** bird, dust, leaf (key value: มี leaf!)
- **Expected format:** likely YOLO bbox (need to verify)

### Dataset 2: solar-panel-o6dwf  
- **URL:** https://universe.roboflow.com/solar-panel-damage-detectionsegmentation/solar-panel-o6dwf
- **Workspace:** `solar-panel-damage-detectionsegmentation`
- **Project:** `solar-panel-o6dwf`
- **Expected classes:** Physical-Damage + others (need to verify)
- **Expected format:** Instance Segmentation (polygon) ✅

### Dataset 3: solar-panel-b1cmz
- **URL:** https://universe.roboflow.com/dhanashree-meshram-rcbmv/solar-panel-b1cmz
- **Workspace:** `dhanashree-meshram-rcbmv`
- **Project:** `solar-panel-b1cmz`
- **Expected classes:** Physical-Damage + others (need to verify)
- **Expected format:** Instance Segmentation (polygon) ✅

---

## 🛠️ Technical Requirements

### Environment
- Google Colab (preferred)
- Mount Google Drive ถ้าเก็บ data ที่นั่น
- Use Roboflow Python SDK

### Dependencies
```bash
pip install roboflow ultralytics matplotlib pillow pyyaml
```

### Download format
- Dataset 1: try `yolov8` first, fallback to `yolov5`
- Dataset 2: use `yolov8` for segmentation (polygons)
- Dataset 3: use `yolov8` for segmentation (polygons)

⚠️ **Important:** ใช้ format ที่ preserve segmentation polygons ถ้ามี
- ถ้า dataset เป็น instance seg ให้ download เป็น YOLO segmentation format
- จะได้ label file ที่มี polygon coordinates (>5 columns per line)

---

## 📊 Required Output (ต่อ dataset)

### 1. Folder structure tree (max 3 levels)

### 2. data.yaml content (full dump)

### 3. Counts table
```
Split   | Images | Labels | Annotations | Class breakdown
--------|--------|--------|-------------|----------------
train   | ?      | ?      | ?           | class_a=?, class_b=?
valid   | ?      | ?      | ?           | ...
test    | ?      | ?      | ?           | ...
TOTAL   | ?      | ?      | ?           | ...
```

### 4. Format detection
Per dataset, report:
- `bbox` (5 columns per line) — YOLO detection
- `polygon` (>5 columns per line) — YOLO segmentation
- `mixed` — both formats present
- → **Critical for Pipeline B decision**

### 5. Visualizations (matplotlib)

**5a. Random samples (9 images)** จาก train set พร้อม:
- bbox (rectangle) ถ้า bbox format
- polygon (filled translucent + outline) ถ้า segmentation format
- color-coded ตาม class
- class name label บน annotation

**5b. Per-class samples** (6 images per class)
- เลือกภาพที่มีแต่ class นั้น
- หรือถ้าไม่มี → ภาพที่มี class นั้น dominant

**5c. Class distribution bar chart**
- Per-split breakdown
- Total breakdown

**5d. Annotation size distribution**
- For bbox: width × height histogram
- For polygon: area (in pixels²) histogram

### 6. Summary report (ASCII box format)
```
╔══════════════════════════════════════════════════════════════╗
║  [Dataset name] — Verification Report                        ║
╠══════════════════════════════════════════════════════════════╣
║  Source:    [workspace/project]                              ║
║  License:   [from Roboflow]                                  ║
║  Format:    [bbox / polygon / mixed]                         ║
║                                                              ║
║  📊 Dataset Composition:                                     ║
║     Total images:       ?                                    ║
║     Total annotations:  ?                                    ║
║                                                              ║
║  🏷️  Classes:                                                ║
║     [class_name] (id=?):    ? annotations                    ║
║     ...                                                      ║
║                                                              ║
║  📐 Image format:                                            ║
║     Resolution: ?×?                                          ║
║                                                              ║
║  🎯 For Pipeline B (Instance Segmentation):                  ║
║     ✅/⚠️ Format: [bbox/polygon]                              ║
║     → [need conversion? / ready to use?]                     ║
║                                                              ║
║  ✅ Quality:                                                 ║
║     - Class balance: ?                                       ║
║     - Annotation quality: [visual inspection result]         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 🔄 Cross-Dataset Comparison (final cell)

หลังจาก verify ครบ 3 datasets + IA-Cobotics ที่ verify แล้ว ให้สร้าง **comparison table**:

| Metric | IA-Cobotics | Dataset 1 | Dataset 2 | Dataset 3 |
|---|---|---|---|---|
| Total images | 302 | ? | ? | ? |
| Format | bbox | ? | ? | ? |
| Pipeline B ready | ⚠️ | ? | ? | ? |
| Class: dust | 473 | ? | - | - |
| Class: bird | 637 | ? | - | - |
| Class: leaf | - | ? | - | - |
| Class: physical_damage | - | - | ? | ? |
| License | CC BY 4.0 | ? | ? | ? |

---

## 🗺️ Class Mapping Plan

หลัง audit ให้เติม mapping table:

```python
CLASS_MAPPING = {
    'IA-Cobotics': {
        'Dust': 'dust',
        'bird-droppings': 'bird_drop',
    },
    'bird_dust_leaf': {
        # ←  fill from actual class names
    },
    'solar-panel-o6dwf': {
        # ←  fill from actual class names
    },
    'solar-panel-b1cmz': {
        # ←  fill from actual class names
    },
}
```

Target unified taxonomy: `clean, dust, bird_drop, physical_damage, leaf`

Flag any source class ที่ map ไม่ได้ตรงๆ (e.g., "snow" — ตัดสินใจว่าจะ map ไป "other" หรือ skip)

---

## 📋 Acceptance Criteria

Notebook ผ่าน QA เมื่อ:

- [ ] 1. Download ทั้ง 3 datasets สำเร็จ (หรือ document ถ้า fail พร้อม reason)
- [ ] 2. Folder structure + data.yaml ของแต่ละ dataset แสดงชัด
- [ ] 3. Counts table ครบทั้ง 3 datasets
- [ ] 4. Format detection ถูกต้อง (verify ด้วยการเปิด label file จริง)
- [ ] 5. Visualization 5a, 5b, 5c, 5d ครบทุก dataset
- [ ] 6. Summary report 4 reports (3 datasets + comparison)
- [ ] 7. Class mapping plan ตามจริง (ไม่ใช่ placeholder)
- [ ] 8. Cross-dataset comparison table
- [ ] 9. Code มี error handling (Roboflow API fail, missing files)
- [ ] 10. Cell ทุกตัวรันได้ independent (สามารถ re-run cell ใด cell หนึ่ง)

---

## 🚨 Risks & Mitigation

| Risk | Mitigation |
|---|---|
| Roboflow API key invalid | Add try/except + clear error message |
| Dataset is private | Document และ skip — แจ้ง project owner |
| Format unexpected | Log raw label file content for inspection |
| Polygons vs bbox confusion | Implement `detect_format()` function ตามที่ใช้ใน IA-Cobotics notebook |
| Memory issue (large datasets) | Sample inspection (first 100 images) instead of all |
| Path with spaces | Use `Path` objects + quote strings in shell commands |

---

## 📤 Deliverables

1. **Notebook file:** `verify_3_roboflow_datasets.ipynb`
2. **Output cells:** All cells executed with results visible
3. **Summary section** ที่บอก project owner ว่า:
   - Total data after merge = ? images
   - Class taxonomy coverage = ?/5 classes
   - Pipeline B ready = ?/4 datasets
   - Recommended next step
