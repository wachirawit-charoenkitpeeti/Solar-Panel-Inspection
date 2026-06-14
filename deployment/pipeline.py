"""Two-stage raster-mask inference and per-panel severity computation."""

import base64
import io
import json
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from urgency import DEFECT_CLASSES as URGENCY_DEFECT_CLASSES
from urgency import RULE_VERSION, get_panel_urgency

DEFECT_CLASSES = URGENCY_DEFECT_CLASSES

PANEL_CLASS_ID = 0
DEFECT_CLASS_IDS = (1, 2, 3, 4)

CLASS_NAMES = {
    0: "panel",
    1: "dust",
    2: "bird_drop",
    3: "physical_damage",
    4: "leaf",
}

STAGE2_TO_PIPELINE = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
}

MASK_COLORS = {
    "panel": (27, 178, 210),
    "dust": (245, 166, 35),
    "bird_drop": (211, 78, 155),
    "physical_damage": (224, 67, 54),
    "leaf": (59, 166, 90),
}

MODEL_VERSION = "stage1-panel-v1__stage2-defect-v2-negatives"
PIPELINE_VERSION = "phase8-raster-v2"

PANEL_CONF = 0.25
DEFECT_CONF = 0.25
NMS_IOU = 0.70
IMAGE_SIZE = 640
MAX_DETECTIONS = 300
DEFECT_PANEL_OVERLAP = 0.50
MAX_IMAGE_PIXELS = 25_000_000

DEFAULT_MODEL_REPO = "wcahca/solar-panel-inspection"
DEFAULT_STAGE1_FILENAME = "stage1_panel.pt"
DEFAULT_STAGE2_FILENAME = "stage2_defect_v2.pt"


def _normalized_model_names(model):
    names = model.names
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    raise TypeError(f"Unsupported model names format: {type(names)}")


def _validate_model_classes(stage1, stage2):
    stage1_names = _normalized_model_names(stage1)
    stage2_names = _normalized_model_names(stage2)
    expected_stage1 = {0: "panel"}
    expected_stage2 = {
        0: "dust",
        1: "bird_drop",
        2: "physical_damage",
        3: "leaf",
    }
    if stage1_names != expected_stage1:
        raise ValueError(
            "Stage 1 checkpoint has unexpected classes: "
            f"{stage1_names}; expected {expected_stage1}."
        )
    if stage2_names != expected_stage2:
        raise ValueError(
            "Stage 2 checkpoint has unexpected classes: "
            f"{stage2_names}; expected {expected_stage2}."
        )


def mask_to_bbox(mask):
    """Return an exclusive-maximum pixel bounding box for a boolean mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    ]


def decode_mask_png(value):
    """Decode one base64 PNG mask from the Phase 6 cache."""
    raw = base64.b64decode(value.encode("ascii"))
    try:
        image = Image.open(io.BytesIO(raw)).convert("L")
    except Exception as exc:
        raise ValueError("Failed to decode cached PNG mask.") from exc
    return np.asarray(image) > 0


def cached_rows_to_instances(rows):
    """Convert serialized Phase 6 cache rows to inference instances."""
    instances = []
    for row in rows:
        mask = decode_mask_png(row["mask_png_base64"])
        class_id = int(row["class_id"])
        if class_id not in CLASS_NAMES:
            raise ValueError(f"Unknown cached class id: {class_id}")
        instances.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "score": float(row.get("score", 0.0)),
                "mask": mask,
                "area": int(mask.sum()),
                "bbox": mask_to_bbox(mask),
            }
        )
    return instances


def validate_image(image):
    """Normalize an uploaded image and reject excessive dimensions."""
    if image is None:
        raise ValueError("กรุณาอัปโหลดภาพก่อนเริ่มวิเคราะห์")
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("ภาพไม่มีขนาดที่ใช้งานได้")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("ภาพมีขนาดเกิน 25 megapixels")
    return image


def _resize_mask(mask, width, height):
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def _result_to_instances(result, original_shape, stage):
    height, width = original_shape
    if result.masks is None or result.boxes is None:
        return []

    raw_masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    instances = []

    for raw_mask, raw_class, score in zip(raw_masks, classes, scores):
        mask = raw_mask > 0.5
        if mask.shape != (height, width):
            mask = _resize_mask(mask, width, height)
        if not mask.any():
            continue

        if stage == 1:
            class_id = PANEL_CLASS_ID
        else:
            if int(raw_class) not in STAGE2_TO_PIPELINE:
                raise ValueError(f"Unknown Stage 2 class id: {raw_class}")
            class_id = STAGE2_TO_PIPELINE[int(raw_class)]

        instances.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "score": float(score),
                "mask": mask,
                "area": int(mask.sum()),
                "bbox": mask_to_bbox(mask),
            }
        )
    return instances


def spatial_split_defects(panels, defects):
    """Filter orphan defects and assign shared pixels to one panel."""
    if panels:
        panel_union = np.logical_or.reduce(
            [panel["mask"] for panel in panels]
        )
    elif defects:
        panel_union = np.zeros_like(defects[0]["mask"], dtype=bool)
    else:
        panel_union = None

    fragment_masks = [
        {
            class_id: np.zeros_like(panel["mask"], dtype=bool)
            for class_id in DEFECT_CLASS_IDS
        }
        for panel in panels
    ]
    assigned_instances = [set() for _ in panels]
    retained = []
    orphan = []

    for defect_index, defect in enumerate(defects):
        defect_area = int(defect["mask"].sum())
        overlap_union = (
            int(np.logical_and(defect["mask"], panel_union).sum())
            if panel_union is not None
            else 0
        )
        overlap_ratio = overlap_union / defect_area if defect_area else 0.0
        record = {
            **defect,
            "source_index": defect_index,
            "panel_overlap_ratio": float(overlap_ratio),
        }

        # Phase 6/7 keeps only defects with overlap strictly greater than 0.50.
        if overlap_ratio <= DEFECT_PANEL_OVERLAP:
            orphan.append(record)
            continue

        retained.append(record)
        overlaps = [
            np.logical_and(defect["mask"], panel["mask"])
            for panel in panels
        ]
        overlap_areas = [int(mask.sum()) for mask in overlaps]
        panel_order = sorted(
            range(len(panels)),
            key=lambda index: (-overlap_areas[index], index),
        )

        claimed = np.zeros_like(defect["mask"], dtype=bool)
        for panel_index in panel_order:
            fragment = np.logical_and(overlaps[panel_index], ~claimed)
            if not fragment.any():
                continue
            fragment_masks[panel_index][defect["class_id"]] |= fragment
            assigned_instances[panel_index].add(defect_index)
            claimed |= fragment

    return retained, orphan, fragment_masks, assigned_instances


def panel_severity_records(panels, fragment_masks, assigned_instances):
    """Calculate union severity and per-class severity for each panel."""
    rows = []
    for panel_index, panel in enumerate(panels):
        overall_union = np.zeros_like(panel["mask"], dtype=bool)
        class_areas = {}

        for class_id in DEFECT_CLASS_IDS:
            class_mask = fragment_masks[panel_index][class_id]
            overall_union |= class_mask
            class_areas[class_id] = int(class_mask.sum())

        panel_area = int(panel["mask"].sum())
        defect_area = int(overall_union.sum())
        severity = 100.0 * defect_area / panel_area if panel_area else 0.0
        class_severities = {
            CLASS_NAMES[class_id]: float(
                100.0 * class_areas[class_id] / panel_area
                if panel_area
                else 0.0
            )
            for class_id in DEFECT_CLASS_IDS
        }
        urgency = get_panel_urgency(class_severities)
        positive_classes = [
            class_id
            for class_id, area in class_areas.items()
            if area > 0
        ]
        dominant_id = (
            sorted(
                positive_classes,
                key=lambda class_id: (
                    -class_areas[class_id],
                    class_id,
                ),
            )[0]
            if positive_classes
            else None
        )

        if defect_area > panel_area:
            raise AssertionError("Defect union area exceeds panel area.")

        rows.append(
            {
                "panel_index": int(panel_index),
                "panel_area": panel_area,
                "defect_area": defect_area,
                "severity_pct": float(np.clip(severity, 0, 100)),
                "state": "defective" if defect_area > 0 else "clean",
                "n_assigned_defects": int(
                    len(assigned_instances[panel_index])
                ),
                "dominant_defect_class": (
                    CLASS_NAMES[dominant_id]
                    if dominant_id is not None
                    else "none"
                ),
                "per_class_severity_pct": class_severities,
                "per_class_actions": urgency["class_actions"],
                "recommended_action": urgency["panel_action"],
                "action_priority": urgency["panel_priority"],
                "trigger_classes": urgency["trigger_classes"],
                "bbox": [float(value) for value in panel["bbox"]],
                "panel_confidence": float(panel.get("score", 0.0)),
            }
        )
    return rows


def compute_reports_from_instances(panels, defects):
    """Run the shared Phase 6/7 raster post-processing."""
    retained, orphan, fragments, assigned = spatial_split_defects(
        panels,
        defects,
    )
    panel_reports = panel_severity_records(panels, fragments, assigned)
    return {
        "panel_reports": panel_reports,
        "retained_defects": retained,
        "orphan_defects": orphan,
        "fragment_masks": fragments,
        "assigned_instances": assigned,
    }


def _mask_boundary(mask):
    if not mask.any():
        return mask
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def draw_overlay(image_rgb, panels, details):
    """Draw panel and retained-defect masks on the original image."""
    canvas = image_rgb.astype(np.float32).copy()
    colored = canvas.copy()
    instances = panels + details["retained_defects"]

    for instance in instances:
        color = np.asarray(
            MASK_COLORS[instance["class_name"]],
            dtype=np.float32,
        )
        colored[instance["mask"]] = color

    changed = np.any(colored != image_rgb, axis=2, keepdims=True)
    canvas = np.where(
        changed,
        0.58 * canvas + 0.42 * colored,
        canvas,
    )
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    for instance in instances:
        boundary = _mask_boundary(instance["mask"])
        canvas[boundary] = MASK_COLORS[instance["class_name"]]

    annotated = Image.fromarray(canvas)
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for row in details["panel_reports"]:
        x1, y1, _, _ = row["bbox"]
        label = (
            f'Panel {row["panel_index"] + 1}: '
            f'{row["severity_pct"]:.1f}% | '
            f'{row["recommended_action"]}'
        )
        position = (max(0, int(x1)), max(0, int(y1)))
        text_box = draw.textbbox(position, label, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text(position, label, fill=(255, 255, 255), font=font)
    return annotated


def load_models_from_paths(stage1_path, stage2_path):
    """Load two local Ultralytics checkpoints."""
    from ultralytics import YOLO

    stage1 = YOLO(str(stage1_path))
    stage2 = YOLO(str(stage2_path))
    _validate_model_classes(stage1, stage2)
    return stage1, stage2


@lru_cache(maxsize=1)
def load_models_from_hub(
    repo_id=None,
    stage1_filename=DEFAULT_STAGE1_FILENAME,
    stage2_filename=DEFAULT_STAGE2_FILENAME,
):
    """Download public checkpoints once, then cache the loaded models."""
    from huggingface_hub import hf_hub_download

    repo_id = repo_id or os.getenv("MODEL_REPO", DEFAULT_MODEL_REPO)
    stage1_path = hf_hub_download(
        repo_id=repo_id,
        filename=stage1_filename,
    )
    stage2_path = hf_hub_download(
        repo_id=repo_id,
        filename=stage2_filename,
    )
    return load_models_from_paths(stage1_path, stage2_path)


def build_json_report(
    source_name,
    width,
    height,
    panels,
    details,
):
    """Build the JSON-compatible public report."""
    panel_reports = details["panel_reports"]
    highest_action = (
        max(
            panel_reports,
            key=lambda row: row["action_priority"],
        )["recommended_action"]
        if panel_reports
        else None
    )
    return {
        "source_name": source_name,
        "image": {
            "width": width,
            "height": height,
        },
        "status": "ok" if panels else "no_panel_detected",
        "model_version": MODEL_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "rule_version": RULE_VERSION,
        "thresholds": {
            "panel_conf": PANEL_CONF,
            "defect_conf": DEFECT_CONF,
            "nms_iou": NMS_IOU,
            "imgsz": IMAGE_SIZE,
            "defect_panel_overlap_strictly_greater_than": (
                DEFECT_PANEL_OVERLAP
            ),
        },
        "n_panels": len(panels),
        "n_retained_defects": len(details["retained_defects"]),
        "n_orphan_defects": len(details["orphan_defects"]),
        "highest_priority_recommended_action": highest_action,
        "panels": panel_reports,
        "limitations": {
            "bird_drop_mask_recall": 0.566,
            "urgent_action_coverage_approx": 0.74,
            "threshold_status": (
                "prototype heuristic, not a validated maintenance policy"
            ),
            "bird_drop_warning": (
                "No bird-drop prediction does not prove that bird drop "
                "is absent."
            ),
        },
        "disclaimer": (
            "Actions are recommendations. Confirm findings with a "
            "qualified operator before maintenance."
        ),
    }


def run_pipeline(
    image,
    stage1,
    stage2,
    source_name="uploaded_image",
):
    """Run both models and return the overlay plus structured report."""
    image = validate_image(image)
    image_rgb = np.asarray(image)
    height, width = image_rgb.shape[:2]

    stage1_result = stage1.predict(
        image,
        conf=PANEL_CONF,
        iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        max_det=MAX_DETECTIONS,
        retina_masks=True,
        verbose=False,
        device="cpu",
    )[0]
    stage2_result = stage2.predict(
        image,
        conf=DEFECT_CONF,
        iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        max_det=MAX_DETECTIONS,
        retina_masks=True,
        verbose=False,
        device="cpu",
    )[0]

    panels = _result_to_instances(
        stage1_result,
        (height, width),
        stage=1,
    )
    defects = _result_to_instances(
        stage2_result,
        (height, width),
        stage=2,
    )
    details = compute_reports_from_instances(panels, defects)
    annotated = draw_overlay(image_rgb, panels, details)
    report = build_json_report(
        source_name,
        width,
        height,
        panels,
        details,
    )
    return {
        "annotated_image": annotated,
        "panel_reports": details["panel_reports"],
        "report": report,
        "report_json": json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        "panels": panels,
        "defects": defects,
        "details": details,
    }
