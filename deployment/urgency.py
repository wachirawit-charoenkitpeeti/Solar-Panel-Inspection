"""Prototype maintenance urgency rules from the evaluated Phase 7 policy."""

from math import isfinite

DEFECT_CLASSES = ("dust", "bird_drop", "physical_damage", "leaf")

ACTION_PRIORITY = {
    "no_action": 0,
    "schedule_cleaning": 1,
    "cleaning_soon": 2,
    "urgent_cleaning": 3,
    "remove_immediately": 4,
    "urgent_inspection": 5,
}

ACTION_LABEL_TH = {
    "no_action": "ไม่ต้องดำเนินการ",
    "schedule_cleaning": "วางแผนทำความสะอาด",
    "cleaning_soon": "ทำความสะอาดเร็ว ๆ นี้",
    "urgent_cleaning": "ทำความสะอาดด่วน",
    "remove_immediately": "นำใบไม้ออกทันที",
    "urgent_inspection": "ตรวจสอบด่วน (ความเสียหาย)",
}

RULE_VERSION = "phase7-prototype-v1"


def validate_severity(value):
    """Return a finite severity percentage within the supported range."""
    if value is None or not isfinite(float(value)):
        raise ValueError(f"Severity must be finite, got {value!r}.")
    severity = float(value)
    if severity < 0 or severity > 100:
        raise ValueError(f"Severity outside [0, 100]: {severity}")
    return severity


def get_class_action(defect_class, severity_pct):
    """Map one defect class and severity percentage to an action."""
    severity = validate_severity(severity_pct)

    if defect_class == "dust":
        if severity < 5:
            return "no_action"
        if severity < 15:
            return "schedule_cleaning"
        if severity < 25:
            return "cleaning_soon"
        return "urgent_cleaning"

    if defect_class == "bird_drop":
        if severity == 0:
            return "no_action"
        return "urgent_cleaning" if severity > 2 else "schedule_cleaning"

    if defect_class == "physical_damage":
        return "urgent_inspection" if severity > 0 else "no_action"

    if defect_class == "leaf":
        return "remove_immediately" if severity > 0 else "no_action"

    raise KeyError(f"Unknown defect class: {defect_class}")


def get_panel_urgency(panel_severities):
    """Use the highest class priority as the action for one panel."""
    normalized = {
        name: validate_severity(panel_severities.get(name, 0.0))
        for name in DEFECT_CLASSES
    }
    class_actions = {
        name: get_class_action(name, severity)
        for name, severity in normalized.items()
    }
    panel_action = max(
        class_actions.values(),
        key=lambda action: ACTION_PRIORITY[action],
    )
    top_priority = ACTION_PRIORITY[panel_action]
    trigger_classes = sorted(
        name
        for name, action in class_actions.items()
        if action != "no_action"
        and ACTION_PRIORITY[action] == top_priority
    )
    return {
        "panel_action": panel_action,
        "panel_priority": top_priority,
        "class_actions": class_actions,
        "trigger_classes": trigger_classes,
    }


def run_boundary_tests():
    """Run the locked Phase 7 boundary cases."""
    cases = [
        ("dust", 0, "no_action"),
        ("dust", 4.999999, "no_action"),
        ("dust", 5, "schedule_cleaning"),
        ("dust", 14.999999, "schedule_cleaning"),
        ("dust", 15, "cleaning_soon"),
        ("dust", 24.999999, "cleaning_soon"),
        ("dust", 25, "urgent_cleaning"),
        ("dust", 100, "urgent_cleaning"),
        ("bird_drop", 0, "no_action"),
        ("bird_drop", 1e-9, "schedule_cleaning"),
        ("bird_drop", 2.0, "schedule_cleaning"),
        ("bird_drop", 2.000001, "urgent_cleaning"),
        ("physical_damage", 0, "no_action"),
        ("physical_damage", 1e-9, "urgent_inspection"),
        ("leaf", 0, "no_action"),
        ("leaf", 1e-9, "remove_immediately"),
    ]
    for defect_class, severity, expected in cases:
        actual = get_class_action(defect_class, severity)
        assert actual == expected, (
            defect_class,
            severity,
            expected,
            actual,
        )

    invalid_values = [float("nan"), float("inf"), -0.0001, 100.0001]
    for invalid in invalid_values:
        try:
            get_class_action("dust", invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid severity accepted: {invalid}")

    try:
        get_class_action("unknown", 1)
    except KeyError:
        pass
    else:
        raise AssertionError("Unknown defect class accepted.")

    return len(cases) + len(invalid_values) + 1
