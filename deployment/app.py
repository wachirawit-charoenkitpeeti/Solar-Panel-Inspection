"""Gradio interface for the two-stage solar-panel inspection pipeline."""

from pathlib import Path

import gradio as gr

from pipeline import load_models_from_hub, run_pipeline
from urgency import ACTION_LABEL_TH

APP_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = APP_DIR / "samples"

TABLE_HEADERS = [
    "แผง",
    "สถานะ",
    "Severity",
    "Defect หลัก",
    "คำแนะนำ",
]


def _panel_table_markdown(rows):
    """Render every detected panel without a fixed-height table viewport."""
    header = "| " + " | ".join(TABLE_HEADERS) + " |"
    separator = "| " + " | ".join(["---"] * len(TABLE_HEADERS)) + " |"
    if not rows:
        return "\n".join(
            [
                "### ผลต่อแผง",
                "",
                header,
                separator,
                "| - | ไม่พบแผง | - | - | - |",
            ]
        )
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join(["### ผลต่อแผง", "", header, separator, *body])


def available_samples():
    """Return only owner-provided images that currently exist."""
    supported = {".jpg", ".jpeg", ".png"}
    return [
        str(path)
        for path in sorted(SAMPLES_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in supported
    ]


def _panel_table(panel_reports):
    state_labels = {
        "clean": "ไม่พบ defect",
        "defective": "พบ defect",
    }
    return [
        [
            row["panel_index"] + 1,
            state_labels[row["state"]],
            f'{row["severity_pct"]:.1f}%',
            row["dominant_defect_class"],
            ACTION_LABEL_TH[row["recommended_action"]],
        ]
        for row in panel_reports
    ]


def analyze(image):
    """Run inference after the user explicitly presses the analyze button."""
    if image is None:
        message = "กรุณาอัปโหลดภาพก่อนเริ่มวิเคราะห์"
        gr.Warning(message)
        return None, _panel_table_markdown([])

    try:
        stage1, stage2 = load_models_from_hub()
        result = run_pipeline(
            image,
            stage1,
            stage2,
            source_name="uploaded_image",
        )
    except Exception as exc:
        raise gr.Error(
            f"วิเคราะห์ภาพไม่สำเร็จ: {type(exc).__name__}: {exc}"
        ) from exc

    table = _panel_table(result["panel_reports"])
    return result["annotated_image"], _panel_table_markdown(table)


def build_demo():
    """Create the public Gradio interface without loading either model."""
    with gr.Blocks(title="Solar Panel Inspection") as demo:
        gr.Markdown(
            """
            # Solar Panel Inspection

            อัปโหลดภาพแผงโซลาร์เพื่อหาแผงและ defect 4 กลุ่ม:
            **dust** (<span style="color:#F5A623">สีเหลือง</span>),
            **bird_drop** (<span style="color:#D34E9B">สีชมพู</span>),
            **physical_damage** (<span style="color:#E04336">สีแดง</span>)
            และ **leaf** (<span style="color:#3BA65A">สีเขียว</span>)

            ระบบคำนวณพื้นที่ defect ต่อแผงและแนะนำการดูแลเบื้องต้น
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    type="pil",
                    label="ภาพแผงโซลาร์",
                )
                analyze_button = gr.Button(
                    "วิเคราะห์ภาพ",
                    variant="primary",
                )

                samples = available_samples()
                if samples:
                    gr.Examples(
                        examples=[[path] for path in samples],
                        inputs=input_image,
                        label="ภาพตัวอย่าง",
                    )
                else:
                    gr.Markdown(
                        "_ยังไม่มีภาพตัวอย่าง "
                        "เพิ่มภาพที่คุณเป็นเจ้าของใน `samples/`_"
                    )

            with gr.Column():
                output_image = gr.Image(
                    label="ผลการตรวจ panel และ defect",
                )
                output_table = gr.Markdown(
                    value=_panel_table_markdown([]),
                )

        analyze_button.click(
            analyze,
            inputs=input_image,
            outputs=[
                output_image,
                output_table,
            ],
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
