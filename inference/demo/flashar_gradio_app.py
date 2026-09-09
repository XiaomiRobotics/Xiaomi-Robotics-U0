from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xr_u0_flashar.demo_runtime import (  # noqa: E402
    SUPPORTED_TASKS,
    TASK_SCENE,
    TASK_T2I,
    TASK_TRANSFER,
    TASK_X2I,
    pil_image_to_base64,
    task_defaults,
)
from xr_u0_flashar.demo_examples import demo_example_for, example_choices  # noqa: E402


ROBOT_ARM_TYPES = ["AgiBot G1", "Agibot G2", "Agilex Piper", "ARX Robot"]
IMAGE_STYLES = ["Real", "Simulator"]
TRANSFER_DEPTH_INPUT = "Depth map"
TRANSFER_RGB_INPUT = "RGB image"


def image_from_base64(value: str) -> Image.Image:
    data = base64.b64decode(value)
    return Image.open(io.BytesIO(data)).convert("RGB")


def api_post_generate(api_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{api_url.rstrip('/')}/api/generate",
        json=payload,
        timeout=3600,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise gr.Error(f"Generation failed: {detail}")
    return response.json()


def build_ui(api_url: str) -> gr.Blocks:
    initial_example = demo_example_for(TASK_T2I)

    with gr.Blocks(title="Xiaomi-Robotics-U0 FlashAR Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Xiaomi-Robotics-U0 FlashAR Demo")

        with gr.Row():
            with gr.Column(scale=1):
                task_input = gr.Radio(
                    choices=list(SUPPORTED_TASKS),
                    value=TASK_T2I,
                    label="Task",
                )
                example_input = gr.Dropdown(
                    choices=example_choices(TASK_T2I),
                    value=initial_example.id,
                    label="Example",
                    allow_custom_value=False,
                )
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value=initial_example.prompt,
                    lines=7,
                    placeholder="A clean product photograph of a glass teapot beside green tea leaves.",
                )
                reference_image = gr.Image(
                    label="Reference image",
                    type="pil",
                    value=initial_example.reference_path,
                    visible=False,
                )
                transfer_input_type = gr.Radio(
                    choices=[TRANSFER_DEPTH_INPUT, TRANSFER_RGB_INPUT],
                    value=TRANSFER_RGB_INPUT if initial_example.input_image_type == "rgb" else TRANSFER_DEPTH_INPUT,
                    label="Transfer reference type",
                    visible=False,
                )
                robot_arm_type = gr.Dropdown(
                    choices=ROBOT_ARM_TYPES,
                    value=initial_example.robot_arm_type,
                    label="Robot arm type",
                    visible=False,
                )
                image_style = gr.Dropdown(
                    choices=IMAGE_STYLES,
                    value=initial_example.image_style,
                    label="Image style",
                    visible=False,
                )

                with gr.Accordion("Advanced", open=False):
                    cfg_scale = gr.Number(label="CFG scale", value=initial_example.cfg_scale, precision=2)
                    height_tokens = gr.Number(
                        label="Height tokens",
                        value=initial_example.height,
                        precision=0,
                        visible=True,
                    )
                    width_tokens = gr.Number(
                        label="Width tokens",
                        value=initial_example.width,
                        precision=0,
                        visible=True,
                    )
                    seed_input = gr.Number(label="Seed", value=42, precision=0)
                    temperature = gr.Number(label="Temperature", value=initial_example.temperature, precision=2)
                    top_k = gr.Number(label="Top-k", value=initial_example.top_k, precision=0)
                    top_p = gr.Number(label="Top-p", value=initial_example.top_p, precision=2)
                    source_image_area = gr.Number(
                        label="Source image area",
                        value=initial_example.source_image_area,
                        precision=0,
                        visible=False,
                    )
                    da3_model_path = gr.Textbox(
                        label="DA3 model path",
                        value="",
                        placeholder="Leave blank to use the API server default or XR_U0_DA3_MODEL_PATH",
                        visible=False,
                    )
                    da3_device = gr.Textbox(label="DA3 device", value=initial_example.da3_device, visible=False)
                    da3_process_res = gr.Number(
                        label="DA3 process resolution",
                        value=initial_example.da3_process_res,
                        precision=0,
                        visible=False,
                    )

                submit = gr.Button("Generate", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                result_image = gr.Image(label="Generated image", type="pil", height=620)
                output_path = gr.Textbox(label="Output path", interactive=False)
                audit_json = gr.JSON(label="Audit")

        def prompt_placeholder(task_type: str) -> str:
            return {
                TASK_T2I: "A clean product photograph of a glass teapot beside green tea leaves.",
                TASK_X2I: "Keep the same scene and change the main object color to matte blue.",
                TASK_SCENE: "Workspace: A dark slate table. Objects: sushi ingredients, chopsticks, and a soy sauce dish. Lighting: focused overhead light. Background: a minimalist kitchen.",
                TASK_TRANSFER: "Workspace: A blue felt-covered table. Task objects: a yellow lemon held by the left robot arm. Background: a dark wooden cabinet.",
            }[task_type]

        def example_updates(task_type: str, example_id: str | None = None):
            example = demo_example_for(task_type, example_id)
            task = example.task_type
            needs_reference = task in (TASK_X2I, TASK_TRANSFER)
            is_scene = task == TASK_SCENE
            is_transfer = task == TASK_TRANSFER
            is_rgb_transfer = is_transfer and example.input_image_type == "rgb"
            uses_reference_shape = task in (TASK_X2I, TASK_TRANSFER)
            uses_explicit_shape = task in (TASK_T2I, TASK_SCENE)
            return (
                gr.update(value=example.prompt, placeholder=prompt_placeholder(task)),
                gr.update(value=example.reference_path, visible=needs_reference),
                gr.update(
                    visible=is_transfer,
                    value=TRANSFER_RGB_INPUT if is_rgb_transfer else TRANSFER_DEPTH_INPUT,
                ),
                gr.update(visible=is_scene, value=example.robot_arm_type),
                gr.update(visible=is_scene or is_transfer, value=example.image_style),
                gr.update(value=example.cfg_scale),
                gr.update(value=example.height, visible=uses_explicit_shape),
                gr.update(value=example.width, visible=uses_explicit_shape),
                gr.update(value=example.temperature),
                gr.update(value=example.top_k),
                gr.update(value=example.top_p),
                gr.update(value=example.source_image_area, visible=uses_reference_shape),
                gr.update(visible=is_rgb_transfer, value=""),
                gr.update(visible=is_rgb_transfer, value=example.da3_device),
                gr.update(visible=is_rgb_transfer, value=example.da3_process_res),
            )

        example_outputs = [
            prompt_input,
            reference_image,
            transfer_input_type,
            robot_arm_type,
            image_style,
            cfg_scale,
            height_tokens,
            width_tokens,
            temperature,
            top_k,
            top_p,
            source_image_area,
            da3_model_path,
            da3_device,
            da3_process_res,
        ]

        def on_task_change(task_type: str):
            choices = example_choices(task_type)
            example = demo_example_for(task_type)
            return (
                gr.update(choices=choices, value=example.id),
                *example_updates(task_type, example.id),
            )

        task_input.change(
            fn=on_task_change,
            inputs=[task_input],
            outputs=[example_input, *example_outputs],
        )

        def on_example_change(task_type: str, example_id: str):
            return example_updates(task_type, example_id)

        example_input.change(
            fn=on_example_change,
            inputs=[task_input, example_input],
            outputs=example_outputs,
        )

        def on_transfer_input_change(image_type: str):
            use_rgb = image_type == TRANSFER_RGB_INPUT
            return (
                gr.update(visible=use_rgb, value=""),
                gr.update(visible=use_rgb),
                gr.update(visible=use_rgb),
            )

        transfer_input_type.change(
            fn=on_transfer_input_change,
            inputs=[transfer_input_type],
            outputs=[
                da3_model_path,
                da3_device,
                da3_process_res,
            ],
        )

        def on_submit(
            task_type: str,
            prompt: str,
            ref_image: Image.Image | None,
            reference_mode: str,
            arm_type: str,
            style: str,
            cfg: float,
            height: float,
            width: float,
            seed: float,
            temp: float,
            k: float,
            p: float,
            image_area: float,
            da3_path: str,
            da3_dev: str,
            da3_res: float,
        ):
            task = task_defaults(task_type)["task_type"]
            if not prompt or not prompt.strip():
                raise gr.Error("Prompt is required")
            if task in (TASK_X2I, TASK_TRANSFER) and ref_image is None:
                raise gr.Error(f"{task} requires a reference image")
            input_image_type = "rgb" if task == TASK_TRANSFER and reference_mode == TRANSFER_RGB_INPUT else "depth"
            uses_reference_shape = task in (TASK_X2I, TASK_TRANSFER)

            height_value = None if uses_reference_shape else int(height) if height and int(height) > 0 else None
            width_value = None if uses_reference_shape else int(width) if width and int(width) > 0 else None
            references = []
            if ref_image is not None:
                references.append(pil_image_to_base64(ref_image.convert("RGB")))

            payload = {
                "task_type": task,
                "prompt": prompt,
                "reference_images": references,
                "height": height_value,
                "width": width_value,
                "cfg_scale": float(cfg),
                "temperature": float(temp),
                "top_k": int(k),
                "top_p": float(p),
                "seed": int(seed) if seed is not None else None,
                "source_image_area": int(image_area),
                "robot_arm_type": arm_type,
                "image_style": style,
                "input_image_type": input_image_type,
                "da3_model_path": str(da3_path).strip() if da3_path else None,
                "da3_device": str(da3_dev).strip() if da3_dev else "cuda:0",
                "da3_process_res": int(da3_res),
            }
            result = api_post_generate(api_url, payload)
            image = image_from_base64(result["image_base64"])
            audit = result.get("audit", {})
            status_text = (
                f"{result['task_type']} done in {result['total_seconds']:.2f}s "
                f"(generation {result['generation_seconds']:.2f}s)"
            )
            if not audit.get("visual_token_complete", False):
                status_text += " | incomplete visual tokens"
            return image, result["output_path"], audit, status_text

        submit.click(
            fn=on_submit,
            inputs=[
                task_input,
                prompt_input,
                reference_image,
                transfer_input_type,
                robot_arm_type,
                image_style,
                cfg_scale,
                height_tokens,
                width_tokens,
                seed_input,
                temperature,
                top_k,
                top_p,
                source_image_area,
                da3_model_path,
                da3_device,
                da3_process_res,
            ],
            outputs=[result_image, output_path, audit_json, status],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Xiaomi-Robotics-U0 FlashAR Gradio app.")
    parser.add_argument("--api-url", default=os.environ.get("XR_U0_FLASHAR_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_ui(args.api_url)
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
