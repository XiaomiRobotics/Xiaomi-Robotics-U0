from __future__ import annotations

from typing import Any

from .common import ar_image_sampling, flashar_sampling, override_or_examples


TRANSFER_SYSTEM_PROMPT = (
    "You are an advanced controllable image generation model. You will be provided with "
    "a scene description and a grid of monocular depth maps from multiple viewpoints "
    "(where pixel intensity represents distance from the camera - brighter means closer, "
    "darker means farther). The depth maps are arranged in a grid layout. Your objective "
    "is to generate a corresponding grid of photorealistic images that strictly respect "
    "the 3D geometry defined by each depth map. The output canvas is one horizontally "
    "concatenated multi-view image. You should keep the consistency across different "
    "views in the grid."
)

EXAMPLES: list[dict[str, str]] = [
    {
        "id": "transfer_towel",
        "task_type": "Transfer",
        "reference_image": "examples/assets/transfer/towel_depth_triptych.png",
        "visual_placeholder": "<|VIS_PLH|>",
        "supervised_start": "<|extra_100|>",
        "supervised_end": "<|extra_101|>",
        "text_prompt": (
            TRANSFER_SYSTEM_PROMPT
            + " Image Style: Real. Scene Description: Workspace: Black granite slab with polished, "
            "reflective surface and thin silver metal edge visible on the left side.\n"
            "Task objects: Beige towel with soft, looped texture and brown border, partially folded, "
            "located at the center of the workspace. Black robotic arm with metallic joints and matte "
            "plastic housing, gripping the towel near the right edge of the workspace.\n"
            "Irrelevant objects: Dark brown woven basket with matte texture and visible wicker pattern, "
            "positioned near the left edge of the workspace.\n"
            "Lighting: Even overhead lighting casting soft shadows, creating glossy reflections on the "
            "floor tiles and subtle highlights on the robot arm's metallic surfaces.\n"
            "Background: Distant white tiled floor with glossy finish and visible grout lines, reflecting "
            "overhead lighting. <|VIS_PLH|><|extra_100|><|VIS_PLH|><|extra_101|>"
        ),
    },
    {
        "id": "transfer_sort_fruit",
        "task_type": "Transfer",
        "reference_image": "examples/assets/transfer/sort_fruit_depth_triptych.png",
        "visual_placeholder": "<|VIS_PLH|>",
        "supervised_start": "<|extra_100|>",
        "supervised_end": "<|extra_101|>",
        "text_prompt": (
            TRANSFER_SYSTEM_PROMPT
            + " Image Style: Simulator. Scene Description: Workspace: A white rectangular table "
            "with a smooth matte surface and rounded corners.\n"
            "Task objects: A white marble sphere with a polished stone surface and spherical shape "
            "held by the robot arm gripper at the right edge of the workspace. A white metallic "
            "robot arm with a smooth finish and articulated joints extending from the right side "
            "of the workspace. The gripper has black matte plastic fingers with white circular "
            "pads and is gripping the marble sphere.\n"
            "Irrelevant objects: A matte green plastic rectangular container with smooth surfaces "
            "on the left side of the workspace. A matte pink plastic rectangular container with "
            "smooth surfaces near the center-left of the workspace. An orange fruit with a glossy "
            "peel and spherical shape inside the green container. A red pomegranate with a glossy "
            "surface and spherical shape inside the pink container.\n"
            "Lighting: Bright overhead lighting casting soft shadows beneath the robot arm and "
            "containers on the workspace.\n"
            "Background: A beige wall with a smooth matte finish behind the workspace. A light "
            "gray tiled floor with visible grout lines below the workspace. A wooden cabinet with "
            "a light brown finish and open shelves is visible in the far background. "
            "<|VIS_PLH|><|extra_100|><|VIS_PLH|><|extra_101|>"
        ),
    },
    {
        "id": "transfer_lemon_depth",
        "task_type": "Transfer",
        "reference_image": "examples/assets/transfer/legacy_depth_triptych.png",
        "visual_placeholder": "<|VIS_PLH|>",
        "supervised_start": "<|extra_100|>",
        "supervised_end": "<|extra_101|>",
        "text_prompt": (
            TRANSFER_SYSTEM_PROMPT
            + " Image Style: Simulator. Scene Description: Workspace: A blue felt-covered table "
            "with a soft fabric surface and slightly rounded edges.\n"
            "Task objects: A yellow plastic lemon with a matte surface and oval shape held by "
            "the left robot arm at the center-left of the workspace. A white and black metallic "
            "robot arm with a smooth finish and articulated joints on the left side of the "
            "workspace, gripping the lemon. A second white and black metallic robot arm with a "
            "smooth finish and articulated joints on the right side of the workspace, positioned "
            "above the workspace surface.\n"
            "Irrelevant objects: A black plastic bottle with a glossy surface and white label "
            "near the back-left edge of the workspace. A green plastic apple with a matte finish "
            "near the front-right edge of the workspace. A yellow-green sponge with a rough "
            "texture and rectangular shape near the center of the workspace. A white ceramic cup "
            "with a glossy surface and printed design near the right edge of the workspace.\n"
            "Lighting: Bright overhead lighting casting soft shadows beneath objects on the "
            "workspace.\n"
            "Background: A dark brown wooden cabinet with multiple shelves containing colorful "
            "items in the far background. A smooth gray concrete floor beneath the workspace. "
            "<|VIS_PLH|><|extra_100|><|VIS_PLH|><|extra_101|>"
        ),
    },
]


def build_prompts(
    *,
    num_samples: int | None = None,
    prompt: str | None = None,
    reference_images: list[str] | None = None,
    legacy_video_jsonl: str | None = None,
) -> dict[str, Any]:
    del legacy_video_jsonl
    return override_or_examples(
        prefix="transfer",
        examples=EXAMPLES,
        num_samples=num_samples,
        prompt=prompt,
        reference_images=reference_images,
        prompt_field="text_prompt",
    )


def task_config(
    engine: str,
    *,
    num_samples: int | None = None,
    prompt: str | None = None,
    reference_images: list[str] | None = None,
    legacy_video_jsonl: str | None = None,
) -> dict[str, Any]:
    prompts = build_prompts(
        num_samples=num_samples,
        prompt=prompt,
        reference_images=reference_images,
        legacy_video_jsonl=legacy_video_jsonl,
    )
    config: dict[str, Any] = {
        "task_type": "Transfer",
        "input_image_type": "depth",
        "da3_model_path": "depth-anything/DA3-LARGE-1.1",
        "da3_device": "cuda:0",
        "da3_min_depth": 0.7,
        "da3_max_depth": 2.0,
        "da3_process_res": 504,
        "template": "{text}",
        "unc_prompt": "<|extra_203|><|extra_100|>",
        "classifier_free_guidance": 3.0,
        "prompts": prompts,
    }
    if engine == "ar":
        config.update(
            {
                "force_same_image_size": False,
                "stop_on_image_end": True,
                "image_area": 1024 * 1024,
                "sampling_params": ar_image_sampling(32768),
            }
        )
    else:
        config.update({"source_image_area": 1024 * 1024, "match_reference_shape": True, **flashar_sampling()})
    return config
