from __future__ import annotations

from typing import Any

from .common import ar_image_sampling, flashar_sampling, override_or_examples


SCENE_SYSTEM_PROMPT = (
    "You are an advanced controllable image generation model specialized in synthesizing "
    "high-quality robotic vision data. You will receive a scene description of the robot "
    "workspace. Your task is to generate the high-quality initial observation containing "
    "3 views with one full size view on top and two half size views side by side at the "
    "bottom. You should keep the consistency across different views. Image Style: Real."
)

EXAMPLES: list[dict[str, str]] = [
    {
        "id": "scene_floral_foam",
        "task_type": "Scene Gen",
        "text_prompt": (
            SCENE_SYSTEM_PROMPT
            + " Robot Arm Type: AgiBot G1. Scene Description: Workspace: A block of floral foam "
            "soaked in water, vibrant green and porous, used for arrangements. The surface is "
            "damp, crumbly, and rectangular.\n"
            "Objects: A bundle of long-stemmed red roses with thorns stripped lies on the left. "
            "A roll of green floral tape sits upright near the center. A pair of pruning shears "
            "with red handles rests on the right edge. A handful of baby's breath, delicate and "
            "white, scatters near the front. A spool of gold ribbon, shiny and narrow, unspools "
            "toward the back. A plastic vase, clear and cylindrical, stands empty in the far corner.\n"
            "Lighting: Bright, colorful studio lights illuminate the flowers, enhancing their vivid "
            "hues. Soft shadows fall behind the stems, creating depth in the arrangement.\n"
            "Background: Walls lined with shelves of dried flowers and ribbons in every color "
            "surround the area. Large windows look out onto a bustling flower market street. "
            "Buckets of fresh blooms fill the floor space in the background. A chalkboard menu "
            "listing flower prices hangs on the back wall. <|extra_100|>"
        ),
        "uncond_prompt": SCENE_SYSTEM_PROMPT + " <|extra_100|>",
        "visual_placeholder": "<|VIS_PLH|>",
        "supervised_start": "<|extra_100|>",
        "supervised_end": "<|extra_101|>",
    },
    {
        "id": "scene_printing_press",
        "task_type": "Scene Gen",
        "text_prompt": (
            SCENE_SYSTEM_PROMPT
            + " Robot Arm Type: ARX Robot. Scene Description: Workspace: An ancient, "
            "ink-stained wooden printing press bed, carved with intricate floral patterns "
            "and smelling of turpentine and old paper.\n"
            "Objects: A heavy lead type tray filled with reversed letters sits on the left "
            "side. A roller covered in thick black ink rests diagonally across the center. "
            "A sheet of handmade rice paper with a fresh crimson print lies on the right. "
            "A small porcelain bowl of water for cleaning brushes is near the front edge. "
            "A wooden mallet with a worn handle leans against the ink roller. Scattered "
            "copper shavings from type casting gleam near the back corner.\n"
            "Lighting: Soft, directional daylight streams through a high clerestory window, "
            "illuminating dust motes dancing above the ink-stained wood.\n"
            "Background: Tall shelves packed with thousands of wooden type cases line the "
            "dimly lit walls. A large hanging pulley system for lifting paper reams dominates "
            "the ceiling space. The floor is concrete, stained with decades of spilled ink. "
            "Faint sounds of a city street drift in through the open window above. <|extra_100|>"
        ),
        "uncond_prompt": SCENE_SYSTEM_PROMPT + " <|extra_100|>",
        "visual_placeholder": "<|VIS_PLH|>",
        "supervised_start": "<|extra_100|>",
        "supervised_end": "<|extra_101|>",
    },
    {
        "id": "scene_soviet_spacecraft",
        "task_type": "Scene Gen",
        "text_prompt": (
            SCENE_SYSTEM_PROMPT
            + " Robot Arm Type: Agibot G2. Scene Description: Workspace: The cockpit dashboard "
            "of a derelict 1980s Soviet spacecraft, covered in a fine layer of gray lunar dust. "
            "The panel is a chaotic mix of analog dials, toggle switches, and cracked glass screens.\n"
            "Objects: A heavy-duty wrench with a red handle lies diagonally across the central "
            "console. A crumpled foil food packet labeled in Cyrillic rests near the oxygen gauge. "
            "An analog stopwatch with a stopped face is taped to the right side. A small potted "
            "basil plant in a tin can sits surprisingly vibrant in the center. A roll of gray duct "
            "tape is partially unpeeled on the left edge. A pair of thick woolen gloves is tossed "
            "carelessly into the far corner.\n"
            "Lighting: Harsh, stark white light from a single LED strip above cuts through the "
            "darkness, creating high-contrast black shadows. Dust particles float visibly in the beam.\n"
            "Background: Through the scratched cockpit window, the black void of space is dotted "
            "with bright, unmoving stars. The curved horizon of Earth glows with a thin blue "
            "atmospheric line in the distance. Interior walls of the ship are lined with peeling "
            "silver thermal insulation foil. Wires hang loosely from the ceiling panels where "
            "components have been removed. <|extra_100|>"
        ),
        "uncond_prompt": SCENE_SYSTEM_PROMPT + " <|extra_100|>",
        "visual_placeholder": "<|VIS_PLH|>",
        "supervised_start": "<|extra_100|>",
        "supervised_end": "<|extra_101|>",
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
        prefix="scene_gen",
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
        "task_type": "Scene Gen",
        "template": "{text}",
        "unc_prompt": "<|extra_100|>",
        "classifier_free_guidance": 2.0,
        "image_area": 1024 * 1024,
        "prompts": prompts,
    }
    if engine == "ar":
        config.update(
            {
                "force_same_image_size": False,
                "stop_on_image_end": True,
                "sampling_params": ar_image_sampling(32768),
            }
        )
    else:
        config.update({"height": 32, "width": 128, **flashar_sampling()})
    return config
