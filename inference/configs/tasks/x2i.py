from __future__ import annotations

from typing import Any

from xr_u0_ar.task_prompts import X2I_PROMPT_TEMPLATE, X2I_UNCOND_PROMPT

from .common import ar_image_sampling, flashar_sampling, override_or_examples


EXAMPLES: list[dict[str, str]] = [
    {
        "id": "x2i_adjust_tortoise",
        "reference_image": "examples/assets/x2i/adjust_tortoise.png",
        "prompt": "Change the tortoise's shell texture to a smooth surface.",
    },
    {
        "id": "x2i_style_ukiyoe",
        "reference_image": "examples/assets/x2i/style_ukiyoe.png",
        "prompt": "Transfer the image into a traditional ukiyo-e woodblock-print style.",
    },
    {
        "id": "x2i_background_snowy_mountain",
        "reference_image": "examples/assets/x2i/background_snowy_mountain.png",
        "prompt": "Change the beach and ocean environment in the picture to a snowy mountain.",
    },
    {
        "id": "x2i_extract_striped_top",
        "reference_image": "examples/assets/x2i/extract_striped_top.png",
        "prompt": "Extract the colorful striped top worn by the person in the image.",
    },
    {
        "id": "x2i_remove_red_trolley",
        "reference_image": "examples/assets/x2i/remove_red_trolley.png",
        "prompt": "Remove the red trolley marked 77 and labeled WEST CHESTER from the railway track in the foreground.",
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
        prefix="x2i",
        examples=EXAMPLES,
        num_samples=num_samples,
        prompt=prompt,
        reference_images=reference_images,
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
        "task_type": "X2I",
        "template": X2I_PROMPT_TEMPLATE,
        "unc_prompt": X2I_UNCOND_PROMPT,
        "classifier_free_guidance": 3.0,
        "prompts": prompts,
    }
    if engine == "ar":
        config.update(
            {
                "target_height": 64,
                "target_width": 64,
                "image_area": 1024 * 1024,
                "sampling_params": ar_image_sampling(5120),
            }
        )
    else:
        config.update({"source_image_area": 1024 * 1024, "match_reference_shape": True, **flashar_sampling()})
    return config
