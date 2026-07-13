from __future__ import annotations

from typing import Any

from xr_u0_ar.task_prompts import T2I_PROMPT_TEMPLATE, T2I_UNCOND_PROMPT

from .common import ar_image_sampling, flashar_sampling, override_or_examples


EXAMPLES: list[dict[str, str]] = [
    {
        "id": "t2i_glass_teapot",
        "prompt": "A clean product photograph of a glass teapot beside green tea leaves, soft daylight, neutral background.",
    },
    {
        "id": "t2i_workshop_camera",
        "prompt": "A compact industrial inspection camera on a brushed steel workbench, precise studio lighting.",
    },
    {
        "id": "t2i_robotic_gripper",
        "prompt": "A white robotic gripper holding a translucent blue cube above a matte black calibration board.",
    },
    {
        "id": "t2i_ceramic_vase",
        "prompt": "A handmade ceramic vase with a pale celadon glaze on a walnut table, morning window light.",
    },
    {
        "id": "t2i_lab_sample",
        "prompt": "A labeled glass sample vial standing in a tidy robotics lab, shallow depth of field.",
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
        prefix="t2i",
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
        "task_type": "T2I",
        "template": T2I_PROMPT_TEMPLATE,
        "unc_prompt": T2I_UNCOND_PROMPT,
        "classifier_free_guidance": 3.0,
        "image_area": 1024 * 1024,
        "prompts": prompts,
    }
    if engine == "ar":
        config.update({"target_height": 64, "target_width": 64, "sampling_params": ar_image_sampling(5120)})
    else:
        config.update({"height": 64, "width": 64, **flashar_sampling()})
    return config
