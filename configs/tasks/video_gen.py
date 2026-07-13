from __future__ import annotations

from typing import Any

from .common import active_examples, ar_image_sampling, load_legacy_video_cases, repeat_case, set_reference


EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "video_gen_reach_yellow_block",
        "text_prompt": (
            "You are a helpful assistant for embodied video prediction. You are given a task "
            "instruction and the initial observation. Your task is to predict future image chunks "
            "at 3 FPS with 1 views. Robot Arm Type: AgiBot G1. Instruction: In a fixed robotic "
            "workspace, generate a rigid, physically consistent embodied robotic arm. The arm "
            "maintains high stability with no deformation and enters the frame to Reach forward "
            "with the left arm to grasp the yellow block and lift it upward "
            "<|VIS_PLH|><|extra_100|><|extra_101|>"
        ),
        "visual_placeholder": "<|VIS_PLH|>",
        "supervised_start": "<|extra_100|>",
        "supervised_end": "<|extra_101|>",
        "image_list": ["examples/assets/video-gen/initial_0000.png"],
    },
    # {
    #     "id": "video_gen_extend_and_retract",
    #     "text_prompt": (
    #         "You are a helpful assistant for embodied video prediction. You are given a task "
    #         "instruction and the initial observation. Your task is to predict future image chunks "
    #         "at 3 FPS with 1 views. Robot Arm Type: AgiBot G1. Instruction: In a fixed robotic "
    #         "workspace, generate a rigid, physically consistent embodied robotic arm. The arm "
    #         "maintains high stability with no deformation and enters the frame to Extend the left "
    #         "arm forward to its full reach and then retract to the starting position "
    #         "<|VIS_PLH|><|extra_100|><|extra_101|>"
    #     ),
    #     "visual_placeholder": "<|VIS_PLH|>",
    #     "supervised_start": "<|extra_100|>",
    #     "supervised_end": "<|extra_101|>",
    #     "image_list": ["examples/assets/video-gen/initial_0001.png"],
    # },
    # {
    #     "id": "video_gen_grasp_cup",
    #     "text_prompt": (
    #         "You are a helpful assistant for embodied video prediction. You are given a task "
    #         "instruction and the initial observation. Your task is to predict future image chunks "
    #         "at 3 FPS with 1 views. Robot Arm Type: AgiBot G1. Instruction: In a fixed robotic "
    #         "workspace, generate a rigid, physically consistent embodied robotic arm. The arm "
    #         "maintains high stability with no deformation and enters the frame to Bring the left "
    #         "arm down at an angle to grasp the cup near the table edge "
    #         "<|VIS_PLH|><|extra_100|><|extra_101|>"
    #     ),
    #     "visual_placeholder": "<|VIS_PLH|>",
    #     "supervised_start": "<|extra_100|>",
    #     "supervised_end": "<|extra_101|>",
    #     "image_list": ["examples/assets/video-gen/initial_0002.png"],
    # },
]

OVERRIDE_CASE = {
    "reference_image": "examples/assets/video-gen/initial_0000.png",
    "prompt": "Predict the next observations for this scene.",
}


def build_prompts(
    *,
    num_samples: int | None = None,
    prompt: str | None = None,
    reference_images: list[str] | None = None,
    legacy_video_jsonl: str | None = None,
) -> dict[str, Any]:
    legacy_count = len(EXAMPLES) if num_samples is None else int(num_samples)
    legacy_cases = load_legacy_video_cases(legacy_count, legacy_video_jsonl)
    if legacy_cases is not None:
        return legacy_cases

    if prompt is None and not reference_images:
        return active_examples("video_gen", EXAMPLES, limit=num_samples)

    case = dict(OVERRIDE_CASE)
    if prompt is not None:
        case["prompt"] = prompt
    set_reference(case, reference_images)
    return repeat_case("video_gen", case, num_samples)


def task_config(
    engine: str,
    *,
    num_samples: int | None = None,
    prompt: str | None = None,
    reference_images: list[str] | None = None,
    legacy_video_jsonl: str | None = None,
) -> dict[str, Any]:
    del engine
    prompts = build_prompts(
        num_samples=num_samples,
        prompt=prompt,
        reference_images=reference_images,
        legacy_video_jsonl=legacy_video_jsonl,
    )
    return {
        "task_type": "Video Gen",
        "template": (
            "<|extra_203|>You are a helpful assistant for embodied video prediction. "
            "USER: {text} Initial observation: <|IMAGE|> ASSISTANT: <|extra_100|>"
        ),
        "unc_prompt": (
            "<|extra_203|>You are a helpful assistant for embodied video prediction. "
            "USER: Initial observation: <|IMAGE|> ASSISTANT: <|extra_100|>"
        ),
        "classifier_free_guidance": 3.0,
        "image_area": 512 * 512,
        "force_same_image_size": False,
        "stop_on_image_end": False,
        "output_fps": 1,
        "sampling_params": {
            **ar_image_sampling(32768),
            "text_top_k": 200,
            "text_top_p": 0.5,
            "text_temperature": 0.7,
            "image_top_k": 2560,
            "cfg_decay": "linear",
            "cfg_min": 1.0,
            "cfg_total_images": 15,
        },
        "prompts": prompts,
    }
