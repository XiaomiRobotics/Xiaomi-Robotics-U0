from __future__ import annotations

from copy import deepcopy


def build_sequence_config(task, example, *, num_samples=None, prompt=None, reference_images=None):
    count = 1 if num_samples is None else int(num_samples)
    if count < 1:
        raise ValueError("num_samples must be positive")
    case = deepcopy(example)
    if prompt is not None:
        case["text_prompt"] = prompt
    if reference_images:
        case["image_list"] = list(reference_images)
    if case["text_prompt"].count("<|VIS_PLH|>") != len(case["image_list"]):
        raise ValueError("Sequence prompt placeholders must match the input images")
    return {
        "task_type": task,
        "template": "{text}",
        "unc_prompt": "{text}",
        "seed": 42,
        "classifier_free_guidance": 3.0,
        "force_same_image_size": False,
        "stop_on_image_end": False,
        "output_fps": 1,
        "sampling_params": {
            "max_new_tokens": 0,
            "image_temperature": 1.0,
            "image_top_k": 5120,
            "image_top_p": 1.0,
            "control_greedy": True,
            "cfg_min": 1.0,
            "cfg_total_images": 15,
            "max_images": 32,
            "max_image_tokens": 32768,
            "max_header_tokens": 32,
        },
        "prompts": {
            case["id"] if count == 1 else f'{case["id"]}_{i:03d}': deepcopy(case)
            for i in range(count)
        },
    }
