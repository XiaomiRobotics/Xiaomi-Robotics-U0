from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from configs.tasks import task_config
from xr_u0_flashar.demo_runtime import (
    DEFAULT_DA3_DEVICE,
    DEFAULT_DA3_MAX_DEPTH,
    DEFAULT_DA3_MIN_DEPTH,
    DEFAULT_DA3_PROCESS_RES,
    SUPPORTED_TASKS,
    TASK_SCENE,
    TASK_T2I,
    TASK_TRANSFER,
    TASK_X2I,
    normalize_task_type,
    task_defaults,
)


ROOT = Path(__file__).resolve().parents[1]
SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
STOP_TOKEN_RE = re.compile(r"<\|VIS_PLH\|>|<\|extra_100\|>|<\|extra_101\|>")
TASK_CONFIG_KEYS = {
    TASK_T2I: "t2i",
    TASK_X2I: "x2i",
    TASK_SCENE: "scene_gen",
    TASK_TRANSFER: "transfer",
}


@dataclass(frozen=True)
class FlashARDemoExample:
    id: str
    task_type: str
    prompt: str
    reference_image: str | None
    cfg_scale: float
    height: int
    width: int
    source_image_area: int
    temperature: float
    top_k: int
    top_p: float
    input_image_type: str
    robot_arm_type: str
    image_style: str
    da3_model_path: str | None
    da3_device: str
    da3_min_depth: float
    da3_max_depth: float
    da3_process_res: int

    @property
    def reference_path(self) -> str | None:
        if not self.reference_image:
            return None
        return str((ROOT / self.reference_image).resolve())


def _clean_prompt(text: str) -> str:
    value = text.split("Scene Description:", 1)[-1]
    value = STOP_TOKEN_RE.split(value, maxsplit=1)[0]
    value = SPECIAL_TOKEN_RE.sub("", value)
    return value.strip()


def _extract_robot_arm(text: str) -> str:
    match = re.search(r"Robot Arm Type:\s*(.*?)\.\s*Scene Description:", text, flags=re.DOTALL)
    return match.group(1).strip() if match else "AgiBot G1"


def _extract_image_style(text: str, fallback: str) -> str:
    match = re.search(r"Image Style:\s*([^.<\n]+)", text)
    return match.group(1).strip() if match else fallback


def _case_prompt(task: str, case: dict[str, Any]) -> str:
    if task in (TASK_T2I, TASK_X2I):
        return str(case.get("prompt") or "").strip()
    return _clean_prompt(str(case.get("text_prompt") or case.get("prompt") or ""))


def _build_example(task: str, cfg: dict[str, Any], case_id: str, case: dict[str, Any]) -> FlashARDemoExample:
    defaults = task_defaults(task)
    raw_prompt = str(case.get("text_prompt") or case.get("prompt") or "")
    reference = case.get("reference_image")
    reference_image = str(reference) if reference else None
    fallback_style = "Simulator" if task == TASK_TRANSFER else "Real"

    return FlashARDemoExample(
        id=case_id,
        task_type=task,
        prompt=_case_prompt(task, case),
        reference_image=reference_image,
        cfg_scale=float(cfg.get("classifier_free_guidance", defaults["cfg_scale"])),
        height=int(cfg.get("height") or defaults["height"] or 0),
        width=int(cfg.get("width") or defaults["width"] or 0),
        source_image_area=int(cfg.get("source_image_area") or cfg.get("image_area") or 1024 * 1024),
        temperature=float(cfg.get("temperature", 1.0)),
        top_k=int(cfg.get("top_k", 5120)),
        top_p=float(cfg.get("top_p", 1.0)),
        input_image_type=str(cfg.get("input_image_type") or "depth"),
        robot_arm_type=_extract_robot_arm(raw_prompt),
        image_style=_extract_image_style(raw_prompt, fallback_style),
        da3_model_path=None,
        da3_device=str(cfg.get("da3_device") or DEFAULT_DA3_DEVICE),
        da3_min_depth=float(cfg.get("da3_min_depth", DEFAULT_DA3_MIN_DEPTH)),
        da3_max_depth=float(cfg.get("da3_max_depth", DEFAULT_DA3_MAX_DEPTH)),
        da3_process_res=int(cfg.get("da3_process_res", DEFAULT_DA3_PROCESS_RES)),
    )


@lru_cache(maxsize=1)
def flashar_demo_examples() -> dict[str, list[FlashARDemoExample]]:
    examples: dict[str, list[FlashARDemoExample]] = {}
    for task in SUPPORTED_TASKS:
        cfg = task_config("flashar", TASK_CONFIG_KEYS[task])
        task_examples = []
        for case_id, raw_case in cfg["prompts"].items():
            if not isinstance(raw_case, dict):
                raw_case = {"prompt": str(raw_case)}
            task_examples.append(_build_example(task, cfg, str(case_id), raw_case))
        examples[task] = task_examples
    return examples


def example_choices(task_type: str) -> list[str]:
    task = normalize_task_type(task_type)
    return [example.id for example in flashar_demo_examples()[task]]


def demo_example_for(task_type: str, example_id: str | None = None) -> FlashARDemoExample:
    task = normalize_task_type(task_type)
    examples = flashar_demo_examples()[task]
    if not examples:
        raise ValueError(f"{task} has no configured demo examples")
    if not example_id:
        return examples[0]
    for example in examples:
        if example.id == example_id:
            return example
    raise ValueError(f"unknown {task} example: {example_id}")
