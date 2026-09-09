from __future__ import annotations

from .sequence_common import build_sequence_config

EXAMPLE = {'id': 'embodied_interleave_subtask', 'text_prompt': '<|extra_203|>You are a helpful assistant for embodied subtask planning. You are given a task instruction and the initial observation. Your task is to predict the interleaved subtask and observation to finish the task. Robot Arm Type: AgiBot G1. Instruction: pickup red apple, orange, and peach and place in shopping cart bag. Finish the task with 6 steps.\n<|VIS_PLH|>\n<|VIS_PLH|>\n<|VIS_PLH|>\n<|extra_100|>', 'image_list': ['examples/assets/interleave/subtask/input_000.png', 'examples/assets/interleave/subtask/input_001.png', 'examples/assets/interleave/subtask/input_002.png'], 'target_grid_shapes': [[28, 37], [28, 37], [28, 37], [28, 37], [28, 37], [28, 37], [28, 37], [28, 37], [28, 37], [28, 37], [28, 37], [28, 37], [28, 37], [28, 37], [28, 37], [28, 37], [28, 37], [28, 37]]}

def task_config(engine, *, num_samples=None, prompt=None, reference_images=None, legacy_video_jsonl=None):
    return build_sequence_config('interleave_subtask', EXAMPLE, num_samples=num_samples,
                                 prompt=prompt, reference_images=reference_images)
