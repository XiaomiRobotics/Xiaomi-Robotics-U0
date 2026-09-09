from __future__ import annotations

from .sequence_common import build_sequence_config

EXAMPLE = {'id': 'embodied_interleave_video', 'text_prompt': '<|extra_203|>You are a helpful assistant for embodied video prediction. You are given a task instruction and the initial observation. Your task is to predict future image chunks at 1 FPS with 1 views. Robot Arm Type: aloha-agilex. Instruction: In a fixed robotic workspace, generate a rigid, physically consistent embodied robotic arm. The arm maintains high stability with no deformation and enters the frame to Lift the long thin bottle from the table upright.\n<|VIS_PLH|>\n<|extra_100|>', 'image_list': ['examples/assets/interleave/video/input_000.png'], 'target_grid_shapes': [[28, 37], [28, 37], [28, 37], [28, 37], [28, 37]]}

def task_config(engine, *, num_samples=None, prompt=None, reference_images=None, legacy_video_jsonl=None):
    return build_sequence_config('interleave_video', EXAMPLE, num_samples=num_samples,
                                 prompt=prompt, reference_images=reference_images)
