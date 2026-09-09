# vLLM patches required by FlashAR

These patches modify vLLM 0.11.0 to support FlashAR inference. Apply them once
after installing vLLM:

```bash
pip install vllm==0.11.0
python tools/apply_vllm_patches.py
```

The script is idempotent — running it again on an already-patched vLLM is a
no-op. To revert: `python tools/apply_vllm_patches.py --revert`.

## Patch provenance

Most patches in this directory are sourced from
[BAAI/Xiaomi-Robotics-U0](https://github.com/baaivision/Xiaomi-Robotics-U0)'s `third_party/vllm/`,
licensed under Apache-2.0 (Copyright 2025 BAAI). They add Xiaomi-Robotics-U0 model support
to upstream vLLM (model class registration, hybrid CFG batch scheduler, custom
logits processor, etc.).

`model_executor/models/unis.py.patch` has been extended with FlashAR-specific
additions (the `aux_hidden_state_layers` capture mechanism for vertical_block
input). The added lines are clearly marked with `# FlashAR:` comments.

## Files touched by these patches

After applying, the following parts of vLLM are modified:
- `model_executor/models/{unis.py, registry.py}` — Xiaomi-Robotics-U0 + FlashAR model
- `transformers_utils/configs/{unis.py, __init__.py}`, `config.py` — config
- `inputs/{data.py, preprocess.py}` — multimodal hybrid CFG inputs
- `v1/engine/{llm_engine.py, output_processor.py, parallel_sampling.py, processor.py}`
- `v1/core/sched/{batch_scheduler.py, batch_manager.py, output.py}` — hybrid CFG scheduling
- `v1/sample/logits_processor/{__init__.py, builtin.py, interface.py, state.py}` — CFG logits
- `v1/worker/{gpu_model_runner.py, gpu_input_batch.py}` — hybrid CFG worker support
