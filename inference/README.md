## Quick Start

### Environment Setup

Create one conda environment for the backend you plan to run:

| Use case | Conda environment | Notes |
| -------- | ----------------- | ----- |
| Eager inference | `xr-u0-eager` | Works for both `--engine ar` and `--engine flashar`. |
| AR vLLM inference | `xr-u0-ar-vllm` | Applies the AR vLLM patch set. |
| FlashAR vLLM inference | `xr-u0-flashar-vllm` | Applies the FlashAR vLLM patch set for speed-up. |

```bash
git clone https://github.com/XiaomiRobotics/Xiaomi-Robotics-U0.git
cd Xiaomi-Robotics-U0/inference
```

For eager inference:

```bash
conda create -n xr-u0-eager python=3.10 -y
conda activate xr-u0-eager
pip install -U pip
pip install -r requirements-ar.txt
pip install -e .
```

For AR vLLM inference:

```bash
conda create -n xr-u0-ar-vllm python=3.12 -y
conda activate xr-u0-ar-vllm
pip install -U pip
pip install -r requirements-vllm.txt
pip install -e .
python -m xr_u0_ar.apply_vllm_patches
```

For Xiaomi-Robotics-U0-FlashAR vLLM inference:

```bash
conda create -n xr-u0-flashar-vllm python=3.12 -y
conda activate xr-u0-flashar-vllm
pip install -U pip
pip install -r requirements-vllm.txt
pip install -e .
python -m xr_u0_flashar.apply_vllm_patches
```

Keep the AR and FlashAR vLLM patch sets in separate conda environments. The
patch scripts check that the installed vLLM version is exactly `0.11.0`.

### RGB Transfer Depth Setup (Optional)

RGB Transfer uses Depth Anything 3 (DA3) to convert RGB reference images into
inverse-depth maps before Xiaomi-Robotics-U0 inference. Install the optional DA3
dependencies in the same conda environment that will run Xiaomi-Robotics-U0:

```bash
pip install -e ".[depth]"
python -m pip install --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
```

The default DA3 model is `depth-anything/DA3-LARGE-1.1`. You can pass the Hub ID
directly, or override it with a local `DA3-LARGE-1.1` directory:

```bash
python scripts/inference.py \
  --engine flashar --backend vllm --task transfer \
  --input-image-type rgb \
  --da3-model-path depth-anything/DA3-LARGE-1.1
```

You can also download it first and pass the local directory:

```bash
huggingface-cli download depth-anything/DA3-LARGE-1.1 \
  --local-dir <local-da3-model-dir>
```

### Configuration

Xiaomi-Robotics-U0 uses composable Python config files. Values such as `model_path`,
`tokenizer_path`, and `vq_path` can be local directories or HuggingFace Hub IDs
for automatic download.

| File | What to edit |
| ---- | ------------ |
| `configs/base.py` | Model, tokenizer, and `VisionTokenizer` paths: `model_path`, `tokenizer_path`, `vq_path`. |
| `configs/tasks/*.py` | Per-task examples, prompts, CFG, shapes, and input images. |
| `configs/tasks/common.py` | Shared task helpers and common sampling parameters. |
| `configs/runtimes.py` | Eager/vLLM runtime parameters such as `max_num_seqs`, `max_num_batched_tokens`, and `gpu_memory_utilization`. |
| `configs/profiles.py` | Resource profiles. `multi-gpu` sets eager device mapping or vLLM tensor parallelism. |

CLI arguments override the config files, which is useful for quick tests:

```bash
python scripts/inference.py \
  --engine ar --backend eager --task t2i \
  --model-path <Xiaomi-Robotics-U0-HF-ID-or-local-path> \
  --tokenizer-path <Xiaomi-Robotics-U0-HF-ID-or-local-path> \
  --vq-path <VisionTokenizer-HF-ID-or-local-path> \
  --dry-run
```

Use `--dry-run` whenever you want to inspect the final composed config without
loading a model.

### Inference

All tasks use the same entry point:

```bash
python scripts/inference.py \
  --engine <ar|flashar> \
  --backend <eager|vllm> \
  --task <t2i|x2i|scene-gen|transfer>
```

Use `--engine ar` with `Xiaomi-Robotics-U0` for `T2I`, `X2I`, `Scene Gen`, and `Transfer`.
Use `--engine flashar` with `Xiaomi-Robotics-U0-FlashAR` to speed up those tasks.
The AR checkpoint is 34B, while the FlashAR checkpoint is 38B.
Sequence checkpoints support `--engine ar --backend eager` with
`--task interleave_subtask` or `--task interleave_video`. Select the 4B or 34B
Sequence checkpoint with `--model-size 4b` or `--model-size 34b`.
The examples retain their input image dimensions and interleaved prompt layout.
Image tasks save PNGs; subtask planning saves ordered text/image JSON and images;
video prediction saves an MP4 at 1 FPS. Runtime summaries are printed to the log.

Task examples come from `configs/tasks/*.py`. Override them from the CLI with
`--prompt` and `--reference-image` when needed. Transfer uses depth-map
references by default; RGB Transfer needs `--input-image-type rgb`; see
[RGB Transfer Depth Setup](#rgb-transfer-depth-setup).

Xiaomi-Robotics-U0-FlashAR vLLM keeps `enable_prefix_caching = False` by default.

The 4B checkpoint is available for AR inference only. From the `inference/`
directory, select it with `--model-size 4b`; the default AR model remains 34B:

```bash
python scripts/inference.py \
  --engine ar --backend eager --model-size 4b --task t2i
```

The expected local path is `../ckpt/Xiaomi-Robotics-U0-4B`.

### Distributed Inference

Set visible GPUs and select the multi-GPU profile:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python scripts/inference.py --engine flashar --backend vllm --task t2i --profile multi-gpu
```

## 3. Gradio Demo

The demo runs Xiaomi-Robotics-U0-FlashAR through a FastAPI model server and a Gradio UI. It
covers `T2I`, `X2I`, `Scene Gen`, and `Transfer`.

Use the `xr-u0-flashar-vllm` environment and install the UI dependencies:

```bash
conda activate xr-u0-flashar-vllm
pip install -e ".[demo]"
```

Set paths with environment variables or the matching server CLI arguments:

```bash
export XR_U0_FLASHAR_MODEL_DIR=<flashar-model-or-local-path>
export XR_U0_FLASHAR_TOKENIZER_DIR=<flashar-tokenizer-or-local-path>
export XR_U0_VISION_TOKENIZER_DIR=<VisionTokenizer-HF-ID-or-local-path>
# Optional; only needed when using RGB images for Transfer.
export XR_U0_DA3_MODEL_PATH=depth-anything/DA3-LARGE-1.1
```

`XR_U0_DA3_MODEL_PATH` is optional. Depth-map Transfer examples do not use it.
For RGB Transfer, first install the optional DA3 dependencies as described in
[RGB Transfer Depth Setup](#rgb-transfer-depth-setup).

Start the API server and UI in two terminals:

```bash
CUDA_VISIBLE_DEVICES=0 python demo/flashar_api_server.py --load-on-startup
# CUDA_VISIBLE_DEVICES=0,1 \
# python demo/flashar_api_server.py --load-on-startup --tensor-parallel-size 2
# export XR_U0_FLASHAR_TP=2
```

```bash
python demo/flashar_gradio_app.py --api-url http://127.0.0.1:8000
```

Open `http://127.0.0.1:7860`. Outputs are saved under
`outputs/gradio_flashar/` with neighboring audit JSON files.

## 4. Citation

If you find this work useful, please cite:

```bibtex
@misc{li2026xiaomiroboticsu0,
  title         = {{Xiaomi-Robotics-U0}: Unified Embodied Synthesis with World Foundation Model},
  author        = {Xinghang Li and Jun Guo and Qiwei Li and Long Qian and Hang Lai and Yueze Wang and Hongyu Yan and Jiahang Cao and Xi Chen and Jingen Qu and Jiaxi Song and Nan Sun and Hanye Zhao and Futeng Liu and Wanli Peng and Heyun Wang and Yunhong Wang and Caoyu Xia and Jack Zhao and Diyun Xiang and Hangjun Ye and Heng Qu and Huaping Liu and Jason Li},
  year          = {2026},
  eprint        = {2607.11643},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2607.11643}
}
```
