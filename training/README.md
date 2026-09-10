<div align="center">
<h1>Xiaomi-Robotics-U0 Training Framework</h1>
<p>FSDP training framework for Xiaomi-Robotics-U0 world foundation models</p>
<p>
<a href="https://arxiv.org/abs/2607.11643"><img src="https://img.shields.io/badge/arXiv-2607.11643-b31b1b.svg" alt="Paper" /></a>
<a href="https://robotics.xiaomi.com/xiaomi-robotics-u0.html"><img src="https://img.shields.io/badge/Project-Page-2e7d32.svg" alt="Project Page" /></a>
<a href="https://github.com/XiaomiRobotics/Xiaomi-Robotics-U0"><img src="https://img.shields.io/badge/GitHub-Xiaomi--Robotics--U0-181717.svg" alt="Xiaomi-Robotics-U0" /></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License" /></a>
</p>
</div>

Xiaomi-Robotics-U0 is a world foundation model that learns environment states,
motion dynamics, and task evolution from multimodal spatiotemporal data. It
provides a foundation for research in robot perception, prediction, planning,
and control.

This repository provides the PyTorch FSDP training framework for the two
distilled 4B Xiaomi-Robotics-U0 models and the 34B sequence-generation model.
It supports multi-node training, long-context packed data, Hugging Face
checkpoints, and Ulysses sequence parallelism. Inference code and runnable
examples are maintained in the
[Xiaomi-Robotics-U0 inference branch](https://github.com/XiaomiRobotics/Xiaomi-Robotics-U0/tree/dev/4b-ar-inference).


## 1. Installation

The recommended environment is Python 3.10, CUDA, PyTorch 2.7, Transformers
4.57, and FlashAttention 2.

```bash
git clone <repository-url>
cd Xiaomi-Robotics-U0/training

conda create -n xr-u0-train python=3.10 -y
conda activate xr-u0-train

pip install -U pip
pip install -r requirements.txt
pip install -e . --no-deps
```

Install a FlashAttention 2 build compatible with the local PyTorch and CUDA
versions before training.

## 2. Training

The repository provides the following training presets:

| Model | Configuration | Launcher |
| :-- | :-- | :-- |
| 4B | `configs/train/u0_4b.yaml` | `scripts/train/run_u0_4b.sh` |
| 34B | `configs/train/u0_34b.yaml` | `scripts/train/run_u0_34b.sh` |
| 4B sequence | `configs/train/u0_4b_sequence.yaml` | `scripts/train/run_u0_4b_sequence.sh` |
| 34B sequence | `configs/train/u0_34b_sequence.yaml` | `scripts/train/run_u0_34b_sequence.sh` |

The single-step and sequence-generation 4B models share the same training
entry point. Select the task by providing the corresponding checkpoint and
tokenized training data.

Create a local data catalog from the public template, then replace its
placeholder dataset path. The resulting catalog is ignored by Git.

```bash
cp configs/data/full_16k_megatron_packed.example.json \
  configs/data/full_16k_megatron_packed.json

cp configs/data/sequence_32k_megatron_packed.example.json \
  configs/data/sequence_32k_megatron_packed.json
```

Set the checkpoint and data paths in the YAML file, or override them when
launching:

```bash
WM_FSDP_MODEL_CHECKPOINT=/path/to/model-hf \
WM_FSDP_TOKENIZER_PATH=/path/to/tokenizer \
WM_FSDP_SPECIAL_TOKENS_FILE=/path/to/unis_vision_tokens.txt \
DATASET_CONFIG=/path/to/packed-data.json \
RESULTS_DIR=/path/to/outputs/u0-34b \
bash scripts/train/run_u0_34b.sh
```

To enable Ulysses SP=2 for 34B, set `parallel.ulysses_size: 2` in
`configs/train/u0_34b.yaml` and use the standard launcher:

```bash
bash scripts/train/run_u0_34b.sh
```

## 3. Project Structure

```text
wm-fsdp-release/
├── configs/                  # Reproducible data and training configurations
│   ├── data/                 # Public catalog examples and data-format notes
│   └── train/                # 4B/34B full-span and sequence presets
├── scripts/                  # Command-line tools and launchers
│   ├── data/                 # Packed-catalog builders and validators
│   └── train/                # Multi-node FSDP training launchers
├── src/wm_fsdp/              # Core training package
│   ├── data/                 # Dataset loading, sampling, and collation
│   ├── models/               # Model construction and Ulysses adapters
│   ├── tokenizers/           # Text tokenizer and native IBQ VQ tokenizer
│   └── train/                # Distributed loop, FSDP, checkpointing, and resume
├── README.md                 # Project overview and usage
├── LICENSE                   # Apache 2.0 license
├── pyproject.toml            # Package metadata and build configuration
└── requirements.txt          # Python runtime dependencies
```

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
