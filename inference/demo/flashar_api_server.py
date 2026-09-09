from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xr_u0_flashar.demo_runtime import (  # noqa: E402
    DEFAULT_DA3_MODEL_PATH,
    SUPPORTED_TASKS,
    image_path_to_base64,
    normalize_task_type,
    pil_image_from_base64,
    render_flashar_request,
    save_flashar_result,
)


@dataclass
class ServerConfig:
    model_dir: str = os.environ.get("XR_U0_FLASHAR_MODEL_DIR", "checkpoints/Xiaomi-Robotics-U0-FlashAR")
    tokenizer_dir: str = os.environ.get("XR_U0_FLASHAR_TOKENIZER_DIR", "checkpoints/Xiaomi-Robotics-U0-FlashAR")
    vision_tokenizer_dir: str = os.environ.get("XR_U0_VISION_TOKENIZER_DIR", "checkpoints/VisionTokenizer")
    vision_tokenizer_type: str = os.environ.get("XR_U0_VISION_TOKENIZER_TYPE", "ibq")
    vision_device: str = os.environ.get("XR_U0_VISION_DEVICE", "cuda:0")
    output_dir: str = os.environ.get("XR_U0_FLASHAR_DEMO_OUTPUT_DIR", "outputs/gradio_flashar")
    da3_model_path: str = os.environ.get("XR_U0_DA3_MODEL_PATH", DEFAULT_DA3_MODEL_PATH)
    da3_device: str = os.environ.get("XR_U0_DA3_DEVICE", "cuda:0")
    da3_min_depth: float = float(os.environ.get("XR_U0_DA3_MIN_DEPTH", "0.7"))
    da3_max_depth: float = float(os.environ.get("XR_U0_DA3_MAX_DEPTH", "2.0"))
    da3_process_res: int = int(os.environ.get("XR_U0_DA3_PROCESS_RES", "504"))
    tensor_parallel_size: int = int(os.environ.get("XR_U0_FLASHAR_TP", "1"))
    gpu_memory_utilization: float = float(os.environ.get("XR_U0_FLASHAR_GPU_MEMORY_UTILIZATION", "0.85"))
    max_model_len: int = int(os.environ.get("XR_U0_FLASHAR_MAX_MODEL_LEN", "16384"))
    max_num_seqs: int = int(os.environ.get("XR_U0_FLASHAR_MAX_NUM_SEQS", "16"))
    max_num_batched_tokens: int = int(os.environ.get("XR_U0_FLASHAR_MAX_BATCHED_TOKENS", "32768"))
    seed: int = int(os.environ.get("XR_U0_FLASHAR_SEED", "42"))
    enable_chunked_prefill: bool = os.environ.get("XR_U0_FLASHAR_CHUNKED_PREFILL", "1") != "0"
    strict_visual_tokens: bool = os.environ.get("XR_U0_FLASHAR_STRICT_VISUAL_TOKENS", "1") != "0"
    load_on_startup: bool = os.environ.get("XR_U0_FLASHAR_LOAD_ON_STARTUP", "0") == "1"


class GenerateRequest(BaseModel):
    task_type: str = Field("T2I", description=f"One of: {', '.join(SUPPORTED_TASKS)}")
    prompt: str = Field(..., description="User prompt or scene description.")
    reference_images: list[str] = Field(default_factory=list, description="PNG/JPEG images as base64 or data URLs.")
    height: int | None = Field(None, description="Output height in visual tokens. Auto for X2I/Transfer.")
    width: int | None = Field(None, description="Output width in visual tokens. Auto for X2I/Transfer.")
    cfg_scale: float | None = Field(None, description="Classifier-free guidance scale.")
    temperature: float = 1.0
    top_k: int = 5120
    top_p: float = 1.0
    seed: int | None = None
    source_image_area: int = 1024 * 1024
    robot_arm_type: str = "AgiBot G1"
    image_style: str = "Real"
    input_image_type: str = Field("depth", description="Transfer reference type: depth or rgb.")
    da3_model_path: str | None = Field(None, description="DA3 model path or HF repo for RGB Transfer input.")
    da3_device: str | None = Field(None, description="Device used by DA3 preprocessing.")
    da3_min_depth: float | None = Field(None, description="DA3 raw depth clip lower bound.")
    da3_max_depth: float | None = Field(None, description="DA3 raw depth clip upper bound.")
    da3_process_res: int | None = Field(None, description="DA3 processing resolution.")


class GenerateResponse(BaseModel):
    request_id: str
    task_type: str
    image_base64: str
    output_path: str
    audit_path: str
    audit: dict[str, Any]
    metadata: dict[str, Any]
    total_seconds: float
    generation_seconds: float


class FlashARService:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.llm = None
        self.vision_tokenizer = None
        self.visual_offset = None
        self.loaded_at: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.llm is not None and self.vision_tokenizer is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        from xr_u0_ar.vision_tokenizer import build_vision_tokenizer
        from xr_u0_flashar.vllm import LLM

        self.llm = LLM.from_pretrained(
            self.config.model_dir,
            tokenizer_dir=self.config.tokenizer_dir,
            tensor_parallel_size=self.config.tensor_parallel_size,
            max_model_len=self.config.max_model_len,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            seed=self.config.seed,
            enable_prefix_caching=False,
            enable_chunked_prefill=self.config.enable_chunked_prefill,
            strict_visual_tokens=self.config.strict_visual_tokens,
        )
        self.vision_tokenizer = build_vision_tokenizer(
            self.config.vision_tokenizer_type,
            self.config.vision_tokenizer_dir,
            device=self.config.vision_device,
        )
        self.visual_offset = self.llm.tokenizer.encode("<|image end|>")[0] + 1
        self.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.load()
        assert self.llm is not None
        assert self.vision_tokenizer is not None
        assert self.visual_offset is not None

        started = time.perf_counter()
        request_id = uuid.uuid4().hex[:8]
        task_slug = normalize_task_type(request.task_type).lower().replace(" ", "_")
        da3_artifact_dir = Path(self.config.output_dir) / task_slug / "da3_preprocess" / request_id
        reference_images = [pil_image_from_base64(item) for item in request.reference_images]
        rendered = render_flashar_request(
            task_type=request.task_type,
            text=request.prompt,
            tokenizer=self.llm.tokenizer,
            vision_tokenizer=self.vision_tokenizer,
            reference_images=reference_images,
            height=request.height,
            width=request.width,
            cfg_scale=request.cfg_scale,
            source_image_area=request.source_image_area,
            robot_arm_type=request.robot_arm_type,
            image_style=request.image_style,
            input_image_type=request.input_image_type,
            da3_model_path=request.da3_model_path or self.config.da3_model_path or None,
            da3_device=request.da3_device or self.config.da3_device,
            da3_min_depth=(
                self.config.da3_min_depth if request.da3_min_depth is None else request.da3_min_depth
            ),
            da3_max_depth=(
                self.config.da3_max_depth if request.da3_max_depth is None else request.da3_max_depth
            ),
            da3_process_res=(
                self.config.da3_process_res if request.da3_process_res is None else request.da3_process_res
            ),
            da3_artifact_dir=da3_artifact_dir,
        )

        generation_started = time.perf_counter()
        results = self.llm.generate(
            [rendered.prompt],
            height=rendered.height,
            width=rendered.width,
            cfg_scale=rendered.cfg_scale,
            temperature=float(request.temperature),
            top_k=int(request.top_k),
            top_p=float(request.top_p),
            uncond_prompt=[rendered.uncond_prompt],
            seeds=None if request.seed is None else [int(request.seed)],
            prompt_template="{text}",
        )
        generation_seconds = time.perf_counter() - generation_started
        if not results:
            raise RuntimeError("FlashAR vLLM returned no result")

        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / task_slug
        output_path = output_dir / f"{ts}_{request_id}.png"
        metadata = {
            **rendered.metadata,
            "request_id": request_id,
            "prompt": request.prompt,
            "temperature": float(request.temperature),
            "top_k": int(request.top_k),
            "top_p": float(request.top_p),
            "seed": request.seed,
            "model_dir": self.config.model_dir,
            "tokenizer_dir": self.config.tokenizer_dir,
            "vision_tokenizer_dir": self.config.vision_tokenizer_dir,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "strict_visual_tokens": self.config.strict_visual_tokens,
        }
        saved = save_flashar_result(
            result=results[0],
            vision_tokenizer=self.vision_tokenizer,
            visual_offset=self.visual_offset,
            output_path=output_path,
            metadata=metadata,
        )
        request_json_path = output_path.with_suffix(".request.json")
        request_json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        return GenerateResponse(
            request_id=request_id,
            task_type=rendered.task_type,
            image_base64=image_path_to_base64(saved["output_path"]),
            output_path=saved["output_path"],
            audit_path=saved["audit_path"],
            audit=saved["audit"],
            metadata={**metadata, "request_path": str(request_json_path.resolve())},
            total_seconds=time.perf_counter() - started,
            generation_seconds=generation_seconds,
        )


CONFIG = ServerConfig()
SERVICE = FlashARService(CONFIG)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if CONFIG.load_on_startup:
        SERVICE.load()
    yield


app = FastAPI(title="Xiaomi-Robotics-U0 FlashAR Demo API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "loaded": SERVICE.is_loaded,
        "loaded_at": SERVICE.loaded_at,
        "model_dir": CONFIG.model_dir,
        "tokenizer_dir": CONFIG.tokenizer_dir,
        "vision_tokenizer_dir": CONFIG.vision_tokenizer_dir,
        "output_dir": CONFIG.output_dir,
        "da3_model_path": CONFIG.da3_model_path or None,
    }


@app.post("/api/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        return SERVICE.generate(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Xiaomi-Robotics-U0 FlashAR Gradio demo API.")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--model-dir", default=CONFIG.model_dir)
    parser.add_argument("--tokenizer-dir", default=CONFIG.tokenizer_dir)
    parser.add_argument("--vision-tokenizer-dir", default=CONFIG.vision_tokenizer_dir)
    parser.add_argument("--vision-device", default=CONFIG.vision_device)
    parser.add_argument("--output-dir", default=CONFIG.output_dir)
    parser.add_argument("--da3-model-path", default=CONFIG.da3_model_path)
    parser.add_argument("--da3-device", default=CONFIG.da3_device)
    parser.add_argument("--da3-min-depth", type=float, default=CONFIG.da3_min_depth)
    parser.add_argument("--da3-max-depth", type=float, default=CONFIG.da3_max_depth)
    parser.add_argument("--da3-process-res", type=int, default=CONFIG.da3_process_res)
    parser.add_argument("--tensor-parallel-size", type=int, default=CONFIG.tensor_parallel_size)
    parser.add_argument("--gpu-memory-utilization", type=float, default=CONFIG.gpu_memory_utilization)
    parser.add_argument("--max-model-len", type=int, default=CONFIG.max_model_len)
    parser.add_argument("--max-num-seqs", type=int, default=CONFIG.max_num_seqs)
    parser.add_argument("--max-num-batched-tokens", type=int, default=CONFIG.max_num_batched_tokens)
    parser.add_argument("--load-on-startup", action="store_true", default=CONFIG.load_on_startup)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CONFIG.model_dir = args.model_dir
    CONFIG.tokenizer_dir = args.tokenizer_dir
    CONFIG.vision_tokenizer_dir = args.vision_tokenizer_dir
    CONFIG.vision_device = args.vision_device
    CONFIG.output_dir = args.output_dir
    CONFIG.da3_model_path = args.da3_model_path
    CONFIG.da3_device = args.da3_device
    CONFIG.da3_min_depth = args.da3_min_depth
    CONFIG.da3_max_depth = args.da3_max_depth
    CONFIG.da3_process_res = args.da3_process_res
    CONFIG.tensor_parallel_size = args.tensor_parallel_size
    CONFIG.gpu_memory_utilization = args.gpu_memory_utilization
    CONFIG.max_model_len = args.max_model_len
    CONFIG.max_num_seqs = args.max_num_seqs
    CONFIG.max_num_batched_tokens = args.max_num_batched_tokens
    CONFIG.load_on_startup = args.load_on_startup

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
