from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from wm_fsdp.tokenizers.u0_text import load_u0_tokenizer
from wm_fsdp.models.u0_ulysses import enable_u0_ulysses


def grouped_multimodal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    visual_token_offset: int,
    image_loss_weight: float = 1.0,
    text_loss_weight: float = 1.0,
    chunk_tokens: int = 256,
) -> dict[str, torch.Tensor]:
    """Compute pooled image and text/control CE means for packed AR training."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(
            f"Expected logits [B,S,V] and labels [B,S], got {tuple(logits.shape)} and "
            f"{tuple(labels.shape)}"
        )
    if logits.size(1) < 2:
        raise ValueError("Grouped multimodal loss requires at least two tokens")
    if chunk_tokens <= 0:
        raise ValueError("loss.chunk_tokens must be positive")
    if image_loss_weight <= 0.0 or text_loss_weight <= 0.0:
        raise ValueError("Image and text loss weights must be positive")

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(-100)
    image_valid = valid & shift_labels.ge(int(visual_token_offset))
    text_valid = valid & ~image_valid
    image_counts = image_valid.sum(dim=1)
    text_counts = text_valid.sum(dim=1)

    zero = shift_logits[:, 0, 0].float() * 0.0
    image_sums = zero.clone()
    text_sums = zero.clone()
    batch_size = int(shift_logits.size(0))
    vocab_size = int(shift_logits.size(-1))

    def chunk_cross_entropy(chunk_logits: torch.Tensor, chunk_labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            chunk_logits.reshape(-1, vocab_size),
            chunk_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(batch_size, -1)

    sequence_length = int(shift_labels.size(1))
    for start in range(0, sequence_length, int(chunk_tokens)):
        end = min(start + int(chunk_tokens), sequence_length)
        chunk_logits = shift_logits[:, start:end, :].contiguous()
        chunk_labels = shift_labels[:, start:end].contiguous()
        if torch.is_grad_enabled() and chunk_logits.requires_grad:
            losses = checkpoint(
                chunk_cross_entropy,
                chunk_logits,
                chunk_labels,
                use_reentrant=False,
            ).float()
        else:
            losses = chunk_cross_entropy(chunk_logits, chunk_labels).float()
        image_sums = image_sums + (
            losses * image_valid[:, start:end].float()
        ).sum(dim=1)
        text_sums = text_sums + (
            losses * text_valid[:, start:end].float()
        ).sum(dim=1)

    sample_image_ce = image_sums / image_counts.clamp(min=1).float()
    sample_text_ce = text_sums / text_counts.clamp(min=1).float()
    sample_loss = (
        float(image_loss_weight) * sample_image_ce
        + float(text_loss_weight) * sample_text_ce
    )
    image_ce = image_sums.sum() / image_counts.sum().clamp(min=1).float()
    text_ce = text_sums.sum() / text_counts.sum().clamp(min=1).float()
    loss = float(image_loss_weight) * image_ce + float(text_loss_weight) * text_ce
    return {
        "loss": loss,
        # The trainer averages this objective over equal-sized local micro-batches.
        "loss_sum": loss * batch_size,
        "sample_loss": sample_loss,
        "image_ce": image_ce,
        "text_ce": text_ce,
        "sample_image_ce": sample_image_ce,
        "sample_text_ce": sample_text_ce,
        "image_token_count": image_counts.sum(),
        "text_token_count": text_counts.sum(),
    }


def selective_grouped_multimodal_cross_entropy(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    output_head: nn.Module,
    *,
    visual_token_offset: int,
    image_loss_weight: float = 1.0,
    text_loss_weight: float = 1.0,
    chunk_tokens: int = 256,
) -> dict[str, torch.Tensor]:
    """Apply the LM head only to causally shifted, supervised positions."""
    if hidden_states.ndim != 3 or labels.ndim != 2 or hidden_states.shape[:2] != labels.shape:
        raise ValueError(
            f"Expected hidden states [B,S,H] and labels [B,S], got "
            f"{tuple(hidden_states.shape)} and {tuple(labels.shape)}"
        )
    if hidden_states.size(1) < 2:
        raise ValueError("Grouped multimodal loss requires at least two tokens")
    if chunk_tokens <= 0:
        raise ValueError("loss.chunk_tokens must be positive")
    if image_loss_weight <= 0.0 or text_loss_weight <= 0.0:
        raise ValueError("Image and text loss weights must be positive")

    batch_size = int(hidden_states.size(0))
    sequence_length = int(hidden_states.size(1) - 1)
    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(-100)
    valid_positions = valid.reshape(-1).nonzero(as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        # The trainer rejects target-free micro-batches. Keep this helper
        # well-defined for validation and malformed-input diagnostics.
        zero = hidden_states.reshape(-1)[0].float() * 0.0
        zeros = hidden_states.new_zeros(batch_size, dtype=torch.float32)
        zero_count = labels.new_zeros((), dtype=torch.long)
        return {
            "loss": zero,
            "loss_sum": zero,
            "sample_loss": zeros,
            "image_ce": zero,
            "text_ce": zero,
            "sample_image_ce": zeros,
            "sample_text_ce": zeros,
            "image_token_count": zero_count,
            "text_token_count": zero_count.clone(),
        }

    flat_hidden = shift_hidden.reshape(-1, shift_hidden.size(-1))
    flat_labels = shift_labels.reshape(-1)
    selected_hidden = flat_hidden.index_select(0, valid_positions)
    selected_labels = flat_labels.index_select(0, valid_positions)
    selected_batches = torch.div(valid_positions, sequence_length, rounding_mode="floor")
    selected_image = selected_labels.ge(int(visual_token_offset))
    selected_text = ~selected_image

    image_counts = torch.bincount(
        selected_batches[selected_image], minlength=batch_size
    )
    text_counts = torch.bincount(
        selected_batches[selected_text], minlength=batch_size
    )
    image_sums = hidden_states.new_zeros(batch_size, dtype=torch.float32)
    text_sums = hidden_states.new_zeros(batch_size, dtype=torch.float32)

    def chunk_cross_entropy(
        chunk_hidden: torch.Tensor, chunk_labels: torch.Tensor
    ) -> torch.Tensor:
        # U0's reference forward promotes logits to FP32 before CE. Preserve
        # that numerical contract without materializing [B,S,V] logits.
        return F.cross_entropy(
            output_head(chunk_hidden).float(),
            chunk_labels,
            reduction="none",
        )

    for start in range(0, int(valid_positions.numel()), int(chunk_tokens)):
        end = min(start + int(chunk_tokens), int(valid_positions.numel()))
        chunk_hidden = selected_hidden[start:end]
        chunk_labels = selected_labels[start:end]
        if torch.is_grad_enabled() and chunk_hidden.requires_grad:
            # Checkpoint the head and CE together so each chunk's vocabulary
            # logits are discarded after forward and recomputed in backward.
            losses = checkpoint(
                chunk_cross_entropy,
                chunk_hidden,
                chunk_labels,
                use_reentrant=False,
            ).float()
        else:
            losses = chunk_cross_entropy(chunk_hidden, chunk_labels).float()
        chunk_batches = selected_batches[start:end]
        chunk_image = selected_image[start:end]
        chunk_text = selected_text[start:end]
        image_sums = image_sums.index_add(
            0, chunk_batches[chunk_image], losses[chunk_image]
        )
        text_sums = text_sums.index_add(
            0, chunk_batches[chunk_text], losses[chunk_text]
        )

    sample_image_ce = image_sums / image_counts.clamp(min=1).float()
    sample_text_ce = text_sums / text_counts.clamp(min=1).float()
    sample_loss = (
        float(image_loss_weight) * sample_image_ce
        + float(text_loss_weight) * sample_text_ce
    )
    image_ce = image_sums.sum() / image_counts.sum().clamp(min=1).float()
    text_ce = text_sums.sum() / text_counts.sum().clamp(min=1).float()
    loss = float(image_loss_weight) * image_ce + float(text_loss_weight) * text_ce
    return {
        "loss": loss,
        "loss_sum": loss * batch_size,
        "sample_loss": sample_loss,
        "image_ce": image_ce,
        "text_ce": text_ce,
        "sample_image_ce": sample_image_ce,
        "sample_text_ce": sample_text_ce,
        "image_token_count": image_counts.sum(),
        "text_token_count": text_counts.sum(),
    }


def _find_transformer_layers(model: nn.Module) -> tuple[type, ...]:
    for path in ("model.layers", "transformer.h", "model.decoder.layers", "decoder.layers"):
        module: Any = model
        for attribute in path.split("."):
            module = getattr(module, attribute, None)
            if module is None:
                break
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            return (type(module[0]),)
    raise ValueError(
        "Unable to find transformer layers. Configure a model adapter for the checkpoint architecture."
    )


def _dropout_probability(config: Any, name: str, default: float) -> float:
    value = float(config.get(name, default))
    if not 0.0 <= value < 1.0:
        raise ValueError(f"model.llm.{name} must be in [0, 1), got {value}")
    return value


def _set_hidden_dropout(model: nn.Module, probability: float) -> None:
    decoder = model.get_decoder()
    embedding_dropout = getattr(decoder, "dropout", None)
    layers = getattr(decoder, "layers", None)
    if not isinstance(embedding_dropout, nn.Dropout) or not isinstance(layers, nn.ModuleList):
        raise ValueError(
            "llm.hidden_dropout requires a decoder with dropout and layers attributes"
        )
    residual_dropouts = [getattr(layer, "dropout", None) for layer in layers]
    if not residual_dropouts or not all(
        isinstance(dropout, nn.Dropout) for dropout in residual_dropouts
    ):
        raise ValueError(
            "llm.hidden_dropout requires every decoder layer to expose residual dropout"
        )
    embedding_dropout.p = probability
    for dropout in residual_dropouts:
        dropout.p = probability


def _load_model_config(checkpoint: str) -> tuple[Any, type[nn.Module] | None]:
    config_dict, _ = PretrainedConfig.get_config_dict(
        checkpoint,
        local_files_only=True,
    )
    if str(config_dict.get("model_type", "")).lower() == "unis":
        from wm_fsdp.models.configuration_unis import UNISConfig
        from wm_fsdp.models.modeling_unis import UNISForCausalLM

        return (
            UNISConfig.from_pretrained(checkpoint, local_files_only=True),
            UNISForCausalLM,
        )
    return (
        AutoConfig.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            local_files_only=True,
        ),
        None,
    )


class CausalLMModel(nn.Module):
    """FSDP-facing wrapper for a Hugging Face causal language model."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        llm_config = config.llm
        checkpoint = str(llm_config.checkpoint)
        tokenizer_path = str(llm_config.tokenizer_path)
        self.tokenizer = load_u0_tokenizer(
            tokenizer_path,
            llm_config.get("special_tokens_file", None),
        )

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        attention_implementation = llm_config.get("attention_implementation", None)
        if attention_implementation:
            load_kwargs["attn_implementation"] = str(attention_implementation)
        dtype_name = llm_config.get("torch_dtype", None)
        if dtype_name:
            dtype = getattr(torch, str(dtype_name), None)
            if dtype is None:
                raise ValueError(f"Unsupported llm.torch_dtype: {dtype_name}")
            load_kwargs["dtype"] = dtype

        model_config, local_model_class = _load_model_config(checkpoint)
        attention_dropout = _dropout_probability(
            llm_config,
            "attention_dropout",
            float(getattr(model_config, "attention_dropout", 0.0)),
        )
        hidden_dropout = _dropout_probability(
            llm_config,
            "hidden_dropout",
            attention_dropout,
        )
        model_config.attention_dropout = attention_dropout
        model_config.hidden_dropout = hidden_dropout
        if attention_implementation:
            model_config._attn_implementation = str(attention_implementation)
        load_kwargs["config"] = model_config

        if bool(llm_config.get("init_on_meta", False)):
            with torch.device("meta"):
                if local_model_class is None:
                    config_kwargs: dict[str, Any] = {"trust_remote_code": True}
                    if attention_implementation:
                        config_kwargs["attn_implementation"] = str(attention_implementation)
                    self.llm_model = AutoModelForCausalLM.from_config(
                        model_config,
                        **config_kwargs,
                    )
                else:
                    self.llm_model = local_model_class(model_config)
        else:
            if local_model_class is None:
                self.llm_model = AutoModelForCausalLM.from_pretrained(checkpoint, **load_kwargs)
            else:
                local_load_kwargs = dict(load_kwargs)
                local_load_kwargs.pop("trust_remote_code", None)
                self.llm_model = local_model_class.from_pretrained(
                    checkpoint,
                    **local_load_kwargs,
                )
        self.llm_config = self.llm_model.config
        _set_hidden_dropout(self.llm_model, hidden_dropout)
        decoder = self.llm_model.get_decoder()
        output_head = self.llm_model.get_output_embeddings()
        if not isinstance(decoder, nn.Module) or not isinstance(output_head, nn.Module):
            raise TypeError("Causal LM must expose decoder and output embeddings modules")
        compute_dtype = load_kwargs.get("dtype")
        if compute_dtype in {torch.float16, torch.bfloat16}:
            # U0 RMSNorm can promote activations to the FP32 master-weight
            # dtype. Its FlashAttention adapter uses this field to restore the
            # configured compute dtype before entering the fused kernel.
            self.llm_config._pre_quantization_dtype = compute_dtype
        self.visual_token_offset = int(
            llm_config.get("visual_token_offset", getattr(self.llm_config, "visual_token_offset", 151854))
        )
        self.visual_vocab_size = int(llm_config.get("visual_vocab_size", 131072))
        loss_config = config.get("loss", {})
        self.image_loss_weight = float(loss_config.get("image_weight", 1.0))
        self.text_loss_weight = float(loss_config.get("text_weight", 1.0))
        self.loss_chunk_tokens = int(loss_config.get("chunk_tokens", 256))
        if self.image_loss_weight <= 0.0 or self.text_loss_weight <= 0.0:
            raise ValueError("model.loss image_weight and text_weight must be positive")
        if self.loss_chunk_tokens <= 0:
            raise ValueError("model.loss.chunk_tokens must be positive")
        if self.visual_token_offset + self.visual_vocab_size != int(self.llm_config.vocab_size):
            raise ValueError(
                "U0 visual vocabulary does not match the checkpoint: "
                f"offset={self.visual_token_offset} size={self.visual_vocab_size} "
                f"model_vocab={self.llm_config.vocab_size}"
            )
        self.fsdp_transformer_layer_cls = _find_transformer_layers(self.llm_model)

    def enable_ulysses(self) -> None:
        enable_u0_ulysses(self)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        labels: torch.LongTensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        outputs = self.llm_model.get_decoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        return selective_grouped_multimodal_cross_entropy(
            outputs[0],
            labels,
            self.llm_model.get_output_embeddings(),
            visual_token_offset=self.visual_token_offset,
            image_loss_weight=self.image_loss_weight,
            text_loss_weight=self.text_loss_weight,
            chunk_tokens=self.loss_chunk_tokens,
        )


__all__ = [
    "CausalLMModel",
    "grouped_multimodal_cross_entropy",
    "selective_grouped_multimodal_cross_entropy",
]
