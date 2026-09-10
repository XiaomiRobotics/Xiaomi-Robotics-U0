from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import torch

from wm_fsdp.train.ulysses import (
    gather_sequence,
    get_ulysses_state,
    head_to_sequence,
    pad_sequence_for_ulysses,
    sequence_to_head,
    shard_position_ids,
    shard_sequence,
)


@dataclass
class _ForwardState:
    original_length: int = 0
    padded_length: int = 0


_ADAPTER_MARKER = "_wm_fsdp_ulysses_enabled"


def _modeling_symbol(module: torch.nn.Module, name: str) -> Any:
    modeling_module = __import__(module.__class__.__module__, fromlist=[name])
    symbol = getattr(modeling_module, name, None)
    if symbol is None:
        raise TypeError(
            f"Ulysses adapter requires {name} in {module.__class__.__module__}"
        )
    return symbol


def _validate_u0_ulysses_model(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.ModuleList]:
    """Validate the remote-code contract before mutating any model module."""

    state = get_ulysses_state()
    llm = getattr(model, "llm_model", model)
    decoder = getattr(llm, "model", None)
    layers = getattr(decoder, "layers", None)
    if decoder is None or not isinstance(layers, torch.nn.ModuleList) or not layers:
        raise TypeError("Ulysses adapter requires a U0 model with model.layers")
    if not isinstance(getattr(decoder, "norm", None), torch.nn.Module):
        raise TypeError("Ulysses adapter requires a U0 model with model.norm")

    config = getattr(llm, "config", None)
    if config is None:
        raise TypeError("Ulysses adapter requires a model config")
    if str(getattr(config, "_attn_implementation", "")) != "flash_attention_2":
        raise ValueError("Ulysses currently requires attention_implementation=flash_attention_2")
    if int(getattr(config, "pretraining_tp", 1)) != 1:
        raise ValueError("Ulysses currently requires pretraining_tp=1")

    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError(
            f"query_heads={query_heads} must be a positive multiple of kv_heads={kv_heads}"
        )
    if query_heads % state.size or kv_heads % state.size:
        raise ValueError(
            f"ulysses_size={state.size} must divide query_heads={query_heads} and kv_heads={kv_heads}"
        )

    required_attributes = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "rotary_emb",
        "num_heads",
        "num_key_value_heads",
        "head_dim",
    )
    for layer_index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        missing = [name for name in required_attributes if not hasattr(attention, name)]
        if attention is None or missing:
            raise TypeError(
                f"Ulysses layer {layer_index} is not a compatible U0 attention; missing={missing}"
            )
        if not hasattr(attention, "_flash_attention_forward"):
            raise TypeError("Ulysses adapter requires U0 FlashAttention decoder layers")
        if int(attention.num_heads) != query_heads or int(attention.num_key_value_heads) != kv_heads:
            raise ValueError(
                f"Ulysses layer {layer_index} head counts disagree with the model config"
            )
        for symbol_name in (
            "apply_rotary_pos_emb",
            "flash_attn_func",
            "flash_attn_varlen_func",
            "_get_unpad_data",
            "index_first_axis",
            "pad_input",
        ):
            _modeling_symbol(attention, symbol_name)
    return llm, decoder, layers


def _padded_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    original_length: int,
    padded_length: int,
    device: torch.device,
) -> torch.Tensor | None:
    if attention_mask is None:
        if original_length == padded_length:
            return None
        attention_mask = torch.ones(
            (batch_size, original_length), dtype=torch.long, device=device
        )
    if attention_mask.ndim != 2 or attention_mask.shape != (batch_size, original_length):
        raise ValueError(
            "Ulysses U0 expects a full 2-D padding mask with shape "
            f"{(batch_size, original_length)}, got {tuple(attention_mask.shape)}"
        )
    attention_mask = attention_mask.to(device=device, dtype=torch.long)
    if padded_length > original_length:
        attention_mask = torch.cat(
            [
                attention_mask,
                attention_mask.new_zeros(batch_size, padded_length - original_length),
            ],
            dim=1,
        )
    return attention_mask


def enable_u0_ulysses(model: torch.nn.Module) -> None:
    """Patch a loaded U0 model for Ulysses sequence parallel training.

    The adapter keeps decoder residual/MLP activations sequence-sharded. Q/K/V
    use differentiable all-to-all exchanges inside attention, and the final
    hidden state is gathered once before the language-model head.
    """

    state = get_ulysses_state()
    if not state.enabled:
        return
    llm = getattr(model, "llm_model", model)
    enabled_size = getattr(llm, _ADAPTER_MARKER, None)
    if enabled_size is not None:
        if int(enabled_size) != state.size:
            raise RuntimeError(
                f"U0 model is already patched for ulysses_size={enabled_size}, "
                f"cannot reconfigure it for ulysses_size={state.size}"
            )
        return
    llm, decoder, layers = _validate_u0_ulysses_model(model)
    config = llm.config
    reference_attention = layers[0].self_attn
    apply_rotary_pos_emb = _modeling_symbol(reference_attention, "apply_rotary_pos_emb")
    flash_attn_func = _modeling_symbol(reference_attention, "flash_attn_func")
    flash_attn_varlen_func = _modeling_symbol(reference_attention, "flash_attn_varlen_func")
    get_unpad_data = _modeling_symbol(reference_attention, "_get_unpad_data")
    index_first_axis = _modeling_symbol(reference_attention, "index_first_axis")
    pad_input = _modeling_symbol(reference_attention, "pad_input")

    runtime = _ForwardState()

    def shard_decoder_input(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        del module
        if args:
            hidden_states = args[0]
        else:
            hidden_states = kwargs.get("hidden_states")
        if not torch.is_tensor(hidden_states):
            raise ValueError("U0 decoder layer did not receive tensor hidden_states")
        position_ids = args[2] if len(args) > 2 else kwargs.get("position_ids")
        padded, padded_positions, original_length, padded_length = pad_sequence_for_ulysses(
            hidden_states, position_ids
        )
        runtime.original_length = original_length
        runtime.padded_length = padded_length
        local_positions = shard_position_ids(padded_positions, padded_length)
        updated_args = list(args)
        if updated_args:
            updated_args[0] = shard_sequence(padded)
        else:
            kwargs["hidden_states"] = shard_sequence(padded)
        if len(updated_args) > 2:
            updated_args[2] = local_positions
        else:
            kwargs["position_ids"] = local_positions
        return tuple(updated_args), kwargs

    def gather_decoder_output(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        del module, args
        if runtime.original_length <= 0:
            raise RuntimeError("Ulysses decoder output was reached before its input was sharded")
        return gather_sequence(output, runtime.original_length)

    for layer in layers:
        attention = getattr(layer, "self_attn", None)

        def ulysses_attention_forward(
            self: torch.nn.Module,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            past_key_value: Any = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, None, None]:
            del kwargs
            if output_attentions or use_cache or past_key_value is not None:
                raise ValueError("Ulysses training does not support attention outputs or KV cache")
            if runtime.padded_length <= 0:
                raise RuntimeError("Ulysses attention was reached before sequence sharding")
            batch_size, local_length, _ = hidden_states.shape
            if position_ids is not None and position_ids.size(1) != local_length:
                if position_ids.size(1) != runtime.original_length:
                    raise ValueError(
                        "Ulysses position_ids must be either full or local length, got "
                        f"{position_ids.size(1)} for full={runtime.original_length} local={local_length}"
                    )
                if runtime.padded_length > runtime.original_length:
                    position_ids = torch.cat(
                        [
                            position_ids,
                            position_ids.new_zeros(
                                position_ids.size(0),
                                runtime.padded_length - runtime.original_length,
                            ),
                        ],
                        dim=1,
                    )
                position_ids = shard_position_ids(position_ids, runtime.padded_length)

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            query_states = self.q_norm(
                query_states.view(batch_size, local_length, self.num_heads, self.head_dim)
            ).transpose(1, 2)
            key_states = self.k_norm(
                key_states.view(batch_size, local_length, self.num_key_value_heads, self.head_dim)
            ).transpose(1, 2)
            value_states = value_states.view(
                batch_size, local_length, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            compute_dtype = getattr(config, "_pre_quantization_dtype", self.q_proj.weight.dtype)
            if compute_dtype in {torch.float16, torch.bfloat16}:
                query_states = query_states.to(compute_dtype)
                key_states = key_states.to(compute_dtype)
                value_states = value_states.to(compute_dtype)

            rotary_length = runtime.padded_length
            if position_ids is not None:
                rotary_length = max(rotary_length, int(position_ids.max().item()) + 1)
            cos, sin = self.rotary_emb(value_states, seq_len=rotary_length)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

            query_states = sequence_to_head(query_states.contiguous()).transpose(1, 2)
            key_states = sequence_to_head(key_states.contiguous()).transpose(1, 2)
            value_states = sequence_to_head(value_states.contiguous()).transpose(1, 2)
            full_mask = _padded_attention_mask(
                attention_mask,
                batch_size=batch_size,
                original_length=runtime.original_length,
                padded_length=runtime.padded_length,
                device=hidden_states.device,
            )
            dropout = self.attention_dropout if self.training else 0.0
            if full_mask is None:
                attention_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    causal=True,
                )
            else:
                indices, cumulative_lengths, max_length = get_unpad_data(full_mask)
                query_heads_per_rank = int(query_states.size(2))
                kv_heads_per_rank = int(key_states.size(2))
                query_unpadded = index_first_axis(
                    query_states.reshape(-1, query_heads_per_rank, self.head_dim),
                    indices,
                )
                key_unpadded = index_first_axis(
                    key_states.reshape(-1, kv_heads_per_rank, self.head_dim),
                    indices,
                )
                value_unpadded = index_first_axis(
                    value_states.reshape(-1, kv_heads_per_rank, self.head_dim),
                    indices,
                )
                output_unpadded = flash_attn_varlen_func(
                    query_unpadded,
                    key_unpadded,
                    value_unpadded,
                    cu_seqlens_q=cumulative_lengths,
                    cu_seqlens_k=cumulative_lengths,
                    max_seqlen_q=max_length,
                    max_seqlen_k=max_length,
                    dropout_p=dropout,
                    causal=True,
                )
                attention_output = pad_input(
                    output_unpadded,
                    indices,
                    batch_size,
                    runtime.padded_length,
                )
            attention_output = head_to_sequence(
                attention_output.transpose(1, 2).contiguous()
            )
            attention_output = attention_output.transpose(1, 2).reshape(
                batch_size, local_length, -1
            )
            return self.o_proj(attention_output), None, None

        attention.forward = types.MethodType(ulysses_attention_forward, attention)

    # Register hooks only after every decoder layer has passed validation and
    # received its replacement, so an incompatible remote-code model is never
    # left partially hooked.
    layers[0].register_forward_pre_hook(shard_decoder_input, with_kwargs=True)
    decoder.norm.register_forward_hook(gather_decoder_output)
    setattr(llm, _ADAPTER_MARKER, state.size)


__all__ = ["enable_u0_ulysses"]
