# -*- coding: utf-8 -*-
"""
Xiaomi-Robotics-U0-FlashAR wrapper.

Combines proximity attention masking with horizontal/vertical prediction heads
to enable diagonal-step parallel decoding of image tokens.

Usage:
    model = UNISFlashAR(backbone.model, vocab_size=..., hidden_size=...)
    # Training:
    out = model(input_ids=image_ids, height=H, width=W, text_input_ids=text_ids)
    loss = out["loss"]
    # Generation:
    grid = model.generate(height=H, width=W, device=device, text_input_ids=text_ids)
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache


# ============================================================================
# Sampling helpers
# ============================================================================

def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def _sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    sample_logits: bool = True,
) -> torch.Tensor:
    idx = []
    for i in range(logits.shape[1]):
        one_logits = logits[:, i, :] / max(temperature, 1e-5)
        if top_k > 0 or top_p < 1.0:
            one_logits = _top_k_top_p_filtering(one_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(one_logits, dim=-1)
        if sample_logits:
            one_idx = torch.multinomial(probs, num_samples=1)
        else:
            _, one_idx = torch.topk(probs, k=1, dim=-1)
        idx.append(one_idx.view(-1, 1))
    return torch.cat(idx, dim=-1)


# ============================================================================
# Proximity attention mask
# ============================================================================

def _build_step_id(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Manhattan-distance step id for each position in the H×W grid."""
    rows = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
    cols = torch.arange(width, device=device).unsqueeze(0).expand(height, width)
    return (rows + cols).reshape(-1)


def _build_proximity_allow(
    height: int, width: int, device: torch.device
) -> torch.Tensor:
    """Boolean allow matrix: allow[q, k] = True iff k can be attended by q."""
    total = height * width
    allow = torch.zeros((total, total), device=device, dtype=torch.bool)
    previous: List[int] = []
    for c in range(height + width - 1):
        current: List[int] = []
        for h in range(height):
            w = c - h
            if 0 <= w < width:
                idx = h * width + w
                current.append(idx)
                previous.append(idx)
        for idx in current:
            allow[idx, previous] = True
    return allow


def build_t2i_flashar_mask(
    prefix_ids: Optional[torch.Tensor],
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    text_attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a 4-D mask for pure T2I:
      - prefix (text/special tokens) uses causal mask
      - image tokens attend all valid prefix tokens
      - image block uses diagonal-step proximity mask
    """
    if prefix_ids is None:
        prefix_ids = torch.empty((batch_size, 0), device=device, dtype=torch.long)
    if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.unsqueeze(0)
    prefix_len = prefix_ids.size(1)

    t_steps = height * width
    total = prefix_len + t_steps
    attn = torch.full(
        (batch_size, 1, total, total), float("-inf"), device=device, dtype=dtype
    )

    if prefix_len > 0:
        causal = torch.tril(
            torch.ones(prefix_len, prefix_len, device=device, dtype=torch.bool)
        )
        diag = torch.eye(prefix_len, device=device, dtype=torch.bool)
        for b in range(batch_size):
            allow = causal.clone()
            if text_attention_mask is not None:
                valid = text_attention_mask[b].to(device=device).bool()
                allow &= valid.unsqueeze(0) & valid.unsqueeze(1)
                allow |= diag
                attn[b, 0, prefix_len:, :prefix_len] = torch.where(
                    valid.unsqueeze(0), 0.0, float("-inf")
                )
            else:
                attn[b, 0, prefix_len:, :prefix_len] = 0.0
            attn[b, 0, :prefix_len, :prefix_len].masked_fill_(allow, 0.0)

    step_id = _build_step_id(height, width, device)
    allowed = _build_proximity_allow(height, width, device)
    attn[:, 0, prefix_len:, prefix_len:].masked_fill_(allowed, 0.0)
    return attn, step_id


def build_t2i_causal_mask(
    prefix_ids: Optional[torch.Tensor],
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    text_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build a standard AR causal 4-D mask for T2I sequence:
      [text_prefix][image_tokens]
    """
    if prefix_ids is None:
        prefix_ids = torch.empty((batch_size, 0), device=device, dtype=torch.long)
    if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.unsqueeze(0)
    prefix_len = prefix_ids.size(1)
    image_len = height * width
    total = prefix_len + image_len

    causal = torch.tril(torch.ones((total, total), device=device, dtype=torch.bool))
    eye = torch.eye(total, device=device, dtype=torch.bool)
    attn = torch.full((batch_size, 1, total, total), float("-inf"), device=device, dtype=dtype)
    image_valid = torch.ones((image_len,), device=device, dtype=torch.bool)
    for b in range(batch_size):
        if prefix_len > 0 and text_attention_mask is not None:
            prefix_valid = text_attention_mask[b, :prefix_len].to(device=device).bool()
        else:
            prefix_valid = torch.ones((prefix_len,), device=device, dtype=torch.bool)
        valid = torch.cat([prefix_valid, image_valid], dim=0)
        allow = causal & valid.unsqueeze(0) & valid.unsqueeze(1)
        allow |= eye
        attn[b, 0].masked_fill_(allow, 0.0)
    return attn


# ============================================================================
# UNISFlashAR
# ============================================================================

class UNISFlashAR(nn.Module):
    """
    Wrap an AR backbone for Xiaomi-Robotics-U0-FlashAR diagonal-step decoding.

    Architecture:
        backbone[:vertical_start_layer]      → shared hidden
        backbone[vertical_start_layer:]      → horizontal_head → h_logits (right neighbor)
        [vertical_block from same split]     → vertical_head   → v_logits (down neighbor)
        fused_logits = shift-and-merge(h_logits, v_logits)
    """

    def __init__(
        self,
        pretrained_backbone: nn.Module,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int = -100,
        mask_token_id: Optional[int] = None,
        visual_token_offset: Optional[int] = None,
        use_vertical_block: bool = True,
        vertical_layers: int = 1,
        vertical_start_layer: int = -1,
        learnable_fuse: bool = False,
        lm_head: Optional[nn.Linear] = None,
        split_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = pretrained_backbone
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = (
            mask_token_id if mask_token_id is not None else pad_token_id
        )
        self.visual_token_offset = visual_token_offset
        self.use_vertical_block = use_vertical_block
        self.learnable_fuse = bool(learnable_fuse)

        cfg = getattr(pretrained_backbone, "config", None)
        backbone_layers = getattr(pretrained_backbone, "layers", None)
        self.backbone_num_layers = int(len(backbone_layers)) if backbone_layers is not None else 0
        self.vertical_layers = max(1, int(vertical_layers)) if use_vertical_block else 0
        self.vertical_start_layer = self.backbone_num_layers
        if use_vertical_block:
            if backbone_layers is None or self.backbone_num_layers <= 0:
                raise ValueError("pretrained_backbone.layers is required for vertical_block.")
            if vertical_start_layer < 0:
                vertical_start_layer = self.backbone_num_layers - self.vertical_layers
            self.vertical_start_layer = int(vertical_start_layer)
            if self.vertical_start_layer < 0:
                raise ValueError(
                    f"vertical_start_layer must be >= 0, got {self.vertical_start_layer}."
                )
            if self.vertical_start_layer + self.vertical_layers > self.backbone_num_layers:
                raise ValueError(
                    "vertical_start_layer + vertical_layers exceeds backbone depth. "
                    f"start={self.vertical_start_layer} layers={self.vertical_layers} "
                    f"backbone_num_layers={self.backbone_num_layers}."
                )

        self.split_backbone = bool(split_backbone) and use_vertical_block and (
            self.vertical_start_layer < self.backbone_num_layers
        )

        # --- prediction heads ---
        self.horizontal_head = nn.Linear(hidden_size, vocab_size, bias=False)

        if use_vertical_block:
            # Reuse Xiaomi-Robotics-U0 transformer parameters by cloning decoder layers from backbone.
            self.vertical_block = nn.ModuleList(
                [
                    copy.deepcopy(layer)
                    for layer in backbone_layers[
                        self.vertical_start_layer : self.vertical_start_layer + self.vertical_layers
                    ]
                ]
            )
            for layer in self.vertical_block:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "is_causal"):
                    layer.self_attn.is_causal = False
            rms_eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
            from xr_u0_ar.modeling_unis import UNISRMSNorm

            self.vertical_norm = UNISRMSNorm(hidden_size, eps=rms_eps)
        else:
            self.vertical_block = None
            self.vertical_norm = None

        self.vertical_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Per-position hv gate on SAME target-position features (train/inference consistent).
        gate_proj_dim = max(64, hidden_size // 8)
        self.hv_gate_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, gate_proj_dim, bias=False),
            nn.SiLU(),
            nn.Linear(gate_proj_dim, 1, bias=True),
        )
        self.hv_gate_corner = nn.Linear(hidden_size, 1, bias=True)
        # Start from symmetric mix (sigmoid(0)=0.5).
        nn.init.zeros_(self.hv_gate_mlp[-1].weight)
        nn.init.zeros_(self.hv_gate_mlp[-1].bias)
        nn.init.zeros_(self.hv_gate_corner.weight)
        nn.init.zeros_(self.hv_gate_corner.bias)

        # initialise heads from lm_head if provided
        if lm_head is not None:
            with torch.no_grad():
                self.horizontal_head.weight.copy_(lm_head.weight)
                self.vertical_head.weight.copy_(lm_head.weight)

        self._sync_dtype_with_backbone()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sync_dtype_with_backbone(self) -> None:
        try:
            backbone_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            return
        self.horizontal_head.to(dtype=backbone_dtype)
        if self.vertical_block is not None:
            self.vertical_block.to(dtype=backbone_dtype)
        if self.vertical_norm is not None:
            self.vertical_norm.to(dtype=backbone_dtype)
        self.vertical_head.to(dtype=backbone_dtype)

    def _vertical_capture_layer(self) -> Optional[int]:
        if self.vertical_block is None:
            return None
        if self.vertical_start_layer >= self.backbone_num_layers:
            return None
        return int(self.vertical_start_layer)

    def _reshape_grid(
        self, seq: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        bsz, seq_len = seq.shape[:2]
        if seq_len != height * width:
            raise ValueError(f"Expected seq_len={height * width}, got {seq_len}")
        return seq.view(bsz, height, width, -1)

    def _hv_gate_from_pair(
        self,
        h_feat: torch.Tensor,
        v_feat: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        if h_feat.shape != v_feat.shape:
            raise ValueError("h_feat and v_feat must share shape.")
        if h_feat.numel() == 0:
            shape = (*h_feat.shape[:-1], 1)
            return torch.full(shape, 0.5, device=h_feat.device, dtype=out_dtype)
        gate_dtype = self.hv_gate_mlp[0].weight.dtype
        feat = torch.cat([h_feat, v_feat], dim=-1).to(gate_dtype)
        gate = torch.sigmoid(self.hv_gate_mlp(feat))
        return gate.to(dtype=out_dtype)

    def _hv_gate_corner(
        self,
        cond_hidden: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        gate_dtype = self.hv_gate_corner.weight.dtype
        gate = torch.sigmoid(self.hv_gate_corner(cond_hidden.to(gate_dtype)))
        return gate.to(dtype=out_dtype)

    @staticmethod
    def _binary_gate_entropy(prob: torch.Tensor) -> torch.Tensor:
        prob = prob.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
        return entropy / 0.6931471805599453

    @staticmethod
    def _build_step_mask_2d(step_id: torch.Tensor) -> torch.Tensor:
        """Bool mask where True means blocked attention."""
        allowed = step_id[None, :] <= step_id[:, None]
        return ~allowed

    @staticmethod
    def _build_step_attn_4d(
        step_mask_2d: torch.Tensor, batch_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        seq_len = int(step_mask_2d.size(0))
        mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len),
            device=step_mask_2d.device,
            dtype=dtype,
        )
        if step_mask_2d.any():
            mask_val = torch.finfo(dtype).min
            mask.masked_fill_(step_mask_2d.view(1, 1, seq_len, seq_len), mask_val)
        return mask

    def _fuse_logits(
        self,
        h_logits: torch.Tensor,
        v_logits: torch.Tensor,
        cond_h_logits: torch.Tensor,
        cond_v_logits: torch.Tensor,
        h_hidden: torch.Tensor,
        v_hidden: torch.Tensor,
        cond_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Shift-and-merge horizontal / vertical logits into per-position predictions.

        h_logits[b, h, w] predicts position (h, w+1);  after roll → position (h, w).
        v_logits[b, h, w] predicts position (h+1, w);  after roll → position (h, w).
        """
        rw_corner = self._hv_gate_corner(cond_hidden, out_dtype=h_logits.dtype)  # [B,1,1]
        cond_logits = (
            rw_corner[:, 0, :] * cond_h_logits[:, 0, :]
            + (1.0 - rw_corner[:, 0, :]) * cond_v_logits[:, 0, :]
        )
        h_shift = h_logits.roll(shifts=1, dims=2)   # prediction for right neighbor
        v_shift = v_logits.roll(shifts=1, dims=1)   # prediction for bottom neighbor
        fused = torch.zeros_like(v_shift)
        fused[:, 0, 0, :] = cond_logits              # corner: hv_gate
        fused[:, 0, 1:, :] = h_shift[:, 0, 1:, :]   # first row: horizontal only
        fused[:, 1:, 0, :] = v_shift[:, 1:, 0, :]   # first col: vertical only
        if h_shift.size(1) > 1 and h_shift.size(2) > 1:
            h_feat_shift = h_hidden.roll(shifts=1, dims=2)
            v_feat_shift = v_hidden.roll(shifts=1, dims=1)
            rw_interior = self._hv_gate_from_pair(
                h_feat_shift[:, 1:, 1:, :],
                v_feat_shift[:, 1:, 1:, :],
                out_dtype=h_logits.dtype,
            )
            fused[:, 1:, 1:, :] = (
                rw_interior * h_shift[:, 1:, 1:, :]
                + (1.0 - rw_interior) * v_shift[:, 1:, 1:, :]
            )
            gate_mean = rw_interior.mean().detach().reshape(())
            gate_reg = self._hv_gate_from_pair(
                h_feat_shift[:, 1:, 1:, :].detach(),
                v_feat_shift[:, 1:, 1:, :].detach(),
                out_dtype=h_logits.dtype,
            )
            gate_entropy = self._binary_gate_entropy(gate_reg).mean().reshape(())
        else:
            gate_mean = rw_corner.mean().detach().reshape(())
            gate_reg = self._hv_gate_corner(
                cond_hidden.detach(),
                out_dtype=h_logits.dtype,
            )
            gate_entropy = self._binary_gate_entropy(gate_reg).mean().reshape(())
        corner_mean = rw_corner.mean().detach().reshape(())
        gate_stats = {
            "hv_gate_h": gate_mean,
            "hv_gate_v": (1.0 - gate_mean).detach().reshape(()),
            "hv_gate_h_corner": corner_mean,
            "hv_gate_v_corner": (1.0 - corner_mean).detach().reshape(()),
            "hv_gate_entropy": gate_entropy.detach().reshape(()),
            "loss_gate_collapse": (1.0 - gate_entropy).reshape(()),
        }
        return fused, gate_stats

    def _apply_vertical_block(
        self,
        image_hidden: torch.Tensor,
        step_mask_2d: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        if self.vertical_block is not None:
            bsz, seq_len, _ = image_hidden.shape
            attn_mask = self._build_step_attn_4d(step_mask_2d, bsz, image_hidden.dtype)
            position_ids = (
                torch.arange(seq_len, device=image_hidden.device, dtype=torch.long)
                + int(position_offset)
            ).unsqueeze(0).expand(bsz, -1)
            v_hidden = image_hidden
            # torch.utils.checkpoint needs at least one grad-requiring input.
            # During vertical-only warmup the frozen backbone feeds detached
            # hidden states into this block, so forcing checkpoint here would
            # silently drop gradients for vertical_block parameters.
            use_checkpoint = (
                bool(getattr(self.backbone, "gradient_checkpointing", False))
                and self.training
                and image_hidden.requires_grad
            )
            for layer in self.vertical_block:
                if use_checkpoint:
                    def _layer_forward(hidden_states: torch.Tensor, _layer: nn.Module = layer) -> torch.Tensor:
                        return _layer(
                            hidden_states=hidden_states,
                            attention_mask=attn_mask,
                            position_ids=position_ids,
                            output_attentions=False,
                            use_cache=False,
                        )[0]

                    v_hidden = torch.utils.checkpoint.checkpoint(
                        _layer_forward,
                        v_hidden,
                        use_reentrant=False,
                    )
                else:
                    v_hidden = layer(
                        hidden_states=v_hidden,
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False,
                    )[0]
            return self.vertical_norm(v_hidden)
        return image_hidden

    def _compute_logits(
        self,
        cond_horizontal_hidden: torch.Tensor,
        cond_vertical_hidden: torch.Tensor,
        horizontal_hidden: torch.Tensor,
        vertical_input_hidden: torch.Tensor,
        height: int,
        width: int,
        step_mask_2d: torch.Tensor,
        position_offset: int = 0,
    ) -> Dict[str, torch.Tensor]:
        cond_h_logits = self.horizontal_head(cond_horizontal_hidden)
        cond_v_logits = self.vertical_head(cond_vertical_hidden)
        h_grid = self._reshape_grid(horizontal_hidden, height, width)
        h_logits = self.horizontal_head(h_grid)

        v_hidden = self._apply_vertical_block(
            vertical_input_hidden,
            step_mask_2d,
            position_offset=position_offset,
        )
        v_grid = self._reshape_grid(v_hidden, height, width)
        v_logits = self.vertical_head(v_grid)

        fused, gate_stats = self._fuse_logits(
            h_logits,
            v_logits,
            cond_h_logits,
            cond_v_logits,
            h_grid,
            v_grid,
            cond_horizontal_hidden,
        )
        return {"fused": fused, "h_logits": h_logits, "v_logits": v_logits, **gate_stats}

    def _run_backbone_remaining_layers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run backbone layers [vertical_start_layer:] on already-computed hidden states."""
        backbone_layers = self.backbone.layers
        use_checkpoint = (
            bool(getattr(self.backbone, "gradient_checkpointing", False))
            and self.training
        )
        for layer_idx in range(self.vertical_start_layer, self.backbone_num_layers):
            layer = backbone_layers[layer_idx]
            if use_checkpoint:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    False,
                    False,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                    use_cache=False,
                )
            hidden_states = layer_outputs[0]
        hidden_states = self.backbone.norm(hidden_states)
        return hidden_states

    def _run_backbone(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.split_backbone:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
                stop_after_layer=self.vertical_start_layer,
            )
            shared_hidden = outputs.last_hidden_state
            last_hidden = self._run_backbone_remaining_layers(
                shared_hidden, attention_mask,
            )
            return last_hidden, shared_hidden

        capture_layer = self._vertical_capture_layer()
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
            capture_hidden_before_layer=capture_layer,
        )
        last_hidden = outputs.last_hidden_state
        vertical_input_hidden = (
            outputs.captured_hidden_state if capture_layer is not None else last_hidden
        )
        if vertical_input_hidden is None:
            raise RuntimeError(
                "backbone did not return captured_hidden_state for vertical branch."
            )
        return last_hidden, vertical_input_hidden

    def _split_hidden_states(
        self,
        hidden: torch.Tensor,
        prefix_len: int,
        text_attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split backbone hidden states into:
          - cond_hidden: one prefix state per sample used for (0, 0)
          - image_hidden: H*W image token states

        When text prefix is padded, use the last valid prefix token per sample
        instead of the padded tail position.
        """
        if prefix_len <= 0:
            return hidden[:, :1, :], hidden

        prefix_hidden = hidden[:, :prefix_len, :]
        image_hidden = hidden[:, prefix_len:, :]
        if text_attention_mask is None:
            return prefix_hidden[:, -1:, :], image_hidden

        if text_attention_mask.dim() == 1:
            text_attention_mask = text_attention_mask.unsqueeze(0)
        if text_attention_mask.size(0) != hidden.size(0):
            raise ValueError("text_attention_mask batch size mismatch.")
        if text_attention_mask.size(1) < prefix_len:
            raise ValueError("text_attention_mask length is smaller than prefix length.")

        valid_lens = text_attention_mask[:, :prefix_len].to(device=hidden.device).long().sum(dim=1)
        valid_lens = valid_lens.clamp(min=1, max=prefix_len)
        gather_idx = (valid_lens - 1).view(-1, 1, 1).expand(-1, 1, prefix_hidden.size(-1))
        cond_hidden = prefix_hidden.gather(dim=1, index=gather_idx)
        return cond_hidden, image_hidden

    def _cross_entropy_4d(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        chunked_loss: bool,
        chunk_rows: int = 4,
    ) -> torch.Tensor:
        """
        Cross entropy over [B, H, W, V] logits with optional row chunking.
        Chunked mode reduces peak activation memory in loss computation.
        """
        if not chunked_loss:
            return F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=self.pad_token_id,
            )

        height = int(logits.size(1))
        loss_sum = logits.new_zeros(())
        token_count = 0
        chunk_rows = max(1, int(chunk_rows))
        for row_start in range(0, height, chunk_rows):
            row_end = min(row_start + chunk_rows, height)
            chunk_logits = logits[:, row_start:row_end, :, :].reshape(-1, self.vocab_size)
            chunk_targets = targets[:, row_start:row_end, :].reshape(-1)
            valid = chunk_targets.ne(self.pad_token_id)
            valid_tokens = int(valid.sum().item())
            if valid_tokens == 0:
                continue
            chunk_loss = F.cross_entropy(
                chunk_logits,
                chunk_targets,
                ignore_index=self.pad_token_id,
                reduction="sum",
            )
            loss_sum = loss_sum + chunk_loss
            token_count += valid_tokens

        if token_count == 0:
            return logits.new_zeros(())
        return loss_sum / float(token_count)

    # ------------------------------------------------------------------
    # Mask helpers (public, for external callers)
    # ------------------------------------------------------------------

    def build_mask(
        self,
        image_ids: torch.Tensor,
        height: int,
        width: int,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Build full input ids, prefix length, 4-D attention mask, and step mask.
        """
        mask_dtype = next(self.backbone.parameters()).dtype
        if text_input_ids is not None:
            if text_input_ids.dim() == 1:
                text_input_ids = text_input_ids.unsqueeze(0)
            prefix_len = text_input_ids.size(1)
            full_input_ids = torch.cat([text_input_ids, image_ids], dim=1)
            attn_mask, step_id = build_t2i_flashar_mask(
                prefix_ids=text_input_ids,
                height=height,
                width=width,
                batch_size=image_ids.size(0),
                device=image_ids.device,
                dtype=mask_dtype,
                text_attention_mask=text_attention_mask,
            )
        else:
            prefix_len = 0
            full_input_ids = image_ids
            attn_mask, step_id = build_t2i_flashar_mask(
                prefix_ids=None,
                height=height,
                width=width,
                batch_size=image_ids.size(0),
                device=image_ids.device,
                dtype=mask_dtype,
            )
        step_mask_2d = self._build_step_mask_2d(step_id)
        return full_input_ids, prefix_len, attn_mask, step_mask_2d

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        height: int,
        width: int,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        step_positions: Optional[torch.Tensor] = None,
        prev_positions: Optional[torch.Tensor] = None,
        chunked_loss: bool = False,
        ar_distill: bool = False,
        ar_distill_temperature: float = 2.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute fused Xiaomi-Robotics-U0-FlashAR loss.

        Args:
            input_ids:  (B, H*W)  flattened image token ids.
            height, width: spatial dimensions of the image grid.
            text_input_ids:  (B, T) optional text prefix.
            text_attention_mask: (B, T) optional padding mask for prefix.
            step_positions: 1-D positions to predict (generation mode).
            prev_positions: 1-D positions whose hidden states are used.
        """
        bsz = input_ids.size(0)

        # --- build mask ---
        full_input_ids, prefix_len, attn_mask, step_mask_2d = self.build_mask(
            image_ids=input_ids,
            height=height,
            width=width,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
        )

        # --- backbone forward ---
        hidden, vertical_input_hidden = self._run_backbone(full_input_ids, attn_mask)
        cond_horizontal_hidden, image_hidden = self._split_hidden_states(
            hidden=hidden,
            prefix_len=prefix_len,
            text_attention_mask=text_attention_mask,
        )
        cond_vertical_hidden, vertical_image_hidden = self._split_hidden_states(
            hidden=vertical_input_hidden,
            prefix_len=prefix_len,
            text_attention_mask=text_attention_mask,
        )

        # === step-based generation path ===
        if step_positions is not None and prev_positions is not None:
            return self._forward_step(
                cond_horizontal_hidden,
                cond_vertical_hidden,
                image_hidden,
                vertical_image_hidden,
                step_mask_2d,
                step_positions,
                prev_positions,
                height,
                width,
                input_ids.device,
                prefix_len,
            )

        # === full loss path ===
        outputs = self._forward_full(
            cond_horizontal_hidden,
            cond_vertical_hidden,
            image_hidden,
            vertical_image_hidden,
            step_mask_2d,
            input_ids,
            height,
            width,
            prefix_len=prefix_len,
            chunked_loss=chunked_loss,
        )
        if ar_distill:
            outputs["loss_distill"] = self._compute_ar_distill_loss(
                full_input_ids=full_input_ids,
                prefix_len=prefix_len,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                student_v_logits=outputs["logits_v"],
                height=height,
                width=width,
                temperature=ar_distill_temperature,
            )
        return outputs

    # ------------------------------------------------------------------
    # Forward sub-paths
    # ------------------------------------------------------------------

    def _forward_step(
        self,
        cond_horizontal_hidden: torch.Tensor,
        cond_vertical_hidden: torch.Tensor,
        horizontal_hidden: torch.Tensor,
        vertical_input_hidden: torch.Tensor,
        step_mask_2d: torch.Tensor,
        step_positions: torch.Tensor,
        prev_positions: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        position_offset: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Per-step logits for generation (no loss)."""
        bsz = horizontal_hidden.size(0)
        if step_positions.dim() != 1 or prev_positions.dim() != 1:
            raise ValueError("step_positions and prev_positions must be 1-D.")
        if step_positions.numel() == 0:
            return {
                "step_logits": torch.empty(
                    (bsz, 0, self.vocab_size), device=device
                )
            }

        v_hidden = self._apply_vertical_block(
            vertical_input_hidden,
            step_mask_2d,
            position_offset=position_offset,
        )
        cond_h_logits = self.horizontal_head(cond_horizontal_hidden)
        cond_v_logits = self.vertical_head(cond_vertical_hidden)

        h_prev = self.horizontal_head(horizontal_hidden[:, prev_positions, :])
        v_prev = self.vertical_head(v_hidden[:, prev_positions, :])
        h_prev_feat = horizontal_hidden[:, prev_positions, :]
        v_prev_feat = v_hidden[:, prev_positions, :]

        total = height * width
        pos_to_idx = torch.full((total,), -1, device=device, dtype=torch.long)
        pos_to_idx[prev_positions] = torch.arange(
            prev_positions.numel(), device=device
        )

        rows = step_positions // width
        cols = step_positions % width
        step_logits = torch.empty(
            (bsz, step_positions.numel(), self.vocab_size),
            device=device,
            dtype=h_prev.dtype,
        )

        left_mask = cols > 0
        up_mask = rows > 0
        both_mask = left_mask & up_mask
        corner_mask = ~left_mask & ~up_mask

        # first-row non-corner: horizontal only
        h_only = left_mask & ~up_mask
        if h_only.any():
            left_idx = pos_to_idx[step_positions[h_only] - 1]
            step_logits[:, h_only, :] = h_prev[:, left_idx, :]

        # first-col non-corner: vertical only
        v_only = up_mask & ~left_mask
        if v_only.any():
            up_idx = pos_to_idx[step_positions[v_only] - width]
            step_logits[:, v_only, :] = v_prev[:, up_idx, :]

        # interior: per-position hv gate
        if both_mask.any():
            left_idx = pos_to_idx[step_positions[both_mask] - 1]
            up_idx = pos_to_idx[step_positions[both_mask] - width]
            rw = self._hv_gate_from_pair(
                h_prev_feat[:, left_idx, :],
                v_prev_feat[:, up_idx, :],
                out_dtype=h_prev.dtype,
            )
            step_logits[:, both_mask, :] = (
                rw * h_prev[:, left_idx, :] + (1.0 - rw) * v_prev[:, up_idx, :]
            )

        # corner (0,0): per-sample corner gate
        if corner_mask.any():
            rw_corner = self._hv_gate_corner(cond_horizontal_hidden, out_dtype=h_prev.dtype)
            cond_logits = rw_corner * cond_h_logits[:, :1, :] + (1.0 - rw_corner) * cond_v_logits[:, :1, :]
            step_logits[:, corner_mask, :] = cond_logits.expand(-1, int(corner_mask.sum().item()), -1)

        return {"step_logits": step_logits}

    def _forward_full(
        self,
        cond_horizontal_hidden: torch.Tensor,
        cond_vertical_hidden: torch.Tensor,
        horizontal_hidden: torch.Tensor,
        vertical_input_hidden: torch.Tensor,
        step_mask_2d: torch.Tensor,
        input_ids: torch.Tensor,
        height: int,
        width: int,
        prefix_len: int = 0,
        chunked_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Standard full-grid loss."""
        bsz = input_ids.size(0)
        device = input_ids.device
        logits = self._compute_logits(
            cond_horizontal_hidden,
            cond_vertical_hidden,
            horizontal_hidden,
            vertical_input_hidden,
            height,
            width,
            step_mask_2d,
            position_offset=prefix_len,
        )
        fused = logits["fused"]
        h_logits = logits["h_logits"]
        v_logits = logits["v_logits"]

        target_grid = input_ids.view(bsz, height, width)
        loss = self._cross_entropy_4d(
            fused, target_grid, chunked_loss=chunked_loss
        )

        if width > 1:
            loss_h = self._cross_entropy_4d(
                h_logits[:, :, :-1, :],
                target_grid[:, :, 1:],
                chunked_loss=chunked_loss,
            )
        else:
            loss_h = torch.tensor(0.0, device=device, dtype=loss.dtype)

        if height > 1:
            loss_v = self._cross_entropy_4d(
                v_logits[:, :-1, :, :],
                target_grid[:, 1:, :],
                chunked_loss=chunked_loss,
            )
        else:
            loss_v = torch.tensor(0.0, device=device, dtype=loss.dtype)

        return {
            "loss": loss,
            "loss_h": loss_h,
            "loss_v": loss_v,
            "logits_h": h_logits,
            "logits_v": v_logits,
            "logits": fused,
            "hv_gate_h": logits["hv_gate_h"],
            "hv_gate_v": logits["hv_gate_v"],
            "hv_gate_h_corner": logits["hv_gate_h_corner"],
            "hv_gate_v_corner": logits["hv_gate_v_corner"],
            "hv_gate_entropy": logits["hv_gate_entropy"],
            "loss_gate_collapse": logits["loss_gate_collapse"],
        }

    def _compute_ar_distill_loss(
        self,
        full_input_ids: torch.Tensor,
        prefix_len: int,
        text_input_ids: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor],
        student_v_logits: torch.Tensor,
        height: int,
        width: int,
        temperature: float,
    ) -> torch.Tensor:
        """
        AR distillation for vertical branch:
          student at (r,c) predicting (r+1,c) vs AR teacher at prev token (q-1).
        """
        bsz = int(full_input_ids.size(0))
        image_len = int(height * width)
        if height <= 1:
            return student_v_logits.new_zeros(())

        student_flat = student_v_logits[:, :-1, :, :].reshape(bsz, -1, self.vocab_size)
        q = torch.arange(width, image_len, device=full_input_ids.device, dtype=torch.long)

        if student_flat.size(1) != int(q.numel()):
            raise RuntimeError(
                f"Distill alignment mismatch: student={student_flat.size(1)} teacher_targets={int(q.numel())}"
            )

        mask_dtype = next(self.backbone.parameters()).dtype
        ar_attn = build_t2i_causal_mask(
            prefix_ids=text_input_ids,
            height=height,
            width=width,
            batch_size=bsz,
            device=full_input_ids.device,
            dtype=mask_dtype,
            text_attention_mask=text_attention_mask,
        )
        with torch.no_grad():
            ar_hidden, _ = self._run_backbone(full_input_ids, ar_attn)
        ar_image_hidden = ar_hidden[:, prefix_len:, :] if prefix_len > 0 else ar_hidden

        # teacher at previous token predicts current target q in AR factorization.
        prev_idx = q - 1

        temp = max(float(temperature), 1e-5)
        chunk = 64
        kl_sum = student_flat.new_zeros(())
        token_count = 0
        for start in range(0, int(prev_idx.numel()), chunk):
            end = min(start + chunk, int(prev_idx.numel()))
            idx = prev_idx[start:end]
            with torch.no_grad():
                teacher_logits = self.horizontal_head(ar_image_hidden[:, idx, :])
            student_logits = student_flat[:, start:end, :]
            log_p = F.log_softmax(student_logits.float() / temp, dim=-1)
            q_prob = F.softmax(teacher_logits.float() / temp, dim=-1)
            kl = F.kl_div(log_p, q_prob, reduction="none").sum(dim=-1)
            kl_sum = kl_sum + kl.sum()
            token_count += int(kl.numel())

        if token_count == 0:
            return student_v_logits.new_zeros(())
        return (kl_sum / float(token_count)) * (temp * temp)

    def _build_cfg_text_batch(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
        unconditional_text_input_ids: torch.Tensor,
        unconditional_text_attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if text_input_ids.dim() == 1:
            text_input_ids = text_input_ids.unsqueeze(0)
        if unconditional_text_input_ids.dim() == 1:
            unconditional_text_input_ids = unconditional_text_input_ids.unsqueeze(0)

        if text_attention_mask is None:
            text_attention_mask = torch.ones_like(text_input_ids, dtype=torch.long)
        elif text_attention_mask.dim() == 1:
            text_attention_mask = text_attention_mask.unsqueeze(0)
        if unconditional_text_attention_mask is None:
            unconditional_text_attention_mask = torch.ones_like(
                unconditional_text_input_ids, dtype=torch.long
            )
        elif unconditional_text_attention_mask.dim() == 1:
            unconditional_text_attention_mask = unconditional_text_attention_mask.unsqueeze(0)

        if text_input_ids.size(0) != unconditional_text_input_ids.size(0):
            raise ValueError("CFG text batch size mismatch between cond and uncond prompts.")

        max_len = max(text_input_ids.size(1), unconditional_text_input_ids.size(1))
        pad_id = int(self.mask_token_id)

        def _pad(ids: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            if ids.size(1) == max_len:
                return ids, mask
            padded_ids = ids.new_full((ids.size(0), max_len), pad_id)
            padded_mask = mask.new_zeros((mask.size(0), max_len))
            padded_ids[:, : ids.size(1)] = ids
            padded_mask[:, : mask.size(1)] = mask
            return padded_ids, padded_mask

        uncond_ids, uncond_mask = _pad(
            unconditional_text_input_ids,
            unconditional_text_attention_mask.to(dtype=torch.long),
        )
        cond_ids, cond_mask = _pad(
            text_input_ids,
            text_attention_mask.to(dtype=torch.long),
        )
        return (
            torch.cat([uncond_ids, cond_ids], dim=0),
            torch.cat([uncond_mask, cond_mask], dim=0),
        )

    @staticmethod
    def _normalize_text_batch(
        text_input_ids: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if text_input_ids is None:
            return None, None
        if text_input_ids.dim() == 1:
            text_input_ids = text_input_ids.unsqueeze(0)
        if text_attention_mask is None:
            text_attention_mask = torch.ones_like(text_input_ids, dtype=torch.long)
        elif text_attention_mask.dim() == 1:
            text_attention_mask = text_attention_mask.unsqueeze(0)
        return text_input_ids, text_attention_mask.to(
            device=text_input_ids.device, dtype=torch.long
        )

    @staticmethod
    def _gather_last_valid_hidden(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, -1:, :]
        valid_lens = attention_mask.to(device=hidden_states.device).long().sum(dim=1).clamp(
            min=1, max=hidden_states.size(1)
        )
        gather_idx = (valid_lens - 1).view(-1, 1, 1).expand(
            -1, 1, hidden_states.size(-1)
        )
        return hidden_states.gather(dim=1, index=gather_idx)

    @staticmethod
    def _build_kv_attention_mask(
        *,
        batch_size: int,
        current_len: int,
        past_len: int,
        device: torch.device,
        dtype: torch.dtype,
        prefix_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prefix_len = 0 if prefix_attention_mask is None else int(prefix_attention_mask.size(1))
        total_kv = prefix_len + past_len + current_len
        attn_mask = torch.zeros(
            (batch_size, 1, current_len, total_kv),
            device=device,
            dtype=dtype,
        )
        if prefix_len > 0:
            invalid_prefix = ~prefix_attention_mask.to(device=device).bool()
            if invalid_prefix.any():
                attn_mask[:, :, :, :prefix_len] = attn_mask[:, :, :, :prefix_len].masked_fill(
                    invalid_prefix[:, None, None, :],
                    torch.finfo(dtype).min,
                )
        return attn_mask

    @staticmethod
    def _build_kv_attention_mask_2d(
        *,
        batch_size: int,
        current_len: int,
        past_len: int,
        device: torch.device,
        prefix_attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        prefix_len = 0 if prefix_attention_mask is None else int(prefix_attention_mask.size(1))
        if prefix_len <= 0:
            return None
        prefix_attention_mask = prefix_attention_mask.to(device=device, dtype=torch.long)
        if bool(prefix_attention_mask.all()):
            return None
        total_kv = prefix_len + past_len + current_len
        attn_mask = torch.ones(
            (batch_size, total_kv),
            device=device,
            dtype=prefix_attention_mask.dtype,
        )
        attn_mask[:, :prefix_len] = prefix_attention_mask
        return attn_mask

    def _backbone_uses_flash_attention_2(self) -> bool:
        return bool(getattr(self.backbone, "_use_flash_attention_2", False))

    def _vertical_uses_flash_attention_2(self) -> bool:
        if self.vertical_block is None or len(self.vertical_block) == 0:
            return False
        first_attn = getattr(self.vertical_block[0], "self_attn", None)
        cfg = getattr(first_attn, "config", None)
        return bool(getattr(cfg, "_attn_implementation", "") == "flash_attention_2")

    @contextmanager
    def _temporary_backbone_non_causal(self):
        if not self._backbone_uses_flash_attention_2():
            yield
            return
        original_flags = []
        for layer in getattr(self.backbone, "layers", []):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None or not hasattr(self_attn, "is_causal"):
                continue
            original_flags.append((self_attn, bool(self_attn.is_causal)))
            self_attn.is_causal = False
        try:
            yield
        finally:
            for self_attn, is_causal in original_flags:
                self_attn.is_causal = is_causal

    def _prefill_generation_prefix(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, DynamicCache, int, torch.Tensor]:
        if self.split_backbone:
            shared_cache = DynamicCache()
            shared_outputs = self.backbone(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                past_key_values=shared_cache,
                use_cache=True,
                return_dict=True,
                stop_after_layer=self.vertical_start_layer,
            )
            shared_hidden = shared_outputs.last_hidden_state
            cond_vertical_hidden = self._gather_last_valid_hidden(
                shared_hidden, text_attention_mask,
            )
            remaining_cache = DynamicCache()
            batch_size, seq_len = text_input_ids.shape
            position_ids = torch.arange(seq_len, device=text_input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            h_hidden = shared_hidden
            for layer_idx in range(self.vertical_start_layer, self.backbone_num_layers):
                layer = self.backbone.layers[layer_idx]
                layer_outputs = layer(
                    h_hidden,
                    attention_mask=text_attention_mask if text_attention_mask.dim() <= 2 else None,
                    position_ids=position_ids,
                    past_key_value=remaining_cache,
                    output_attentions=False,
                    use_cache=True,
                )
                h_hidden = layer_outputs[0]
                remaining_cache = layer_outputs[1] if len(layer_outputs) > 1 else remaining_cache
            h_hidden = self.backbone.norm(h_hidden)
            cond_horizontal_hidden = self._gather_last_valid_hidden(
                h_hidden, text_attention_mask,
            )
            combined_cache = self._merge_caches(shared_outputs.past_key_values, remaining_cache)
            return (
                cond_horizontal_hidden,
                cond_vertical_hidden,
                combined_cache,
                int(text_input_ids.size(1)),
                text_attention_mask,
            )

        capture_layer = self._vertical_capture_layer()
        prefix_cache = DynamicCache()
        outputs = self.backbone(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            past_key_values=prefix_cache,
            use_cache=True,
            return_dict=True,
            capture_hidden_before_layer=capture_layer,
        )
        cond_horizontal_hidden = self._gather_last_valid_hidden(
            outputs.last_hidden_state,
            text_attention_mask,
        )
        vertical_prefix_hidden = (
            outputs.captured_hidden_state if capture_layer is not None else outputs.last_hidden_state
        )
        if vertical_prefix_hidden is None:
            raise RuntimeError(
                "backbone did not return captured_hidden_state for vertical prefix branch."
            )
        cond_vertical_hidden = self._gather_last_valid_hidden(
            vertical_prefix_hidden,
            text_attention_mask,
        )
        return (
            cond_horizontal_hidden,
            cond_vertical_hidden,
            outputs.past_key_values,
            int(text_input_ids.size(1)),
            text_attention_mask,
        )

    @staticmethod
    def _merge_caches(cache_a: DynamicCache, cache_b: DynamicCache) -> DynamicCache:
        """Merge two DynamicCaches (from shared layers and remaining layers) into one."""
        merged = DynamicCache()
        for layer_idx in range(len(cache_a)):
            k, v = cache_a[layer_idx]
            merged.update(k, v, layer_idx)
        offset = len(cache_a)
        for layer_idx in range(len(cache_b)):
            k, v = cache_b[layer_idx]
            merged.update(k, v, offset + layer_idx)
        return merged

    def _append_backbone_kv_step(
        self,
        step_token_ids: torch.Tensor,
        step_positions: torch.Tensor,
        prefix_len: int,
        prefix_attention_mask: Optional[torch.Tensor],
        past_key_values: DynamicCache,
        past_image_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, DynamicCache]:
        batch_size = int(step_token_ids.size(0))
        current_len = int(step_token_ids.size(1))

        if self._backbone_uses_flash_attention_2():
            attention_mask = self._build_kv_attention_mask_2d(
                batch_size=batch_size,
                current_len=current_len,
                past_len=past_image_len,
                device=step_token_ids.device,
                prefix_attention_mask=prefix_attention_mask,
            )
        else:
            mask_dtype = next(self.backbone.parameters()).dtype
            attention_mask = self._build_kv_attention_mask(
                batch_size=batch_size,
                current_len=current_len,
                past_len=past_image_len,
                device=step_token_ids.device,
                dtype=mask_dtype,
                prefix_attention_mask=prefix_attention_mask,
            )
        position_ids = (
            step_positions.to(device=step_token_ids.device, dtype=torch.long)
            + int(prefix_len)
        ).unsqueeze(0).expand(batch_size, -1)

        if self.split_backbone:
            with self._temporary_backbone_non_causal():
                shared_outputs = self.backbone(
                    input_ids=step_token_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    stop_after_layer=self.vertical_start_layer,
                )
            shared_hidden = shared_outputs.last_hidden_state
            remaining_cache = DynamicCache()
            num_shared_layers = self.vertical_start_layer
            for cache_layer_idx in range(num_shared_layers, len(past_key_values)):
                k, v = past_key_values[cache_layer_idx]
                remaining_cache.update(k, v, cache_layer_idx - num_shared_layers)

            remaining_attn = self._build_kv_attention_mask(
                batch_size=batch_size,
                current_len=current_len,
                past_len=past_image_len,
                device=step_token_ids.device,
                dtype=shared_hidden.dtype,
                prefix_attention_mask=prefix_attention_mask,
            ) if not self._backbone_uses_flash_attention_2() else attention_mask

            h_hidden = shared_hidden
            new_remaining_cache = DynamicCache()
            for rel_idx, layer_idx in enumerate(range(self.vertical_start_layer, self.backbone_num_layers)):
                layer = self.backbone.layers[layer_idx]
                layer_past = DynamicCache()
                if rel_idx < len(remaining_cache):
                    k, v = remaining_cache[rel_idx]
                    layer_past.update(k, v, 0)

                layer_outputs = layer(
                    h_hidden,
                    attention_mask=remaining_attn,
                    position_ids=position_ids,
                    past_key_value=new_remaining_cache,
                    output_attentions=False,
                    use_cache=True,
                )
                h_hidden = layer_outputs[0]
            h_hidden = self.backbone.norm(h_hidden)
            merged_cache = self._merge_caches(shared_outputs.past_key_values, new_remaining_cache)
            return h_hidden, shared_hidden, merged_cache

        capture_layer = self._vertical_capture_layer()
        with self._temporary_backbone_non_causal():
            outputs = self.backbone(
                input_ids=step_token_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                capture_hidden_before_layer=capture_layer,
            )
        current_h_hidden = outputs.last_hidden_state
        current_v_input_hidden = (
            outputs.captured_hidden_state if capture_layer is not None else current_h_hidden
        )
        if current_v_input_hidden is None:
            raise RuntimeError(
                "backbone did not return captured_hidden_state for vertical KV branch."
            )
        return current_h_hidden, current_v_input_hidden, outputs.past_key_values

    def _append_vertical_kv_step(
        self,
        step_hidden: torch.Tensor,
        step_positions: torch.Tensor,
        prefix_len: int,
        past_key_values: Optional[DynamicCache],
        past_image_len: int,
    ) -> Tuple[torch.Tensor, Optional[DynamicCache]]:
        if self.vertical_block is None:
            return step_hidden, past_key_values

        cache = past_key_values if past_key_values is not None else DynamicCache()
        batch_size = int(step_hidden.size(0))
        current_len = int(step_hidden.size(1))
        if self._vertical_uses_flash_attention_2():
            attention_mask = None
        else:
            attention_mask = self._build_kv_attention_mask(
                batch_size=batch_size,
                current_len=current_len,
                past_len=past_image_len,
                device=step_hidden.device,
                dtype=step_hidden.dtype,
            )
        position_ids = (
            step_positions.to(device=step_hidden.device, dtype=torch.long)
            + int(prefix_len)
        ).unsqueeze(0).expand(batch_size, -1)
        v_hidden = step_hidden
        for layer in self.vertical_block:
            layer_outputs = layer(
                hidden_states=v_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=cache,
                output_attentions=False,
                use_cache=True,
            )
            v_hidden = layer_outputs[0]
            cache = layer_outputs[1]
        return self.vertical_norm(v_hidden), cache

    def _compute_step_logits_from_prev(
        self,
        cond_horizontal_hidden: torch.Tensor,
        cond_vertical_hidden: torch.Tensor,
        prev_h_hidden: Optional[torch.Tensor],
        prev_v_hidden: Optional[torch.Tensor],
        step_positions: torch.Tensor,
        prev_positions: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = int(cond_horizontal_hidden.size(0))
        cond_h_logits = self.horizontal_head(cond_horizontal_hidden)
        cond_v_logits = self.vertical_head(cond_vertical_hidden)
        work_device = cond_h_logits.device
        if cond_v_logits.device != work_device:
            cond_v_logits = cond_v_logits.to(work_device)
        step_positions_work = step_positions.to(work_device)
        prev_positions_work = prev_positions.to(work_device)
        step_logits = torch.empty(
            (batch_size, step_positions.numel(), self.vocab_size),
            device=work_device,
            dtype=cond_h_logits.dtype,
        )

        rows = step_positions_work // width
        cols = step_positions_work % width
        left_mask = cols > 0
        up_mask = rows > 0
        both_mask = left_mask & up_mask
        corner_mask = ~left_mask & ~up_mask
        h_only = left_mask & ~up_mask
        v_only = up_mask & ~left_mask

        if prev_h_hidden is not None and prev_v_hidden is not None and prev_positions.numel() > 0:
            h_prev = self.horizontal_head(prev_h_hidden)
            v_prev = self.vertical_head(prev_v_hidden)
            if h_prev.device != work_device:
                h_prev = h_prev.to(work_device)
            if v_prev.device != work_device:
                v_prev = v_prev.to(work_device)
            prev_h_for_gate = prev_h_hidden.to(work_device)
            prev_v_for_gate = prev_v_hidden.to(work_device)
            total = int(height * width)
            pos_to_idx = torch.full((total,), -1, device=work_device, dtype=torch.long)
            pos_to_idx[prev_positions_work] = torch.arange(prev_positions.numel(), device=work_device)

            if h_only.any():
                left_idx = pos_to_idx[step_positions_work[h_only] - 1]
                step_logits[:, h_only, :] = h_prev[:, left_idx, :]

            if v_only.any():
                up_idx = pos_to_idx[step_positions_work[v_only] - width]
                step_logits[:, v_only, :] = v_prev[:, up_idx, :]

            if both_mask.any():
                left_idx = pos_to_idx[step_positions_work[both_mask] - 1]
                up_idx = pos_to_idx[step_positions_work[both_mask] - width]
                rw = self._hv_gate_from_pair(
                    prev_h_for_gate[:, left_idx, :],
                    prev_v_for_gate[:, up_idx, :],
                    out_dtype=h_prev.dtype,
                )
                if rw.device != work_device:
                    rw = rw.to(work_device)
                step_logits[:, both_mask, :] = (
                    rw * h_prev[:, left_idx, :] + (1.0 - rw) * v_prev[:, up_idx, :]
                )
        elif h_only.any() or v_only.any() or both_mask.any():
            raise RuntimeError("Previous diagonal hidden states are missing for non-corner prediction.")

        if corner_mask.any():
            rw_corner = self._hv_gate_corner(
                cond_horizontal_hidden.to(work_device), out_dtype=cond_h_logits.dtype
            )
            if rw_corner.device != work_device:
                rw_corner = rw_corner.to(work_device)
            cond_logits = (
                rw_corner * cond_h_logits[:, :1, :]
                + (1.0 - rw_corner) * cond_v_logits[:, :1, :]
            )
            step_logits[:, corner_mask, :] = cond_logits.expand(
                -1, int(corner_mask.sum().item()), -1
            )
        return step_logits

    def _generate_with_kv_cache(
        self,
        *,
        height: int,
        width: int,
        device: torch.device,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
        unconditional_text_input_ids: Optional[torch.Tensor],
        unconditional_text_attention_mask: Optional[torch.Tensor],
        cfg_scale: float,
        temperature: float,
        top_k: int,
        top_p: float,
        sample_logits: bool,
    ) -> torch.Tensor:
        step_id = _build_step_id(height, width, device)
        max_step = int(step_id.max().item())
        grid = torch.full(
            (1, height, width), self.mask_token_id, device=device, dtype=torch.long
        )

        text_input_ids, text_attention_mask = self._normalize_text_batch(
            text_input_ids,
            text_attention_mask,
        )
        if text_input_ids is None or text_attention_mask is None:
            raise ValueError("KV-cache generation requires text_input_ids.")

        use_cfg = cfg_scale > 1.0 and unconditional_text_input_ids is not None
        if use_cfg:
            kv_text_ids, kv_text_mask = self._build_cfg_text_batch(
                text_input_ids,
                text_attention_mask,
                unconditional_text_input_ids,
                unconditional_text_attention_mask,
            )
        else:
            kv_text_ids, kv_text_mask = text_input_ids, text_attention_mask

        cond_horizontal_hidden, cond_vertical_hidden, backbone_cache, prefix_len, prefix_mask = self._prefill_generation_prefix(
            kv_text_ids,
            kv_text_mask,
        )

        batch_size = int(kv_text_ids.size(0))
        vertical_cache = DynamicCache() if self.vertical_block is not None else None
        past_image_len = 0
        prev_positions = torch.empty((0,), device=device, dtype=torch.long)
        prev_h_hidden = None
        prev_v_hidden = None

        for step in range(0, max_step + 1):
            step_positions = (step_id == step).nonzero(as_tuple=False).view(-1).to(
                device=device, dtype=torch.long
            )
            if step_positions.numel() == 0:
                continue

            branch_step_logits = self._compute_step_logits_from_prev(
                cond_horizontal_hidden=cond_horizontal_hidden,
                cond_vertical_hidden=cond_vertical_hidden,
                prev_h_hidden=prev_h_hidden,
                prev_v_hidden=prev_v_hidden,
                step_positions=step_positions,
                prev_positions=prev_positions,
                height=height,
                width=width,
                device=device,
            )

            if use_cfg:
                u_step_logits, c_step_logits = branch_step_logits.chunk(2, dim=0)
                step_logits = u_step_logits + cfg_scale * (c_step_logits - u_step_logits)
            else:
                step_logits = branch_step_logits

            if self.visual_token_offset is not None:
                step_logits = step_logits.clone()
                step_logits[:, :, : self.visual_token_offset] = float("-inf")

            step_pred = _sample_logits(
                step_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample_logits=sample_logits,
            )
            grid.view(1, -1)[:, step_positions] = step_pred.to(grid.device)

            step_token_ids = step_pred.expand(batch_size, -1).contiguous()
            current_h_hidden, current_v_input_hidden, backbone_cache = self._append_backbone_kv_step(
                step_token_ids=step_token_ids,
                step_positions=step_positions,
                prefix_len=prefix_len,
                prefix_attention_mask=prefix_mask,
                past_key_values=backbone_cache,
                past_image_len=past_image_len,
            )
            current_v_hidden, vertical_cache = self._append_vertical_kv_step(
                step_hidden=current_v_input_hidden,
                step_positions=step_positions,
                prefix_len=prefix_len,
                past_key_values=vertical_cache,
                past_image_len=past_image_len,
            )

            prev_positions = step_positions
            prev_h_hidden = current_h_hidden
            prev_v_hidden = current_v_hidden
            past_image_len += int(step_positions.numel())
        return grid[0]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        height: int,
        width: int,
        device: torch.device,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        unconditional_text_input_ids: Optional[torch.Tensor] = None,
        unconditional_text_attention_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        sample_logits: bool = True,
    ) -> torch.Tensor:
        """
        Diagonal-step decoding.

        Returns:
            (H, W) int64 token grid.
        """
        if text_input_ids is None:
            raise ValueError(
                "UNISFlashAR.generate now requires text_input_ids and always uses KV-cache generation."
            )
        return self._generate_with_kv_cache(
            height=height,
            width=width,
            device=device,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            unconditional_text_input_ids=unconditional_text_input_ids,
            unconditional_text_attention_mask=unconditional_text_attention_mask,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_logits=sample_logits,
        )


__all__ = [
    "UNISFlashAR",
    "build_t2i_flashar_mask",
    "build_t2i_causal_mask",
    "_sample_logits",
]
