"""Optional logits dump support for Xiaomi-Robotics-U0-FlashAR vLLM regression checks."""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch


class LogitsDumpStore:
    """Per-process sparse logits dump aggregator.

    When ``XR_U0_FLASHAR_DUMP_LOGITS`` is set, each diagonal sampling step stores
    the raw conditional and unconditional logits top-k, sampled token ids, and
    final sparse sampling distribution. The dump can be teacher-forced through
    an eager implementation for high-precision regression analysis.
    """

    def __init__(self):
        self.records: dict[str, list[dict]] = {}
        self.metadata: dict[str, dict] = {}
        self.dump_dir: Optional[str] = None
        self.topk: int = int(os.environ.get("XR_U0_FLASHAR_DUMP_LOGITS_TOPK", "2048"))

    def is_active(self) -> bool:
        return bool(self.dump_dir)

    def configure(self, dump_dir: str, topk: int) -> None:
        self.dump_dir = dump_dir
        self.topk = topk
        os.makedirs(dump_dir, exist_ok=True)

    def register(self, parent_id: str, manifest: dict) -> None:
        self.records.setdefault(parent_id, [])
        self.metadata[parent_id] = manifest

    def record_step(
        self,
        parent_id: str,
        diag_idx: int,
        step_positions: torch.Tensor,
        cond_logits: torch.Tensor,
        uncond_logits: torch.Tensor,
        sampled: torch.Tensor,
        final_probs: Optional[torch.Tensor] = None,
    ) -> None:
        if parent_id not in self.records:
            return
        topk = min(self.topk, cond_logits.size(-1))
        cond_topk_vals, cond_topk_idx = torch.topk(cond_logits.float(), topk, dim=-1)
        uncond_topk_vals, uncond_topk_idx = torch.topk(uncond_logits.float(), topk, dim=-1)
        rec = {
            "diag_idx": int(diag_idx),
            "step_positions": step_positions.detach().to("cpu").to(torch.int32).numpy(),
            "cond_topk_vals": cond_topk_vals.detach().to("cpu").to(torch.float16).numpy(),
            "cond_topk_idx": cond_topk_idx.detach().to("cpu").to(torch.int32).numpy(),
            "uncond_topk_vals": uncond_topk_vals.detach().to("cpu").to(torch.float16).numpy(),
            "uncond_topk_idx": uncond_topk_idx.detach().to("cpu").to(torch.int32).numpy(),
            "sampled": sampled.detach().to("cpu").to(torch.int32).numpy(),
        }
        if final_probs is not None:
            fp = final_probs.detach().float()
            nz = fp > 0
            fp_idx_list, fp_val_list = [], []
            for row in range(fp.size(0)):
                idx = nz[row].nonzero(as_tuple=False).view(-1)
                fp_idx_list.append(idx.to("cpu").to(torch.int32).numpy())
                fp_val_list.append(fp[row, idx].to("cpu").to(torch.float32).numpy())
            rec["final_probs_idx"] = np.array(fp_idx_list, dtype=object)
            rec["final_probs_val"] = np.array(fp_val_list, dtype=object)
        self.records[parent_id].append(rec)

    def flush(self) -> list[str]:
        """Write each parent's records to disk and return written directories."""
        if not self.dump_dir:
            return []
        written: list[str] = []
        for parent_id, recs in self.records.items():
            if not recs:
                continue
            pdir = os.path.join(self.dump_dir, parent_id)
            os.makedirs(pdir, exist_ok=True)
            save_kwargs = dict(
                diag_idxs=np.array([r["diag_idx"] for r in recs], dtype=np.int32),
                step_positions=np.array([r["step_positions"] for r in recs], dtype=object),
                cond_topk_vals=np.array([r["cond_topk_vals"] for r in recs], dtype=object),
                cond_topk_idx=np.array([r["cond_topk_idx"] for r in recs], dtype=object),
                uncond_topk_vals=np.array([r["uncond_topk_vals"] for r in recs], dtype=object),
                uncond_topk_idx=np.array([r["uncond_topk_idx"] for r in recs], dtype=object),
                sampled=np.array([r["sampled"] for r in recs], dtype=object),
            )
            if all("final_probs_idx" in r for r in recs):
                save_kwargs["final_probs_idx"] = np.array(
                    [r["final_probs_idx"] for r in recs], dtype=object)
                save_kwargs["final_probs_val"] = np.array(
                    [r["final_probs_val"] for r in recs], dtype=object)
            np.savez_compressed(os.path.join(pdir, "dump.npz"), **save_kwargs)
            with open(os.path.join(pdir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(self.metadata.get(parent_id, {}), f, indent=2)
            written.append(pdir)
        self.records = {}
        self.metadata = {}
        return written


LOGITS_DUMP = LogitsDumpStore()
_dump_dir = os.environ.get("XR_U0_FLASHAR_DUMP_LOGITS", "")
if _dump_dir:
    LOGITS_DUMP.configure(_dump_dir, LOGITS_DUMP.topk)
