# -*- coding: utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from vllm import SamplingParams


def _build_inputs(input_ids, unconditional_ids):
    return {
        "prompt_token_ids": input_ids.tolist()[0],
        "uncond_prompt_token_ids": unconditional_ids.tolist()[0],
    }


def _build_sampling_params(cfg, tokenizer):
    cfg_decay = cfg.sampling_params.get("cfg_decay", "none")
    extra_args = {
        "guidance_scale": cfg.classifier_free_guidance,
        "text_top_k": cfg.sampling_params["text_top_k"],
        "text_top_p": cfg.sampling_params["text_top_p"],
        "text_temperature": cfg.sampling_params["text_temperature"],
        "visual_top_k": cfg.sampling_params["image_top_k"],
        "visual_top_p": cfg.sampling_params["image_top_p"],
        "visual_temperature": cfg.sampling_params["image_temperature"],
        "width": getattr(cfg, "target_width", None),
        "height": getattr(cfg, "target_height", None),
        "area": cfg.image_area if getattr(cfg, "target_width", None) else None,
        "cfg_decay": cfg_decay,
        "cfg_min": cfg.sampling_params.get("cfg_min", 1.0),
        "cfg_total_images": cfg.sampling_params.get("cfg_total_images", 1),
    }
    if cfg.task_type in ["t2i", "x2i"] or getattr(cfg, "stop_on_image_end", False):
        stop_token_ids = tokenizer.encode("<|image end|>")
    else:
        stop_token_ids = tokenizer.encode("<|extra_204|>")
    return SamplingParams(
        top_k=cfg.sampling_params["top_k"],
        top_p=cfg.sampling_params["top_p"],
        temperature=cfg.sampling_params["temperature"],
        max_tokens=cfg.sampling_params["max_new_tokens"],
        detokenize=False,
        extra_args=extra_args,
        stop_token_ids=stop_token_ids,
    )


@torch.no_grad()
def generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
):

    if getattr(cfg, "streaming", False):
        raise ValueError("Streaming generation is not supported in VLLM yet.")
    else:
        yield non_streaming_generate(
            cfg, model, tokenizer, input_ids, unconditional_ids,
        )


def non_streaming_generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
):
    inputs = _build_inputs(input_ids, unconditional_ids)
    sampling_params = _build_sampling_params(cfg, tokenizer)

    results = model.generate(inputs, sampling_params=sampling_params)
    gen_token_ids = np.array(results[0].outputs[0].token_ids)

    return gen_token_ids


def _select_parent_results(results, n_inputs: int):
    if len(results) == n_inputs:
        return results

    if len(results) == 2 * n_inputs:
        cond_by_parent = {}
        for result in results:
            request_id = str(getattr(result, "request_id", ""))
            if len(request_id) < 2 or request_id[0] not in {"0", "1"}:
                continue
            parent_id = request_id[1:]
            if not parent_id.isdigit():
                continue
            parent_index = int(parent_id)
            if parent_index >= n_inputs:
                continue
            if request_id[0] == "0":
                cond_by_parent[parent_index] = result
        if len(cond_by_parent) == n_inputs:
            return [cond_by_parent[index] for index in range(n_inputs)]

    result_ids = [str(getattr(result, "request_id", index)) for index, result in enumerate(results[:16])]
    raise RuntimeError(
        f"vLLM returned {len(results)} result(s) for {n_inputs} input(s); "
        f"sample request ids: {result_ids}"
    )


@torch.no_grad()
def generate_batch(
    cfg,
    model,
    tokenizer,
    input_ids_batch,
    unconditional_ids_batch,
):
    if len(input_ids_batch) != len(unconditional_ids_batch):
        raise ValueError("input_ids_batch and unconditional_ids_batch must have the same length")
    if not input_ids_batch:
        return []

    inputs = [
        _build_inputs(input_ids, unconditional_ids)
        for input_ids, unconditional_ids in zip(input_ids_batch, unconditional_ids_batch)
    ]
    sampling_params = _build_sampling_params(cfg, tokenizer)
    results = model.generate(inputs, sampling_params=sampling_params)
    results = _select_parent_results(results, len(inputs))

    outputs = []
    for index, result in enumerate(results):
        if not result.outputs:
            raise RuntimeError(f"vLLM result {index} has no outputs")
        outputs.append(np.array(result.outputs[0].token_ids))
    return outputs
