"""APE-style parallel-context generation for scratchpad evaluations."""

from __future__ import annotations

import math
from typing import Any

import torch

from ape import (
    enable_attention_prefill_context,
    enable_attention_prefill_prefix,
    enable_attention_prefill_query,
)
from .rendering import build_context_fields, build_prefix_field, build_query_field, scratchpad_text


def wrap_prefix(model_name: str, prefix_field: str) -> str:
    prefix_field = prefix_field.strip()
    if "llama" in model_name.lower():
        return f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n{prefix_field}\n\n"
    if "mistral" in model_name.lower():
        return f"<s>[INST]{prefix_field}\n\n"
    if "gemma" in model_name.lower():
        return f"<bos><start_of_turn>user\n{prefix_field}\n\n"
    return f"{prefix_field}\n\n" if prefix_field else ""


def wrap_query(model_name: str, query_field: str, scratchpad_tokens: list[str] | None = None) -> str:
    scratch = scratchpad_text(scratchpad_tokens or [])
    name = model_name.lower()
    if "llama" in name:
        return f"{query_field}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>{scratch}"
    if "mistral" in name:
        return f"{query_field}[/INST]{scratch}"
    if "gemma" in name:
        return f"{query_field}<end_of_turn>\n<start_of_turn>model\n{scratch}"
    return query_field + scratch


def context_strings(example: dict[str, Any]) -> list[str]:
    return build_context_fields(example)


@torch.no_grad()
def generate_with_ape(
    model: Any,
    tokenizer: Any,
    model_name: str,
    example: dict[str, Any],
    scratchpad_tokens: list[str] | None = None,
    temperature: float = 1.0,
    scale: float = 1.0,
    max_new_tokens: int = 64,
    max_context_tokens: int = 4096,
    query_position_shift: int = 0,
) -> str:
    """Generate with APE prefix/context/query prefill.

    This follows the original demo structure in the APE repo. The scaling
    arguments are inference-only, so training remains ordinary causal LM SFT.
    """

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prefix = wrap_prefix(model_name, build_prefix_field(example))
    query = wrap_query(
        model_name,
        build_query_field(example, scratchpad_tokens=None),
        scratchpad_tokens=scratchpad_tokens,
    )
    contexts = context_strings(example)
    if not contexts:
        contexts = [""]

    prefix_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
    query_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
    len_prefix = prefix_ids.shape[1]
    len_query = query_ids.shape[1]
    per_context_max = max(1, int(max_context_tokens) - len_prefix - len_query - int(max_new_tokens))
    per_context_max = max(1, math.floor(per_context_max / max(len(contexts), 1)))
    context_ids = tokenizer(
        contexts,
        return_tensors="pt",
        truncation=True,
        max_length=per_context_max,
        padding=True,
        add_special_tokens=False,
    ).input_ids
    context_mask_cpu = (context_ids != tokenizer.pad_token_id).reshape(-1)
    context_mask = context_mask_cpu.to(model.device)

    enable_attention_prefill_prefix(model_name, model)
    outputs = model(prefix_ids.to(model.device), past_key_values=None, use_cache=True)

    prefix_cache = []
    for layer_cache in outputs.past_key_values:
        bsz, _ = context_ids.shape
        prefix_cache.append(
            (
                layer_cache[0].repeat(bsz, 1, 1, 1),
                layer_cache[1].repeat(bsz, 1, 1, 1),
                layer_cache[2],
            )
        )

    enable_attention_prefill_context(model_name, model)
    outputs = model(context_ids.to(model.device), past_key_values=prefix_cache, use_cache=True)

    merged_cache = []
    for layer_cache in outputs.past_key_values:
        bsz = context_ids.shape[0]
        past_key = torch.cat(
            [
                layer_cache[0][:1, :, :len_prefix, :],
                layer_cache[0][:, :, len_prefix:, :].transpose(1, 2).flatten(0, 1)[context_mask].unsqueeze(0).transpose(1, 2),
            ],
            dim=2,
        )
        past_value = torch.cat(
            [
                layer_cache[1][:1, :, :len_prefix, :],
                layer_cache[1][:, :, len_prefix:, :].transpose(1, 2).flatten(0, 1)[context_mask].unsqueeze(0).transpose(1, 2),
            ],
            dim=2,
        )
        past_position = torch.cat(
            [
                layer_cache[2][:, :len_prefix],
                layer_cache[2][:, len_prefix:].repeat(bsz, 1).flatten()[context_mask].unsqueeze(0),
            ],
            dim=1,
        )
        merged_cache.append((past_key, past_value, past_position, len(contexts)))

    enable_attention_prefill_query(
        model_name,
        model,
        float(temperature),
        float(scale),
        position_shift=int(query_position_shift),
    )
    outputs = model(
        input_ids=query_ids.to(model.device),
        past_key_values=merged_cache,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    eos_token_id = tokenizer.eos_token_id
    generated = []
    for _ in range(int(max_new_tokens)):
        token_id = int(next_token[0, 0].item())
        if eos_token_id is not None and token_id == int(eos_token_id):
            break
        generated.append(next_token)
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    if not generated:
        return ""
    output = torch.cat(generated, dim=-1)[0]
    return tokenizer.decode(output, skip_special_tokens=True).strip()
