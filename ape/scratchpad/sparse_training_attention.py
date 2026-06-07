"""Block-sparse training attention for scratchpad SFT.

The intended training pattern is APE-like:

- prefix attends causally to prefix
- each document attends causally to prefix + itself
- query/scratchpad/answer attends causally to prefix + all documents + itself

The flash backend below implements that by issuing separate FlashAttention
calls per contiguous block. It currently supports Qwen2/Qwen3 attention modules,
which is the default model family in the scratchpad config.
"""

from __future__ import annotations

import types
from typing import Any

import torch

from .rendering import SEGMENT_DOC_START, SEGMENT_PAD, SEGMENT_PREFIX, SEGMENT_SUFFIX

try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        apply_rotary_pos_emb,
        repeat_kv,
    )
except Exception:  # pragma: no cover - optional dependency
    Qwen2Attention = None
    apply_rotary_pos_emb = None
    repeat_kv = None

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    if apply_rotary_pos_emb is None or repeat_kv is None:
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb,
            repeat_kv,
        )
except Exception:  # pragma: no cover - optional dependency
    Qwen3Attention = None

QWEN_ATTENTION_CLASSES = tuple(cls for cls in (Qwen2Attention, Qwen3Attention) if cls is not None)


def _flash_attn_func():
    try:
        from flash_attn import flash_attn_func
    except Exception as exc:  # pragma: no cover - depends on local CUDA build
        raise RuntimeError(
            "train.sparse_attention_backend=flash_block requires a working flash-attn install. "
            "Use train.sparse_attention_backend=sdpa_mask for the correctness fallback. "
            f"flash-attn import error: {exc}"
        ) from exc
    return flash_attn_func


def _shape_qkv(self: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    num_heads = query_states.shape[-1] // self.head_dim
    num_key_value_heads = key_states.shape[-1] // self.head_dim
    query_shape = (bsz, q_len, num_heads, self.head_dim)
    key_value_shape = (bsz, q_len, num_key_value_heads, self.head_dim)
    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states.view(query_shape))
    else:
        query_states = query_states.view(query_shape)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states.view(key_value_shape))
    else:
        key_states = key_states.view(key_value_shape)
    value_states = value_states.view(key_value_shape)
    return query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2)


def _apply_rope(query_states, key_states, position_embeddings):
    cos, sin = position_embeddings
    try:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)
    except TypeError:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, None)


def _contiguous_span(mask: torch.Tensor) -> tuple[int, int] | None:
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        return None
    start = int(indices[0].item())
    end = int(indices[-1].item()) + 1
    return start, end


def _flash_block(
    flash_attn_func,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout_p: float,
    softmax_scale: float | None,
) -> torch.Tensor:
    return flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
    )


def _block_sparse_flash_attention(
    self: Any,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    flash_attn_func = _flash_attn_func()
    num_key_value_groups = getattr(
        self,
        "num_key_value_groups",
        max(1, query_states.shape[1] // max(key_states.shape[1], 1)),
    )
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = repeat_kv(key_states, num_key_value_groups).transpose(1, 2).contiguous()
    value_states = repeat_kv(value_states, num_key_value_groups).transpose(1, 2).contiguous()

    batch, seq_len, _, _ = query_states.shape
    output = torch.zeros_like(query_states)
    dropout_p = float(getattr(self, "attention_dropout", 0.0)) if self.training else 0.0
    softmax_scale = getattr(self, "scaling", None)

    for batch_idx in range(batch):
        seg = segment_ids[batch_idx]
        real_span = _contiguous_span(seg != SEGMENT_PAD)
        if real_span is None:
            continue
        real_start, real_end = real_span
        prefix_span = _contiguous_span(seg == SEGMENT_PREFIX)

        if prefix_span is not None:
            start, end = prefix_span
            output[batch_idx : batch_idx + 1, start:end] = _flash_block(
                flash_attn_func,
                query_states[batch_idx : batch_idx + 1, start:end],
                key_states[batch_idx : batch_idx + 1, start:end],
                value_states[batch_idx : batch_idx + 1, start:end],
                causal=True,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
            )

        prefix_k = None
        prefix_v = None
        if prefix_span is not None:
            prefix_start, prefix_end = prefix_span
            prefix_k = key_states[batch_idx : batch_idx + 1, prefix_start:prefix_end]
            prefix_v = value_states[batch_idx : batch_idx + 1, prefix_start:prefix_end]

        doc_ids = torch.unique(seg[(seg >= SEGMENT_DOC_START) & (seg != SEGMENT_PAD)]).tolist()
        for doc_id in sorted(int(doc_id) for doc_id in doc_ids):
            doc_span = _contiguous_span(seg == doc_id)
            if doc_span is None:
                continue
            start, end = doc_span
            doc_k = key_states[batch_idx : batch_idx + 1, start:end]
            doc_v = value_states[batch_idx : batch_idx + 1, start:end]
            if prefix_k is not None:
                block_k = torch.cat([prefix_k, doc_k], dim=1)
                block_v = torch.cat([prefix_v, doc_v], dim=1)
            else:
                block_k = doc_k
                block_v = doc_v
            output[batch_idx : batch_idx + 1, start:end] = _flash_block(
                flash_attn_func,
                query_states[batch_idx : batch_idx + 1, start:end],
                block_k,
                block_v,
                causal=True,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
            )

        suffix_span = _contiguous_span(seg == SEGMENT_SUFFIX)
        if suffix_span is not None:
            start, end = suffix_span
            output[batch_idx : batch_idx + 1, start:end] = _flash_block(
                flash_attn_func,
                query_states[batch_idx : batch_idx + 1, start:end],
                key_states[batch_idx : batch_idx + 1, real_start:real_end],
                value_states[batch_idx : batch_idx + 1, real_start:real_end],
                causal=True,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
            )
    return output


def qwen_block_sparse_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs,
):
    segment_ids = kwargs.get("segment_ids")
    if segment_ids is None or past_key_values is not None:
        return self._ape_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    query_states, key_states, value_states = _shape_qkv(self, hidden_states)
    query_states, key_states = _apply_rope(query_states, key_states, position_embeddings)
    attn_output = _block_sparse_flash_attention(
        self,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        segment_ids=segment_ids.to(hidden_states.device),
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def install_qwen_block_sparse_attention(model: torch.nn.Module) -> int:
    if not QWEN_ATTENTION_CLASSES:
        raise ImportError("Qwen2/Qwen3 attention classes are not available in this Transformers install")
    installed = 0
    for module in model.modules():
        if isinstance(module, QWEN_ATTENTION_CLASSES):
            if not hasattr(module, "_ape_original_forward"):
                module._ape_original_forward = module.forward
            module.forward = types.MethodType(qwen_block_sparse_attention_forward, module)
            installed += 1
    return installed
