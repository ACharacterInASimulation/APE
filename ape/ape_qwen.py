"""APE attention patches for Qwen2/Qwen3-style decoder models.

This mirrors the tuple-cache implementation used by the original APE Llama
patches. It is intended for the same Transformers generation stack as the
existing APE repo; newer DynamicCache-only releases may require pinning
Transformers to the APE-compatible version.
"""

from __future__ import annotations

import math
import types
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        Qwen2Model,
        apply_rotary_pos_emb,
        repeat_kv,
    )
except Exception:  # pragma: no cover - optional model family
    Qwen2Attention = None
    Qwen2Model = None

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3Model
except Exception:  # pragma: no cover - optional model family
    Qwen3Attention = None
    Qwen3Model = None

from flash_attn import flash_attn_func


QWEN_ATTENTION_CLASSES = tuple(cls for cls in (Qwen2Attention, Qwen3Attention) if cls is not None)
QWEN_MODEL_CLASSES = tuple(cls for cls in (Qwen2Model, Qwen3Model) if cls is not None)


def _shape_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, q_len, _ = hidden_states.size()
    head_dim = self.head_dim
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    num_heads = query_states.shape[-1] // head_dim
    num_key_value_heads = key_states.shape[-1] // head_dim
    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states.view(bsz, q_len, num_heads, head_dim))
    else:
        query_states = query_states.view(bsz, q_len, num_heads, head_dim)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states.view(bsz, q_len, num_key_value_heads, head_dim))
    else:
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim)
    return query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2)


def _apply_rope(self, query_states, key_states, value_states, position_ids, position_embeddings=None):
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    try:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    except TypeError:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)


def qwen_attention_prefill_prefix(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    if past_key_value is None:
        past_key_value = kwargs.get("past_key_values", None)
    bsz, q_len, _ = hidden_states.size()
    query_states, key_states, value_states = _shape_qkv(self, hidden_states)
    query_states, key_states = _apply_rope(
        self, query_states, key_states, value_states, position_ids, position_embeddings
    )
    past_key_value = (key_states, value_states, position_ids) if use_cache else None
    self._ape_last_past_key_value = past_key_value
    self.len_prefix = q_len
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_output = flash_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        causal=q_len > 1,
    )
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None if not output_attentions else None


def qwen_attention_prefill_context(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    if past_key_value is None:
        past_key_value = kwargs.get("past_key_values", None)
    bsz, q_len, _ = hidden_states.size()
    query_states, key_states, value_states = _shape_qkv(self, hidden_states)
    query_states, key_states = _apply_rope(
        self, query_states, key_states, value_states, position_ids, position_embeddings
    )
    assert past_key_value is not None
    past_key, past_value, past_position = past_key_value
    key_states = torch.cat([past_key, key_states], dim=2)
    value_states = torch.cat([past_value, value_states], dim=2)
    position_states = torch.cat([past_position, position_ids], dim=-1)
    past_key_value = (key_states, value_states, position_states) if use_cache else None
    self._ape_last_past_key_value = past_key_value
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_output = flash_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        causal=q_len > 1,
    )
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None if not output_attentions else None


def qwen_attention_prefill_query(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    temperature: float = 1.0,
    scale: float = 1.0,
    position_shift: int = 0,
    **kwargs,
):
    if past_key_value is None:
        past_key_value = kwargs.get("past_key_values", None)
    bsz, q_len, _ = hidden_states.size()
    query_states, key_states, value_states = _shape_qkv(self, hidden_states)
    assert past_key_value is not None
    if len(past_key_value) == 4:
        past_key, past_value, past_position = past_key_value[0], past_key_value[1], past_key_value[2]
        current_position = past_position.max().item() + 1 + int(position_shift)
        self.len_context = past_key.shape[2] - self.len_prefix
    else:
        past_key, past_value, past_position = past_key_value
        current_position = past_position.max().item() + 1
    key_position_ids = position_ids - position_ids.min().item() + current_position
    query_states, key_states = _apply_rope(
        self, query_states, key_states, value_states, key_position_ids, position_embeddings
    )
    key_states = torch.cat([past_key, key_states], dim=2)
    value_states = torch.cat([past_value, value_states], dim=2)
    position_states = torch.cat([past_position, key_position_ids], dim=-1)
    past_key_value = (key_states, value_states, position_states) if use_cache else None
    self._ape_last_past_key_value = past_key_value
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    key_states_context = key_states[:, :, self.len_prefix : self.len_prefix + self.len_context]
    key_states_other = torch.cat(
        [key_states[:, :, : self.len_prefix], key_states[:, :, self.len_prefix + self.len_context :]],
        dim=-2,
    )
    value_states_context = value_states[:, :, self.len_prefix : self.len_prefix + self.len_context]
    value_states_other = torch.cat(
        [value_states[:, :, : self.len_prefix], value_states[:, :, self.len_prefix + self.len_context :]],
        dim=-2,
    )

    attn_output_context, lse_context, _ = flash_attn_func(
        query_states.transpose(1, 2),
        key_states_context.transpose(1, 2),
        value_states_context.transpose(1, 2),
        causal=False,
        softmax_scale=1 / (math.sqrt(self.head_dim) * float(temperature)),
        return_attn_probs=True,
    )
    attn_output_other, lse_other, _ = flash_attn_func(
        query_states.transpose(1, 2),
        key_states_other.transpose(1, 2),
        value_states_other.transpose(1, 2),
        causal=True,
        return_attn_probs=True,
    )
    lse_context = lse_context.transpose(1, 2).unsqueeze(-1).to(query_states.dtype)
    lse_other = lse_other.transpose(1, 2).unsqueeze(-1).to(query_states.dtype)
    lse_context = lse_context * (float(scale) * float(temperature))
    attn_weights = torch.cat([lse_context, lse_other], dim=-1).unsqueeze(dim=-2)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    value_states = torch.cat([attn_output_context.unsqueeze(-2), attn_output_other.unsqueeze(-2)], dim=-2)
    attn_output = torch.matmul(attn_weights, value_states).squeeze(dim=-2)
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None if not output_attentions else attn_weights


def qwen_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if cache_position is None:
        past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
    if position_ids is None:
        if past_key_values is not None and past_key_values and len(past_key_values[0]) in {3, 4}:
            past_position = past_key_values[0][2]
            position_shift = int(getattr(self, "_ape_query_position_shift", 0)) if len(past_key_values[0]) == 4 else 0
            position_start = int(past_position.max().item()) + 1 + position_shift
            position_ids = torch.arange(
                position_start,
                position_start + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0)
        else:
            position_ids = cache_position.unsqueeze(0)
    causal_mask = None
    if hasattr(self, "_update_causal_mask"):
        try:
            causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, None, output_attentions)
        except TypeError:
            causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None
    position_embeddings = self.rotary_emb(hidden_states, position_ids) if hasattr(self, "rotary_emb") else None
    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        past_key_value = past_key_values[idx] if past_key_values is not None else None
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        if isinstance(layer_outputs, tuple):
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        else:
            hidden_states = layer_outputs
        if use_cache:
            layer_cache = getattr(getattr(decoder_layer, "self_attn", None), "_ape_last_past_key_value", None)
            if layer_cache is None and isinstance(layer_outputs, tuple) and len(layer_outputs) > 2:
                layer_cache = layer_outputs[2 if output_attentions else 1]
            if layer_cache is None:
                raise RuntimeError("Qwen APE attention did not produce a cache entry")
            next_decoder_cache += (layer_cache,)
    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def _enable(model, attention_forward):
    if not QWEN_ATTENTION_CLASSES:
        raise ImportError("No Qwen2/Qwen3 attention classes are available in this Transformers install")
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            _enable(module, attention_forward)
        if isinstance(module, QWEN_ATTENTION_CLASSES):
            model._modules[name].forward = types.MethodType(attention_forward, model._modules[name])
        if QWEN_MODEL_CLASSES and isinstance(module, QWEN_MODEL_CLASSES):
            model._modules[name].forward = types.MethodType(qwen_forward, model._modules[name])


def enable_qwen_attention_prefill_prefix(model):
    _enable(model, qwen_attention_prefill_prefix)


def enable_qwen_attention_prefill_context(model):
    _enable(model, qwen_attention_prefill_context)


def enable_qwen_attention_prefill_query(model, temperature, scale, position_shift=0):
    _enable(
        model,
        partial(
            qwen_attention_prefill_query,
            temperature=temperature,
            scale=scale,
            position_shift=position_shift,
        ),
    )
    if QWEN_MODEL_CLASSES:
        for module in model.modules():
            if isinstance(module, QWEN_MODEL_CLASSES):
                module._ape_query_position_shift = int(position_shift)
