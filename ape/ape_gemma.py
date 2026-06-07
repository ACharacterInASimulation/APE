from transformers.models.gemma2.modeling_gemma2 import (
    repeat_kv,
    Gemma2Attention,
    Gemma2Model,
    apply_rotary_pos_emb
)

from transformers.modeling_outputs import BaseModelOutputWithPast

import math
import types
import torch

from torch import nn

from flash_attn import flash_attn_func
from typing import List, Optional, Tuple, Union

from functools import partial

def gemma_attention_prefill_prefix(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    past_key_value = (key_states, value_states, position_ids) if use_cache else None

    self.len_prefix = q_len

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    flash_kwargs = {}
    flash_kwargs["softcap"] = self.config.attn_logit_softcapping

    if q_len > 1:
        attn_output = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), causal=True, **flash_kwargs)
    else:
        attn_output = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), causal=False, **flash_kwargs)
   
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def gemma_attention_prefill_context(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    assert past_key_value is not None

    (past_key, past_value, past_position) = past_key_value
    
    key_states = torch.cat([past_key, key_states], dim=2)
    value_states = torch.cat([past_value, value_states], dim=2)
    position_states = torch.cat([past_position, position_ids], dim=-1)

    past_key_value = (key_states, value_states, position_states) if use_cache else None
        
    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    flash_kwargs = {}
    flash_kwargs["softcap"] = self.config.attn_logit_softcapping

    if q_len > 1:
        attn_output = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), causal=True, **flash_kwargs)
    else:
        attn_output = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), causal=False, **flash_kwargs)

    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def gemma_attention_prefill_query(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    temperature=1,
    scale=1,
    position_shift=0,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    assert past_key_value is not None

    if len(past_key_value) == 4:
        past_key, past_value, past_position = past_key_value[0], past_key_value[1], past_key_value[2]
        current_position = past_position.max().item() + 1 + int(position_shift)
        self.len_context = past_key.shape[2] - self.len_prefix
    else:
        (past_key, past_value, past_position) = past_key_value
        current_position = past_position.max().item() + 1

    key_position_ids = position_ids - position_ids.min().item() + current_position

    cos, sin = self.rotary_emb(value_states, key_position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, key_position_ids)

    key_states= torch.cat([past_key, key_states], dim=2)
    value_states = torch.cat([past_value, value_states], dim=2)
    position_states = torch.cat([past_position, key_position_ids], dim=-1)
    
    past_key_value = (key_states, value_states, position_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    key_states_context = torch.cat([key_states[:, :, :self.len_prefix], key_states[:, :, self.len_prefix+self.len_context:]], dim=-2)
    key_states_other = key_states[:, :, self.len_prefix:self.len_prefix+self.len_context]
    value_states_context = torch.cat([value_states[:, :, :self.len_prefix], value_states[:, :, self.len_prefix+self.len_context:]], dim=-2)
    value_states_other = value_states[:, :, self.len_prefix:self.len_prefix+self.len_context]

    flash_kwargs = {}
    flash_kwargs["softcap"] = self.config.attn_logit_softcapping

    attn_output_context, lse_context, _ = flash_attn_func(query_states.transpose(1, 2), key_states_context.transpose(1, 2), value_states_context.transpose(1, 2), causal=False, softmax_scale = 1 / (math.sqrt(self.head_dim) * temperature), return_attn_probs=True, **flash_kwargs)
    attn_output_other, lse_other, _ = flash_attn_func(query_states.transpose(1, 2), key_states_other.transpose(1, 2), value_states_other.transpose(1, 2), causal=True, return_attn_probs=True, **flash_kwargs)
    lse_context = lse_context.transpose(1, 2).unsqueeze(-1).to(query_states.dtype)
    lse_other = lse_other.transpose(1, 2).unsqueeze(-1).to(query_states.dtype)

    scale = scale * temperature
    lse_context = lse_context * scale

    attn_weights = torch.cat([lse_context, lse_other], dim=-1).unsqueeze(dim=-2)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    value_states = torch.cat([attn_output_context.unsqueeze(-2), attn_output_other.unsqueeze(-2)], dim=-2)
    attn_output = torch.matmul(attn_weights, value_states).squeeze(dim=-2)
    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def gemma_forward(
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
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )


    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, None, output_attentions
    )

    # embed positions
    hidden_states = inputs_embeds

    # normalized
    # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
    # See https://github.com/huggingface/transformers/pull/29402
    normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
    hidden_states = hidden_states * normalizer

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
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

def gemma_update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: List[torch.FloatTensor],
    output_attentions: bool,
):
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    if past_key_values is not None:
        target_length = past_key_values[0][0].shape[2] + sequence_length
    else:
        target_length = attention_mask.shape[-1] if attention_mask is not None else input_tensor.shape[1]

    if attention_mask is not None and attention_mask.dim() == 4:
        # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
        if attention_mask.max() != 0:
            raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
        causal_mask = attention_mask
    else:
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    return causal_mask


def enable_gemma_attention_prefill_prefix(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_gemma_attention_prefill_prefix(
                module,
            )

        if isinstance(module, Gemma2Attention):
            model._modules[name].forward = types.MethodType(
                gemma_attention_prefill_prefix, model._modules[name]
            )

        if isinstance(module, Gemma2Model):
            model._modules[name].forward = types.MethodType(
                gemma_forward, model._modules[name]
            )
            model._modules[name]._update_causal_mask = types.MethodType(
                gemma_update_causal_mask, model._modules[name]
            )

def enable_gemma_attention_prefill_context(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_gemma_attention_prefill_context(
                module,
            )

        if isinstance(module, Gemma2Attention):
            model._modules[name].forward = types.MethodType(
                gemma_attention_prefill_context, model._modules[name]
            )

def enable_gemma_attention_prefill_query(model, temperature, scale, position_shift=0):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_gemma_attention_prefill_query(
                module, temperature, scale, position_shift=position_shift
            )

        if isinstance(module, Gemma2Attention):
            model._modules[name].forward = types.MethodType(
                partial(
                    gemma_attention_prefill_query,
                    temperature=temperature,
                    scale=scale,
                    position_shift=position_shift,
                ),
                model._modules[name],
            )
