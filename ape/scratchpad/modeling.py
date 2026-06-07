"""Model loading, trainable scratchpad embeddings, and generation helpers."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRATCHPAD_EMBEDDINGS_FILE = "scratchpad_embeddings.pt"
SCRATCHPAD_CONFIG_FILE = "scratchpad_config.json"


def dtype_from_string(dtype: str | None) -> torch.dtype | None:
    if dtype is None or str(dtype).lower() in {"auto", "none", ""}:
        return None
    key = str(dtype).lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def set_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.unk_token_id is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})


def add_special_tokens(tokenizer: Any, model: nn.Module, tokens: list[str]) -> list[int]:
    existing = set(tokenizer.get_vocab())
    to_add = [token for token in tokens if token not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        model.resize_token_embeddings(len(tokenizer))
    return [int(tokenizer.convert_tokens_to_ids(token)) for token in tokens]


class TrainableTokenEmbedding(nn.Module):
    """Wrap an embedding so only selected token rows are trainable.

    This avoids saving or optimizing the full input embedding matrix just to
    learn the scratchpad/gist-token slots.
    """

    def __init__(self, base_embedding: nn.Embedding, token_ids: list[int]) -> None:
        super().__init__()
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        self.base_embedding = base_embedding
        self.base_embedding.weight.requires_grad_(False)
        self.token_ids = [int(token_id) for token_id in token_ids]
        device = base_embedding.weight.device
        mapping = torch.full(
            (base_embedding.num_embeddings,),
            -1,
            dtype=torch.long,
            device=device,
        )
        for slot, token_id in enumerate(self.token_ids):
            mapping[int(token_id)] = slot
        self.register_buffer("token_id_to_slot", mapping, persistent=False)
        self.trainable = nn.Embedding(len(self.token_ids), base_embedding.embedding_dim)
        with torch.no_grad():
            ids = torch.tensor(self.token_ids, dtype=torch.long, device=device)
            self.trainable.weight.copy_(base_embedding(ids).to(self.trainable.weight.device))

    @property
    def weight(self) -> torch.Tensor:
        return self.base_embedding.weight

    @property
    def num_embeddings(self) -> int:
        return self.base_embedding.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.base_embedding.embedding_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = self.base_embedding(input_ids)
        slots = self.token_id_to_slot[input_ids]
        mask = slots >= 0
        if mask.any():
            base = base.clone()
            base[mask] = self.trainable(slots[mask]).to(base.dtype)
        return base

    def scratchpad_weight(self) -> torch.Tensor:
        return self.trainable.weight.detach()

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):  # type: ignore[override]
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()  # type: ignore[attr-defined]
        base_weight = self.base_embedding.weight if keep_vars else self.base_embedding.weight.detach()
        trainable_weight = self.trainable.weight if keep_vars else self.trainable.weight.detach()
        merged_weight = base_weight.clone()
        token_ids = torch.tensor(self.token_ids, dtype=torch.long, device=merged_weight.device)
        merged_weight[token_ids] = trainable_weight.to(merged_weight.device, merged_weight.dtype)
        destination[prefix + "weight"] = merged_weight
        return destination


def install_trainable_token_embeddings(model: nn.Module, token_ids: list[int]) -> TrainableTokenEmbedding:
    wrapper = TrainableTokenEmbedding(model.get_input_embeddings(), token_ids)
    model.set_input_embeddings(wrapper)
    return wrapper


def initialize_trainable_token_embeddings_from_text(
    model: nn.Module,
    tokenizer: Any,
    init_text: str,
) -> list[int]:
    wrapper = find_trainable_token_embedding(model)
    if wrapper is None:
        raise ValueError("scratchpad embeddings must be installed before initialization")
    init_ids = tokenizer.encode(init_text, add_special_tokens=False)
    if not init_ids:
        raise ValueError(f"scratchpad init text produced no tokens: {init_text!r}")
    ids = torch.tensor(init_ids, dtype=torch.long, device=wrapper.base_embedding.weight.device)
    with torch.no_grad():
        source = wrapper.base_embedding(ids).to(torch.float32).mean(dim=0)
        source = source.to(wrapper.trainable.weight.device, wrapper.trainable.weight.dtype)
        wrapper.trainable.weight.copy_(source.unsqueeze(0).expand_as(wrapper.trainable.weight))
    return [int(token_id) for token_id in init_ids]


def find_trainable_token_embedding(model: nn.Module) -> TrainableTokenEmbedding | None:
    embedding = model.get_input_embeddings()
    if isinstance(embedding, TrainableTokenEmbedding):
        return embedding
    for module in model.modules():
        if isinstance(module, TrainableTokenEmbedding):
            return module
    return None


def save_scratchpad_state(
    model: nn.Module,
    tokenizer: Any,
    output_dir: str | Path,
    tokens: list[str],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in tokens]
    wrapper = find_trainable_token_embedding(model)
    if wrapper is not None:
        weights = wrapper.scratchpad_weight().cpu()
    else:
        with torch.no_grad():
            ids = torch.tensor(token_ids, dtype=torch.long, device=model.get_input_embeddings().weight.device)
            weights = model.get_input_embeddings()(ids).detach().cpu()
    torch.save({"tokens": tokens, "token_ids": token_ids, "embeddings": weights}, output_dir / SCRATCHPAD_EMBEDDINGS_FILE)
    with (output_dir / SCRATCHPAD_CONFIG_FILE).open("w", encoding="utf-8") as handle:
        json.dump({"tokens": tokens, "token_ids": token_ids}, handle, indent=2)


def load_scratchpad_state(model: nn.Module, tokenizer: Any, checkpoint_dir: str | Path | None) -> None:
    if checkpoint_dir is None:
        return
    path = Path(checkpoint_dir) / SCRATCHPAD_EMBEDDINGS_FILE
    if not path.exists():
        return
    state = torch.load(path, map_location="cpu")
    tokens = list(state["tokens"])
    token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in tokens]
    weights = state["embeddings"]
    embedding = model.get_input_embeddings()
    with torch.no_grad():
        for slot, token_id in enumerate(token_ids):
            embedding.weight[int(token_id)].copy_(weights[slot].to(embedding.weight.device, embedding.weight.dtype))
    if getattr(model.config, "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
        model.tie_weights()


def load_tokenizer(path: str, fallback_path: str | None = None) -> Any:
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception:
        if fallback_path is None:
            raise
        tokenizer = AutoTokenizer.from_pretrained(fallback_path, trust_remote_code=True)
    set_pad_token(tokenizer)
    return tokenizer


def checkpoint_scratchpad_tokens(checkpoint_dir: str | Path | None) -> list[str]:
    if checkpoint_dir is None:
        return []
    config_path = Path(checkpoint_dir) / SCRATCHPAD_CONFIG_FILE
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        return list(state.get("tokens", []))
    embeddings_path = Path(checkpoint_dir) / SCRATCHPAD_EMBEDDINGS_FILE
    if embeddings_path.exists():
        state = torch.load(embeddings_path, map_location="cpu")
        return list(state.get("tokens", []))
    return []


def ensure_tokenizer_tokens(tokenizer: Any, tokens: list[str]) -> None:
    if not tokens:
        return
    existing = set(tokenizer.get_vocab())
    to_add = [token for token in tokens if token not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})


def load_causal_lm(
    model_name_or_path: str,
    dtype: str | None = "bfloat16",
    attn_implementation: str | None = None,
    load_in_4bit: bool = False,
    device_map: str | dict[str, Any] | None = None,
) -> nn.Module:
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    torch_dtype = dtype_from_string(dtype)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
        )
    if device_map is not None:
        kwargs["device_map"] = device_map
    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def load_model_for_eval(
    base_model: str,
    checkpoint: str | None = None,
    dtype: str | None = "bfloat16",
    attn_implementation: str | None = None,
    load_in_4bit: bool = False,
    device_map: str | dict[str, Any] | None = None,
) -> tuple[Any, nn.Module]:
    checkpoint_path = Path(checkpoint) if checkpoint else None
    adapter_config = checkpoint_path / "adapter_config.json" if checkpoint_path else None
    tokenizer_path = str(checkpoint_path) if checkpoint_path and (checkpoint_path / "tokenizer_config.json").exists() else base_model
    tokenizer = load_tokenizer(tokenizer_path, fallback_path=base_model)
    ensure_tokenizer_tokens(tokenizer, checkpoint_scratchpad_tokens(checkpoint_path))

    if adapter_config and adapter_config.exists():
        model = load_causal_lm(
            base_model,
            dtype=dtype,
            attn_implementation=attn_implementation,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
        )
        model.resize_token_embeddings(len(tokenizer))
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(checkpoint_path))
    else:
        model_path = str(checkpoint_path) if checkpoint_path else base_model
        model = load_causal_lm(
            model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
        )
        model.resize_token_embeddings(len(tokenizer))
    load_scratchpad_state(model, tokenizer, checkpoint_path)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def greedy_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    max_new_tokens: int = 64,
    eos_token_id: int | list[int] | None = None,
) -> torch.Tensor:
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    eos_ids = set()
    if isinstance(eos_token_id, int):
        eos_ids.add(int(eos_token_id))
    elif eos_token_id is not None:
        eos_ids.update(int(token_id) for token_id in eos_token_id)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    generated = []
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    cur_attention_mask = attention_mask
    cur_position = None
    if position_ids is not None:
        cur_position = position_ids[:, -1:] + 1

    for _ in range(int(max_new_tokens)):
        token_id = int(next_token[0, 0].item())
        if eos_ids and token_id in eos_ids:
            break
        generated.append(next_token)
        cur_attention_mask = torch.cat(
            [cur_attention_mask, torch.ones_like(next_token, dtype=cur_attention_mask.dtype)],
            dim=-1,
        )
        outputs = model(
            input_ids=next_token,
            attention_mask=cur_attention_mask,
            position_ids=cur_position,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        if cur_position is not None:
            cur_position = cur_position + 1

    if not generated:
        return torch.empty((input_ids.shape[0], 0), dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat(generated, dim=-1)


def trainable_parameter_count(model: nn.Module) -> tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total
