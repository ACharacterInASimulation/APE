"""APE-style prompt fields and tokenization for scratchpad/gist-token SFT."""

from __future__ import annotations

from typing import Any

import torch

DEFAULT_SCRATCHPAD_LEN = 32
DEFAULT_QUESTION_POSITION_GAP = 512
SEGMENT_PAD = -2
SEGMENT_SUFFIX = -1
SEGMENT_PREFIX = 0
SEGMENT_DOC_START = 1


def build_scratchpad_tokens(
    length: int = DEFAULT_SCRATCHPAD_LEN,
    token_prefix: str = "<scratchpad_",
    token_suffix: str = ">",
) -> list[str]:
    return [f"{token_prefix}{idx}{token_suffix}" for idx in range(int(length))]


def scratchpad_text(tokens: list[str]) -> str:
    if not tokens:
        return ""
    return "\n" + " ".join(tokens) + "\n"


def format_context_document(doc: dict[str, Any], index: int | None = None) -> str:
    """Format one context chunk for APE's parallel context field."""

    title = str(doc.get("title", "") or "").strip()
    text = str(doc.get("text", "") or "").strip()
    if title:
        return f"{title}\n{text}"
    return text


def format_document(doc: dict[str, Any], index: int) -> str:
    return format_context_document(doc, index=index)


def build_prefix_field(example: dict[str, Any]) -> str:
    instruction = str(example.get("instruction", "") or "Use the provided contexts to answer the question.").strip()
    return f"{instruction}\n\n" if instruction else ""


def build_context_fields(example: dict[str, Any]) -> list[str]:
    return [
        format_context_document(doc, idx + 1)
        for idx, doc in enumerate(example.get("documents", []))
        if str(doc.get("text", "") or "").strip()
    ]


def build_query_field(example: dict[str, Any], scratchpad_tokens: list[str] | None = None) -> str:
    question = str(example.get("question", "") or "").strip()
    query = f"\n\nQuestion: {question}\nAnswer:"
    if scratchpad_tokens:
        query += scratchpad_text(scratchpad_tokens)
    return query


def ape_prompt_fields(
    example: dict[str, Any],
    scratchpad_tokens: list[str] | None = None,
) -> dict[str, Any]:
    """Return the APE decomposition: prefix, parallel contexts, query."""

    return {
        "prefix": build_prefix_field(example),
        "contexts": build_context_fields(example),
        "query": build_query_field(example, scratchpad_tokens=scratchpad_tokens),
    }


def prompt_segments(example: dict[str, Any]) -> dict[str, str]:
    fields = ape_prompt_fields(example, scratchpad_tokens=None)
    contexts_text = "\n\n".join(fields["contexts"])
    return {
        "prefix": fields["prefix"],
        "contexts": f"{contexts_text}" if contexts_text else "",
        "query": build_query_field(example, scratchpad_tokens=None),
    }


def render_prompt(example: dict[str, Any], scratchpad_tokens: list[str] | None = None) -> str:
    fields = ape_prompt_fields(example, scratchpad_tokens=scratchpad_tokens)
    contexts_text = "\n\n".join(fields["contexts"])
    return fields["prefix"] + contexts_text + fields["query"]


def render_answer(example: dict[str, Any], eos_token: str | None = None) -> str:
    answer = str(example.get("answer", "") or "").strip()
    if not answer:
        answers = example.get("answers", [])
        if answers:
            answer = str(answers[0]).strip()
    answer = " " + answer
    if eos_token:
        answer += eos_token
    return answer


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _standard_position_ids(length: int) -> list[int]:
    return list(range(length))


def _ape_parallel_position_ids(
    prefix_len: int,
    context_lens: list[int],
    suffix_len: int,
    gap: int = DEFAULT_QUESTION_POSITION_GAP,
) -> list[int]:
    prefix_positions = list(range(prefix_len))
    context_positions = [
        position
        for context_len in context_lens
        for position in range(prefix_len, prefix_len + context_len)
    ]
    suffix_start = prefix_len + (max(context_lens) if context_lens else 0) + int(gap)
    suffix_positions = list(range(suffix_start, suffix_start + suffix_len))
    return prefix_positions + context_positions + suffix_positions


def encode_prompt(
    tokenizer: Any,
    example: dict[str, Any],
    scratchpad_tokens: list[str] | None = None,
    position_strategy: str = "standard",
    question_position_gap: int = DEFAULT_QUESTION_POSITION_GAP,
) -> dict[str, list[int]]:
    fields = ape_prompt_fields(example, scratchpad_tokens=scratchpad_tokens)
    prefix_ids = encode_text(tokenizer, fields["prefix"])
    context_ids = [encode_text(tokenizer, context) for context in fields["contexts"]]
    query_ids = encode_text(tokenizer, fields["query"])
    input_ids = prefix_ids + [token_id for doc_ids in context_ids for token_id in doc_ids] + query_ids
    segment_ids = (
        [SEGMENT_PREFIX] * len(prefix_ids)
        + [
            segment_id
            for doc_index, doc_ids in enumerate(context_ids)
            for segment_id in [SEGMENT_DOC_START + doc_index] * len(doc_ids)
        ]
        + [SEGMENT_SUFFIX] * len(query_ids)
    )
    if position_strategy in {"standard", "ape_parallel"}:
        gap = 0
        position_ids = _ape_parallel_position_ids(
            prefix_len=len(prefix_ids),
            context_lens=[len(doc_ids) for doc_ids in context_ids],
            suffix_len=len(query_ids),
            gap=gap,
        )
    elif position_strategy in {"question_after_docs_plus_gap", "ape_parallel_pos512"}:
        gap = int(question_position_gap)
        position_ids = _ape_parallel_position_ids(
            prefix_len=len(prefix_ids),
            context_lens=[len(doc_ids) for doc_ids in context_ids],
            suffix_len=len(query_ids),
            gap=gap,
        )
    else:
        raise ValueError(f"Unknown position_strategy: {position_strategy}")
    query_start_position = position_ids[-len(query_ids)] if query_ids else None

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "prompt_length": len(input_ids),
        "prefix_length": len(prefix_ids),
        "context_lengths": [len(doc_ids) for doc_ids in context_ids],
        "query_start_position": query_start_position,
    }


def encode_training_example(
    tokenizer: Any,
    example: dict[str, Any],
    scratchpad_tokens: list[str],
    max_seq_len: int,
    append_eos: bool = True,
    position_strategy: str = "standard",
    question_position_gap: int = DEFAULT_QUESTION_POSITION_GAP,
) -> dict[str, list[int]]:
    prompt = encode_prompt(
        tokenizer=tokenizer,
        example=example,
        scratchpad_tokens=scratchpad_tokens,
        position_strategy=position_strategy,
        question_position_gap=question_position_gap,
    )
    answer_ids = encode_text(
        tokenizer,
        render_answer(example, tokenizer.eos_token if append_eos else None),
    )
    input_ids = prompt["input_ids"] + answer_ids
    if len(input_ids) > int(max_seq_len):
        raise ValueError(f"Encoded example has {len(input_ids)} tokens, above max_seq_len={max_seq_len}")
    prompt_length = int(prompt["prompt_length"])
    labels = [-100] * prompt_length + answer_ids
    segment_ids = prompt["segment_ids"] + [SEGMENT_SUFFIX] * len(answer_ids)
    tail_start = prompt["position_ids"][-1] + 1 if prompt["position_ids"] else 0
    answer_positions = list(range(tail_start, tail_start + len(answer_ids)))
    position_ids = prompt["position_ids"] + answer_positions
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "labels": labels,
    }


def encode_decoder_training_example(
    tokenizer: Any,
    example: dict[str, Any],
    max_seq_len: int,
    append_eos: bool = True,
) -> dict[str, list[int]]:
    prompt_ids = encode_text(tokenizer, render_prompt(example, scratchpad_tokens=None))
    answer_ids = encode_text(
        tokenizer,
        render_answer(example, tokenizer.eos_token if append_eos else None),
    )
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > int(max_seq_len):
        raise ValueError(f"Encoded example has {len(input_ids)} tokens, above max_seq_len={max_seq_len}")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def build_sparse_block_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """Build APE block-sparse causal attention.

    Prefix rows see causal prefix. Document rows see prefix plus their own
    document causally. Suffix rows (query/scratch/answer) see every previous
    real token, including all documents.
    """

    if segment_ids.ndim != 2:
        raise ValueError(f"segment_ids must have shape [batch, seq], got {tuple(segment_ids.shape)}")
    batch, seq_len = segment_ids.shape
    device = segment_ids.device
    row_segments = segment_ids[:, :, None]
    col_segments = segment_ids[:, None, :]
    rows = torch.arange(seq_len, device=device)[:, None]
    cols = torch.arange(seq_len, device=device)[None, :]
    causal = cols <= rows
    query_real = row_segments != SEGMENT_PAD
    key_real = col_segments != SEGMENT_PAD
    prefix_key = col_segments == SEGMENT_PREFIX
    prefix_query = row_segments == SEGMENT_PREFIX
    doc_query = row_segments >= SEGMENT_DOC_START
    same_doc = row_segments == col_segments
    suffix_query = row_segments == SEGMENT_SUFFIX

    allowed_by_block = suffix_query | (prefix_query & prefix_key) | (doc_query & (prefix_key | same_doc))
    mask = causal.unsqueeze(0) & query_real & key_real & allowed_by_block

    pad_positions = segment_ids == SEGMENT_PAD
    if pad_positions.any():
        eye = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)
        mask = mask | (pad_positions[:, :, None] & pad_positions[:, None, :] & eye)
    return mask.unsqueeze(1)


def pad_batch(
    batch: list[dict[str, list[int]]],
    pad_token_id: int,
    label_pad_id: int = -100,
    pad_to_multiple_of: int | None = None,
) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    if pad_to_multiple_of and max_len % int(pad_to_multiple_of):
        max_len = ((max_len + int(pad_to_multiple_of) - 1) // int(pad_to_multiple_of)) * int(pad_to_multiple_of)

    padded = {"input_ids": [], "attention_mask": [], "position_ids": [], "segment_ids": [], "labels": []}
    for item in batch:
        length = len(item["input_ids"])
        pad_len = max_len - length
        padded["input_ids"].append(item["input_ids"] + [pad_token_id] * pad_len)
        padded["attention_mask"].append(item["attention_mask"] + [0] * pad_len)
        last_pos = item["position_ids"][-1] if item["position_ids"] else 0
        padded["position_ids"].append(item["position_ids"] + [last_pos] * pad_len)
        padded["segment_ids"].append(item["segment_ids"] + [SEGMENT_PAD] * pad_len)
        labels = item.get("labels", [-100] * length)
        padded["labels"].append(labels + [label_pad_id] * pad_len)

    return {
        key: torch.tensor(value, dtype=torch.long)
        for key, value in padded.items()
    }


class ScratchpadCollator:
    def __init__(
        self,
        pad_token_id: int,
        pad_to_multiple_of: int | None = None,
        sparse_attention_backend: str = "flash_block",
        sdpa_mask_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.sparse_attention_backend = sparse_attention_backend
        self.sdpa_mask_dtype = sdpa_mask_dtype

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        padded = pad_batch(
            batch,
            pad_token_id=self.pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        if self.sparse_attention_backend == "sdpa_mask":
            sparse_mask = build_sparse_block_mask(padded["segment_ids"])
            padded["attention_mask"] = torch.zeros(
                sparse_mask.shape,
                dtype=self.sdpa_mask_dtype,
                device=sparse_mask.device,
            ).masked_fill(~sparse_mask, torch.finfo(self.sdpa_mask_dtype).min)
        elif self.sparse_attention_backend in {"flash_block", "dense"}:
            pass
        else:
            raise ValueError(f"Unknown sparse_attention_backend: {self.sparse_attention_backend}")
        if self.sparse_attention_backend != "flash_block":
            padded.pop("segment_ids", None)
        return padded
