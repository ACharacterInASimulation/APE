#!/usr/bin/env python
"""Empirically benchmark sparse training attention backends."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ape.scratchpad.rendering import (  # noqa: E402
    SEGMENT_DOC_START,
    SEGMENT_PREFIX,
    SEGMENT_SUFFIX,
    ScratchpadCollator,
    _ape_parallel_position_ids,
)
from ape.scratchpad.sparse_training_attention import install_qwen_block_sparse_attention  # noqa: E402


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    key = name.lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def tiny_qwen_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=args.seq_len + args.position_gap + 64,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config)
    model.config.use_cache = False
    model.train()
    return model.to(device=device, dtype=dtype_from_name(args.dtype, device))


def make_item(args: argparse.Namespace) -> dict[str, list[int]]:
    if args.seq_len <= args.prefix_len + args.suffix_len + args.doc_count:
        raise ValueError("seq_len is too small for the requested prefix/suffix/doc_count")
    doc_total = args.seq_len - args.prefix_len - args.suffix_len
    base_doc_len = doc_total // args.doc_count
    remainder = doc_total % args.doc_count
    context_lens = [base_doc_len + (1 if idx < remainder else 0) for idx in range(args.doc_count)]
    segment_ids = [SEGMENT_PREFIX] * args.prefix_len
    for doc_idx, doc_len in enumerate(context_lens):
        segment_ids.extend([SEGMENT_DOC_START + doc_idx] * doc_len)
    segment_ids.extend([SEGMENT_SUFFIX] * args.suffix_len)
    position_ids = _ape_parallel_position_ids(args.prefix_len, context_lens, args.suffix_len, gap=args.position_gap)
    input_ids = [3 + (idx % (args.vocab_size - 3)) for idx in range(args.seq_len)]
    labels = [-100] * (args.seq_len - args.suffix_len) + [
        3 + ((idx + 17) % (args.vocab_size - 3)) for idx in range(args.suffix_len)
    ]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * args.seq_len,
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "labels": labels,
    }


def cuda_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_backend(backend: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(int(args.seed))
    model = tiny_qwen_model(args, device)
    if backend in {"eager_block", "flash_block"}:
        installed = install_qwen_block_sparse_attention(model, backend=backend)
        if installed <= 0:
            raise RuntimeError("no Qwen attention layers were patched")
    elif backend != "sdpa_mask":
        raise ValueError(f"unknown backend: {backend}")

    item = make_item(args)
    batch = ScratchpadCollator(
        pad_token_id=0,
        sparse_attention_backend=backend,
        sdpa_mask_dtype=dtype_from_name(args.dtype, device),
    )([item])
    batch = {key: value.to(device) for key, value in batch.items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch, use_cache=False)
        loss = output.loss
        loss.backward()
        optimizer.step()
        return float(loss.detach().float().cpu())

    for _ in range(args.warmup):
        step()
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    losses = []
    start = time.perf_counter()
    for _ in range(args.steps):
        losses.append(step())
    sync(device)
    elapsed = time.perf_counter() - start
    memory = cuda_memory()
    del optimizer, model, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "status": "ok",
        "backend": backend,
        "steps": args.steps,
        "mean_step_ms": 1000.0 * elapsed / max(args.steps, 1),
        "last_loss": losses[-1] if losses else None,
        **memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--doc-count", type=int, default=8)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--suffix-len", type=int, default=256)
    parser.add_argument("--position-gap", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--backends", default="sdpa_mask,eager_block,flash_block")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"torch: {torch.__version__}")

    results = []
    for backend in [item.strip() for item in args.backends.split(",") if item.strip()]:
        try:
            result = run_backend(backend, args, device)
        except Exception as exc:
            result = {
                "status": "failed",
                "backend": backend,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append(result)
        print(json.dumps(result, indent=2))

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                    "args": vars(args),
                    "results": results,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
