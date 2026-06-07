#!/usr/bin/env python
"""Materialize fast, length-filtered scratchpad SFT JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ape.scratchpad.data import (  # noqa: E402
    DEFAULT_INSTRUCTION,
    DEFAULT_SOURCES,
    load_source_examples,
    truncate_documents_with_tokenizer,
    write_examples_jsonl,
)
from ape.scratchpad.modeling import load_tokenizer  # noqa: E402
from ape.scratchpad.rendering import (  # noqa: E402
    DEFAULT_SCRATCHPAD_LEN,
    build_scratchpad_tokens,
    encode_training_example,
)


def read_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def cfg_get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def keep_by_length(
    tokenizer: Any,
    example: dict[str, Any],
    scratchpad_tokens: list[str],
    max_seq_len: int,
    append_eos: bool,
) -> bool:
    try:
        encode_training_example(
            tokenizer=tokenizer,
            example=example,
            scratchpad_tokens=scratchpad_tokens,
            max_seq_len=max_seq_len,
            append_eos=append_eos,
        )
        return True
    except ValueError:
        return False


def materialize_split(
    split: str,
    output_path: Path,
    tokenizer: Any,
    sources: list[dict[str, Any]],
    examples_per_source: int,
    instruction: str,
    max_docs: int | None,
    min_docs: int,
    max_doc_tokens: int,
    max_seq_len: int,
    scratchpad_tokens: list[str],
    seed: int,
    append_eos: bool,
) -> int:
    prepared = []
    for source_idx, source_cfg in enumerate(sources):
        target = int(source_cfg.get(f"{split}_examples", source_cfg.get("examples", examples_per_source)))
        source_name = source_cfg.get("name", source_cfg.get("hf_path"))
        kept = 0
        attempts = max(target * 4, target + 1000)
        raw_examples = load_source_examples(
            source_cfg=source_cfg,
            split=split,
            count=attempts,
            instruction=instruction,
            max_docs=max_docs,
            min_docs=min_docs,
            seed=seed + source_idx,
            show_progress=True,
        )
        for example in tqdm(raw_examples, desc=f"{source_name}:{split}:filter", dynamic_ncols=True):
            example = truncate_documents_with_tokenizer(example, tokenizer, max_doc_tokens=max_doc_tokens)
            if not keep_by_length(tokenizer, example, scratchpad_tokens, max_seq_len, append_eos=append_eos):
                continue
            prepared.append(example)
            kept += 1
            if kept >= target:
                break
        print(f"{source_name}:{split} kept={kept} target={target}")
    return write_examples_jsonl(prepared, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scratchpad_multihop.yaml")
    parser.add_argument("--split", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--max-doc-tokens", type=int, default=None)
    parser.add_argument("--samples-per-source", type=int, default=None)
    parser.add_argument("--eval-samples-per-source", type=int, default=None)
    parser.add_argument("--scratchpad-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    config = read_yaml(args.config)
    model_name = args.tokenizer or cfg_get(config, "model.name_or_path", "Qwen/Qwen3-1.7B")
    tokenizer = load_tokenizer(model_name)
    scratchpad_len = args.scratchpad_len or int(cfg_get(config, "scratchpad.length", DEFAULT_SCRATCHPAD_LEN))
    scratchpad_tokens = build_scratchpad_tokens(
        scratchpad_len,
        token_prefix=str(cfg_get(config, "scratchpad.token_prefix", "<scratchpad_")),
        token_suffix=str(cfg_get(config, "scratchpad.token_suffix", ">")),
    )
    max_seq_len = args.max_seq_len or int(cfg_get(config, "data.max_seq_len", 4096))
    max_doc_tokens = args.max_doc_tokens or int(cfg_get(config, "data.max_doc_tokens", 512))
    train_per_source = args.samples_per_source or int(cfg_get(config, "data.examples_per_source", 20_000))
    eval_per_source = args.eval_samples_per_source or int(cfg_get(config, "data.eval_examples_per_source", 1000))
    instruction = str(cfg_get(config, "data.instruction", DEFAULT_INSTRUCTION))
    max_docs_raw = cfg_get(config, "data.max_docs", None)
    max_docs = None if max_docs_raw in {None, "all", "none", "null"} else int(max_docs_raw)
    min_docs = int(cfg_get(config, "data.min_docs", 2))
    seed = args.seed if args.seed is not None else int(cfg_get(config, "seed", 42))
    append_eos = args.append_eos if args.append_eos is not None else bool(cfg_get(config, "data.append_eos_token", True))
    sources = list(cfg_get(config, "data.sources", DEFAULT_SOURCES))
    output_dir = Path(args.output_dir or cfg_get(config, "data.output_dir", "data/scratchpad_multihop"))
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_name_or_tokenizer": model_name,
        "max_seq_len": max_seq_len,
        "max_doc_tokens": max_doc_tokens,
        "scratchpad_len": scratchpad_len,
        "scratchpad_tokens": scratchpad_tokens,
        "sources": sources,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    splits = ["train", "eval"] if args.split == "both" else [args.split]
    for split in splits:
        per_source = train_per_source if split == "train" else eval_per_source
        output_path = output_dir / f"{split}.jsonl"
        count = materialize_split(
            split=split,
            output_path=output_path,
            tokenizer=tokenizer,
            sources=sources,
            examples_per_source=per_source,
            instruction=instruction,
            max_docs=max_docs,
            min_docs=min_docs,
            max_doc_tokens=max_doc_tokens,
            max_seq_len=max_seq_len,
            scratchpad_tokens=scratchpad_tokens,
            seed=seed,
            append_eos=append_eos,
        )
        print(f"wrote {count} examples to {output_path}")


if __name__ == "__main__":
    main()
