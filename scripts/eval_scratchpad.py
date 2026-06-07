#!/usr/bin/env python
"""Evaluate normal decoder, base APE, and trained scratchpad checkpoint variants."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ape.scratchpad.ape_generation import generate_with_ape  # noqa: E402
from ape.scratchpad.data import load_examples_from_jsonl, open_text  # noqa: E402
from ape.scratchpad.metrics import mean_metrics, score_qa  # noqa: E402
from ape.scratchpad.modeling import greedy_generate, load_model_for_eval  # noqa: E402
from ape.scratchpad.rendering import (  # noqa: E402
    DEFAULT_SCRATCHPAD_LEN,
    build_context_fields,
    build_prefix_field,
    build_query_field,
    build_scratchpad_tokens,
)

ALL_METHODS = [
    "decoder",
    "ape_scaled",
    "ape_scaled_pos64",
    "ape_scaled_pos128",
    "ape_scaled_pos512",
    "scratchpad_noscale",
    "scratchpad_scaled",
    "scratchpad_scaled_pos512",
]

LITM_POSITION_FILES = {
    10: {"start": 0, "middle": 4, "end": 9},
    20: {"start": 0, "middle": 9, "end": 19},
    30: {"start": 0, "middle": 14, "end": 29},
}


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


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_config_csv(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return parse_csv(str(value))


def expand_methods(methods: str) -> list[str]:
    requested = parse_csv(methods)
    if requested == ["all"] or "all" in requested:
        return ALL_METHODS
    for method in requested:
        if method not in ALL_METHODS:
            raise ValueError(f"Unknown method {method}; choose from {ALL_METHODS} or all")
    return requested


def reorder_gold(example: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant == "as_is":
        return example
    docs = list(example.get("documents", []))
    gold = [doc for doc in docs if bool(doc.get("is_gold", False))]
    other = [doc for doc in docs if not bool(doc.get("is_gold", False))]
    if not gold:
        return example
    if variant == "gold_start":
        ordered = gold + other
    elif variant == "gold_end":
        ordered = other + gold
    elif variant == "gold_middle":
        mid = len(other) // 2
        ordered = other[:mid] + gold + other[mid:]
    else:
        raise ValueError(f"Unknown order variant: {variant}")
    copied = dict(example)
    copied["documents"] = ordered
    return copied


def examples_with_order_variants(
    examples: Iterable[dict[str, Any]],
    variants: list[str],
) -> Iterable[tuple[str, dict[str, Any]]]:
    for example in examples:
        for variant in variants:
            yield variant, reorder_gold(example, variant)


def iter_litm_file(path: Path, doc_count: int, position_name: str, gold_index: int, limit: int | None = None):
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            docs = []
            for idx, ctx in enumerate(row.get("ctxs", [])):
                docs.append(
                    {
                        "id": ctx.get("id", f"{doc_count}-{gold_index}-{idx}"),
                        "title": ctx.get("title", ""),
                        "text": ctx.get("text", ""),
                        "is_gold": bool(ctx.get("isgold", False) or idx == gold_index),
                        "metadata": {
                            "hasanswer": ctx.get("hasanswer"),
                            "original_retrieval_index": ctx.get("original_retrieval_index"),
                        },
                    }
                )
            yield {
                "id": f"litm_nq-{doc_count}-{gold_index}-{count}",
                "source": "litm_nq",
                "task": "qa",
                "instruction": "Write a high-quality answer using only the provided search results. Some documents may be irrelevant.",
                "question": row.get("question", ""),
                "documents": docs,
                "answers": row.get("answers", []),
                "answer": row.get("answers", [""])[0] if row.get("answers") else "",
                "metadata": {
                    "doc_count": doc_count,
                    "gold_index": gold_index,
                    "position": position_name,
                },
            }
            count += 1
            if limit is not None and count >= int(limit):
                break


def load_litm_examples(
    litm_dir: Path,
    doc_counts: list[int],
    positions: list[str],
    limit_per_file: int | None,
) -> list[tuple[str, dict[str, Any]]]:
    pairs = []
    for doc_count in doc_counts:
        for position in positions:
            gold_index = LITM_POSITION_FILES[doc_count][position]
            filename = f"nq-open-{doc_count}_total_documents_gold_at_{gold_index}.jsonl.gz"
            path = litm_dir / f"{doc_count}_total_documents" / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing LITM file {path}; run scripts/download_litm_nq.py first")
            for example in iter_litm_file(path, doc_count, position, gold_index, limit=limit_per_file):
                pairs.append((f"litm_{doc_count}_{position}", example))
    return pairs


def should_evaluate_pair(
    method: str,
    variant: str,
    example: dict[str, Any],
    parallel_litm_positions: set[str],
) -> bool:
    if method == "decoder":
        return True
    if example.get("source") == "litm_nq":
        position = str(example.get("metadata", {}).get("position", ""))
        return position in parallel_litm_positions
    if variant != "as_is":
        return False
    return True


def method_uses_scratchpad(method: str) -> bool:
    return method.startswith("scratchpad_")


def method_uses_ape(method: str) -> bool:
    return method.startswith("ape_") or method_uses_scratchpad(method)


def method_uses_scaling(method: str) -> bool:
    return method.startswith("ape_scaled") or method.startswith("scratchpad_scaled")


def method_position_shift(method: str) -> int:
    fixed_shifts = {
        "ape_scaled_pos64": 64,
        "ape_scaled_pos128": 128,
        "ape_scaled_pos512": 512,
        "scratchpad_scaled_pos512": 512,
    }
    return int(fixed_shifts.get(method, 0))


@torch.no_grad()
def generate_decoder(
    model: Any,
    tokenizer: Any,
    example: dict[str, Any],
    max_new_tokens: int,
    max_context_tokens: int,
) -> str:
    prefix = build_prefix_field(example)
    query = build_query_field(example, scratchpad_tokens=None)
    contexts = build_context_fields(example) or [""]
    prefix_ids = tokenizer(prefix, truncation=False, add_special_tokens=False).input_ids
    query_ids = tokenizer(query, truncation=False, add_special_tokens=False).input_ids
    separator_ids = tokenizer("\n\n", truncation=False, add_special_tokens=False).input_ids
    prompt_budget = max(1, int(max_context_tokens) - int(max_new_tokens))
    context_budget = max(1, prompt_budget - len(prefix_ids) - len(query_ids))
    per_context_max = max(1, math.floor(context_budget / max(len(contexts), 1)))
    context_ids = []
    for context_index, context in enumerate(contexts):
        if context_index > 0:
            context_ids.extend(separator_ids)
        context_ids.extend(
            tokenizer(
                context,
                truncation=True,
                max_length=per_context_max,
                add_special_tokens=False,
            ).input_ids
        )
    input_ids = prefix_ids + context_ids + query_ids
    if len(input_ids) > prompt_budget:
        keep_query = min(len(query_ids), prompt_budget)
        remaining = prompt_budget - keep_query
        input_ids = input_ids[:remaining] + query_ids[-keep_query:]

    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    generated = greedy_generate(
        model,
        input_ids=input_tensor,
        attention_mask=attention_mask,
        position_ids=None,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        metrics = row["metrics"]
        groups["overall"].append(metrics)
        groups[f"method:{row['method']}"].append(metrics)
        groups[f"variant:{row['variant']}"].append(metrics)
        groups[f"method_variant:{row['method']}:{row['variant']}"].append(metrics)
        source = row.get("source", "")
        if source:
            groups[f"source:{source}"].append(metrics)
            groups[f"method_source:{row['method']}:{source}"].append(metrics)
    return {
        group: {"num_examples": len(items), "metrics": mean_metrics(items)}
        for group, items in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scratchpad_multihop.yaml")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--decoder-checkpoint", default=None)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--litm-dir", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--output-jsonl", default="outputs/scratchpad_eval/predictions.jsonl")
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--limit-per-litm-file", type=int, default=None)
    parser.add_argument("--litm-doc-counts", default="10,20,30")
    parser.add_argument("--litm-positions", default="start,middle,end")
    parser.add_argument("--parallel-litm-positions", default=None)
    parser.add_argument("--order-variants", default="as_is,gold_start,gold_middle,gold_end")
    parser.add_argument("--scratchpad-len", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--ape-model-name", default=None)
    parser.add_argument("--ape-temperature", type=float, default=None)
    parser.add_argument("--ape-scale", type=float, default=None)
    args = parser.parse_args()

    config = read_yaml(args.config)
    base_model = args.base_model or str(cfg_get(config, "model.name_or_path", "Qwen/Qwen3-1.7B"))
    scratchpad_len = args.scratchpad_len or int(cfg_get(config, "scratchpad.length", DEFAULT_SCRATCHPAD_LEN))
    scratchpad_tokens = build_scratchpad_tokens(
        scratchpad_len,
        token_prefix=str(cfg_get(config, "scratchpad.token_prefix", "<scratchpad_")),
        token_suffix=str(cfg_get(config, "scratchpad.token_suffix", ">")),
    )
    max_new_tokens = args.max_new_tokens or int(cfg_get(config, "eval.max_new_tokens", 64))
    max_context_tokens = args.max_context_tokens or int(cfg_get(config, "data.max_seq_len", 4096))
    ape_temperature = args.ape_temperature if args.ape_temperature is not None else float(cfg_get(config, "eval.ape_temperature", 0.9))
    ape_scale = args.ape_scale if args.ape_scale is not None else float(cfg_get(config, "eval.ape_scale", 0.9))
    ape_model_name = args.ape_model_name or str(cfg_get(config, "eval.ape_model_name", base_model)).lower()
    method_config = cfg_get(config, "eval.methods", ALL_METHODS)
    method_string = args.methods if args.methods else ",".join(parse_config_csv(method_config))
    methods = expand_methods(method_string)
    parallel_litm_positions = set(
        parse_csv(args.parallel_litm_positions)
        if args.parallel_litm_positions
        else parse_config_csv(cfg_get(config, "eval.parallel_litm_positions", ["start"]))
    )

    example_pairs: list[tuple[str, dict[str, Any]]] = []
    if args.input_jsonl:
        examples = load_examples_from_jsonl(args.input_jsonl, limit=args.max_examples)
        variants = parse_csv(args.order_variants)
        example_pairs.extend(examples_with_order_variants(examples, variants))
    if args.litm_dir:
        doc_counts = [int(item) for item in parse_csv(args.litm_doc_counts)]
        positions = parse_csv(args.litm_positions)
        litm_pairs = load_litm_examples(
            Path(args.litm_dir),
            doc_counts=doc_counts,
            positions=positions,
            limit_per_file=args.limit_per_litm_file or args.max_examples,
        )
        example_pairs.extend(litm_pairs)
    if not example_pairs:
        input_jsonl = cfg_get(config, "data.eval_jsonl", None)
        if not input_jsonl:
            raise ValueError("Provide --input-jsonl or --litm-dir")
        examples = load_examples_from_jsonl(input_jsonl, limit=args.max_examples)
        example_pairs.extend(examples_with_order_variants(examples, parse_csv(args.order_variants)))

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_json) if args.metrics_json else output_path.with_suffix(".metrics.json")
    all_rows = []

    for method in methods:
        if method_uses_scratchpad(method) and args.checkpoint is None:
            raise ValueError(f"{method} requires --checkpoint pointing to the trained scratchpad checkpoint")
        method_checkpoint = (
            args.decoder_checkpoint
            if method == "decoder"
            else args.checkpoint
            if method_uses_scratchpad(method)
            else None
        )
        tokenizer, model = load_model_for_eval(
            base_model=base_model,
            checkpoint=method_checkpoint,
            dtype=str(cfg_get(config, "model.dtype", "bfloat16")),
            attn_implementation=cfg_get(config, "model.attn_implementation", None),
            load_in_4bit=bool(cfg_get(config, "model.load_in_4bit", False)),
            device_map=cfg_get(config, "model.device_map", None),
        )
        if cfg_get(config, "model.device_map", None) is None:
            model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        scaled = method_uses_scaling(method)
        temp = ape_temperature if scaled else 1.0
        scale = ape_scale if scaled else 1.0
        query_position_shift = method_position_shift(method)
        method_pairs = [
            (variant, example)
            for variant, example in example_pairs
            if should_evaluate_pair(method, variant, example, parallel_litm_positions)
        ]
        for variant, example in tqdm(method_pairs, desc=method, dynamic_ncols=True):
            if method == "decoder":
                prediction = generate_decoder(
                    model=model,
                    tokenizer=tokenizer,
                    example=example,
                    max_new_tokens=max_new_tokens,
                    max_context_tokens=max_context_tokens,
                )
            elif method_uses_ape(method):
                prediction = generate_with_ape(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=ape_model_name,
                    example=example,
                    scratchpad_tokens=scratchpad_tokens if method_uses_scratchpad(method) else None,
                    temperature=temp,
                    scale=scale,
                    max_new_tokens=max_new_tokens,
                    max_context_tokens=max_context_tokens,
                    query_position_shift=query_position_shift,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")
            metrics = score_qa(prediction, list(example.get("answers", [])))
            row = {
                "method": method,
                "variant": variant,
                "id": example.get("id"),
                "source": example.get("source"),
                "question": example.get("question"),
                "answers": example.get("answers", []),
                "prediction": prediction,
                "metrics": metrics,
                "ape_temperature": temp,
                "ape_scale": scale,
                "query_position_shift": query_position_shift,
                "metadata": example.get("metadata", {}),
            }
            all_rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open_text(output_path, "wt") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = summarize(all_rows)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"predictions {output_path}")
    print(f"metrics {metrics_path}")


if __name__ == "__main__":
    main()
