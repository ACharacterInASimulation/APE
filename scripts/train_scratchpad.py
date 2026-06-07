#!/usr/bin/env python
"""Fine-tune a causal LM with 32 learned scratchpad/gist tokens."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ape.scratchpad.data import load_examples_from_jsonl  # noqa: E402
from ape.scratchpad.modeling import (  # noqa: E402
    add_special_tokens,
    install_trainable_token_embeddings,
    load_causal_lm,
    load_tokenizer,
    save_scratchpad_state,
    trainable_parameter_count,
)
from ape.scratchpad.rendering import (  # noqa: E402
    DEFAULT_QUESTION_POSITION_GAP,
    DEFAULT_SCRATCHPAD_LEN,
    ScratchpadCollator,
    build_scratchpad_tokens,
    encode_training_example,
)
from ape.scratchpad.sparse_training_attention import install_qwen_block_sparse_attention  # noqa: E402


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


class ScratchpadSFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Any,
        scratchpad_tokens: list[str],
        max_seq_len: int,
        append_eos: bool,
        position_strategy: str,
        question_position_gap: int,
        max_examples: int | None = None,
    ) -> None:
        self.examples = load_examples_from_jsonl(jsonl_path, limit=max_examples)
        self.tokenizer = tokenizer
        self.scratchpad_tokens = scratchpad_tokens
        self.max_seq_len = int(max_seq_len)
        self.append_eos = bool(append_eos)
        self.position_strategy = position_strategy
        self.question_position_gap = int(question_position_gap)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return encode_training_example(
            tokenizer=self.tokenizer,
            example=self.examples[index],
            scratchpad_tokens=self.scratchpad_tokens,
            max_seq_len=self.max_seq_len,
            append_eos=self.append_eos,
            position_strategy=self.position_strategy,
            question_position_gap=self.question_position_gap,
        )


class SaveScratchpadCallback(TrainerCallback):
    def __init__(self, tokenizer: Any, scratchpad_tokens: list[str]) -> None:
        self.tokenizer = tokenizer
        self.scratchpad_tokens = scratchpad_tokens

    def on_save(self, args, state, control, **kwargs):  # type: ignore[override]
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        model = kwargs.get("model")
        if model is not None:
            save_scratchpad_state(model, self.tokenizer, checkpoint_dir, self.scratchpad_tokens)
        return control


def make_training_arguments(**kwargs: Any) -> TrainingArguments:
    parameters = inspect.signature(TrainingArguments).parameters
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in parameters and "eval_strategy" in parameters:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    supported = {key: value for key, value in kwargs.items() if key in parameters}
    return TrainingArguments(**supported)


def apply_lora(model: torch.nn.Module, config: dict[str, Any]) -> torch.nn.Module:
    if not bool(cfg_get(config, "model.use_lora", True)):
        return model
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if bool(cfg_get(config, "model.load_in_4bit", False)):
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=int(cfg_get(config, "model.lora_r", 16)),
        lora_alpha=int(cfg_get(config, "model.lora_alpha", 32)),
        lora_dropout=float(cfg_get(config, "model.lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(
            cfg_get(
                config,
                "model.lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    )
    return get_peft_model(model, lora_config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scratchpad_multihop.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--eval-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scratchpad-len", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--sparse-attention-backend", choices=["eager_block", "flash_block", "sdpa_mask", "dense"], default=None)
    parser.add_argument("--base-attn-implementation", default=None)
    parser.add_argument(
        "--position-strategy",
        choices=["standard", "ape_parallel", "question_after_docs_plus_gap", "ape_parallel_pos512"],
        default=None,
    )
    parser.add_argument("--question-position-gap", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    args = parser.parse_args()

    config = read_yaml(args.config)
    seed = int(cfg_get(config, "seed", 42))
    set_seed(seed)

    model_name = args.model or str(cfg_get(config, "model.name_or_path", "Qwen/Qwen3-1.7B"))
    output_dir = args.output_dir or str(cfg_get(config, "output_dir", "outputs/scratchpad_qwen3_1_7b"))
    train_jsonl = args.train_jsonl or str(cfg_get(config, "data.train_jsonl", "data/scratchpad_multihop/train.jsonl"))
    eval_jsonl = args.eval_jsonl or cfg_get(config, "data.eval_jsonl", "data/scratchpad_multihop/eval.jsonl")
    scratchpad_len = args.scratchpad_len or int(cfg_get(config, "scratchpad.length", DEFAULT_SCRATCHPAD_LEN))
    scratchpad_tokens = build_scratchpad_tokens(
        scratchpad_len,
        token_prefix=str(cfg_get(config, "scratchpad.token_prefix", "<scratchpad_")),
        token_suffix=str(cfg_get(config, "scratchpad.token_suffix", ">")),
    )
    max_seq_len = args.max_seq_len or int(cfg_get(config, "data.max_seq_len", 4096))
    append_eos = bool(cfg_get(config, "data.append_eos_token", True))
    position_strategy = args.position_strategy or str(cfg_get(config, "train.position_strategy", "standard"))
    question_position_gap = args.question_position_gap or int(
        cfg_get(config, "train.question_position_gap", DEFAULT_QUESTION_POSITION_GAP)
    )
    sparse_attention_backend = args.sparse_attention_backend or str(cfg_get(config, "train.sparse_attention_backend", "flash_block"))
    if sparse_attention_backend not in {"eager_block", "flash_block", "sdpa_mask", "dense"}:
        raise ValueError("train.sparse_attention_backend must be one of eager_block, flash_block, sdpa_mask, dense")
    if sparse_attention_backend in {"eager_block", "flash_block", "sdpa_mask"}:
        load_attn_implementation = args.base_attn_implementation or str(cfg_get(config, "train.base_attn_implementation", "sdpa"))
    else:
        load_attn_implementation = cfg_get(config, "model.attn_implementation", "flash_attention_2")

    tokenizer = load_tokenizer(model_name)
    model = load_causal_lm(
        model_name,
        dtype=str(cfg_get(config, "model.dtype", "bfloat16")),
        attn_implementation=load_attn_implementation,
        load_in_4bit=bool(cfg_get(config, "model.load_in_4bit", False)),
        device_map=cfg_get(config, "model.device_map", None),
    )
    if sparse_attention_backend in {"eager_block", "flash_block"}:
        installed = install_qwen_block_sparse_attention(model, backend=sparse_attention_backend)
        print(f"installed {sparse_attention_backend} sparse attention on {installed} Qwen attention layers")
    elif sparse_attention_backend == "sdpa_mask":
        print("using SDPA with a 4D APE block-sparse attention mask")
    else:
        print("using dense causal attention")
    if sparse_attention_backend in {"eager_block", "flash_block", "sdpa_mask"}:
        model.config.use_cache = False
    add_special_tokens(tokenizer, model, scratchpad_tokens)
    model = apply_lora(model, config)
    token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in scratchpad_tokens]
    install_trainable_token_embeddings(model, token_ids)

    if bool(cfg_get(config, "train.gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    trainable, total = trainable_parameter_count(model)
    print(f"trainable parameters: {trainable:,} / {total:,} ({100.0 * trainable / max(total, 1):.4f}%)")

    train_dataset = ScratchpadSFTDataset(
        train_jsonl,
        tokenizer=tokenizer,
        scratchpad_tokens=scratchpad_tokens,
        max_seq_len=max_seq_len,
        append_eos=append_eos,
        position_strategy=position_strategy,
        question_position_gap=question_position_gap,
        max_examples=args.max_train_examples or cfg_get(config, "train.max_train_examples", None),
    )
    eval_dataset = None
    if eval_jsonl:
        eval_path = Path(str(eval_jsonl))
        if eval_path.exists():
            eval_dataset = ScratchpadSFTDataset(
                eval_path,
                tokenizer=tokenizer,
                scratchpad_tokens=scratchpad_tokens,
                max_seq_len=max_seq_len,
                append_eos=append_eos,
                position_strategy=position_strategy,
                question_position_gap=question_position_gap,
                max_examples=args.max_eval_examples or cfg_get(config, "train.max_eval_examples", None),
            )

    training_args = make_training_arguments(
        output_dir=output_dir,
        overwrite_output_dir=bool(cfg_get(config, "train.overwrite_output_dir", False)),
        per_device_train_batch_size=int(cfg_get(config, "train.batch_size", 1)),
        per_device_eval_batch_size=int(cfg_get(config, "train.eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg_get(config, "train.grad_accum_steps", 8)),
        max_steps=args.max_steps or int(cfg_get(config, "train.max_steps", 12_500)),
        learning_rate=args.learning_rate or float(cfg_get(config, "train.learning_rate", 1.0e-5)),
        weight_decay=float(cfg_get(config, "train.weight_decay", 0.0)),
        warmup_steps=int(cfg_get(config, "train.warmup_steps", 100)),
        max_grad_norm=float(cfg_get(config, "train.max_grad_norm", 1.0)),
        logging_steps=int(cfg_get(config, "train.log_every", 10)),
        save_steps=int(cfg_get(config, "train.save_every", 1000)),
        eval_steps=int(cfg_get(config, "train.eval_every", 0)) or None,
        evaluation_strategy="steps" if eval_dataset is not None and int(cfg_get(config, "train.eval_every", 0)) > 0 else "no",
        bf16=str(cfg_get(config, "train.mixed_precision", "bf16")).lower() == "bf16",
        fp16=str(cfg_get(config, "train.mixed_precision", "bf16")).lower() in {"fp16", "float16"},
        dataloader_num_workers=int(cfg_get(config, "train.dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(cfg_get(config, "train.dataloader_pin_memory", True)),
        report_to=list(cfg_get(config, "train.report_to", [])),
        remove_unused_columns=False,
    )

    collator = ScratchpadCollator(
        pad_token_id=int(tokenizer.pad_token_id),
        pad_to_multiple_of=cfg_get(config, "data.pad_to_multiple_of", 8),
        sparse_attention_backend=sparse_attention_backend,
        sdpa_mask_dtype=(
            torch.bfloat16
            if str(cfg_get(config, "train.mixed_precision", "bf16")).lower() == "bf16"
            else torch.float16
            if str(cfg_get(config, "train.mixed_precision", "bf16")).lower() in {"fp16", "float16"}
            else torch.float32
        ),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[SaveScratchpadCallback(tokenizer, scratchpad_tokens)],
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_scratchpad_state(model, tokenizer, output_dir, scratchpad_tokens)
    print(f"saved {output_dir}")


if __name__ == "__main__":
    main()
