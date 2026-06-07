#!/usr/bin/env python
"""Fine-tune a normal dense causal decoder baseline on the scratchpad dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ape.scratchpad.data import load_examples_from_jsonl  # noqa: E402
from ape.scratchpad.modeling import (  # noqa: E402
    load_causal_lm,
    load_tokenizer,
    trainable_parameter_count,
)
from ape.scratchpad.rendering import encode_decoder_training_example  # noqa: E402


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


def make_training_arguments(**kwargs: Any) -> TrainingArguments:
    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        if "evaluation_strategy" in kwargs:
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
        return TrainingArguments(**kwargs)


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


class DecoderSFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Any,
        max_seq_len: int,
        append_eos: bool,
        max_examples: int | None = None,
    ) -> None:
        self.examples = load_examples_from_jsonl(jsonl_path, limit=max_examples)
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.append_eos = bool(append_eos)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return encode_decoder_training_example(
            tokenizer=self.tokenizer,
            example=self.examples[index],
            max_seq_len=self.max_seq_len,
            append_eos=self.append_eos,
        )


class DecoderCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = None, label_pad_id: int = -100) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_id = int(label_pad_id)

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        if self.pad_to_multiple_of and max_len % int(self.pad_to_multiple_of):
            max_len = ((max_len + int(self.pad_to_multiple_of) - 1) // int(self.pad_to_multiple_of)) * int(
                self.pad_to_multiple_of
            )
        padded = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in batch:
            length = len(item["input_ids"])
            pad_len = max_len - length
            padded["input_ids"].append(item["input_ids"] + [self.pad_token_id] * pad_len)
            padded["attention_mask"].append(item["attention_mask"] + [0] * pad_len)
            padded["labels"].append(item["labels"] + [self.label_pad_id] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in padded.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scratchpad_multihop.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--eval-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    args = parser.parse_args()

    config = read_yaml(args.config)
    seed = int(cfg_get(config, "seed", 42))
    set_seed(seed)

    model_name = args.model or str(cfg_get(config, "model.name_or_path", "Qwen/Qwen3-1.7B"))
    output_dir = args.output_dir or str(cfg_get(config, "decoder_output_dir", "outputs/decoder_multihop_qwen3_1_7b"))
    train_jsonl = args.train_jsonl or str(cfg_get(config, "data.train_jsonl", "data/scratchpad_multihop/train.jsonl"))
    eval_jsonl = args.eval_jsonl or cfg_get(config, "data.eval_jsonl", "data/scratchpad_multihop/eval.jsonl")
    max_seq_len = args.max_seq_len or int(cfg_get(config, "data.max_seq_len", 4096))
    append_eos = bool(cfg_get(config, "data.append_eos_token", True))
    attn_implementation = args.attn_implementation or str(
        cfg_get(config, "decoder_train.attn_implementation", "flash_attention_2")
    )

    tokenizer = load_tokenizer(model_name)
    model = load_causal_lm(
        model_name,
        dtype=str(cfg_get(config, "model.dtype", "bfloat16")),
        attn_implementation=attn_implementation,
        load_in_4bit=bool(cfg_get(config, "model.load_in_4bit", False)),
        device_map=cfg_get(config, "model.device_map", None),
    )
    model = apply_lora(model, config)
    if bool(cfg_get(config, "train.gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    trainable, total = trainable_parameter_count(model)
    print(f"using dense causal decoder attention: {attn_implementation}")
    print(f"trainable parameters: {trainable:,} / {total:,} ({100.0 * trainable / max(total, 1):.4f}%)")

    train_dataset = DecoderSFTDataset(
        train_jsonl,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        append_eos=append_eos,
        max_examples=args.max_train_examples or cfg_get(config, "train.max_train_examples", None),
    )
    eval_dataset = None
    if eval_jsonl:
        eval_path = Path(str(eval_jsonl))
        if eval_path.exists():
            eval_dataset = DecoderSFTDataset(
                eval_path,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                append_eos=append_eos,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DecoderCollator(
            pad_token_id=int(tokenizer.pad_token_id),
            pad_to_multiple_of=cfg_get(config, "data.pad_to_multiple_of", 8),
        ),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"saved {output_dir}")


if __name__ == "__main__":
    main()
