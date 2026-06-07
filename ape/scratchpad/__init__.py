"""Scratchpad/gist-token fine-tuning helpers for APE experiments."""

from .data import DEFAULT_SOURCES, load_examples_from_jsonl
from .metrics import score_qa
from .rendering import ape_prompt_fields, build_scratchpad_tokens, encode_prompt, encode_training_example

__all__ = [
    "DEFAULT_SOURCES",
    "build_scratchpad_tokens",
    "ape_prompt_fields",
    "encode_prompt",
    "encode_training_example",
    "load_examples_from_jsonl",
    "score_qa",
]
