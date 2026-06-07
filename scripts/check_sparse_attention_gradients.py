#!/usr/bin/env python
"""Check block-sparse flash training attention against the SDPA mask reference."""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ape.scratchpad import sparse_training_attention as sparse_attention  # noqa: E402
from ape.scratchpad.rendering import (  # noqa: E402
    SEGMENT_DOC_START,
    SEGMENT_PAD,
    SEGMENT_PREFIX,
    SEGMENT_SUFFIX,
    ScratchpadCollator,
    _ape_parallel_position_ids,
    build_sparse_block_mask,
)
from ape.scratchpad.sparse_training_attention import install_qwen_block_sparse_attention  # noqa: E402


def fake_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    return_attn_probs: bool = False,
    **_: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dropout_p:
        raise ValueError("The correctness checker requires dropout_p=0")
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    if causal:
        q_len = q.shape[1]
        k_len = k.shape[1]
        q_idx = torch.arange(q_len, device=q.device)[:, None]
        k_idx = torch.arange(k_len, device=q.device)[None, :]
        allowed = k_idx <= q_idx + (k_len - q_len)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(probs, v_t).transpose(1, 2).contiguous()
    if return_attn_probs:
        lse = torch.logsumexp(scores.float(), dim=-1)
        return out, lse, probs
    return out


def assert_close(name: str, value: float, tolerance: float) -> None:
    print(f"{name}: {value:.8g}")
    if value > tolerance:
        raise AssertionError(f"{name}={value:.8g} exceeds tolerance={tolerance:.8g}")


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def check_mask_semantics() -> None:
    segments = torch.tensor(
        [[SEGMENT_PREFIX, SEGMENT_PREFIX, SEGMENT_DOC_START, SEGMENT_DOC_START, SEGMENT_DOC_START + 1, SEGMENT_DOC_START + 1, SEGMENT_SUFFIX, SEGMENT_SUFFIX, SEGMENT_PAD]],
        dtype=torch.long,
    )
    mask = build_sparse_block_mask(segments)[0, 0]
    assert mask[0, 0] and not mask[0, 1], "prefix must be causal"
    assert mask[2, 0] and mask[2, 2] and not mask[2, 3], "doc rows must see prefix and causal self"
    assert not mask[4, 2] and not mask[5, 3], "document blocks must not cross-attend"
    assert mask[6, 0] and mask[6, 3] and mask[6, 5] and mask[6, 6], "suffix must see prefix/docs/self"
    assert not mask[6, 7], "suffix must remain causal inside suffix"
    assert mask[8, 8] and not mask[8, 7] and not mask[7, 8], "pad rows/cols must be isolated"
    print("mask semantics: ok")


def check_position_ids() -> None:
    prefix_len = 3
    context_lens = [5, 2, 4]
    suffix_len = 6
    positions = _ape_parallel_position_ids(prefix_len, context_lens, suffix_len, gap=0)
    expected = (
        [0, 1, 2]
        + [3, 4, 5, 6, 7]
        + [3, 4]
        + [3, 4, 5, 6]
        + [8, 9, 10, 11, 12, 13]
    )
    assert positions == expected, f"unexpected APE-parallel positions: {positions}"
    shifted = _ape_parallel_position_ids(prefix_len, context_lens, suffix_len, gap=512)
    assert shifted[: len(positions) - suffix_len] == positions[: len(positions) - suffix_len]
    assert shifted[-suffix_len:] == [pos + 512 for pos in positions[-suffix_len:]]
    print("position ids: ok")


def tiny_qwen_model() -> torch.nn.Module:
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3ForCausalLM
    except Exception as exc:  # pragma: no cover - depends on installed transformers
        raise RuntimeError("This checker requires a Transformers install with Qwen3 support") from exc

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config)
    model.config.use_cache = False
    model.eval()
    return model


def make_item(prefix_len: int, context_lens: list[int], suffix_len: int, offset: int) -> dict[str, list[int]]:
    token = 10 + int(offset)
    input_ids: list[int] = []
    segment_ids: list[int] = []
    input_ids.extend(range(token, token + prefix_len))
    segment_ids.extend([SEGMENT_PREFIX] * prefix_len)
    token += prefix_len
    for doc_index, context_len in enumerate(context_lens):
        input_ids.extend(range(token, token + context_len))
        segment_ids.extend([SEGMENT_DOC_START + doc_index] * context_len)
        token += context_len
    input_ids.extend(range(token, token + suffix_len))
    segment_ids.extend([SEGMENT_SUFFIX] * suffix_len)
    labels = [-100] * (len(input_ids) - suffix_len) + list(range(80 + offset, 80 + offset + suffix_len))
    return {
        "input_ids": [int(token_id % 127) for token_id in input_ids],
        "attention_mask": [1] * len(input_ids),
        "position_ids": _ape_parallel_position_ids(prefix_len, context_lens, suffix_len, gap=0),
        "segment_ids": segment_ids,
        "labels": [int(label % 127) if label != -100 else -100 for label in labels],
    }


def make_batches(
    device: torch.device,
    sdpa_mask_dtype: torch.dtype,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    patterns = [
        (2, [3, 2], 3),
        (3, [2, 4, 1], 4),
        (1, [1, 3], 2),
        (4, [2, 1, 2], 3),
    ]
    items = [
        make_item(*patterns[index % len(patterns)], offset=7 * index)
        for index in range(int(batch_size))
    ]
    ref_batch = ScratchpadCollator(
        pad_token_id=0,
        pad_to_multiple_of=8,
        sparse_attention_backend="sdpa_mask",
        sdpa_mask_dtype=sdpa_mask_dtype,
    )(items)
    flash_batch = ScratchpadCollator(
        pad_token_id=0,
        pad_to_multiple_of=8,
        sparse_attention_backend="flash_block",
    )(items)
    for batch in (ref_batch, flash_batch):
        for key, value in list(batch.items()):
            batch[key] = value.to(device)
    real_mask = flash_batch["segment_ids"] != SEGMENT_PAD
    suffix_mask = flash_batch["segment_ids"] == SEGMENT_SUFFIX
    return ref_batch, flash_batch, real_mask, suffix_mask


def run_forward_backward(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> Any:
    model.zero_grad(set_to_none=True)
    output = model(
        **batch,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    output.loss.backward()
    return output


def compare_gradients(reference: torch.nn.Module, candidate: torch.nn.Module) -> tuple[float, str]:
    max_diff = 0.0
    max_name = ""
    ref_params = dict(reference.named_parameters())
    cand_params = dict(candidate.named_parameters())
    for name, ref_param in ref_params.items():
        cand_param = cand_params[name]
        if ref_param.grad is None and cand_param.grad is None:
            continue
        if ref_param.grad is None or cand_param.grad is None:
            return float("inf"), name
        diff = max_abs(ref_param.grad, cand_param.grad)
        if diff > max_diff:
            max_diff = diff
            max_name = name
    return max_diff, max_name


def flash_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fake-flash", action="store_true", help="Use a CPU-compatible fake flash_attn_func.")
    parser.add_argument("--real-flash", action="store_true", help="Require an installed flash-attn backend.")
    args = parser.parse_args()

    check_mask_semantics()
    check_position_ids()

    use_fake_flash = args.fake_flash or not flash_available()
    if args.real_flash and use_fake_flash:
        raise RuntimeError("--real-flash requested, but flash-attn is not importable")
    if use_fake_flash:
        sparse_attention._flash_attn_func = lambda: fake_flash_attn_func
        print("flash backend: fake_flash_attn_func")
    else:
        print("flash backend: installed flash_attn_func")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() and not use_fake_flash else "cpu")
    else:
        device = torch.device(args.device)
    if not use_fake_flash and device.type != "cuda":
        raise RuntimeError("Real flash-attn checks require a CUDA device")
    model_dtype = torch.bfloat16 if not use_fake_flash else torch.float32
    tolerance = float(args.atol) if args.atol is not None else (5.0e-2 if not use_fake_flash else 2.0e-5)
    print(f"device: {device}, dtype: {model_dtype}, tolerance: {tolerance:.8g}")
    torch.manual_seed(int(args.seed))
    reference = tiny_qwen_model().to(device=device, dtype=model_dtype)
    candidate = copy.deepcopy(reference).to(device=device, dtype=model_dtype)
    installed = install_qwen_block_sparse_attention(candidate)
    if installed <= 0:
        raise RuntimeError("No Qwen attention layers were patched for flash_block")

    ref_batch, flash_batch, real_mask, suffix_mask = make_batches(
        device,
        sdpa_mask_dtype=model_dtype,
        batch_size=max(1, int(args.batch_size)),
    )
    ref_output = run_forward_backward(reference, ref_batch)
    cand_output = run_forward_backward(candidate, flash_batch)

    loss_diff = abs(float(ref_output.loss.detach().float()) - float(cand_output.loss.detach().float()))
    all_logits_diff = max_abs(ref_output.logits[real_mask], cand_output.logits[real_mask])
    suffix_logits_diff = max_abs(
        ref_output.logits[suffix_mask],
        cand_output.logits[suffix_mask],
    )
    hidden_diff = 0.0
    for ref_hidden, cand_hidden in zip(ref_output.hidden_states, cand_output.hidden_states):
        hidden_diff = max(hidden_diff, max_abs(ref_hidden[real_mask], cand_hidden[real_mask]))
    grad_diff, grad_name = compare_gradients(reference, candidate)

    assert_close("loss diff", loss_diff, tolerance)
    assert_close("all logits max diff", all_logits_diff, tolerance)
    assert_close("suffix logits max diff", suffix_logits_diff, tolerance)
    assert_close("hidden states max diff", hidden_diff, tolerance)
    assert_close(f"parameter gradients max diff ({grad_name or 'none'})", grad_diff, tolerance)
    print("sparse attention gradient equivalence: ok")


if __name__ == "__main__":
    main()
