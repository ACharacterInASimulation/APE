# <img src="assets/logo.png" width="40" height="40" align="top">  APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding [ICLR 2025]

### [[Paper](https://arxiv.org/abs/2502.05431)] | [[Project](https://infini-ai-lab.github.io/APE-Page)]

## TL;DR

We introduce APE for context-augmented generation with better efficiency and performance.

## Usage

### Environment Setup

```bash
conda create -yn ape python=3.10
conda activate ape

pip install -r requirements.txt
python setup.py install
```

## Run Context-augmented Question Answering with APE

By default, the temperature and scaling factor are set to 0.9, preserving over 90% performance on few-shot tasks.

```bash
CUDA_VISIBLE_DEVICES=0 python demo_APE.py --model llama3-8b-instruct
```

## Experiments

To reproduce the APE results for retrieval-augmented generation (RAG) and in-context learning (ICL) tasks in Section 5, please follow the instructions and use the code provided in the `experiments` directory.

## Scratchpad/Gist-token Positional-bias Experiments

This checkout also includes a lightweight SFT/eval path for positional-bias research:

- trains **32 distinct scratchpad/gist special tokens** after the query and before the answer
- initializes scratch/gist embeddings from the full-stop token by default and trains them with their own LR (`1e-4` by default, versus `1e-5` for LoRA)
- trains with causal LM loss and LoRA; APE temperature/scale is not used during training
- uses APE-style block-sparse training attention: each `prefix + doc_i` stream is isolated, while query/scratch/answer attend prefix plus all docs
- uses APE-parallel positions during training: prefix positions are normal, every document reuses the same post-prefix band, and query/scratch/answer start after `prefix_len + max_doc_len`
- the default `flash_block` backend uses FlashAttention calls per sparse block; `sdpa_mask` and `eager_block` remain available as correctness/debug references
- includes a dense causal decoder-only SFT baseline on the same JSONL data with `flash_attention_2`
- renders prompts as APE fields: `prefix`, parallel `contexts`, then `query`
- evaluates base APE scaled variants, a normal causal `decoder` baseline, and trained scratchpad-checkpoint variants: `scratchpad_noscale`, `scratchpad_scaled`, and `scratchpad_scaled_pos512`
- supports the LITM NaturalQuestions start/middle/end setup for the causal decoder baseline; parallel methods run a single representative LITM position by default
- keeps the requested multi-hop mix in the materialized JSONL path; the original APE experiments only partially cover this set directly

Prepare the six-source 20k-per-source training mix with a fast tokenizer length cutoff at 4096:

```bash
pip install -r requirements-scratchpad.txt
python setup.py install

python scripts/prepare_scratchpad_data.py \
  --config configs/scratchpad_multihop.yaml \
  --split both
```

Train the 32-token scratchpad model:

```bash
python scripts/train_scratchpad.py \
  --config configs/scratchpad_multihop.yaml
```

Train the dense causal decoder baseline on the same dataset:

```bash
python scripts/train_decoder.py \
  --config configs/scratchpad_multihop.yaml
```

Check sparse attention semantics and strict gradient equivalence against the SDPA mask reference:

```bash
python scripts/check_sparse_attention_gradients.py --fake-flash --batch-size 2 --atol 1e-6
```

The default checker uses `eager_block` and also verifies that gradients reach embeddings and Q/K/V/O projections:

```bash
python scripts/check_sparse_attention_gradients.py --batch-size 2 --atol 1e-6
```

To measure installed FlashAttention numeric drift separately:

```bash
python scripts/check_sparse_attention_gradients.py \
  --real-flash \
  --batch-size 2 \
  --allow-real-flash-drift
```

Download the original Lost-in-the-Middle NaturalQuestions files:

```bash
python scripts/download_litm_nq.py \
  --output-dir data/litm_nq \
  --positions start,middle,end
```

Evaluate the normal causal decoder baseline and trained scratchpad checkpoint variants. LITM start/middle/end is reported for `decoder`; scratchpad APE methods run only the representative `--parallel-litm-positions` subset, default `start`, and skip extra gold-doc order variants.

```bash
python scripts/eval_scratchpad.py \
  --config configs/scratchpad_multihop.yaml \
  --checkpoint outputs/scratchpad_multihop_qwen3_1_7b \
  --input-jsonl data/scratchpad_multihop/eval.jsonl \
  --litm-dir data/litm_nq \
  --methods decoder,scratchpad_noscale,scratchpad_scaled,scratchpad_scaled_pos512 \
  --output-jsonl outputs/scratchpad_eval/predictions.jsonl
```

By default, `decoder` loads the base model even when `--checkpoint` is supplied for the scratchpad methods. To evaluate the trained dense decoder baseline, pass `--decoder-checkpoint outputs/decoder_multihop_qwen3_1_7b`.

The `scratchpad_scaled_pos512` method applies the ablation where query/scratch/answer positions start at `prefix_len + max_doc_len + 512` inside APE query prefill.

For longer eval runs, use the suite wrappers after preparing `data/scratchpad_multihop/eval.jsonl` and downloading LITM:

```bash
scripts/run_ape_eval_suite.sh \
  --input-jsonl data/scratchpad_multihop/eval.jsonl \
  --litm-dir data/litm_nq

scripts/run_scratchpad_eval_suite.sh \
  --input-jsonl data/scratchpad_multihop/eval.jsonl \
  --litm-dir data/litm_nq \
  --train-batch-size 4
```

`run_ape_eval_suite.sh` launches `ape_scaled`, `ape_scaled_pos64`, `ape_scaled_pos128`, `ape_scaled_pos512`, and `decoder` in parallel on CUDA device 2 by default. LITM defaults to 1000 examples from the 20-document files; decoder gets `start,middle,end`, while APE methods use the representative `--parallel-litm-positions start`; multi-hop defaults to `as_is` order.

`run_scratchpad_eval_suite.sh` trains the scratchpad checkpoint first when the configured checkpoint is missing, then runs `scratchpad_noscale`, `scratchpad_scaled`, and `scratchpad_scaled_pos512` on CUDA device 3 by default. Pass `--train-batch-size 4` to try batch size 4, `--checkpoint PATH` to choose the train/eval checkpoint path, `--skip-train` to require an existing checkpoint, or `--force-train` to retrain before eval. Both wrappers check that the requested LITM files are present before eval starts.

Scratchpad training defaults can be overridden with `--scratchpad-learning-rate` and `--scratchpad-init-text` on `scripts/train_scratchpad.py`, or `--train-scratchpad-learning-rate` and `--train-scratchpad-init-text` on `scripts/run_scratchpad_eval_suite.sh`.

## TODOs
We will release the code and data in the following order, please stay tuned!

- [x] Release core code of APE, including Llama-3, Llama-3.1, Mistral-v0.3, and Gemma-2.
- [x] Release RAG and ICL evaluation code.
- [x] Release APE context-augmented QA demo
- [ ] Incorporate APE into efficient inference engine

## Citation

If you find APE useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@inproceedings{yang2025ape,
  title={APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding},
  author={Yang, Xinyu and Chen, Tianqi and Chen, Beidi},
  booktitle={ICLR 2025},
  year={2025}
}
```
