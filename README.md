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
- trains with causal LM loss and LoRA; APE temperature/scale is not used during training
- uses APE-style block-sparse training attention by default: each `prefix + doc_i` stream is isolated, while query/scratch/answer attend prefix plus all docs
- uses APE-parallel positions during training: prefix positions are normal, every document reuses the same post-prefix band, and query/scratch/answer start after `prefix_len + max_doc_len`
- the default `flash_block` backend issues FlashAttention calls per block; `sdpa_mask` is available as a correctness fallback
- includes a dense causal decoder-only SFT baseline on the same JSONL data with `flash_attention_2`
- renders prompts as APE fields: `prefix`, parallel `contexts`, then `query`
- evaluates a normal causal `decoder` baseline plus trained scratchpad-checkpoint variants: `scratchpad_noscale`, `scratchpad_scaled`, and `scratchpad_scaled_pos512`
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

Check sparse attention semantics and gradient equivalence before trusting `flash_block`:

```bash
python scripts/check_sparse_attention_gradients.py --fake-flash
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
