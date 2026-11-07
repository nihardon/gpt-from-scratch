# GPT From Scratch

A decoder-only transformer written from first principles in PyTorch — attention, RoPE,
RMSNorm, the training loop, LR schedule, and sampling are all hand-written (no
`nn.Transformer`, no Hugging Face models). A **19.2M-parameter** model trained on
[TinyStories](https://arxiv.org/abs/2305.07759) reaches **1.445 validation loss** and
writes coherent short stories, trained entirely on a laptop GPU (Apple MPS).

```
One day, a brave little boy named Tim went for a walk. He saw a big tree and wanted to
climb it. [...] As Tim climbed, he met a friendly squirrel. Tim said, "Hi, I am Tim. What
are you doing?" The squirrel replied, "I am climbing this tree to see new things. Can you
help me?" [...] But then, something unexpected happened. Tim slipped and fell, and he
started to cry. The squirrel said, "Don't cry, Tim. I will help you." [...] Tim learned
that even when something unexpected happens, good things can happen.
```
<sub>Sample from the best checkpoint, temperature 0.8, top-k 50, seeded with `<|endoftext|>`.</sub>

## Results

| run | params | steps (tokens) | val loss |
|---|---|---|---|
| char-level, tinyshakespeare (early version) | 0.8M | 15K | 1.55 / char |
| GPT-2 BPE, LayerNorm + learned positions, constant LR | 19.2M | 5K (41M) | 1.878 |
| + warmup + cosine LR | 19.2M | 5K (41M) | 1.762 |
| + warmup + cosine LR | 19.2M | 20K (164M) | 1.482 |
| **+ RoPE + RMSNorm** | **19.2M** | **20K (164M)** | **1.445** |

Train and val loss stay within ~0.01 throughout — 20K steps is ~30% of one epoch, so
no story is seen twice and dropout is unnecessary (it's set to 0).

### Ablations
Matched runs: same seed, config, eval schedule and 5K-step budget; one change each.
Eval noise measured at ≈ ±0.005 (same config, different data order).

| change | baseline → change | Δ val | notes |
|---|---|---|---|
| constant LR → warmup (200) + cosine to 0.1× | 1.878 → 1.762 | −0.116 | ahead at every eval; reaches baseline's final loss ~32% sooner |
| LayerNorm → RMSNorm | 1.746 → 1.738 | −0.008 | within ~2× noise: matches LayerNorm. Unfused on MPS: ~4× slower per norm call |
| learned positions → RoPE | 1.738 → 1.679 | −0.059 | 65K fewer params. Gap −0.19 at 1K steps, shrinking to −0.06: an early-training inductive bias that partially persists |

Single seed per arm — treat small deltas accordingly. Logs for every run are in [`runs/`](runs/).

## Model

- **Tokenizer:** GPT-2 BPE via `tiktoken` (50,257 tokens); stories separated by `<|endoftext|>`.
- **Architecture:** 8 pre-norm blocks, d_model 256, 8 heads (head size 32), FFN 4×
  with ReLU, context 256.
- **Attention:** per-head q/k/v, scaled dot-product, causal mask; RoPE on q and k
  (half-split pairs, base 10000).
- **Norm:** hand-written RMSNorm (verified against `nn.RMSNorm`).
- **Embeddings:** input embedding tied to the output layer (no output bias), N(0, 0.02)
  init — tying with PyTorch's default N(0, 1) embedding init gives a step-0 loss of ~71
  instead of ln(50257) ≈ 10.8.
- **Training:** AdamW, batch 32 × 256 tokens, linear warmup + cosine decay
  (1e-3 → 1e-4), best-val checkpointing.
- **Sampling:** temperature + top-k.

`POS = 'learned'` and `NORM = 'layer'` in `config.py` switch back to the baseline components.

## Usage

```bash
pip install -r requirements.txt

# 1. data: TinyStories V2 (~2.2 GB of text)
mkdir -p data
for f in TinyStoriesV2-GPT4-train.txt TinyStoriesV2-GPT4-valid.txt; do
  curl -L -o data/$f https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/$f
done
python prepare.py        # -> data/train.bin (541M tokens), data/val.bin (5.5M), uint16

# 2. train (all settings live in config.py)
python train.py          # 20K steps ≈ 80 min on an Apple M-series GPU

# 3. sample
python sample.py [checkpoint.pt]
```

## Files

| file | |
|---|---|
| `config.py` | every hyperparameter, path and toggle; device selection (cuda / mps / cpu) |
| `tokenizer.py` | GPT-2 BPE wrapper: `encode`, `decode`, `VOCAB_SIZE`, `EOT` |
| `prepare.py` | text → tokenized `uint16` `.bin` files, streamed story by story |
| `dataset.py` | memory-mapped random batches (reads ~4 KB per batch, not the 1 GB file) |
| `model.py` | `RMSNorm`, attention `Head` with RoPE, `MultiHeadAttention`, `FeedForward`, `Block`, `Transformer`, `generate` |
| `train.py` | training loop, LR schedule, loss estimation, checkpointing |
| `sample.py` | load a checkpoint and generate |
| `runs/` | training logs for every result above |
