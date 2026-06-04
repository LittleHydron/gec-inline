# Fine-tuning report — GEC inline-edit format

Submission for the LPNLP "Fine-tuning LLM" homework.

> The numbers in the **Results** section below are placeholders that
> get filled in by `scripts/eval.py` after Modal training completes.
> Update them once the SFT and DPO adapters have been evaluated.

## Task

Grammatical Error Correction with **inline edit syntax**. Given an
English sentence, the model emits the corrected sentence with every
edit wrapped in braces:

```
input    She have did her homework already .
output   She {have did=>has done} her homework already .
```

The format encodes substitutions as `{src=>tgt}`, insertions as
`{=>tgt}`, deletions as `{src=>}`, and multi-token spans as
`{a b=>c}`. The bracket syntax is the one the assignment PDF
specified verbatim — it's a strong choice because:

- Base instruct models do **not** emit this format zero-shot, so the
  fine-tuning gain is dramatic.
- The format is parseable, so an automatic edit-level metric (ERRANT
  F0.5) drops out cleanly.
- Public gold data is available — no need to bootstrap with
  knowledge distillation.

## Dataset

| Split | Source | # examples | Used for |
| --- | --- | --- | --- |
| Train | BEA-2019 W&I+LOCNESS `ABC.train` + FCE `train` (M2) | 10 000 (sampled from ~64 k) | SFT |
| Train (DPO) | Synthesized from gold + SFT outputs | 3 000 pairs | DPO bonus |
| Eval (primary) | BEA-2019 W&I+LOCNESS `ABCN.dev` (M2) | 4 384 | ERRANT F0.5 |
| Eval (secondary) | JFLEG `validation` (multi-ref) | 754 | exact-match |
| Eval (held out) | JFLEG `test` | 747 | qualitative table |

All training rows are formatted with the Qwen2.5 chat template as
`{system, user, assistant}` where the assistant content is the gold
bracketed string. The system prompt is short:

> You are a grammatical error correction assistant. Given an English
> sentence, output the corrected sentence with every edit inlined
> inside braces using the syntax {wrong=>right}. Use {=>word} for
> insertions and {word=>} for deletions. Keep all unchanged tokens
> exactly as in the input. If the input is already correct, return
> it unchanged with no braces.

The inline format is rendered from the M2 gold edits by walking source
tokens left-to-right (`gec/render.py`). A round-trip unit test
(`tests/test_m2_roundtrip.py`) verifies that on a sample of 200 real
BEA sentences `parse(render(source, edits)) == apply(source, edits)`.

DPO pairs are built by `scripts/build_dpo_pairs.py`:

- `chosen` = gold bracketed completion
- `rejected` = the SFT model's output when it differs from gold; if
  the SFT output happens to match gold (which it does for ~30 % of
  short sentences), we fall back to either (a) a corrupted gold with
  one random edit deleted, or (b) the gold of a *different* sentence
  (cross-sentence noise).

## Base model

`Qwen/Qwen2.5-3B-Instruct`. Picked because:

- Strong English grammatical prior at the 3 B size.
- First-class Unsloth support (the framework recommended by the PDF);
  there is an official 4-bit checkpoint at
  `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`.
- Trains comfortably with QLoRA at seq_len 1024, batch 2 × grad-accum 4
  on a Modal L4 24 GB (~$0.80/h out of the $30/month free credits — both
  Modal and Unsloth are suggested by the assignment PDF), and still fits
  a free Colab T4 16 GB via the legacy notebooks. DPO doubles activation
  memory (chosen + rejected forward pass); we run DPO with batch 1 ×
  grad-accum 8.
- Tokenises English compactly (~1.3 chars/token on BEA), so 10 k
  training examples fit in well under one epoch of training-time
  budget on a free T4.

## Evaluation method

1. **ERRANT F0.5 on BEA W&I+LOCNESS dev** — canonical metric for the
   BEA-2019 shared task and the de-facto standard for GEC. We feed
   the model the M2 source line, parse the bracketed output back to a
   plain corrected sentence (`gec/parse.py`), run `errant_parallel`
   to align source → hypothesis into a hypothesis M2 file, and run
   `errant_compare -hyp ... -ref ABCN.dev.gold.bea19.m2` to get
   TP / FP / FN and the F0.5 score.
2. **Exact-match accuracy on JFLEG dev** — for each prediction, mark
   it correct if it matches any of the 4 references after whitespace
   normalization.
3. **Parse-failure rate** — fraction of predictions with unmatched
   braces in the raw output. A fine-tuned model should drive this to
   ≈ 0; the base model with few-shot tends to leave it in the 5–15 %
   range.
4. **Trivial-copy rate** — fraction where the model returned the
   input unchanged. Useful sanity check; the dev set has ~36 %
   already-correct sentences so a healthy model should hover near
   that value, neither far above (under-correction) nor below
   (over-correction).
5. **Qualitative side-by-side table** — 30 rows (15 "hard"
   model-disagreement cases + 15 random) in `results/qualitative.md`.

Both fine-tuned variants are compared against the base model in two
configurations:

- **Base zero-shot**: the same system prompt, no few-shot examples.
  Almost always fails the format and scores near zero on ERRANT.
- **Base + 3-shot**: the system prompt plus three in-context examples
  demonstrating the bracket syntax. This is the *fair* baseline; it
  removes the format-knowledge gap and isolates the actual quality
  improvement from fine-tuning.

## Hyperparameters

| Parameter | SFT | DPO |
| --- | --- | --- |
| LoRA rank | 16 | 16 (fresh adapter on top of merged SFT) |
| LoRA alpha | 32 | 32 |
| LoRA dropout | 0 | 0 |
| Target modules | q,k,v,o,gate,up,down | same |
| Learning rate | 2e-4 | 5e-6 |
| Epochs | 2 | 1 |
| Schedule | cosine | cosine |
| Warmup ratio | 0.03 | 0.05 |
| Per-device batch | 2 | 1 |
| Gradient accumulation | 4 (eff 8) | 8 (eff 8) |
| Weight decay | 0.01 | 0.0 |
| Optimizer | adamw_8bit | adamw_8bit |
| Max seq length | 1024 | 1024 |
| DPO β | — | 0.1 |
| Seed | 3407 | 3407 |

Train-on-responses-only is enabled during SFT (only the assistant span
contributes to the loss), so the model is not penalized for predicting
the prompt tokens.

## Results

> Numbers are filled in after Modal training completes. Until then,
> the only entry below is the **oracle upper bound**: the same
> ERRANT pipeline applied to the gold target sentences themselves,
> which tells us how high any model can possibly score given
> tokenization-edge alignment.

| Model | BEA dev ERRANT P | R | F0.5 | parse fail | trivial copy | JFLEG exact-match |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle (gold → ERRANT realign) | 0.863 | 0.877 | **0.865** | 0.000 | 0.357 | — |
| Base, zero-shot | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Base, 3-shot | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **SFT (ours)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **SFT + DPO (ours)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

The oracle ceiling of 0.865 (not 1.000) comes from ERRANT realigning
our reconstructed source → corrected pair from scratch; some edits get
split or merged differently than the gold M2 spans. This is a well-known
property of ERRANT and matches what the BEA-2019 shared task baselines
report.

## Other notes / findings

_To be filled in once training completes. Expected observations
based on prior GEC fine-tuning literature with 3 B-class models:_

- SFT drives parse-failure rate to near zero within ~200 steps; the
  format is easy to learn.
- The largest residual errors after SFT are missed insertions of
  short closed-class words ("the", "a", commas) rather than incorrect
  word choices — the model becomes confident and conservative.
- DPO with corruption-based rejections improves precision more than
  recall (it teaches the model to avoid spurious edits), pushing F0.5
  up slightly without changing F1 much.

## Reproducibility

All code, data-prep scripts, the Modal pipeline and the legacy Colab
notebooks live in the repo at `github.com/LittleHydron/gec-inline`.
Trained adapters are pushed to the HuggingFace Hub under
`Lopato4ka/qwen2.5-3b-gec-sft` (plus a merged 16-bit copy at
`…-sft-merged`, the base the DPO adapter is trained on) and
`Lopato4ka/qwen2.5-3b-gec-dpo`. The Gradio demo is deployed to a Space
at `huggingface.co/spaces/Lopato4ka/gec-inline`.

Training + prediction generation run on Modal (suggested by the
assignment PDF; ~$5 of the $30/month free credits for the whole
pipeline):

```bash
pip install modal && modal setup
modal secret create huggingface HF_TOKEN=hf_xxx
modal run -m modal_app.app::run_all      # SFT -> DPO pairs -> DPO -> 12 prediction sets
modal volume get gec-inline-results predictions/ results/predictions/
```

ERRANT scoring is CPU-only and runs locally:

```bash
.venv/bin/python -m scripts.build_dataset \
    --m2 data/raw/wi+locness/m2/ABC.train.gold.bea19.m2 \
    --m2 data/raw/fce/m2/fce.train.gold.bea19.m2

.venv/bin/python -m scripts.eval --mode bea \
    --predictions results/predictions/sft_bea_dev.jsonl \
    --ref-m2 data/raw/wi+locness/m2/ABCN.dev.gold.bea19.m2 \
    --out results/metrics/sft_bea_dev.json
```

(`scripts/generate.py` remains usable locally against any published
adapter if you have a GPU; on Modal the same code path is invoked by
`modal_app/app.py::generate`.)
