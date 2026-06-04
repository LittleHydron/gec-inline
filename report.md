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

DPO pairs are built by `scripts/build_dpo_pairs.py` from the SFT
model's own outputs on a 4 000-sentence train subset. **We trained two
DPO checkpoints with identical hyperparameters — only the preference
data differs** (see Results for the consequences):

- **v1 (biased — negative result).** `chosen` = gold; `rejected` = the
  SFT output when it differed from gold, else either gold with one
  random edit *deleted*, or the gold of a *different* sentence.
  Already-correct sentences were skipped. All three choices share a
  hidden bias: `rejected` systematically contains *fewer relevant
  edits* than `chosen`, so the preference gradient reduces to
  "more edits ⇒ preferred".
- **v2 (fixed).** Corruption made symmetric — 50 % delete a gold edit
  (under-correction), 50 % inject a spurious edit (a no-op `{word=>word}`
  or a pointless closed-class swap, the exact shapes v1 hallucinated);
  all 500 already-correct sentences included with the edit-free gold as
  `chosen`; cross-sentence noise dropped. Edit-count direction across
  pairs: 1 257 chosen-more / 1 548 rejected-more / 195 equal.

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

Both DPO checkpoints (v1 and v2) use the **identical** hyperparameters
above — they differ only in the preference data (see Dataset). This
isolates the data bias as the cause of the v1 collapse. SFT final train
loss: 0.223 (2 epochs, 73 min on Modal L4). DPO v1 final train loss:
0.254 — note this *looked healthy* despite producing a broken model.

## Results

All numbers on the same eval sets for every model. BEA dev = ERRANT
P/R/F0.5 over 4 384 sentences; JFLEG = exact-match against any of 4
references over 754 sentences; parse-fail and trivial-copy as defined
above. The **oracle** row is the ERRANT pipeline applied to the gold
target sentences themselves — the alignment-induced ceiling.

| Model | BEA dev ERRANT P | R | F0.5 | parse fail | trivial copy | JFLEG exact-match |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle (gold → ERRANT realign) | 0.863 | 0.877 | **0.865** | 0.000 | 0.357 | — |
| Base, zero-shot | 0.036 | 0.105 | 0.042 | 0.231 | 0.015 | 0.005 |
| Base, 3-shot | 0.143 | 0.121 | 0.138 | 0.047 | 0.192 | 0.133 |
| **SFT (ours)** | 0.440 | 0.399 | **0.431** | 0.002 | 0.321 | **0.293** |
| SFT + DPO **v1** (biased pairs) | 0.120 | 0.380 | 0.139 | 0.065 | 0.029 | 0.021 |
| SFT + DPO **v2** (fixed pairs) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Reading of the table:

- **Fine-tuning gain is 3.1× the fair baseline** (F0.5 0.431 vs 0.138
  for base + 3-shot). Zero-shot is near zero (0.042) — the inline
  format is simply not in the base model's prior, and 23 % of its
  outputs aren't even parseable.
- **SFT nails the format**: parse failures drop to 0.2 %, and its
  trivial-copy rate (0.321) sits just under the dev set's actual
  already-correct rate (0.357) — neither over- nor under-correcting.
  The base + 3-shot model scores most of its JFLEG "exact matches" by
  copying the input (trivial-copy 0.192 ≈ its EM 0.133); SFT's matches
  come from real corrections.
- **DPO v1 is a clean negative result**: recall barely moves
  (0.380 vs SFT's 0.399 — the right edits are still found) but
  precision collapses 0.440 → 0.120 under 20 788 false-positive edits
  (SFT: 3 795), landing F0.5 back at the 3-shot baseline level. See
  the DPO-pairs section above for the data bias that caused it, and
  **Other notes** for the mechanism.

The oracle ceiling of 0.865 (not 1.000) comes from ERRANT realigning
our reconstructed source → corrected pair from scratch; some edits get
split or merged differently than the gold M2 spans. This is a well-known
property of ERRANT and matches what the BEA-2019 shared task baselines
report.

## Other notes / findings

- **The format is easy; the task is the hard part.** SFT drove
  parse failures from 23 % (zero-shot) to 0.2 %. The few SFT parse
  failures left are `max_new_tokens` truncations on very long
  sentences, not malformed braces.
- **The main finding: DPO amplifies preference-data bias far more
  aggressively than SFT absorbs label noise.** v1's pairs encoded
  "more edits ⇒ preferred" three different ways (see Dataset). The
  trained model reward-hacked exactly that proxy: edit density
  exploded from 2.79 (SFT) to **11.22 edits/sentence**, including
  2.11 *no-op* edits per sentence (`{could=>could}`, `{sex=>sex}`)
  and gratuitous paraphrases of correct text (`{purchase=>buy}`).
  Trivial-copy collapsed to 0.029 — the model became *unable* to
  leave a sentence alone. None of these shapes exist in the gold
  data; SFT on the same gold never produced them. As a side effect
  the over-editing also made v1 generation ~2.5× slower per batch
  (longer outputs).
- **Diagnosis path worth noting:** the collapse was caught by the
  *auxiliary* metrics (trivial-copy 0.029 vs the ~0.36 base rate;
  no-op edit counts), not by eyeballing losses — v1's DPO training
  loss looked perfectly healthy (0.693 → 0.254).
- Residual SFT errors skew toward missed insertions of short
  closed-class words ("the", "a", commas) — confident but
  conservative, consistent with prior GEC literature.
- Base + 3-shot mimics the bracket surface form but contributes no
  correction quality: its JFLEG exact matches are nearly all trivial
  copies, and it emits 0.35 no-op edits/sentence copying the few-shot
  examples' pattern.
- JFLEG (fluency-oriented, 4 refs) systematically under-rewards
  BEA-trained models — they make minimal grammatical edits where
  JFLEG references rewrite for fluency. ERRANT F0.5 on BEA dev is the
  metric matched to the training distribution; JFLEG exact-match is a
  format-independent sanity check, not a headline number.
- _[DPO v2 outcome — filled in after the retrain evaluates.]_

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
