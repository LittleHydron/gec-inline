---
title: GEC Inline Edits
emoji: ✏️
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: mit
short_description: Grammatical error correction with inline {wrong=>right} edits.
---

# ✏️ GEC inline-edit fine-tune

LoRA-fine-tuned **Qwen2.5-3B-Instruct** that takes an English sentence
and emits the corrected version with every edit inlined in braces:

```
in    She have did her homework already .
out   She {have did=>has done} her homework already .
```

The bracket syntax is taken verbatim from the LPNLP fine-tuning
assignment prompt. Built for the third LPNLP homework — see the
assignment PDF in the parent directory.

## What's in the box (matches the submission rubric)

| Component | Implementation |
| --- | --- |
| **Task** | Grammatical error correction (GEC) with inline `{src=>tgt}` edits. Insertions render as `{=>tgt}`, deletions as `{src=>}`, multi-token spans as `{a b=>c}`. |
| **Base model** | `Qwen/Qwen2.5-3B-Instruct` (3.1 B parameters, English-strong, mature Unsloth kernels, fits free Colab T4 in 4-bit). |
| **Training data** | BEA-2019 W&I+LOCNESS `ABC.train` + FCE `train`, both as M2 — 10 000 sampled sentences (~95 % with edits, ~5 % already-correct). Built by `scripts/build_dataset.py` from the official Cambridge tarballs. |
| **Evaluation** | (1) ERRANT F0.5 on **BEA-2019 W&I+LOCNESS dev** (`ABCN.dev`, 4 384 sentences) — canonical single-reference GEC metric. (2) Exact-match accuracy on **JFLEG dev** (754 sentences, 4 references each). (3) Parse-failure rate and trivial-copy rate as auxiliary metrics. (4) 30-row qualitative side-by-side table in `results/qualitative.md`. |
| **Training framework** | [Unsloth](https://github.com/unslothai/unsloth) (PDF-recommended; QLoRA in 4-bit with `train_on_responses_only`) on free Colab T4. |
| **Fine-tuning** | LoRA SFT (rank 16, α 32, target = all linear modules, lr 2e-4, 2 epochs, cosine schedule, weight decay 0.01, adamw_8bit, batch 2 × grad-accum 4 = effective 8). |
| **Preference optimisation (bonus)** | DPO on top of the SFT adapter using preference pairs synthesized from gold + SFT outputs (β = 0.1, lr 5e-6, 1 epoch). |
| **Demo** | This Gradio Space. Model dropdown switches between "Base + few-shot" (fair zero-training baseline), "SFT", and "DPO". Output shows the raw bracketed string, an HTML-highlighted diff, and the cleaned-up corrected sentence. |

## Running locally

Python 3.12 is fine; ERRANT (the eval metric) installs cleanly with the
pinned versions in `requirements.txt`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm

# Fetch BEA-2019 + JFLEG, format the train/eval JSONLs:
mkdir -p data/raw && cd data/raw
curl -O https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz
curl -O https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz
tar -xzf wi+locness_v2.1.bea19.tar.gz
tar -xzf fce_v2.1.bea19.tar.gz
cd ../..

.venv/bin/python -m scripts.build_dataset \
    --m2 data/raw/wi+locness/m2/ABC.train.gold.bea19.m2 \
    --m2 data/raw/fce/m2/fce.train.gold.bea19.m2

# Then either open notebooks/01_sft_qwen25_3b.ipynb on Colab,
# OR run inference + eval against a published adapter:
.venv/bin/python -m scripts.generate \
    --eval data/processed/eval_bea_dev.jsonl \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --adapter <user>/qwen2.5-3b-gec-sft \
    --out results/predictions/sft_bea_dev.jsonl

.venv/bin/python -m scripts.eval --mode bea \
    --predictions results/predictions/sft_bea_dev.jsonl \
    --ref-m2 data/raw/wi+locness/m2/ABCN.dev.gold.bea19.m2 \
    --out results/metrics/sft_bea_dev.json
```

To launch the Gradio demo locally with both adapters loaded:

```bash
GEC_SFT_ADAPTER=<user>/qwen2.5-3b-gec-sft \
GEC_DPO_ADAPTER=<user>/qwen2.5-3b-gec-dpo \
.venv/bin/python app.py
```

## Training (Colab)

1. Push this repo (or a fork) to GitHub.
2. Open `notebooks/01_sft_qwen25_3b.ipynb` on Colab with a T4 runtime.
3. Edit `REPO_URL` and `HF_USER` cells; run all. Wall time ≈ 45–60 min.
4. Open `notebooks/02_dpo_qwen25_3b.ipynb` and repeat for the DPO stage.

Notebooks are generated from `notebooks/_build_notebooks.py` — edit the
cell definitions there and re-run the script if you want to tweak the
training recipe.

## Deploying to HuggingFace Spaces

The `README.md` frontmatter at the top already configures Spaces for
Gradio. Use **ZeroGPU** hardware (free for verified users).

```bash
hf repo create <user>/gec-inline --type space --space-sdk gradio --public
hf repo settings <user>/gec-inline --space-hardware zero-a10g
hf upload <user>/gec-inline . . --type space \
    --exclude ".venv/*" --exclude "**/__pycache__/**" \
    --exclude ".git/*" --exclude "data/raw/*"

# Set the adapter env vars on the Space (Settings -> Variables and secrets):
#   GEC_SFT_ADAPTER = <user>/qwen2.5-3b-gec-sft
#   GEC_DPO_ADAPTER = <user>/qwen2.5-3b-gec-dpo
```

## Repo layout

```
gec-inline/
├── app.py                          # Gradio UI (ZeroGPU-ready)
├── gec/
│   ├── m2.py                       # BEA-2019 M2 file parser
│   ├── render.py                   # (source, edits) -> "I {goes=>go} to school"
│   ├── parse.py                    # bracketed string -> corrected sentence
│   ├── prompts.py                  # system prompt + 3-shot few-shot
│   └── inference.py                # transformers + peft loader / generate
├── scripts/
│   ├── build_dataset.py            # M2 -> train.jsonl, JFLEG -> eval.jsonl
│   ├── build_dpo_pairs.py          # gold + SFT outputs -> dpo.jsonl
│   ├── generate.py                 # run a model on an eval set -> predictions
│   ├── eval.py                     # predictions -> ERRANT F0.5 (BEA) or exact-match (JFLEG)
│   └── qualitative_table.py        # side-by-side comparison markdown
├── notebooks/
│   ├── 01_sft_qwen25_3b.ipynb      # Colab SFT (Unsloth)
│   ├── 02_dpo_qwen25_3b.ipynb      # Colab DPO (TRL)
│   └── _build_notebooks.py         # regenerator
├── tests/                          # render/parse round-trip + M2 parser tests
├── data/
│   ├── raw/                        # BEA + JFLEG (gitignored)
│   └── processed/                  # train.jsonl, eval_*.jsonl, dpo.jsonl
├── results/
│   ├── predictions/                # per-checkpoint JSONL
│   ├── metrics/                    # per-checkpoint JSON
│   └── qualitative.md
├── requirements.txt
├── requirements-colab.txt
├── report.md                       # write-up matching the PDF rubric
└── README.md
```
