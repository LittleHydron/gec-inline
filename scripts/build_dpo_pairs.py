"""Build a preference-pair JSONL for DPO from SFT predictions + gold.

Strategy:
  chosen   = the gold bracketed completion
  rejected = sampled from one of three sources, in order of preference:
             1. SFT-model output on the same source if it differs from gold
             2. SFT output on a *different* source (cross-sentence noise)
             3. Programmatic corruption of gold (drop a random brace)

Output JSONL schema (matches TRL DPOTrainer expectations after
``apply_chat_template``)::

    {"prompt": "<chat-template-rendered system+user>",
     "chosen": "<gold completion>",
     "rejected": "<bad completion>"}

The prompt is left in the same chat-template-rendered form as training
so the tokenizer-side templating is consistent between SFT and DPO.

Usage::

    python -m scripts.build_dpo_pairs \
        --train data/processed/train.jsonl \
        --sft-preds results/predictions/sft_train_subset.jsonl \
        --out data/processed/dpo.jsonl \
        --max-pairs 3000 --seed 3407
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from gec.prompts import SYSTEM_PROMPT, build_user_message


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def corrupt(gold: str, rng: random.Random) -> str:
    """Remove one random {...} edit from the gold string (creates an under-correction)."""
    matches = list(re.finditer(r"\{[^{}]*=>[^{}]*\}", gold))
    if not matches:
        return gold + " "  # whitespace-only change is still rejected by exact match
    m = rng.choice(matches)
    return gold[: m.start()] + gold[m.end() :]


def render_prompt(source: str) -> str:
    """A minimal text form of (system, user) that's stable across tokenizers."""
    return f"[SYSTEM] {SYSTEM_PROMPT}\n[USER] {build_user_message(source)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path,
                    help="train.jsonl from build_dataset.py")
    ap.add_argument("--sft-preds", type=Path,
                    help="Predictions JSONL from running SFT model on the training sources "
                         "(optional; if absent we only use corruption-based rejections).")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-pairs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train = read_jsonl(args.train)
    by_source = {row["source"]: row for row in train}

    sft_by_source: dict[str, str] = {}
    if args.sft_preds and args.sft_preds.exists():
        for r in read_jsonl(args.sft_preds):
            sft_by_source[r["source"]] = r["raw"]

    rng.shuffle(train)

    pairs: list[dict] = []
    cross_pool = [row["completion"] for row in train]
    for row in train:
        if len(pairs) >= args.max_pairs:
            break
        if not row["completion"] or "{" not in row["completion"]:
            continue  # skip already-correct sentences — DPO loses signal here

        chosen = row["completion"]
        rejected: str | None = None

        sft_out = sft_by_source.get(row["source"])
        if sft_out and sft_out.strip() != chosen.strip():
            rejected = sft_out.strip()
        else:
            roll = rng.random()
            if roll < 0.5:
                rejected = corrupt(chosen, rng)
            else:
                rejected = rng.choice(cross_pool)
                if rejected.strip() == chosen.strip():
                    rejected = corrupt(chosen, rng)

        if not rejected or rejected.strip() == chosen.strip():
            continue

        pairs.append({
            "prompt": render_prompt(row["source"]),
            "source": row["source"],          # kept for debugging
            "chosen": chosen,
            "rejected": rejected,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} DPO pairs to {args.out}")


if __name__ == "__main__":
    main()
