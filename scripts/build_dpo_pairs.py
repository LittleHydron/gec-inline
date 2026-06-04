"""Build a preference-pair JSONL for DPO from SFT predictions + gold.

Strategy:
  chosen   = the gold bracketed completion
  rejected = SFT-model output on the same source when it differs from
             gold, otherwise a programmatic corruption of gold.

The corruption is deliberately SYMMETRIC: half the time an edit is
*removed* (under-correction), half the time a spurious edit is *added*
(over-correction). Already-correct sentences (gold == source, no braces)
are also included, with the spurious-edit version as rejected.

v1 of this script only removed edits, skipped already-correct
sentences, and used cross-sentence gold as noise — three biases that
all point the same way ("more edits => preferred"). DPO amplified that
proxy into 11 edits/sentence of no-op braces and paraphrases, collapsing
JFLEG exact-match from 29.3 % to 2.1 %. See report.md.

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


def drop_edit(gold: str, rng: random.Random) -> str | None:
    """Remove one random {...} edit from the gold string (under-correction)."""
    matches = list(re.finditer(r"\{[^{}]*=>[^{}]*\}", gold))
    if not matches:
        return None
    m = rng.choice(matches)
    return gold[: m.start()] + gold[m.end() :]


# Closed-class words the over-corrector swaps in — mimics the most common
# spurious-edit shapes (articles, agreement, prepositions, paraphrase).
_SWAP_POOL = ["the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
              "to", "of", "this", "that", "very", "really"]


def add_spurious_edit(gold: str, rng: random.Random) -> str | None:
    """Wrap one plain token in a spurious {...} edit (over-correction).

    50 % a no-op edit ({word=>word}), 50 % a pointless substitution —
    both shapes the v1-trained model actually produced.
    """
    # Candidate tokens: outside any existing brace group.
    plain = [m for m in re.finditer(r"[A-Za-z']+", re.sub(r"\{[^{}]*\}", lambda m: " " * len(m.group()), gold))]
    if not plain:
        return None
    m = rng.choice(plain)
    tok = gold[m.start():m.end()]
    tgt = tok if rng.random() < 0.5 else rng.choice([w for w in _SWAP_POOL if w != tok.lower()])
    return f"{gold[: m.start()]}{{{tok}=>{tgt}}}{gold[m.end():]}"


def corrupt(gold: str, rng: random.Random) -> str | None:
    """Symmetric corruption: under-correct or over-correct with equal odds."""
    ops = [drop_edit, add_spurious_edit]
    rng.shuffle(ops)
    for op in ops:
        bad = op(gold, rng)
        if bad is not None and bad.strip() != gold.strip():
            return bad
    return None


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
    # Already-correct sentences are only ~5 % of the train sample but carry
    # the scarce "don't invent edits" signal — take ALL of them first, then
    # fill the remainder with with-edit rows.
    train.sort(key=lambda row: "{" in (row["completion"] or ""))

    pairs: list[dict] = []
    n_correct = n_sft_rej = n_corrupt_rej = 0
    for row in train:
        if len(pairs) >= args.max_pairs:
            break
        chosen = row["completion"]
        if not chosen:
            continue
        already_correct = "{" not in chosen

        rejected: str | None = None
        sft_out = sft_by_source.get(row["source"])
        if sft_out and sft_out.strip() != chosen.strip():
            rejected = sft_out.strip()
            n_sft_rej += 1
        else:
            rejected = corrupt(chosen, rng)
            if rejected:
                n_corrupt_rej += 1

        if not rejected or rejected.strip() == chosen.strip():
            continue
        if already_correct:
            n_correct += 1

        pairs.append({
            "prompt": render_prompt(row["source"]),
            "source": row["source"],          # kept for debugging
            "chosen": chosen,
            "rejected": rejected,
        })

    print(f"pair sources: {n_sft_rej} real SFT mistakes, {n_corrupt_rej} corruptions; "
          f"{n_correct} pairs with an already-correct (edit-free) chosen")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} DPO pairs to {args.out}")


if __name__ == "__main__":
    main()
