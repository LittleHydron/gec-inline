"""Build the SFT training set and the JFLEG evaluation set as JSONL.

Training JSONL schema (one example per line):
    {
        "source":    "I goes to school .",       # ungrammatical input
        "target":    "I go to school .",         # gold plain corrected sentence
        "completion":"I {goes=>go} to school .", # bracketed string the model must emit
        "messages":  [system, user, assistant]    # chat-template-ready format
    }

Evaluation JSONL schemas:

    JFLEG row (multi-reference, fluency-oriented):
    {
        "source":      "...",
        "corrections": ["ref1", "ref2", "ref3", "ref4"],
    }

    BEA W&I+LOCNESS dev row (single-reference, canonical ERRANT F0.5):
    {
        "source":      "...",
        "target":      "...",
        "completion":  "...",        # gold bracketed string from M2
    }

Usage::
    python -m scripts.build_dataset \
        --m2 data/raw/wi+locness/m2/ABC.train.gold.bea19.m2 \
        --m2 data/raw/fce/m2/fce.train.gold.bea19.m2 \
        --train-out data/processed/train.jsonl \
        --eval-out data/processed/eval.jsonl \
        --max-train 10000 --seed 3407
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from gec.m2 import iter_m2
from gec.prompts import SYSTEM_PROMPT, build_user_message
from gec.render import render_inline


def build_train(
    m2_paths: list[Path],
    max_train: int,
    seed: int,
    keep_identity_fraction: float = 0.05,
) -> list[dict]:
    """Read M2 files, render bracketed completions, return a list of examples.

    Identity examples (source == target, no edits) are mostly dropped: we
    keep ``keep_identity_fraction`` so the model still learns to recognise
    already-correct sentences. The rest of the budget goes to edited ones.
    """
    rng = random.Random(seed)
    edited: list[dict] = []
    identity: list[dict] = []

    for path in m2_paths:
        for sent in iter_m2(path):
            if len(sent.source_tokens) < 3 or len(sent.source_tokens) > 80:
                continue  # skip very short / very long sentences
            rendered = render_inline(sent.source_tokens, sent.edits)
            example = {
                "source": sent.source,
                "target": sent.target,
                "completion": rendered,
            }
            if sent.edits:
                edited.append(example)
            else:
                identity.append(example)

    rng.shuffle(edited)
    rng.shuffle(identity)

    identity_budget = int(max_train * keep_identity_fraction)
    edited_budget = max_train - identity_budget
    chosen = edited[:edited_budget] + identity[:identity_budget]
    rng.shuffle(chosen)

    for ex in chosen:
        ex["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(ex["source"])},
            {"role": "assistant", "content": ex["completion"]},
        ]
    return chosen


def build_eval_bea(m2_path: Path) -> list[dict]:
    """Read a BEA M2 dev/test file and return source/target/completion rows."""
    rows: list[dict] = []
    for sent in iter_m2(m2_path):
        if not sent.source_tokens:
            continue
        rows.append({
            "source": sent.source,
            "target": sent.target,
            "completion": render_inline(sent.source_tokens, sent.edits),
        })
    return rows


def build_eval_jfleg() -> tuple[list[dict], list[dict]]:
    """Return (validation, test) lists of {source, corrections}."""
    ds = load_dataset("jhu-clsp/jfleg")

    def _clean(s: str) -> str:
        return " ".join(s.strip().split())

    def _split(name: str) -> list[dict]:
        out = []
        for row in ds[name]:
            src = _clean(row["sentence"])
            refs = [_clean(c) for c in row["corrections"]]
            if not src:
                continue
            out.append({"source": src, "corrections": refs})
        return out

    return _split("validation"), _split("test")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--m2", action="append", required=True,
        help="Path to a BEA M2 file. Pass --m2 multiple times to concatenate.",
    )
    ap.add_argument("--train-out", type=Path, default=Path("data/processed/train.jsonl"))
    ap.add_argument(
        "--eval-out", type=Path, default=Path("data/processed/eval_jfleg_dev.jsonl"),
        help="JFLEG dev split, the multi-reference fluency benchmark.",
    )
    ap.add_argument(
        "--eval-test-out", type=Path, default=Path("data/processed/eval_jfleg_test.jsonl"),
        help="JFLEG test split, kept separate from the dev split used for tuning.",
    )
    ap.add_argument(
        "--bea-dev-m2", type=Path,
        default=Path("data/raw/wi+locness/m2/ABCN.dev.gold.bea19.m2"),
        help="BEA W&I+LOCNESS dev M2 — the canonical single-reference ERRANT F0.5 set.",
    )
    ap.add_argument(
        "--bea-dev-out", type=Path,
        default=Path("data/processed/eval_bea_dev.jsonl"),
    )
    ap.add_argument("--max-train", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    m2_paths = [Path(p) for p in args.m2]
    for p in m2_paths:
        if not p.exists():
            raise SystemExit(f"M2 file not found: {p}")

    print(f"Building training set from {len(m2_paths)} M2 file(s)…")
    train = build_train(m2_paths, max_train=args.max_train, seed=args.seed)
    write_jsonl(args.train_out, train)
    print(f"  -> wrote {len(train)} examples to {args.train_out}")

    print("Building JFLEG eval set…")
    dev, test = build_eval_jfleg()
    write_jsonl(args.eval_out, dev)
    write_jsonl(args.eval_test_out, test)
    print(f"  -> wrote {len(dev)} JFLEG dev / {len(test)} JFLEG test examples")

    if args.bea_dev_m2.exists():
        print(f"Building BEA dev eval set from {args.bea_dev_m2}…")
        bea_dev = build_eval_bea(args.bea_dev_m2)
        write_jsonl(args.bea_dev_out, bea_dev)
        print(f"  -> wrote {len(bea_dev)} BEA dev examples to {args.bea_dev_out}")
    else:
        print(f"Skipping BEA dev (file not found: {args.bea_dev_m2})")


if __name__ == "__main__":
    main()
