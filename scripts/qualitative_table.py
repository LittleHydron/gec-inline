"""Produce a side-by-side qualitative comparison table for the report.

Reads N prediction JSONL files (one per checkpoint) plus the eval JSONL
with gold targets, picks 15 cherry-picked "hard" examples (longest edits
or examples where models disagree) and 15 random examples, then writes
a markdown table.

Usage::

    python -m scripts.qualitative_table \
        --eval data/processed/eval_bea_dev.jsonl \
        --pred base=results/predictions/base_3shot_bea_dev.jsonl \
        --pred sft=results/predictions/sft_bea_dev.jsonl \
        --pred dpo=results/predictions/dpo_bea_dev.jsonl \
        --out results/qualitative.md \
        --seed 3407
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument(
        "--pred", action="append", required=True,
        help="NAME=path to predictions JSONL. Pass multiple times.",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-hard", type=int, default=15)
    ap.add_argument("--n-random", type=int, default=15)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    gold = read_jsonl(args.eval)
    preds: dict[str, list[dict]] = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        preds[name] = read_jsonl(Path(path))
        if len(preds[name]) != len(gold):
            raise SystemExit(
                f"{name}: {len(preds[name])} predictions vs {len(gold)} gold rows")

    rng = random.Random(args.seed)
    n = len(gold)

    # Hardness heuristic: sentences where (a) gold has many edits AND
    # (b) at least one model disagrees with another model's output.
    def disagree(i: int) -> int:
        outs = {preds[name][i]["corrected"].strip() for name in preds}
        return len(outs)

    def edit_count(i: int) -> int:
        return gold[i].get("completion", "").count("{")

    indices = list(range(n))
    indices.sort(key=lambda i: (-disagree(i), -edit_count(i)))
    hard = indices[: args.n_hard]

    pool = [i for i in range(n) if i not in set(hard)]
    rng.shuffle(pool)
    rand = pool[: args.n_random]

    selected = [("hard", i) for i in hard] + [("random", i) for i in rand]

    cols = ["#", "kind", "source", "gold"] + list(preds.keys())
    out_lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row_no, (kind, i) in enumerate(selected, 1):
        g = gold[i]
        row = [str(row_no), kind, md_escape(g["source"]), md_escape(g.get("completion", g.get("target", "")))]
        for name in preds:
            row.append(md_escape(preds[name][i]["raw"]))
        out_lines.append("| " + " | ".join(row) + " |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} rows to {args.out}")


if __name__ == "__main__":
    main()
