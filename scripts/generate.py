"""Run a model on an eval JSONL and save predictions.

Usage on Colab (after training a LoRA adapter)::

    python -m scripts.generate \
        --eval data/processed/eval_bea_dev.jsonl \
        --base-model Qwen/Qwen2.5-3B-Instruct \
        --adapter <user>/qwen2.5-3b-gec-sft \
        --out results/predictions/sft_bea_dev.jsonl

Predictions JSONL schema:
    {
        "source":     "...",       # copied from the eval row
        "raw":        "...",       # exactly what the model produced
        "corrected":  "...",       # parse_inline(raw)[0]
        "parse_ok":   true/false,
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from gec.inference import generate_batch, load_model


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, type=Path,
                    help="Eval JSONL (must contain a 'source' field).")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", default=None,
                    help="Optional HF Hub id or local path of a LoRA adapter.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "bfloat16", "float16", "4bit"])
    ap.add_argument("--few-shot", action="store_true",
                    help="Prepend 3-shot examples (use for fair base-model eval).")
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="Generate only the first N rows (smoke-test).")
    args = ap.parse_args()

    rows = read_jsonl(args.eval)
    if args.limit:
        rows = rows[: args.limit]
    sentences = [r["source"] for r in rows]
    print(f"Loaded {len(sentences)} sources from {args.eval}")

    tok, model = load_model(args.base_model, adapter_id=args.adapter, dtype=args.dtype)
    print(f"Loaded model {args.base_model}"
          + (f" + adapter {args.adapter}" if args.adapter else "")
          + f" (few_shot={args.few_shot})")

    results = []
    pbar = tqdm(total=len(sentences), desc="generate")
    for start in range(0, len(sentences), args.batch_size):
        chunk = sentences[start : start + args.batch_size]
        out = generate_batch(
            chunk, tok, model,
            include_few_shot=args.few_shot,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        for r in out:
            results.append({
                "source": r.source,
                "raw": r.raw,
                "corrected": r.corrected,
                "parse_ok": r.parse_ok,
            })
        pbar.update(len(chunk))
    pbar.close()

    write_jsonl(args.out, results)
    print(f"Wrote {len(results)} predictions to {args.out}")


if __name__ == "__main__":
    main()
