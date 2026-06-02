"""Score model predictions against gold references.

Two evaluation modes, selected by ``--mode``:

1. ``bea`` -- single-reference ERRANT F0.5 against a BEA M2 file
   (canonical metric for the BEA-2019 shared task). Requires the
   ``errant_parallel`` and ``errant_compare`` CLIs from the ``errant``
   package on PATH.

2. ``jfleg`` -- multi-reference exact-match accuracy on JFLEG
   (sentence matches *any* of the 4 references). Doesn't need ERRANT.

Both modes also report:
- parse-failure rate (how often the model's output had unmatched braces),
- mean edit count per prediction,
- "trivial copy" rate (model emitted the source unchanged).

Usage::

    python -m scripts.eval --mode bea \
        --predictions results/predictions/sft_bea_dev.jsonl \
        --ref-m2     data/raw/wi+locness/m2/ABCN.dev.gold.bea19.m2 \
        --out         results/metrics/sft_bea_dev.json

    python -m scripts.eval --mode jfleg \
        --predictions results/predictions/sft_jfleg_dev.jsonl \
        --jfleg-jsonl data/processed/eval_jfleg_dev.jsonl \
        --out         results/metrics/sft_jfleg_dev.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from gec.parse import parse_inline

WS_RE = re.compile(r"\s+")


def _errant_bin(name: str) -> str:
    """Resolve an ERRANT CLI binary, preferring the current venv."""
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else name


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(s: str) -> str:
    return WS_RE.sub(" ", s.strip())


# -------------- BEA / ERRANT --------------


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def m2_sources_in_order(m2_path: Path) -> list[str]:
    """Return the S-lines from the M2 file (preserves order)."""
    out: list[str] = []
    with m2_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("S "):
                out.append(line[2:].rstrip("\n"))
    return out


def errant_score(hyp_m2: Path, ref_m2: Path) -> dict:
    """Run errant_compare and parse out P / R / F0.5."""
    cp = subprocess.run(
        [_errant_bin("errant_compare"), "-hyp", str(hyp_m2), "-ref", str(ref_m2)],
        check=True, capture_output=True, text=True,
    )
    out = cp.stdout
    # Output ends with a row "TP\tFP\tFN\tPrec\tRec\tF0.5"
    # followed by another with the values.
    lines = [ln for ln in out.splitlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        tokens = ln.split()
        if "Prec" in tokens and "F0.5" in tokens:
            header_idx = i
            break
    if header_idx is None or header_idx + 1 >= len(lines):
        raise RuntimeError(f"Could not parse errant_compare output:\n{out}")
    header = lines[header_idx].split()
    values = lines[header_idx + 1].split()
    metrics = dict(zip(header, values))
    return {
        "TP": int(metrics["TP"]),
        "FP": int(metrics["FP"]),
        "FN": int(metrics["FN"]),
        "P":  float(metrics["Prec"]),
        "R":  float(metrics["Rec"]),
        "F0.5": float(metrics["F0.5"]),
        "raw": out,
    }


def eval_bea(args) -> dict:
    predictions = read_jsonl(args.predictions)
    sources_in_m2 = m2_sources_in_order(args.ref_m2)

    if len(predictions) != len(sources_in_m2):
        raise SystemExit(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(sources_in_m2)} sources in {args.ref_m2}. "
            "Generate against the same eval file used to build the M2.")

    for i, (pred, src) in enumerate(zip(predictions, sources_in_m2)):
        if norm(pred["source"]) != norm(src):
            raise SystemExit(
                f"Source order mismatch at row {i}:\n"
                f"  pred:  {pred['source']!r}\n"
                f"  m2:    {src!r}")

    work = args.out.parent / (args.out.stem + "_work")
    work.mkdir(parents=True, exist_ok=True)
    src_txt = work / "src.txt"
    hyp_txt = work / "hyp.txt"
    hyp_m2 = work / "hyp.m2"
    write_lines(src_txt, [norm(p["source"]) for p in predictions])
    write_lines(hyp_txt, [norm(p["corrected"]) for p in predictions])

    print(f"Building hypothesis M2 via errant_parallel -> {hyp_m2}")
    subprocess.run(
        [_errant_bin("errant_parallel"), "-orig", str(src_txt), "-cor", str(hyp_txt), "-out", str(hyp_m2)],
        check=True,
    )
    print("Scoring with errant_compare …")
    err = errant_score(hyp_m2, args.ref_m2)

    parse_fail = sum(1 for p in predictions if not p["parse_ok"]) / len(predictions)
    trivial = sum(1 for p in predictions if norm(p["corrected"]) == norm(p["source"]))
    return {
        "mode": "bea",
        "n": len(predictions),
        "parse_failure_rate": parse_fail,
        "trivial_copy_rate": trivial / len(predictions),
        "errant": err,
    }


# -------------- JFLEG --------------


def eval_jfleg(args) -> dict:
    predictions = read_jsonl(args.predictions)
    refs = read_jsonl(args.jfleg_jsonl)
    if len(predictions) != len(refs):
        raise SystemExit(
            f"Length mismatch: {len(predictions)} predictions vs {len(refs)} JFLEG rows")

    exact = 0
    for pred, ref in zip(predictions, refs):
        if norm(pred["source"]) != norm(ref["source"]):
            raise SystemExit("Source order mismatch on JFLEG.")
        candidates = {norm(c) for c in ref["corrections"]}
        if norm(pred["corrected"]) in candidates:
            exact += 1

    parse_fail = sum(1 for p in predictions if not p["parse_ok"]) / len(predictions)
    trivial = sum(1 for p in predictions if norm(p["corrected"]) == norm(p["source"]))
    return {
        "mode": "jfleg",
        "n": len(predictions),
        "exact_match_rate": exact / len(predictions),
        "parse_failure_rate": parse_fail,
        "trivial_copy_rate": trivial / len(predictions),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bea", "jfleg"], required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--ref-m2", type=Path, help="BEA M2 reference (mode=bea)")
    ap.add_argument("--jfleg-jsonl", type=Path, help="JFLEG eval JSONL (mode=jfleg)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.mode == "bea":
        if not args.ref_m2:
            raise SystemExit("--ref-m2 is required for mode=bea")
        metrics = eval_bea(args)
    else:
        if not args.jfleg_jsonl:
            raise SystemExit("--jfleg-jsonl is required for mode=jfleg")
        metrics = eval_jfleg(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "errant"}, indent=2))
    if "errant" in metrics:
        e = metrics["errant"]
        print(f"\nERRANT: P={e['P']:.4f} R={e['R']:.4f} F0.5={e['F0.5']:.4f}"
              f"  (TP={e['TP']} FP={e['FP']} FN={e['FN']})")


if __name__ == "__main__":
    main()
