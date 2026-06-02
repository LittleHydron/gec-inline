"""Round-trip integration test on real BEA M2 data.

Skipped automatically when data/raw/*.m2 has not been downloaded yet.
For every M2 sentence we check that render(source_tokens, edits) parsed
back yields exactly the gold target produced by apply_edits.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from gec.m2 import iter_m2
from gec.parse import recover_correction
from gec.render import render_inline


_ROOT = Path(__file__).resolve().parents[1]
_RAW = _ROOT / "data" / "raw"
M2_GLOBS = ["wi+locness/m2/*.m2", "fce/m2/*.m2", "*.m2"]


def _find_m2_files() -> list[Path]:
    if not _RAW.exists():
        return []
    found: list[Path] = []
    for pattern in M2_GLOBS:
        found.extend(_RAW.glob(pattern))
    # dedup while preserving order
    seen, out = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


@pytest.mark.parametrize("seed", [0])
def test_render_parse_recovers_target_on_real_m2(seed: int):
    files = _find_m2_files()
    if not files:
        pytest.skip("No M2 files in data/raw/ — run scripts/download_data.py first.")

    sentences = []
    for f in files:
        sentences.extend(list(iter_m2(f)))
        if len(sentences) >= 500:
            break

    rng = random.Random(seed)
    sample = rng.sample(sentences, min(200, len(sentences)))
    failures = []
    for s in sample:
        rendered = render_inline(s.source_tokens, s.edits)
        recovered = recover_correction(rendered)
        if recovered != s.target:
            failures.append((s.source, s.target, rendered, recovered))

    assert not failures, (
        f"{len(failures)}/{len(sample)} sentences failed round-trip; first 3:\n"
        + "\n---\n".join(
            f"src={src!r}\ngold={gold!r}\nrendered={r!r}\nrecovered={rec!r}"
            for (src, gold, r, rec) in failures[:3]
        )
    )
