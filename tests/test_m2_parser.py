"""Unit tests for the M2 parser using inline-string M2 fragments."""

from __future__ import annotations

import io
from pathlib import Path

from gec.m2 import apply_edits, iter_m2

M2_FIXTURE = """S This are gramamtically incorect .
A 0 2|||R:VERB:SVA|||This is|||REQUIRED|||-NONE-|||0
A 2 3|||R:SPELL|||grammatically|||REQUIRED|||-NONE-|||0
A 3 4|||R:SPELL|||incorrect|||REQUIRED|||-NONE-|||0

S I really go to school .
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0

S She have did it .
A 1 3|||R:VERB:TENSE|||did|||REQUIRED|||-NONE-|||0
"""


def test_iter_m2_basic(tmp_path: Path):
    p = tmp_path / "x.m2"
    p.write_text(M2_FIXTURE, encoding="utf-8")
    sents = list(iter_m2(p))
    assert len(sents) == 3

    assert sents[0].source_tokens == ["This", "are", "gramamtically", "incorect", "."]
    assert sents[0].target == "This is grammatically incorrect ."
    assert [e.error_type for e in sents[0].edits] == [
        "R:VERB:SVA",
        "R:SPELL",
        "R:SPELL",
    ]

    # noop sentence: zero edits, target == source
    assert sents[1].edits == []
    assert sents[1].target == sents[1].source

    # multi-token replacement
    assert sents[2].target == "She did it ."


def test_apply_edits_handles_insertion_and_deletion():
    src = ["I", "to", "school"]
    from gec.m2 import Edit
    edits = [Edit(1, 1, "go")]
    assert apply_edits(src, edits) == ["I", "go", "to", "school"]

    src2 = ["I", "really", "go"]
    edits2 = [Edit(1, 2, "")]
    assert apply_edits(src2, edits2) == ["I", "go"]
