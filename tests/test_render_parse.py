"""Round-trip tests for the inline bracket format.

These tests rely only on the in-process modules (no downloads). The
larger end-to-end round-trip across the real BEA M2 corpus lives in
``test_m2_roundtrip.py`` and is skipped when the data is not present.
"""

from __future__ import annotations

import pytest

from gec.m2 import Edit
from gec.parse import parse_inline, recover_correction
from gec.render import render_inline


# ---------- render ----------


def test_render_simple_substitution():
    src = "I goes to school".split()
    edits = [Edit(1, 2, "go", "R:VERB:SVA")]
    assert render_inline(src, edits) == "I {goes=>go} to school"


def test_render_insertion():
    src = "I to school".split()
    edits = [Edit(1, 1, "go", "M:VERB")]
    assert render_inline(src, edits) == "I {=>go} to school"


def test_render_deletion():
    src = "I really go to school".split()
    edits = [Edit(1, 2, "", "U:ADV")]
    assert render_inline(src, edits) == "I {really=>} go to school"


def test_render_multitoken_replacement():
    src = "She have did it".split()
    edits = [Edit(1, 3, "did", "R:VERB:TENSE")]
    assert render_inline(src, edits) == "She {have did=>did} it"


def test_render_multiple_edits_in_order():
    src = "I goes to schol".split()
    edits = [Edit(1, 2, "go"), Edit(3, 4, "school")]
    assert render_inline(src, edits) == "I {goes=>go} to {schol=>school}"


def test_render_unordered_edits_are_sorted():
    src = "I goes to schol".split()
    edits = [Edit(3, 4, "school"), Edit(1, 2, "go")]
    assert render_inline(src, edits) == "I {goes=>go} to {schol=>school}"


def test_render_no_edits():
    src = "I go to school".split()
    assert render_inline(src, []) == "I go to school"


# ---------- parse ----------


def test_parse_substitution():
    text = "I {goes=>go} to school"
    corrected, edits, ok = parse_inline(text)
    assert ok
    assert corrected == "I go to school"
    assert len(edits) == 1
    assert edits[0].src == ["goes"] and edits[0].tgt == ["go"]


def test_parse_insertion():
    text = "I {=>really} go to school"
    corrected, edits, ok = parse_inline(text)
    assert ok
    assert corrected == "I really go to school"
    assert edits[0].src == [] and edits[0].tgt == ["really"]


def test_parse_deletion():
    text = "I {really=>} go to school"
    corrected, edits, ok = parse_inline(text)
    assert ok
    assert corrected == "I go to school"
    assert edits[0].src == ["really"] and edits[0].tgt == []


def test_parse_multitoken():
    text = "She {have did=>did} it"
    corrected, edits, ok = parse_inline(text)
    assert ok
    assert corrected == "She did it"
    assert edits[0].src == ["have", "did"]
    assert edits[0].tgt == ["did"]


def test_parse_malformed_returns_not_ok():
    text = "I {goes=>go to school"
    corrected, edits, ok = parse_inline(text)
    assert not ok
    # Permissive: we still try to recover something usable.
    assert "I" in corrected


def test_parse_no_edits_passthrough():
    corrected, edits, ok = parse_inline("I go to school")
    assert ok
    assert corrected == "I go to school"
    assert edits == []


# ---------- round-trip ----------


@pytest.mark.parametrize(
    "src, edits, expected_target",
    [
        ("I goes to school".split(), [Edit(1, 2, "go")], "I go to school"),
        ("I to school".split(), [Edit(1, 1, "go")], "I go to school"),
        ("I really go".split(), [Edit(1, 2, "")], "I go"),
        ("She have did it".split(), [Edit(1, 3, "did")], "She did it"),
        (
            "I goes to schol".split(),
            [Edit(1, 2, "go"), Edit(3, 4, "school")],
            "I go to school",
        ),
        ("I go to school".split(), [], "I go to school"),
    ],
)
def test_render_then_parse_recovers_target(src, edits, expected_target):
    rendered = render_inline(src, edits)
    recovered = recover_correction(rendered)
    assert recovered == expected_target
