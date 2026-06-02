"""Render (source tokens, edits) to the inline bracket format from the PDF.

Example:
    source_tokens = ["I", "goes", "to", "school"]
    edits = [Edit(1, 2, "go")]
    render_inline(source_tokens, edits)
        -> "I {goes=>go} to school"

Empty source span -> insertion: "{=>the}"
Empty replacement -> deletion:  "{really=>}"
Multi-token spans are joined by spaces inside the braces.
"""

from __future__ import annotations

from .m2 import Edit


def render_inline(source_tokens: list[str], edits: list[Edit]) -> str:
    edits = sorted(edits, key=lambda e: (e.start, e.end))

    out: list[str] = []
    cursor = 0
    for e in edits:
        if e.start < cursor:
            # Skip edits that overlap a previous one — keep the earlier one,
            # drop the later. Overlaps occur very rarely within one annotator.
            continue
        out.extend(source_tokens[cursor : e.start])
        src_span = " ".join(source_tokens[e.start : e.end])
        out.append("{" + src_span + "=>" + e.replacement + "}")
        cursor = e.end
    out.extend(source_tokens[cursor:])
    return " ".join(out)
