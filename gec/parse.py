"""Parse the inline bracket format back into a clean corrected sentence.

The renderer guarantees a well-formed string, but model output may be
malformed (missing brace, nested, extra whitespace). The parser is
tolerant: anything it cannot understand as `{src=>tgt}` is passed
through as plain text.

Public API:
  parse_inline(text)        -> (corrected_sentence, edits, parse_ok)
  recover_correction(text)  -> corrected_sentence only (shortcut)

Edit tuples returned are (src_tokens: list[str], tgt_tokens: list[str]),
matching the BEA/ERRANT convention of whitespace-tokenized spans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Match {SRC=>TGT}. Both sides can be empty. We are permissive about
# whitespace adjacent to `=>` and inside the braces.
_BRACKET_RE = re.compile(r"\{([^{}=]*?)=>([^{}=]*?)\}")


@dataclass
class ParsedEdit:
    src: list[str]
    tgt: list[str]


def parse_inline(text: str) -> tuple[str, list[ParsedEdit], bool]:
    """Return (corrected_text, edits, parse_ok).

    parse_ok is False when the input contains an unmatched brace
    (`{` or `}` that the bracket regex did not consume) — caller may
    fall back to treating the raw text as the corrected sentence.
    """
    edits: list[ParsedEdit] = []

    def _sub(m: re.Match) -> str:
        src = m.group(1).strip().split()
        tgt = m.group(2).strip().split()
        edits.append(ParsedEdit(src=src, tgt=tgt))
        return m.group(2).strip()

    replaced = _BRACKET_RE.sub(_sub, text)

    # Unmatched braces left behind -> the output was malformed.
    parse_ok = "{" not in replaced and "}" not in replaced

    # Strip duplicate spaces that arise from deletions / empty inserts.
    cleaned = re.sub(r"\s+", " ", replaced).strip()
    return cleaned, edits, parse_ok


def recover_correction(text: str) -> str:
    """Shortcut: return only the corrected sentence."""
    corrected, _, _ = parse_inline(text)
    return corrected
