"""BEA-2019 M2 file parser.

M2 format primer:
    S This are gramamtically incorect .
    A 0 2|||R:VERB:SVA|||This is|||REQUIRED|||-NONE-|||0
    A 2 3|||R:SPELL|||grammatically|||REQUIRED|||-NONE-|||0
    A 3 4|||R:SPELL|||incorrect|||REQUIRED|||-NONE-|||0
    (blank line)
    S Next sentence
    A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0

Each `S` line is a whitespace-tokenized source sentence. Each following
`A` line is an edit by one annotator. `noop` means the annotator found
no errors. We keep edits from one annotator only (default annotator 0)
since edits from different annotators may overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Edit:
    start: int           # inclusive token index in source
    end: int             # exclusive token index in source (== start for insertion)
    replacement: str     # whitespace-tokenized replacement, "" for deletion
    error_type: str = "" # ERRANT label, e.g. "R:VERB:SVA"; informational


@dataclass
class M2Sentence:
    source: str                # raw source line (whitespace-tokenized)
    source_tokens: list[str]   # source.split()
    edits: list[Edit]          # edits from selected annotator
    target_tokens: list[str]   # source after applying edits

    @property
    def target(self) -> str:
        return " ".join(self.target_tokens)


def apply_edits(source_tokens: list[str], edits: list[Edit]) -> list[str]:
    """Apply non-overlapping edits to source_tokens, left to right."""
    out: list[str] = []
    cursor = 0
    for e in sorted(edits, key=lambda x: (x.start, x.end)):
        if e.start < cursor:
            raise ValueError(f"Overlapping edits at index {e.start}")
        out.extend(source_tokens[cursor : e.start])
        if e.replacement:
            out.extend(e.replacement.split())
        cursor = e.end
    out.extend(source_tokens[cursor:])
    return out


def iter_m2(path: str | Path, annotator: int = 0) -> Iterator[M2Sentence]:
    """Stream sentences from a single M2 file."""
    path = Path(path)
    source: str | None = None
    edits: list[Edit] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                if source is not None:
                    yield _build(source, edits)
                source, edits = None, []
                continue
            if line.startswith("S "):
                if source is not None:  # missing blank line — flush
                    yield _build(source, edits)
                source = line[2:]
                edits = []
            elif line.startswith("A "):
                if source is None:
                    continue
                parsed = _parse_a_line(line)
                if parsed is None:
                    continue
                start, end, err, repl, ann = parsed
                if ann != annotator:
                    continue
                if err == "noop":
                    continue
                edits.append(Edit(start=start, end=end, replacement=repl, error_type=err))
    if source is not None:
        yield _build(source, edits)


def _parse_a_line(line: str) -> tuple[int, int, str, str, int] | None:
    # "A <start> <end>|||<type>|||<repl>|||REQUIRED|||-NONE-|||<ann>"
    body = line[2:]
    parts = body.split("|||")
    if len(parts) < 6:
        return None
    span = parts[0].split()
    if len(span) != 2:
        return None
    try:
        start = int(span[0])
        end = int(span[1])
        ann = int(parts[5])
    except ValueError:
        return None
    err = parts[1]
    repl = parts[2] if parts[2] != "-NONE-" else ""
    return start, end, err, repl, ann


def _build(source: str, edits: list[Edit]) -> M2Sentence:
    tokens = source.split()
    target_tokens = apply_edits(tokens, edits)
    return M2Sentence(
        source=source,
        source_tokens=tokens,
        edits=edits,
        target_tokens=target_tokens,
    )


def load_m2_files(paths: list[str | Path], annotator: int = 0) -> Iterator[M2Sentence]:
    for p in paths:
        yield from iter_m2(p, annotator=annotator)
