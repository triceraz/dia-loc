"""JSONL loader for the contrast pairs.

The on-disk schema (one JSON object per line):
    {"id": "d1_001",
     "lang_a": "bm", "lang_b": "nn",
     "text_a": "...", "text_b": "...",
     "source": "..."}

Loader returns a list of dicts in file order. Skips empty lines so a
re-saved file with a trailing newline doesn't blow up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class ContrastPair(TypedDict):
    id: str
    lang_a: str
    lang_b: str
    text_a: str
    text_b: str
    source: str


def load_pairs(path: Path) -> list[ContrastPair]:
    out: list[ContrastPair] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(row)
    return out
