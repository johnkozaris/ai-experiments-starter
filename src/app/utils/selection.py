"""Record selection parsing (1-based user expressions to 0-based indices).

Moved from services to utils; pure helper with no side effects.
"""
from __future__ import annotations


def parse_selection(expression: str | None, dataset_size: int) -> list[int]:
    if dataset_size <= 0:
        return []
    if not expression:
        # default: first record only (consistent with legacy harness)
        return [0]
    indices: set[int] = set()
    for frag in expression.split(","):
        frag = frag.strip()
        if not frag:
            continue
        if "-" in frag:
            start_s, end_s = frag.split("-", 1)
            try:
                a = int(start_s) or 1
                b = int(end_s) or 1
            except ValueError:
                continue
            if a > b:
                a, b = b, a
            for ob in range(a, b + 1):
                if ob >= 1:
                    indices.add(ob - 1)
        else:
            try:
                ob = int(frag) or 1
            except ValueError:
                continue
            if ob >= 1:
                indices.add(ob - 1)
    ordered = sorted(i for i in indices if i < dataset_size)
    return ordered if ordered else [0]

__all__ = ["parse_selection"]
