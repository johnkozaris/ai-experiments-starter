from __future__ import annotations

from typing import Any


def build_dataset(
    records: list[dict[str, Any]], config: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Transform extraction outputs into follow-up dataset.

    - Filters records with at least `min_investors` investors (default 1)
    - Each output row: {"record": <original_output>} ready for templating.
    """
    min_inv = 1
    if config and isinstance(config.get("min_investors"), int):
        min_inv = config["min_investors"]
    out: list[dict[str, Any]] = []
    for r in records:
        investors = r.get("investors") or []
        if len(investors) >= min_inv:
            out.append({"record": r})
    return out
