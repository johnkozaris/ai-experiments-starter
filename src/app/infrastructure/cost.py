"""Cost estimation (isolated) with internal static pricing table.

No dependency on legacy app modules to enforce layering.
Prices are per-million tokens (input/output). Values can be updated centrally.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Price:
    input_per_million: float
    output_per_million: float

    def estimate(self, prompt: int | None, completion: int | None) -> float | None:
        if not prompt and not completion:
            return None
        p_cost = ((prompt or 0) / 1_000_000) * self.input_per_million
        c_cost = ((completion or 0) / 1_000_000) * self.output_per_million
        return round(p_cost + c_cost, 6)


_PRICING: Mapping[str, _Price] = {
    "gpt-4o": _Price(2.50, 10.00),
    "gpt-4o-mini": _Price(0.15, 0.60),
    "gpt-4o-2024-05-13": _Price(5.00, 15.00),
    "gpt-4.1": _Price(2.00, 8.00),
    "gpt-4.1-mini": _Price(0.40, 1.60),
    "gpt-4.1-nano": _Price(0.10, 0.40),
    "gpt-5": _Price(1.25, 10.00),
    "gpt-5-mini": _Price(0.25, 2.00),
    "gpt-5-nano": _Price(0.05, 0.40),
    "gpt-4o-mini-search-preview": _Price(0.15, 0.60),
}


def estimate_run_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    price = _PRICING.get(model)
    if not price:
        return None
    return price.estimate(prompt_tokens, completion_tokens)


__all__ = ["estimate_run_cost"]
