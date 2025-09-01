"""Analytics service functions operating on RunSummary models.

Pure calculations only - no file IO here.
"""
from __future__ import annotations

from collections.abc import Iterable

from app.domain.models import Analytics, ExtremeValue, RunSummary


def build_analytics(summaries: Iterable[RunSummary]) -> Analytics:
    items = list(summaries)
    if not items:
        return Analytics(
            total_runs=0,
        )
    lat_series = [s.avg_latency_ms for s in items if s.avg_latency_ms is not None]
    success_series = [s.success_pct for s in items if s.success_pct is not None]
    token_totals = [s.total_tokens for s in items if s.total_tokens is not None]
    cost_series = [s.est_cost_total for s in items if s.est_cost_total is not None]

    def _extreme(key: str, reverse: bool = False):
        filtered = [s for s in items if getattr(s, key) is not None]
        if not filtered:
            return None
        chosen = max(filtered, key=lambda x: getattr(x, key)) if reverse else min(
            filtered, key=lambda x: getattr(x, key)
        )
        return ExtremeValue(value=getattr(chosen, key), run_dir=chosen.run_dir)

    cumulative_cost = sum(c for c in cost_series)
    avg_cost = (cumulative_cost / len(cost_series)) if cost_series else None
    return Analytics(
        total_runs=len(items),
        cumulative_prompt_tokens=sum(s.prompt_tokens for s in items),
        cumulative_completion_tokens=sum(s.completion_tokens for s in items),
        cumulative_total_tokens=sum(s.total_tokens for s in items),
        cumulative_refusals=sum(s.refusals for s in items),
        cumulative_parse_fallbacks=sum(s.parse_fallbacks for s in items),
        cumulative_cost=cumulative_cost,
        latency_series=lat_series,
        success_series=success_series,
        token_totals_series=token_totals,
        cost_series=cost_series,
        avg_cost=avg_cost,
        fastest_run=_extreme("avg_latency_ms"),
        slowest_run=_extreme("avg_latency_ms", reverse=True),
        lowest_success=_extreme("success_pct"),
        highest_success=_extreme("success_pct", reverse=True),
    )
