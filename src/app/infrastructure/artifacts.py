"""Persistence of run artifacts (manifest, results JSONL, per-record) for new arch."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import orjson

from app.domain.models import PromptSnapshot, RecordResult, RunManifest, RunSummary


def _public_record_view(r: RecordResult) -> dict:
    """Return slim dict for persistence (exclude metrics-style/internal fields).

    Contains:
        index (1-based for readability), output_text, structured_output
    Future: add minimal provenance if needed.
    """
    data = {
        "index": r.index,
        "output_text": r.output_text,
        "structured_output": r.structured_output,
    }
    if r.error_message:
        data["error_message"] = r.error_message
    return data


def ensure_run_dir(base: Path, timestamp: str) -> Path:
    run_dir = base / timestamp
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_prompt_snapshot(run_dir: Path, snapshot: PromptSnapshot) -> Path:
    p = run_dir / "prompts_used.json"
    p.write_bytes(orjson.dumps(snapshot.model_dump(), option=orjson.OPT_INDENT_2))
    return p


def write_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    p = run_dir / "experiment_manifest.json"
    p.write_bytes(orjson.dumps(manifest.model_dump(), option=orjson.OPT_INDENT_2))
    return p


def append_results_jsonl(run_dir: Path, results: Iterable[RecordResult]) -> Path:
    p = run_dir / "experiment_results.jsonl"
    with p.open("ab") as f:
        for r in results:
            f.write(orjson.dumps(_public_record_view(r)) + b"\n")
    return p


def write_record_detail(run_dir: Path, r: RecordResult) -> None:
    rec_dir = run_dir / f"record{r.index + 1}"
    rec_dir.mkdir(parents=True, exist_ok=True)
    # Write input manifest (legacy compatibility + debugging)
    import orjson as _oj  # local import for speed
    (rec_dir / "input_manifest.json").write_bytes(
        _oj.dumps({"index": r.index + 1, "input": r.input}, option=_oj.OPT_INDENT_2)
    )
    (rec_dir / "result.json").write_bytes(
        orjson.dumps(_public_record_view(r), option=orjson.OPT_INDENT_2)
    )


def append_run_summary(summary_path: Path, summary: RunSummary) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("ab") as f:
        f.write(orjson.dumps(summary.model_dump()) + b"\n")
