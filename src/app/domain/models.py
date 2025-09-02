"""Domain models (Pydantic) defining stable contracts for experiment runs."""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "developer", "user"]
Provider = Literal["openai", "azure"]


# -------------------- LLM Usage & Request Logging -------------------- #


class LLMUsage(BaseModel):
    """Token usage returned by model provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMRequestLogEntry(BaseModel):
    """Structured per-call request/response metadata for debugging & audits."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model: str
    provider: Provider
    messages_count: int
    instructions_present: bool
    latency_ms: float | None = None
    refused: bool = False
    parse_fallback: bool = False
    usage: LLMUsage | None = None


# -------------------- Experiment Specification -------------------- #


class DependencySpec(BaseModel):
    """Declares an upstream experiment dependency for chaining outputs."""

    experiment: str
    select: str = "output"
    transform_ref: str | None = None
    transform_config: dict[str, Any] | None = None


class ExperimentSpec(BaseModel):
    """User-authored configuration (YAML) for an experiment run.

    system_prompt now optional - when omitted only the user (and optional developer) prompt
    will be sent. This enables simpler experiments relying solely on a user template.
    """

    name: str
    description: str | None = None
    model: str
    provider: Provider = "openai"
    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logprobs: int | None = None
    system_prompt: str | None = None
    user_prompt: str
    developer_prompt: str | None = None
    instructions: list[str] | str | None = None
    dataset_path: str | None = None
    schema_model: str | None = None  # dotted import path for structured output model
    depends_on: list[DependencySpec] = Field(default_factory=list)
    expected_variables: list[str] | None = None
    record_selection: str | None = None


class ResolvedExperiment(BaseModel):
    """Spec plus resolved absolute paths & loaded dataset (if any) + optional schema.

    The legacy implementation attached dynamic attributes; here we model them explicitly
    for type-safety.
    """

    spec: ExperimentSpec
    root_dir: str
    system_prompt_path: str | None = None
    user_prompt_path: str
    developer_prompt_path: str | None = None
    instruction_paths: list[str] = Field(default_factory=list)
    dataset_records: list[dict[str, Any]] | None = None
    # Structured output (optional)
    schema_model_path: str | None = None
    model_schema_json: dict[str, Any] | None = None

    # NOTE: Pydantic can't easily store arbitrary model subclass type without validation.
    # We expose a private attribute for runtime (non-serialized) access to the imported class.
    model_config = {
        "arbitrary_types_allowed": True,
        "exclude": {"schema_model_cls"},
    }
    schema_model_cls: Any | None = None  # populated at resolve time (not serialized)


# -------------------- Prompt & Record Artifacts -------------------- #


class PromptSnapshot(BaseModel):
    """Snapshot of prompts used for a run (immutable record)."""

    system: str
    user_template: str
    developer: str | None = None
    instructions: list[str] = Field(default_factory=list)


class RecordResult(BaseModel):
    """Outcome of processing a single dataset record."""

    index: int
    input: dict[str, Any]
    output_text: str | None
    structured_output: dict[str, Any] | None = None
    error_message: str | None = None
    refused: bool = False
    validated: bool = False
    validation_errors: list[str] | None = None
    usage: LLMUsage | None = None
    latency_ms: float | None = None
    parse_fallback: bool = False


class RunMetrics(BaseModel):
    """Aggregated metrics for the record loop (embedded in manifest & analytics)."""

    total_records: int = 0
    validated: int = 0
    refusals: int = 0
    parse_fallbacks: int = 0
    success_pct: float | None = None
    avg_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latencies_ms: list[float] = Field(default_factory=list)


class RunManifest(BaseModel):
    """High-level metadata about a run (no per-record outputs).

    Deliberately slim: excludes dynamic metrics, token usage, and cost. Those live
    in per-run `metrics.json` (and longitudinal aggregates in `analytics.json`).
    """

    experiment_name: str
    timestamp: str
    model: str
    provider: Provider
    parameters: dict[str, Any]
    dataset_size: int
    selected_count: int


class RunSummary(BaseModel):
    """Flattened, append-only summary for longitudinal analysis."""

    run_dir: str
    timestamp: str
    count: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    avg_latency_ms: float | None = None
    success_pct: float | None = None
    refusals: int = 0
    parse_fallbacks: int = 0
    est_cost_total: float | None = None


class ExtremeValue(BaseModel):
    value: float | int | None
    run_dir: str


class Analytics(BaseModel):
    """Aggregated longitudinal analytics across runs for an experiment."""

    total_runs: int
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0
    cumulative_total_tokens: int = 0
    cumulative_refusals: int = 0
    cumulative_parse_fallbacks: int = 0
    cumulative_cost: float = 0.0
    latency_series: list[float] = Field(default_factory=list)
    success_series: list[float] = Field(default_factory=list)
    token_totals_series: list[int] = Field(default_factory=list)
    cost_series: list[float] = Field(default_factory=list)
    avg_cost: float | None = None
    fastest_run: ExtremeValue | None = None
    slowest_run: ExtremeValue | None = None
    lowest_success: ExtremeValue | None = None
    highest_success: ExtremeValue | None = None


__all__ = [
    "Analytics",
    "DependencySpec",
    "ExperimentSpec",
    "ExtremeValue",
    "LLMRequestLogEntry",
    "LLMUsage",
    "PromptSnapshot",
    "RecordResult",
    "ResolvedExperiment",
    "RunManifest",
    "RunMetrics",
    "RunSummary",
]
