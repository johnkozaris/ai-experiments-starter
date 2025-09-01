from __future__ import annotations

from app.domain.models import ExperimentSpec


def get_experiment() -> ExperimentSpec:
    # Reuse existing global prompts for simplicity; no schema (free-form summary)
    return ExperimentSpec(
        name="python_extraction",
        description="Free-form summary using Python definition",
        model="gpt-4o-mini",
        provider="azure",
        schema_model=None,
        # Reuse prompts from acq_yaml experiment (relative paths)
        system_prompt="../acq_yaml/prompts/system/extraction.txt",
        user_prompt="../acq_yaml/prompts/user/extraction.jinja",
        instructions=None,
        dataset_path="../../datasets/samples.jsonl",
        temperature=0.3,
        max_output_tokens=400,
    )
