from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationTypeDetailsModel(BaseModel):
    """Actionable tuple describing a concrete configuration edit.

    OpenAI structured outputs requirement: ALL fields required.
    Nullable semantics represented via union with null type at schema layer.
    """

    tool: str = Field(..., description="Exact tool/schemaName or action name from the session")
    target_field: str = Field(
        ..., description="Parameter key within the tool to change (if applicable)"
    )
    operation: str = Field(
        ...,
        description=(
            "One of rebind | rename | update_description | add_default | add_tool | "
            "remove_tool | other"
        ),
    )
    value_hint: str | None = Field(
        None,
        description="Literal replacement or guidance such as 'bind from <prior-step.key>'",
    )


class Recommendation(BaseModel):
    """Structured recommendation output contract used by the recommendation experiment."""

    FailureType: str = Field(
        ..., description="Canonical failure type label (heuristic or inferred)"
    )
    FailureDescription: str = Field(
        ..., description="Brief natural language explanation of how the failure manifested"
    )
    RecommendationToUser: str = Field(
        ..., description="Actionable steps the maker should take inside Copilot Studio"
    )
    RecommendationToAgent: str | None = Field(
        None,
        description=(
            "Optional internal phrasing targeted at an instruction-tuning / agent self-heal system"
        ),
    )
    RecommendationTypeDetails: RecommendationTypeDetailsModel | None = Field(
        None, description="Action tuple for parameter-centric failures; null when not applicable"
    )
    ExtraNotes: str | None = Field(
        None, description="Optional additional context not covered elsewhere"
    )

    model_config = {
        "extra": "forbid",  # disallow unspecified keys per structured output rules
    }
