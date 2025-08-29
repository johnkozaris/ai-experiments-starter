from __future__ import annotations

from pydantic import BaseModel, Field


class AcquisitionExtraction(BaseModel):
    """Normalized acquisition / funding event schema used for structured extraction."""

    acquirer: str = Field(..., description="Acquiring company")
    target: str = Field(..., description="Target company / asset")
    value: str | None = Field(None, description="Deal value with units/currency if present")
    date: str | None = Field(None, description="Announcement or closing date (ISO or raw)")
    deal_type: str | None = Field(
        None,
        description="Transaction type (acquisition, merger, funding)",
    )

    class Config:
        extra = "ignore"
