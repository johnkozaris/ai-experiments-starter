from __future__ import annotations

from pydantic import BaseModel, Field


class Extraction(BaseModel):
    """Example structured extraction schema.

    Adjust fields to match the dataset / prompts. (Legacy placeholder: new canonical
    path is `app.schemas.extraction.AcquisitionExtraction`.)
    """

    acquirer: str = Field(..., description="Acquiring company")
    target: str = Field(..., description="Target company or asset")
    value: str | None = Field(None, description="Deal value (string to preserve units/currency)")
    date: str | None = Field(None, description="Announcement or closing date")
    deal_type: str | None = Field(
        None,
        description="Type of transaction (acquisition, merger, funding, etc.)",
    )

    class Config:
        extra = "ignore"
