"""
Pydantic data schemas for SupportPearlz.
Guarantees schema-validated responses before they reach the user interface.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Structured citation pointing to a verifiable document location."""
    source: str = Field(description="Filename of the cited document (e.g., 'warranty_policy.md')")
    location: str = Field(description="Specific location pointer (e.g. 'Section 3.2' or 'Page 14')")
    snippet: Optional[str] = Field(default=None, description="Short supporting verbatim quote from the chunk")


class RawModelOutput(BaseModel):
    """Raw structured output parsed directly from the LLM."""
    answer: str = Field(description="The customer support response, grounded strictly in provided context.")
    used_labels: List[str] = Field(
        default_factory=list,
        description="The exact context labels cited by the model (e.g. ['S1', 'S2']). Refusal must be empty."
    )
    confidence: Literal["high", "partial", "none"] = Field(
        description="Confidence signal: 'high' if fully answered, 'partial' if partly covered, 'none' if refused."
    )
    answered: bool = Field(
        description="True if the question was answered with company facts; False if refused or unanswerable."
    )
    uncovered_part: Optional[str] = Field(
        default=None,
        description="Explicit description of any part of user query not covered by documentation."
    )


class GroundedResponse(BaseModel):
    """Final validated response returned to CLI, Streamlit UI, or API."""
    answer: str = Field(description="Final user-facing answer text")
    sources: List[SourceCitation] = Field(default_factory=list, description="Validated and de-duplicated citations")
    confidence: Literal["high", "partial", "none"] = Field(default="none")
    answered: bool = Field(default=False)
    refusal_reason: Optional[str] = Field(default=None)
    rewritten_query: Optional[str] = Field(default=None)
    latency_ms: float = Field(default=0.0)
    flags: List[str] = Field(default_factory=list)
