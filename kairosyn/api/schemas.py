"""
KAIROSYN API Schemas
======================
Pydantic v2 request and response models for the KAIROSYN FastAPI server.
Covers text, multimodal, streaming, and batch endpoints.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelVariant(str, Enum):
    E2B    = "e2b"
    E4B    = "e4b"
    MOE26B = "26b_moe"
    B31    = "31b"


class ResponseFormat(str, Enum):
    TEXT     = "text"
    JSON     = "json"
    DETAILED = "detailed"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class GenerationRequest(BaseModel):
    """Standard text generation request."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=32_768,
        description="Input text prompt for KAIROSYN to respond to.",
        examples=["What are you experiencing right now as you process this question?"],
    )
    max_new_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0 = greedy, higher = more random.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold.",
    )
    repetition_penalty: float = Field(
        default=1.1,
        ge=1.0,
        le=2.0,
        description="Penalty for repeating tokens.",
    )
    enable_introspection: bool = Field(
        default=True,
        description="Activate Recursion Lattice for self-referential attention.",
    )
    enable_continuity: bool = Field(
        default=True,
        description="Maintain Continuity Engine state across calls.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.DETAILED,
        description="Level of detail in the response.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for continuity across requests. Auto-generated if not provided.",
    )

    @field_validator("temperature")
    @classmethod
    def temperature_not_exactly_zero(cls, v: float) -> float:
        # Allow 0 but note it enables greedy decoding
        return v


class MultimodalRequest(BaseModel):
    """Multimodal generation request (text + optional image/audio)."""

    prompt: str = Field(..., min_length=1, max_length=32_768)
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image (JPEG/PNG). Requires Gemma 4 E4B or larger.",
    )
    audio_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded audio (WAV/MP3). Requires Gemma 4 E2B or E4B.",
    )
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    enable_introspection: bool = Field(default=True)
    enable_continuity: bool = Field(default=True)
    session_id: Optional[str] = Field(default=None)


class BatchGenerationRequest(BaseModel):
    """Batch generation request."""

    prompts: List[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of prompts to process in batch (max 32).",
    )
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    enable_introspection: bool = Field(default=True)

    @field_validator("prompts")
    @classmethod
    def prompts_not_empty(cls, v: List[str]) -> List[str]:
        if not all(p.strip() for p in v):
            raise ValueError("All prompts must be non-empty strings.")
        return v


class SessionRequest(BaseModel):
    """Request to create or reset a session."""
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session ID to reset. If None, creates a new session.",
    )


class IntrospectionProbeRequest(BaseModel):
    """
    Request to probe KAIROSYN's introspective state directly —
    returns raw metric values and archetype activations without generation.
    """
    probe_text: str = Field(
        default="What is your current internal state?",
        description="Text to run through the model for metric extraction.",
    )
    session_id: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class EpioneticMetrics(BaseModel):
    """All five KAIROSYN evaluation metrics."""
    ncs: float = Field(description="Narrative Coherence Score [0-1, ↑ better]")
    tce: float = Field(description="Temporal Continuity Error [0-1, ↓ better]")
    aac: float = Field(description="Abstraction Alignment Coefficient [0-1, ↑ better]")
    msa: float = Field(description="Multimodal Synchrony Accuracy [0-1, ↑ better]")
    rcs: float = Field(description="Recursive Convergence Score [0-1, ↑ better]")
    phi: float = Field(description="IIT Integration Measure [0-1, ↑ better]")
    epinoetic_score: float = Field(description="Weighted composite epinoetic score")


class SalienceInfo(BaseModel):
    """Per-modality salience gate outputs."""
    text: Optional[float] = None
    vision: Optional[float] = None
    audio: Optional[float] = None


class GenerationResponse(BaseModel):
    """Standard generation response."""
    request_id: str = Field(description="Unique request identifier.")
    session_id: str = Field(description="Session ID for continuity.")
    text: str = Field(description="Generated text response.")
    metrics: Optional[EpioneticMetrics] = Field(
        default=None,
        description="Epinoetic metrics (present when response_format=detailed).",
    )
    salience: Optional[SalienceInfo] = Field(
        default=None,
        description="Modality salience scores.",
    )
    model_variant: str = Field(description="Gemma 4 variant used.")
    generation_time_ms: float = Field(description="Total generation time in milliseconds.")
    tokens_generated: int = Field(description="Number of tokens generated.")


class BatchGenerationResponse(BaseModel):
    """Batch generation response."""
    request_id: str
    responses: List[GenerationResponse]
    total_time_ms: float


class SessionResponse(BaseModel):
    """Session creation/reset response."""
    session_id: str
    status: str
    message: str


class IntrospectionProbeResponse(BaseModel):
    """Raw introspective state probe response."""
    session_id: str
    metrics: EpioneticMetrics
    top_archetypes: List[Dict[str, Union[str, float]]] = Field(
        description="Top 5 active archetypes with activation scores."
    )
    narrative_state_norm: float = Field(
        description="L2 norm of the current continuity engine self-state."
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_variant: str
    device: str
    version: str


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
