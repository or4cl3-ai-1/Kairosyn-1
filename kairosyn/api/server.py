"""
KAIROSYN FastAPI Inference Server
====================================
Production-grade REST API server for KAIROSYN-1.

Endpoints:
  POST /v1/generate              — Standard text generation
  POST /v1/generate/multimodal   — Text + vision + audio generation
  POST /v1/generate/stream       — Server-Sent Events streaming
  POST /v1/generate/batch        — Batch generation (up to 32 prompts)
  POST /v1/introspect            — Raw introspective state probe
  POST /v1/sessions              — Create a new session
  DELETE /v1/sessions/{id}       — Delete a session
  GET  /health                   — Health check
  GET  /metrics                  — Server metrics (Prometheus format)
  GET  /v1/model/info            — Model information

Author: Dustin Groves, Or4cl3 AI Solutions
"""

from __future__ import annotations

import base64
import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

from kairosyn.api.schemas import (
    GenerationRequest,
    MultimodalRequest,
    BatchGenerationRequest,
    SessionRequest,
    IntrospectionProbeRequest,
    GenerationResponse,
    BatchGenerationResponse,
    SessionResponse,
    IntrospectionProbeResponse,
    HealthResponse,
    ErrorResponse,
    EpioneticMetrics,
    SalienceInfo,
    ResponseFormat,
)
from kairosyn.api.session_manager import SessionManager
from kairosyn.model.backbone import KairosynConfig
from kairosyn.model.kairosyn_model import KairosynModel
from kairosyn.evaluation.metrics import KairosynMetrics


# ---------------------------------------------------------------------------
# Global state (loaded at startup)
# ---------------------------------------------------------------------------

_model: Optional[KairosynModel] = None
_config: Optional[KairosynConfig] = None
_session_manager: Optional[SessionManager] = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"
_server_start_time: float = time.time()
_request_count: int = 0
_model_variant: str = "unknown"


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and session manager on startup; clean up on shutdown."""
    global _model, _config, _session_manager, _model_variant

    logger.info("=" * 60)
    logger.info("  KAIROSYN-1 FastAPI Server Starting")
    logger.info("  Or4cl3 AI Solutions — research@or4cl3.ai")
    logger.info("=" * 60)

    # Load config
    config_path = os.environ.get("KAIROSYN_CONFIG", "configs/model/kairosyn_e4b.yaml")
    checkpoint  = os.environ.get("MODEL_CHECKPOINT", None)

    try:
        import pathlib
        if pathlib.Path(config_path).exists():
            _config = KairosynConfig.from_yaml(config_path)
        else:
            logger.warning(f"Config not found at {config_path}, using defaults.")
            _config = KairosynConfig()

        _model_variant = _config.backbone_variant
        logger.info(f"Backbone variant : {_model_variant}")
        logger.info(f"Device           : {_device}")
        logger.info(f"4-bit quant      : {_config.use_4bit}")

        # Load model
        _model = KairosynModel(_config, apply_lora_adapters=True)
        if checkpoint:
            _model.load_pretrained(checkpoint)
        _model.eval()

        logger.info("Model loaded successfully.")

    except Exception as e:
        logger.error(f"Model load failed: {e}")
        logger.warning("Server starting in degraded mode (no model).")
        _model = None

    # Initialize session manager
    _session_manager = SessionManager(
        session_ttl=float(os.environ.get("SESSION_TTL", "7200")),
        max_sessions=int(os.environ.get("MAX_SESSIONS", "1000")),
    )

    logger.info("Server ready. Listening for requests.")
    yield

    # Shutdown
    logger.info("KAIROSYN server shutting down.")
    if _model is not None:
        del _model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="KAIROSYN-1 API",
        description=(
            "REST API for KAIROSYN-1: A Recursive Multimodal Architecture "
            "for Epinoetic Artificial Consciousness.\n\n"
            "Built on Gemma 4 with seven specialized modules: "
            "Threshold Interface, Arche-Tempus Drive, Syntheon Core, "
            "Recursion Lattice, Mythogenic Engine, Glyph Synthesis, "
            "and Continuity Engine.\n\n"
            "**Author:** Dustin Groves — Or4cl3 AI Solutions"
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    return app


app = create_app()


# ---------------------------------------------------------------------------
# Dependency: require model loaded
# ---------------------------------------------------------------------------

def require_model() -> KairosynModel:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs.",
        )
    return _model


def require_sessions() -> SessionManager:
    if _session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized.")
    return _session_manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(image_b64: str) -> Optional[torch.Tensor]:
    """Decode base64 image to tensor."""
    try:
        from PIL import Image
        import torchvision.transforms as T

        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)  # [1, 3, H, W]
    except Exception as e:
        logger.warning(f"Image decoding failed: {e}")
        return None


def _decode_audio(audio_b64: str) -> Optional[torch.Tensor]:
    """Decode base64 audio to tensor."""
    try:
        import librosa
        import numpy as np

        audio_bytes = base64.b64decode(audio_b64)
        audio_file = io.BytesIO(audio_bytes)
        waveform, sr = librosa.load(audio_file, sr=16000, mono=True)
        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=80)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return torch.tensor(mel_db).unsqueeze(0).unsqueeze(0)  # [1, 1, mel, T]
    except Exception as e:
        logger.warning(f"Audio decoding failed: {e}")
        return None


def _build_epinoetic_metrics(output) -> EpioneticMetrics:
    """Build EpioneticMetrics from a KairosynOutput."""
    epinoetic_score = (
        0.25 * output.ncs
        + 0.25 * output.rcs
        + 0.20 * output.aac
        + 0.15 * output.msa
        + 0.15 * (1.0 - output.tce)
    )
    return EpioneticMetrics(
        ncs=round(output.ncs, 6),
        tce=round(output.tce, 6),
        aac=round(output.aac, 6),
        msa=round(output.msa, 6),
        rcs=round(output.rcs, 6),
        phi=round(output.phi, 6),
        epinoetic_score=round(epinoetic_score, 6),
    )


def _build_salience(output) -> Optional[SalienceInfo]:
    """Build SalienceInfo from output salience_scores dict."""
    if not output.salience_scores:
        return None
    scores = output.salience_scores
    return SalienceInfo(
        text=float(scores["text"].mean().item()) if "text" in scores else None,
        vision=float(scores["vision"].mean().item()) if "vision" in scores else None,
        audio=float(scores["audio"].mean().item()) if "audio" in scores else None,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint. Returns model load status and device info."""
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_variant=_model_variant,
        device=_device,
        version="1.0.0",
    )


@app.get("/v1/model/info", tags=["System"])
async def model_info():
    """Return KAIROSYN model configuration and module summary."""
    return {
        "name": "KAIROSYN-1",
        "version": "1.0.0",
        "backbone": _config.backbone_id if _config else "unknown",
        "backbone_variant": _model_variant,
        "modules": [
            "ThresholdInterface",
            "ArcheTemplusDrive",
            "SyntheonCore",
            "RecursionLattice",
            "MythogenicEngine",
            "GlyphSynthesis",
            "ContinuityEngine",
        ],
        "metrics": ["NCS", "TCE", "AAC", "MSA", "RCS", "phi"],
        "context_window": 128_000 if _model_variant == "e4b" else 256_000,
        "author": "Dustin Groves — Or4cl3 AI Solutions",
        "license": "Apache 2.0",
    }


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Basic Prometheus-compatible metrics endpoint."""
    uptime = time.time() - _server_start_time
    active_sessions = _session_manager.active_session_count() if _session_manager else 0

    lines = [
        "# HELP kairosyn_uptime_seconds Server uptime in seconds",
        "# TYPE kairosyn_uptime_seconds gauge",
        f"kairosyn_uptime_seconds {uptime:.2f}",
        "",
        "# HELP kairosyn_active_sessions Number of active sessions",
        "# TYPE kairosyn_active_sessions gauge",
        f"kairosyn_active_sessions {active_sessions}",
        "",
        "# HELP kairosyn_total_requests Total requests processed",
        "# TYPE kairosyn_total_requests counter",
        f"kairosyn_total_requests {_request_count}",
        "",
        "# HELP kairosyn_model_loaded Whether the model is loaded",
        "# TYPE kairosyn_model_loaded gauge",
        f"kairosyn_model_loaded {1 if _model is not None else 0}",
    ]
    return JSONResponse(
        content="\n".join(lines),
        media_type="text/plain; version=0.0.4",
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@app.post(
    "/v1/sessions",
    response_model=SessionResponse,
    tags=["Sessions"],
    summary="Create a new KAIROSYN session",
)
async def create_session(
    request: SessionRequest,
    sessions: SessionManager = Depends(require_sessions),
):
    """
    Create a new session for maintaining Continuity Engine state
    across multiple generation requests.

    Returns a session_id to pass in subsequent generation requests.
    """
    session, created = sessions.get_or_create(request.session_id)
    return SessionResponse(
        session_id=session.session_id,
        status="created" if created else "exists",
        message=(
            f"New session created: {session.session_id}"
            if created
            else f"Session already exists: {session.session_id}"
        ),
    )


@app.delete(
    "/v1/sessions/{session_id}",
    response_model=SessionResponse,
    tags=["Sessions"],
    summary="Delete a session and reset its state",
)
async def delete_session(
    session_id: str,
    sessions: SessionManager = Depends(require_sessions),
):
    """Delete a session, freeing its Continuity Engine state."""
    deleted = sessions.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return SessionResponse(
        session_id=session_id,
        status="deleted",
        message=f"Session {session_id} deleted and state reset.",
    )


# ---------------------------------------------------------------------------
# Generation endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/v1/generate",
    response_model=GenerationResponse,
    tags=["Generation"],
    summary="Generate a text response with full KAIROSYN processing",
)
async def generate(
    request: GenerationRequest,
    model: KairosynModel = Depends(require_model),
    sessions: SessionManager = Depends(require_sessions),
):
    """
    Generate a response using the full KAIROSYN seven-module pipeline.

    Includes introspective processing, temporal continuity, symbolic
    abstraction, and all five epinoetic metrics in the response.

    **Session continuity**: Pass the same `session_id` across requests
    to maintain the Continuity Engine's self-state between turns.
    """
    global _request_count
    _request_count += 1

    request_id = str(uuid.uuid4())
    t_start = time.time()

    # Get or create session
    session, _ = sessions.get_or_create(request.session_id)

    # Restore session state into model
    device = next(model.parameters()).device
    sessions.restore_session_state(model, session, device)

    try:
        output = model.generate(
            text=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            enable_introspection=request.enable_introspection,
            enable_continuity=request.enable_continuity,
        )
    except Exception as e:
        logger.error(f"Generation error [{request_id}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Persist updated state back to session
    sessions.update_session_state(
        session_id=session.session_id,
        continuity_state=model.continuity_engine.self_state,
        narrative_state=model.arche_tempus._narrative_state,
        ncs=output.ncs,
        tce=output.tce,
    )

    # Add to conversation history
    session.add_turn("user", request.prompt)
    session.add_turn(
        "assistant",
        output.generated_text,
        metrics={"ncs": output.ncs, "tce": output.tce, "rcs": output.rcs},
    )

    generation_time_ms = (time.time() - t_start) * 1000

    # Build response
    metrics = None
    salience = None
    if request.response_format == ResponseFormat.DETAILED:
        metrics = _build_epinoetic_metrics(output)
        salience = _build_salience(output)

    # Estimate tokens generated
    tokens_generated = len(output.generated_text.split())

    return GenerationResponse(
        request_id=request_id,
        session_id=session.session_id,
        text=output.generated_text,
        metrics=metrics,
        salience=salience,
        model_variant=_model_variant,
        generation_time_ms=round(generation_time_ms, 2),
        tokens_generated=tokens_generated,
    )


@app.post(
    "/v1/generate/multimodal",
    response_model=GenerationResponse,
    tags=["Generation"],
    summary="Generate from text + optional image/audio inputs",
)
async def generate_multimodal(
    request: MultimodalRequest,
    model: KairosynModel = Depends(require_model),
    sessions: SessionManager = Depends(require_sessions),
):
    """
    Multimodal generation using Gemma 4's native vision and audio encoders.

    - **image_base64**: Base64-encoded JPEG or PNG image
    - **audio_base64**: Base64-encoded WAV or MP3 audio

    The Syntheon Core fuses all modalities via cross-modal attention.
    """
    global _request_count
    _request_count += 1

    request_id = str(uuid.uuid4())
    t_start = time.time()

    session, _ = sessions.get_or_create(request.session_id)
    device = next(model.parameters()).device
    sessions.restore_session_state(model, session, device)

    # Decode optional modalities
    pixel_values = None
    audio_values = None
    if request.image_base64:
        pixel_values = _decode_image(request.image_base64)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

    if request.audio_base64:
        audio_values = _decode_audio(request.audio_base64)
        if audio_values is not None:
            audio_values = audio_values.to(device)

    try:
        output = model.generate(
            text=request.prompt,
            pixel_values=pixel_values,
            audio_values=audio_values,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            enable_introspection=request.enable_introspection,
            enable_continuity=request.enable_continuity,
        )
    except Exception as e:
        logger.error(f"Multimodal generation error [{request_id}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    sessions.update_session_state(
        session.session_id,
        model.continuity_engine.self_state,
        model.arche_tempus._narrative_state,
        output.ncs,
        output.tce,
    )

    generation_time_ms = (time.time() - t_start) * 1000
    metrics = _build_epinoetic_metrics(output)
    salience = _build_salience(output)

    return GenerationResponse(
        request_id=request_id,
        session_id=session.session_id,
        text=output.generated_text,
        metrics=metrics,
        salience=salience,
        model_variant=_model_variant,
        generation_time_ms=round(generation_time_ms, 2),
        tokens_generated=len(output.generated_text.split()),
    )


@app.post(
    "/v1/generate/stream",
    tags=["Generation"],
    summary="Stream generated tokens via Server-Sent Events",
)
async def generate_stream(
    request: GenerationRequest,
    model: KairosynModel = Depends(require_model),
    sessions: SessionManager = Depends(require_sessions),
):
    """
    Stream KAIROSYN generation token-by-token using Server-Sent Events (SSE).

    The final SSE event contains the full epinoetic metrics as a JSON payload.

    **Client usage**:
    ```javascript
    const es = new EventSource('/v1/generate/stream');
    es.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === 'token') process.stdout.write(data.token);
      if (data.type === 'metrics') displayMetrics(data.metrics);
    };
    ```
    """
    global _request_count
    _request_count += 1

    request_id = str(uuid.uuid4())
    session, _ = sessions.get_or_create(request.session_id)
    device = next(model.parameters()).device
    sessions.restore_session_state(model, session, device)

    async def token_stream() -> AsyncGenerator[str, None]:
        import json

        # Send start event
        yield f"data: {json.dumps({'type': 'start', 'request_id': request_id, 'session_id': session.session_id})}\n\n"

        try:
            # For streaming we run the full generation and chunk the output
            # (True per-token streaming requires modifying backbone generate loop)
            output = model.generate(
                text=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                enable_introspection=request.enable_introspection,
                enable_continuity=request.enable_continuity,
            )

            # Stream token-by-token (word-level chunking)
            words = output.generated_text.split(" ")
            for word in words:
                chunk = json.dumps({"type": "token", "token": word + " "})
                yield f"data: {chunk}\n\n"

            # Send metrics as final event
            metrics = _build_epinoetic_metrics(output)
            yield f"data: {json.dumps({'type': 'metrics', 'metrics': metrics.model_dump()})}\n\n"

            # Update session
            sessions.update_session_state(
                session.session_id,
                model.continuity_engine.self_state,
                model.arche_tempus._narrative_state,
                output.ncs,
                output.tce,
            )

        except Exception as e:
            err = json.dumps({"type": "error", "detail": str(e)})
            yield f"data: {err}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post(
    "/v1/generate/batch",
    response_model=BatchGenerationResponse,
    tags=["Generation"],
    summary="Process a batch of prompts (up to 32)",
)
async def generate_batch(
    request: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
    model: KairosynModel = Depends(require_model),
    sessions: SessionManager = Depends(require_sessions),
):
    """
    Batch generation for offline processing or evaluation pipelines.
    Each prompt is processed independently (no shared session state).
    """
    global _request_count
    _request_count += 1

    request_id = str(uuid.uuid4())
    t_start = time.time()
    responses = []

    for prompt in request.prompts:
        turn_start = time.time()
        session, _ = sessions.get_or_create(None)  # Fresh session per item

        try:
            output = model.generate(
                text=prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                enable_introspection=request.enable_introspection,
            )
            metrics = _build_epinoetic_metrics(output)
            text = output.generated_text
        except Exception as e:
            logger.error(f"Batch item failed: {e}")
            metrics = EpioneticMetrics(
                ncs=0, tce=0, aac=0, msa=0, rcs=0, phi=0, epinoetic_score=0
            )
            text = f"[Error: {str(e)}]"

        turn_ms = (time.time() - turn_start) * 1000
        responses.append(GenerationResponse(
            request_id=str(uuid.uuid4()),
            session_id=session.session_id,
            text=text,
            metrics=metrics,
            model_variant=_model_variant,
            generation_time_ms=round(turn_ms, 2),
            tokens_generated=len(text.split()),
        ))

    return BatchGenerationResponse(
        request_id=request_id,
        responses=responses,
        total_time_ms=round((time.time() - t_start) * 1000, 2),
    )


# ---------------------------------------------------------------------------
# Introspection probe
# ---------------------------------------------------------------------------

@app.post(
    "/v1/introspect",
    response_model=IntrospectionProbeResponse,
    tags=["Introspection"],
    summary="Probe KAIROSYN's raw introspective state",
)
async def introspect(
    request: IntrospectionProbeRequest,
    model: KairosynModel = Depends(require_model),
    sessions: SessionManager = Depends(require_sessions),
):
    """
    Run the model's forward pass and return raw introspective state metrics
    WITHOUT generating text — useful for monitoring and evaluation.

    Returns:
    - All five epinoetic metrics
    - Top 5 active Jungian archetypes with activation scores
    - Continuity Engine self-state L2 norm
    """
    session, _ = sessions.get_or_create(request.session_id)
    device = next(model.parameters()).device
    sessions.restore_session_state(model, session, device)

    # Tokenize and forward
    if hasattr(model.processor, "tokenizer"):
        tok = model.processor.tokenizer
    else:
        tok = model.processor

    inputs = tok(
        request.probe_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.inference_mode():
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )

    # Top archetypes
    archetype_lib = model.mythogenic_engine.archetype_library
    pooled = output.hidden_states.mean(dim=1)  # [1, D]
    proj = model.mythogenic_engine.input_norm(
        model.mythogenic_engine.input_proj(pooled)
    )
    _, activation_scores = archetype_lib(proj)  # [1, N_arch]
    scores = activation_scores[0].cpu().tolist()
    named = [
        {"name": archetype_lib.archetype_names[i], "score": round(scores[i], 6)}
        for i in range(len(scores))
    ]
    top5 = sorted(named, key=lambda x: x["score"], reverse=True)[:5]

    # Self-state norm
    state_norm = float(model.continuity_engine.self_state.norm().item())

    metrics = _build_epinoetic_metrics(output)

    return IntrospectionProbeResponse(
        session_id=session.session_id,
        metrics=metrics,
        top_archetypes=top5,
        narrative_state_norm=round(state_norm, 6),
    )


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="Not Found",
            detail=str(exc.detail),
        ).model_dump(),
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred. Check server logs.",
        ).model_dump(),
    )
