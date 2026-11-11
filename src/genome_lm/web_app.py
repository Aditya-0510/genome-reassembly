from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .direct_repeat_inference import DirectRepeatInferenceConfig, DirectRepeatInferenceService
from .repeat_inference import RepeatInferenceConfig, RepeatInferenceService

DEFAULT_CHECKPOINT = "outputs/baseline_cnn_v3/best_model.pt"
DEFAULT_EMBEDDINGS = "outputs/baseline_cnn_v3/embeddings.npz"
DEFAULT_DIRECT_CHECKPOINT = "outputs/direct_repeat_cnn/best_model.pt"

app = FastAPI(title="Genome Repeat Explorer", version="0.1.0")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class SequenceRequest(BaseModel):
    sequence: str = Field(..., min_length=1)
    query_subsequence: str | None = None
    model_choice: str = "direct"


@lru_cache(maxsize=1)
def get_service() -> RepeatInferenceService:
    checkpoint_path = os.getenv("GENOME_CHECKPOINT", DEFAULT_CHECKPOINT)
    embeddings_path = os.getenv("GENOME_EMBEDDINGS", DEFAULT_EMBEDDINGS)
    window_size = int(os.getenv("GENOME_WINDOW_SIZE", "512"))
    stride = int(os.getenv("GENOME_STRIDE", "512"))

    config = RepeatInferenceConfig(
        checkpoint_path=checkpoint_path,
        embeddings_path=embeddings_path,
        window_size=window_size,
        stride=stride,
    )
    return RepeatInferenceService(config)


@lru_cache(maxsize=1)
def get_direct_service() -> DirectRepeatInferenceService:
    direct_checkpoint_path = os.getenv("GENOME_DIRECT_CHECKPOINT", DEFAULT_DIRECT_CHECKPOINT)
    window_size = int(os.getenv("GENOME_WINDOW_SIZE", "512"))
    stride = int(os.getenv("GENOME_STRIDE", "64"))

    config = DirectRepeatInferenceConfig(
        checkpoint_path=direct_checkpoint_path,
        window_size=window_size,
        stride=stride,
    )
    return DirectRepeatInferenceService(config)


def run_prediction(
    model_choice: str,
    sequence: str,
    query_subsequence: str | None,
) -> dict[str, Any]:
    if model_choice == "mlm":
        result = get_service().predict(sequence=sequence, query_subsequence=query_subsequence)
        if not result.get("errors"):
            result["model_choice"] = "mlm"
        return result

    result = get_direct_service().predict(sequence=sequence, query_subsequence=query_subsequence)
    if not result.get("errors"):
        result["model_choice"] = "direct"
    return result


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": None,
            "error": None,
            "input_sequence": "",
            "query_subsequence": "",
            "model_choice": "direct",
        },
    )


@app.post("/analyze", response_class=HTMLResponse)
def analyze_form(
    request: Request,
    sequence: str = Form(...),
    query_subsequence: str = Form(""),
    model_choice: str = Form("direct"),
) -> HTMLResponse:
    try:
        result = run_prediction(
            model_choice=model_choice,
            sequence=sequence,
            query_subsequence=query_subsequence or None,
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "result": None,
                "error": str(exc),
                "input_sequence": sequence,
                "query_subsequence": query_subsequence,
                "model_choice": model_choice,
            },
        )

    errors = result.get("errors", [])
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": result if not errors else None,
            "error": "\n".join(errors) if errors else None,
            "input_sequence": sequence,
            "query_subsequence": query_subsequence,
            "model_choice": model_choice,
        },
    )


@app.post("/api/analyze")
def analyze_api(payload: SequenceRequest) -> JSONResponse:
    try:
        result: dict[str, Any] = run_prediction(
            model_choice=payload.model_choice,
            sequence=payload.sequence,
            query_subsequence=payload.query_subsequence,
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"errors": [str(exc)]})

    if result.get("errors"):
        return JSONResponse(status_code=400, content=result)
    return JSONResponse(status_code=200, content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.genome_lm.web_app:app", host="127.0.0.1", port=8000, reload=False)