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

from .token_repeat_inference import TokenRepeatInferenceConfig, TokenRepeatInferenceService

DEFAULT_TOKEN_CHECKPOINT = "outputs/token_repeat_cnn/best_model.pt"

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


@lru_cache(maxsize=1)
def get_token_service() -> TokenRepeatInferenceService:
    token_checkpoint_path = os.getenv("GENOME_TOKEN_CHECKPOINT", DEFAULT_TOKEN_CHECKPOINT)
    window_size = int(os.getenv("GENOME_WINDOW_SIZE", "512"))
    stride = int(os.getenv("GENOME_STRIDE", "64"))
    position_threshold = float(os.getenv("GENOME_POSITION_THRESHOLD", "0.35"))
    min_interval_bp = int(os.getenv("GENOME_MIN_INTERVAL_BP", "4"))
    min_repeat_fraction = float(os.getenv("GENOME_MIN_REPEAT_FRACTION", "0.02"))

    config = TokenRepeatInferenceConfig(
        checkpoint_path=token_checkpoint_path,
        window_size=window_size,
        stride=stride,
        position_probability_threshold=position_threshold,
        min_interval_bp=min_interval_bp,
        min_repeat_fraction_positive=min_repeat_fraction,
    )
    return TokenRepeatInferenceService(config)


def run_prediction(sequence: str, query_subsequence: str | None) -> dict[str, Any]:
    result = get_token_service().predict(sequence=sequence, query_subsequence=query_subsequence)
    if not result.get("errors"):
        result["model_choice"] = "token"
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
        },
    )


@app.post("/analyze", response_class=HTMLResponse)
def analyze_form(
    request: Request,
    sequence: str = Form(...),
    query_subsequence: str = Form(""),
) -> HTMLResponse:
    try:
        result = run_prediction(sequence=sequence, query_subsequence=query_subsequence or None)
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
        },
    )


@app.post("/api/analyze")
def analyze_api(payload: SequenceRequest) -> JSONResponse:
    try:
        result: dict[str, Any] = run_prediction(
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