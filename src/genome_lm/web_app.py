from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .repeat_inference import RepeatInferenceConfig, RepeatInferenceService

DEFAULT_CHECKPOINT = "outputs/baseline_cnn_v3/best_model.pt"
DEFAULT_EMBEDDINGS = "outputs/baseline_cnn_v3/embeddings.npz"

app = FastAPI(title="Genome Repeat Explorer", version="0.1.0")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


class SequenceRequest(BaseModel):
    sequence: str = Field(..., min_length=1)
    query_subsequence: str | None = None


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


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
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
        result = get_service().predict(sequence=sequence, query_subsequence=query_subsequence or None)
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "error": str(exc),
                "input_sequence": sequence,
                "query_subsequence": query_subsequence,
            },
        )

    errors = result.get("errors", [])
    return templates.TemplateResponse(
        "index.html",
        {
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
        result: dict[str, Any] = get_service().predict(
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