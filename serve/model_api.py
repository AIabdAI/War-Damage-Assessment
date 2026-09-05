#!/usr/bin/env python3
"""Model-distribution API for the future Flutter application.

Serves the compressed on-device models with the version/integrity semantics a
mobile client needs:

  GET /health                              liveness probe
  GET /api/v1/version                      cheap "is there anything new?" check
  GET /api/v1/models                       catalogue (filter: ?kind=&production=)
  GET /api/v1/models/{model_id}            full metadata incl. sha256 + preprocessing
  GET /api/v1/models/{model_id}/download   the artefact (ETag = sha256)
  GET /api/v1/bundle/latest                detector + classifier pair in one call

Client contract (Flutter):
  1. GET /api/v1/bundle/latest and compare `bundle_version` with the stored one.
  2. If unchanged -> nothing to do (no download).
  3. If changed -> download each model, verify sha256 before installing.
  4. Send the stored ETag as If-None-Match; the server answers 304 when the
     device already holds that exact artefact, so bytes are never re-fetched.

Run locally:
  uvicorn serve.model_api:app --host 0.0.0.0 --port 8000

VPS deployment: see serve/README.md (systemd + nginx TLS + optional API key).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse

REGISTRY_PATH = Path(os.environ.get("MODEL_REGISTRY",
                                    Path(__file__).with_name("model_registry.json")))
REPO_ROOT = Path(__file__).resolve().parents[1]
API_KEY = os.environ.get("MODEL_API_KEY")        # unset -> open (local dev)

app = FastAPI(
    title="War Damage Assessment - Model Distribution API",
    version="1.0.0",
    description="Version-checked delivery of on-device detection and damage-"
                "classification models to the mobile application.",
)


def load_registry() -> dict:
    if not REGISTRY_PATH.is_file():
        raise HTTPException(503, "model registry not built - run serve/build_registry.py")
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def check_key(key: str | None) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(401, "invalid or missing API key")


def find_model(registry: dict, model_id: str) -> dict:
    for m in registry["models"]:
        if m["model_id"] == model_id:
            return m
    raise HTTPException(404, f"unknown model_id: {model_id}")


def public_view(m: dict) -> dict:
    """Registry entry without server-side filesystem details."""
    return {k: v for k, v in m.items() if k != "path"}


@app.get("/health")
def health() -> dict:
    ok = REGISTRY_PATH.is_file()
    return {"status": "ok" if ok else "degraded", "registry_present": ok}


@app.get("/api/v1/version")
def version(x_api_key: str | None = Header(default=None)) -> dict:
    check_key(x_api_key)
    reg = load_registry()
    return {"release": reg["release"],
            "bundle_version": reg["bundle"]["bundle_version"],
            "generated_at": reg["generated_at"],
            "registry_version": reg["registry_version"]}


@app.get("/api/v1/models")
def list_models(kind: str | None = Query(default=None),
                production: bool | None = Query(default=None),
                x_api_key: str | None = Header(default=None)) -> dict:
    check_key(x_api_key)
    reg = load_registry()
    items = reg["models"]
    if kind:
        items = [m for m in items if m["kind"] == kind]
    if production is not None:
        items = [m for m in items if m["production"] is production]
    return {"count": len(items), "release": reg["release"],
            "models": [public_view(m) for m in items]}


@app.get("/api/v1/models/{model_id}")
def model_meta(model_id: str, x_api_key: str | None = Header(default=None)) -> dict:
    check_key(x_api_key)
    return public_view(find_model(load_registry(), model_id))


@app.get("/api/v1/models/{model_id}/download")
def download(model_id: str, request: Request,
             if_none_match: str | None = Header(default=None),
             x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    m = find_model(load_registry(), model_id)
    etag = f'"{m["sha256"]}"'
    # device already holds this exact artefact -> no bytes transferred
    if if_none_match and if_none_match.strip() in (etag, m["sha256"]):
        return Response(status_code=304, headers={"ETag": etag})
    path = (REPO_ROOT / m["path"]).resolve()
    if not path.is_file():
        raise HTTPException(410, "artefact missing on server")
    return FileResponse(
        path, media_type="application/octet-stream",
        filename=f"{model_id}.onnx",
        headers={"ETag": etag,
                 "X-Model-Version": m["version"],
                 "X-Model-SHA256": m["sha256"],
                 "Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/api/v1/bundle/latest")
def bundle(x_api_key: str | None = Header(default=None)) -> JSONResponse:
    """Everything the app needs for one full assessment pipeline."""
    check_key(x_api_key)
    reg = load_registry()
    b = reg["bundle"]
    det = find_model(reg, b["detector"]) if b.get("detector") else None
    cls = find_model(reg, b["classifier"]) if b.get("classifier") else None
    return JSONResponse({
        "bundle_version": b["bundle_version"],
        "release": reg["release"],
        "generated_at": reg["generated_at"],
        "detection_classes": reg["detection_classes"],
        "pipeline": b["pipeline"],
        "detector": public_view(det) if det else None,
        "classifier": public_view(cls) if cls else None,
    })
