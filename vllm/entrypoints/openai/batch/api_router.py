# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API router for the Online Batch API (Files + Batches endpoints)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from vllm.entrypoints.openai.batch.protocol import (
    BatchListResponse,
    BatchObject,
    FileListResponse,
)
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.openai.serving_batches import OpenAIServingBatches
from vllm.entrypoints.openai.serving_files import OpenAIServingFiles
from vllm.logger import init_logger

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger

logger = init_logger(__name__)

router = APIRouter()


def serving_files(request: Request) -> Optional[OpenAIServingFiles]:
    return getattr(request.app.state, 'openai_serving_files', None)


def serving_batches(request: Request) -> Optional[OpenAIServingBatches]:
    return getattr(request.app.state, 'openai_serving_batches', None)


def _error_response(message: str, code: int,
                    err_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content=ErrorResponse(
            error=ErrorInfo(message=message, type=err_type, code=code),
        ).model_dump(),
    )


# ── Files endpoints ─────────────────────────────────────────────

@router.post("/v1/files")
async def upload_file(file: UploadFile, raw_request: Request,
                      purpose: str = "batch"):
    handler = serving_files(raw_request)
    if handler is None:
        return _error_response("Batch API is not enabled", 501)
    content = await file.read()
    result = await handler.upload_file(content, file.filename or "upload.jsonl",
                                       purpose)
    return JSONResponse(content=result.model_dump())


@router.get("/v1/files")
async def list_files(raw_request: Request, purpose: str | None = None):
    handler = serving_files(raw_request)
    if handler is None:
        return _error_response("Batch API is not enabled", 501)
    files = await handler.list_files(purpose)
    resp = FileListResponse(object="list", data=files)
    return JSONResponse(content=resp.model_dump())


@router.get("/v1/files/{file_id}")
async def get_file(file_id: str, raw_request: Request):
    handler = serving_files(raw_request)
    if handler is None:
        return _error_response("Batch API is not enabled", 501)
    result = await handler.get_file(file_id)
    if result is None:
        return _error_response("File not found", 404)
    return JSONResponse(content=result.model_dump())


@router.get("/v1/files/{file_id}/content")
async def get_file_content(file_id: str, raw_request: Request):
    handler = serving_files(raw_request)
    if handler is None:
        return _error_response("Batch API is not enabled", 501)
    content = await handler.get_file_content(file_id)
    if content is None:
        return _error_response("File not found", 404)
    return Response(content=content, media_type="application/octet-stream")


@router.delete("/v1/files/{file_id}")
async def delete_file(file_id: str, raw_request: Request):
    handler = serving_files(raw_request)
    if handler is None:
        return _error_response("Batch API is not enabled", 501)
    batch_handler = serving_batches(raw_request)
    result = await handler.delete_file(file_id, batch_handler)
    if result is None:
        return _error_response("File not found", 404)
    if isinstance(result, dict) and "error" in result:
        return JSONResponse(status_code=409, content=result)
    return JSONResponse(content=result.model_dump())


# ── Batches endpoints ───────────────────────────────────────────

@router.post("/v1/batches")
async def create_batch(raw_request: Request):
    handler = serving_batches(raw_request)
    if handler is None:
        return _error_response("Batch API is not enabled", 501)
    body = await raw_request.json()
    result = await handler.create_batch(
        input_file_id=body["input_file_id"],
        endpoint=body["endpoint"],
        completion_window=body.get("completion_window", "24h"),
        metadata=body.get("metadata"),
    )
    if isinstance(result, ErrorResponse):
        err = result.error
        return JSONResponse(status_code=err.code,
                            content=result.model_dump())
    return JSONResponse(content=result.model_dump())


@router.get("/v1/batches")
async def list_batches(raw_request: Request, limit: int = 20,
                       after: str | None = None):
    handler = serving_batches(raw_request)
    if handler is None:
        return _error_response("Batch API is not enabled", 501)
    result = await handler.list_batches(limit=limit, after=after)
    return JSONResponse(content=result.model_dump())


@router.get("/v1/batches/{batch_id}")
async def get_batch(batch_id: str, raw_request: Request):
    handler = serving_batches(raw_request)
    if handler is None:
        return _error_response("Batch API is not enabled", 501)
    result = await handler.get_batch(batch_id)
    if result is None:
        return _error_response("Batch not found", 404)
    return JSONResponse(content=result.model_dump())


@router.post("/v1/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str, raw_request: Request):
    handler = serving_batches(raw_request)
    if handler is None:
        return _error_response("Batch API is not enabled", 501)
    result = await handler.cancel_batch(batch_id)
    if result is None:
        return _error_response("Batch not found", 404)
    if isinstance(result, ErrorResponse):
        err = result.error
        return JSONResponse(status_code=err.code,
                            content=result.model_dump())
    return JSONResponse(content=result.model_dump())


def attach_router(app: FastAPI):
    app.include_router(router)


def init_batch_state(
    state: "State",
    args: "Namespace",
):
    serving_files_instance = OpenAIServingFiles(
        storage_dir=args.batch_storage_dir,
    )
    state.openai_serving_files = serving_files_instance
    state.openai_serving_batches = OpenAIServingBatches(
        storage_dir=args.batch_storage_dir,
        serving_files=serving_files_instance,
        serving_chat=getattr(state, 'openai_serving_chat', None),
        serving_embedding=getattr(state, 'serving_embedding', None),
        serving_score=getattr(state, 'serving_scores', None),
        batch_priority=args.batch_priority,
        retention_hours=args.batch_retention_hours,
    )
