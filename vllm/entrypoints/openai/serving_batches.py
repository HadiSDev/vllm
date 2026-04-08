"""Batch processing for the OpenAI-compatible Batch API."""
import asyncio
import json
import os
import time
from typing import Any, Callable, Optional

from vllm.entrypoints.openai.batch.protocol import (
    BatchError,
    BatchErrors,
    BatchObject,
    BatchRequestCounts,
    BatchRequestInput,
    BatchRequestOutput,
    BatchResponseData,
)
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.openai.serving_files import OpenAIServingFiles
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)

SUPPORTED_ENDPOINTS = {
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/score",
}

# Max concurrent requests per batch to prevent memory exhaustion
_BATCH_CONCURRENCY_LIMIT = 256


class OpenAIServingBatches:
    """Manages batch lifecycle: create, process, cancel, cleanup.

    Does not inherit from OpenAIServing — delegates to existing serving
    handlers for actual request processing.
    """

    def __init__(
        self,
        storage_dir: str,
        serving_files: OpenAIServingFiles,
        serving_chat: Any,  # Optional[OpenAIServingChat]
        serving_embedding: Any,  # Optional[OpenAIServingEmbedding]
        serving_score: Any,  # Optional[ServingScores]
        batch_priority: int = 0,
        retention_hours: int = 24,
    ) -> None:
        self.storage_dir = storage_dir
        self._serving_files = serving_files
        self._serving_chat = serving_chat
        self._serving_embedding = serving_embedding
        self._serving_score = serving_score
        self._batch_priority = batch_priority
        self._retention_hours = retention_hours

        self.metadata_dir = os.path.join(storage_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)

        self._batches: dict[str, BatchObject] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self._metadata_path = os.path.join(self.metadata_dir, "batches.json")
        self._cleanup_task: Optional[asyncio.Task] = None

        self._load_metadata()
        self._recover_crashed_batches()

        if self._retention_hours > 0:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(self._cleanup_loop())

    def _load_metadata(self) -> None:
        if os.path.exists(self._metadata_path):
            with open(self._metadata_path, "r") as f:
                data = json.load(f)
            for item in data:
                bo = BatchObject.model_validate(item)
                self._batches[bo.id] = bo
            logger.info("Loaded %d batches from metadata",
                        len(self._batches))

    def _recover_crashed_batches(self) -> None:
        now = int(time.time())
        for batch in self._batches.values():
            if batch.status in ("validating", "in_progress", "cancelling"):
                batch.status = "failed"
                batch.failed_at = now
                if self._retention_hours > 0:
                    batch.expires_at = now + self._retention_hours * 3600
                if batch.errors is None:
                    batch.errors = BatchErrors(data=[])
                batch.errors.data.append(
                    BatchError(
                        code="server_restart",
                        message="Batch was in progress when server "
                        "stopped. Marked as failed on restart.",
                    ))
                logger.warning("Batch %s marked failed (crash recovery)",
                               batch.id)
        self._save_metadata_sync()

    def _save_metadata_sync(self) -> None:
        data = [bo.model_dump() for bo in self._batches.values()]
        with open(self._metadata_path, "w") as f:
            json.dump(data, f)

    async def _save_metadata(self) -> None:
        self._save_metadata_sync()

    async def create_batch(
        self,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> BatchObject | ErrorResponse:
        file_obj = await self._serving_files.get_file(input_file_id)
        if file_obj is None:
            return ErrorResponse(error=ErrorInfo(
                message=f"File {input_file_id} not found",
                type="invalid_request_error",
                code=404,
            ))

        if endpoint not in SUPPORTED_ENDPOINTS:
            return ErrorResponse(error=ErrorInfo(
                message=f"Endpoint {endpoint} is not supported. "
                f"Supported: {', '.join(sorted(SUPPORTED_ENDPOINTS))}",
                type="invalid_request_error",
                code=400,
            ))

        now = int(time.time())
        batch_id = f"batch-{random_uuid()}"
        expires_at = (now + self._retention_hours * 3600
                      if self._retention_hours > 0 else None)

        batch = BatchObject(
            id=batch_id,
            endpoint=endpoint,
            input_file_id=input_file_id,
            completion_window=completion_window,
            status="validating",
            created_at=now,
            expires_at=expires_at,
            request_counts=BatchRequestCounts(
                total=0, completed=0, failed=0),
            metadata=metadata,
        )

        async with self._lock:
            self._batches[batch_id] = batch
            await self._save_metadata()

        cancel_event = asyncio.Event()
        self._cancel_events[batch_id] = cancel_event
        loop = asyncio.get_event_loop()
        task = loop.create_task(
            self._process_batch(batch_id, cancel_event))
        self._tasks[batch_id] = task

        logger.info("Created batch %s (file=%s, endpoint=%s)",
                     batch_id, input_file_id, endpoint)
        return batch

    def is_file_in_active_batch(self, file_id: str) -> bool:
        """Check if a file is referenced by a non-terminal batch."""
        for batch in self._batches.values():
            if batch.status in ("validating", "in_progress", "cancelling"):
                if file_id in (batch.input_file_id,
                               batch.output_file_id,
                               batch.error_file_id):
                    return True
        return False

    async def get_batch(self, batch_id: str) -> Optional[BatchObject]:
        return self._batches.get(batch_id)

    async def list_batches(
        self,
        limit: int = 20,
        after: Optional[str] = None,
    ) -> tuple[list[BatchObject], bool]:
        all_batches = sorted(
            self._batches.values(),
            key=lambda b: b.created_at,
            reverse=True,
        )

        if after is not None:
            found = False
            filtered = []
            for b in all_batches:
                if found:
                    filtered.append(b)
                if b.id == after:
                    found = True
            all_batches = filtered

        limit = min(limit, 100)
        has_more = len(all_batches) > limit
        return all_batches[:limit], has_more

    async def cancel_batch(
        self, batch_id: str
    ) -> Optional[BatchObject | ErrorResponse]:
        batch = self._batches.get(batch_id)
        if batch is None:
            return None

        if batch.status not in ("validating", "in_progress"):
            return ErrorResponse(error=ErrorInfo(
                message=f"Cannot cancel batch with status '{batch.status}'",
                type="invalid_request_error",
                code=400,
            ))

        batch.status = "cancelling"
        batch.cancelling_at = int(time.time())
        async with self._lock:
            await self._save_metadata()

        cancel_event = self._cancel_events.get(batch_id)
        if cancel_event is not None:
            cancel_event.set()

        return batch

    async def _process_batch(
        self,
        batch_id: str,
        cancel_event: asyncio.Event,
    ) -> None:
        batch = self._batches[batch_id]

        try:
            # 1. Read and validate the input file
            content = await self._serving_files.get_file_content(
                batch.input_file_id)
            if content is None:
                await self._fail_batch(batch, "Input file not found")
                return

            lines = content.decode().strip().split("\n")
            requests: list[BatchRequestInput] = []
            validation_errors: list[BatchError] = []

            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    req = BatchRequestInput.model_validate_json(line)
                    if req.url != batch.endpoint:
                        validation_errors.append(BatchError(
                            code="invalid_request",
                            message=f"Request URL {req.url} does not match "
                            f"batch endpoint {batch.endpoint}",
                            line=line_num,
                        ))
                    else:
                        requests.append(req)
                except Exception as e:
                    validation_errors.append(BatchError(
                        code="invalid_request",
                        message=str(e),
                        line=line_num,
                    ))

            if validation_errors and not requests:
                batch.errors = BatchErrors(data=validation_errors)
                await self._fail_batch(batch, "All requests failed validation")
                return

            # 2. Transition to in_progress
            batch.request_counts.total = len(requests)
            batch.status = "in_progress"
            batch.in_progress_at = int(time.time())
            async with self._lock:
                await self._save_metadata()

            # 3. Process requests with bounded concurrency
            semaphore = asyncio.Semaphore(_BATCH_CONCURRENCY_LIMIT)
            outputs: list[BatchRequestOutput] = []
            error_outputs: list[BatchRequestOutput] = []

            async def process_one(req: BatchRequestInput):
                if cancel_event.is_set():
                    return
                async with semaphore:
                    if cancel_event.is_set():
                        return
                    output = await self._run_single_request(batch, req)
                    if output.error is not None:
                        error_outputs.append(output)
                        batch.request_counts.failed += 1
                    else:
                        outputs.append(output)
                        batch.request_counts.completed += 1
                    async with self._lock:
                        await self._save_metadata()

            tasks = [asyncio.ensure_future(process_one(req))
                     for req in requests]
            await asyncio.gather(*tasks, return_exceptions=True)

            # 4. Check for cancellation
            if cancel_event.is_set():
                await self._finalize_batch(
                    batch, outputs, error_outputs,
                    status="cancelled",
                )
                return

            # 5. Finalize
            await self._finalize_batch(
                batch, outputs, error_outputs, status="completed")

        except Exception as e:
            logger.exception("Batch %s failed with error", batch_id)
            await self._fail_batch(batch, str(e))
        finally:
            self._tasks.pop(batch_id, None)
            self._cancel_events.pop(batch_id, None)

    async def _run_single_request(
        self,
        batch: BatchObject,
        request: BatchRequestInput,
    ) -> BatchRequestOutput:
        try:
            # Force non-streaming and set priority for batch
            if hasattr(request.body, 'stream'):
                request.body.stream = False
            if hasattr(request.body, 'priority'):
                request.body.priority = self._batch_priority

            handler_fn = self._get_handler_fn(request.url)
            if handler_fn is None:
                return BatchRequestOutput(
                    id=f"vllm-{random_uuid()}",
                    custom_id=request.custom_id,
                    response=BatchResponseData(
                        status_code=400,
                        request_id=f"vllm-batch-{random_uuid()}"),
                    error=ErrorResponse(error=ErrorInfo(
                        message=f"No handler for {request.url}",
                        type="invalid_request_error",
                        code=400,
                    )),
                )

            # Call handler with request body only (raw_request=None,
            # same pattern as run_batch.py)
            response = await handler_fn(request.body)

            if isinstance(response, ErrorResponse):
                return BatchRequestOutput(
                    id=f"vllm-{random_uuid()}",
                    custom_id=request.custom_id,
                    response=BatchResponseData(
                        status_code=response.error.code,
                        request_id=f"vllm-batch-{random_uuid()}"),
                    error=response,
                )
            elif response is not None:
                return BatchRequestOutput(
                    id=f"vllm-{random_uuid()}",
                    custom_id=request.custom_id,
                    response=BatchResponseData(
                        body=response,
                        request_id=f"vllm-batch-{random_uuid()}"),
                    error=None,
                )
            else:
                return BatchRequestOutput(
                    id=f"vllm-{random_uuid()}",
                    custom_id=request.custom_id,
                    response=None,
                    error=ErrorResponse(error=ErrorInfo(
                        message="Unexpected response type",
                        type="server_error",
                        code=500,
                    )),
                )
        except Exception as e:
            return BatchRequestOutput(
                id=f"vllm-{random_uuid()}",
                custom_id=request.custom_id,
                response=None,
                error=ErrorResponse(error=ErrorInfo(
                    message=str(e),
                    type="server_error",
                    code=500,
                )),
            )

    def _get_handler_fn(self, url: str) -> Optional[Callable]:
        if url == "/v1/chat/completions" and self._serving_chat:
            return self._serving_chat.create_chat_completion
        elif url == "/v1/embeddings" and self._serving_embedding:
            return self._serving_embedding.create_embedding
        elif url == "/v1/score" and self._serving_score:
            return self._serving_score.create_score
        return None

    async def _fail_batch(self, batch: BatchObject, message: str) -> None:
        now = int(time.time())
        batch.status = "failed"
        batch.failed_at = now
        if self._retention_hours > 0:
            batch.expires_at = now + self._retention_hours * 3600
        if batch.errors is None:
            batch.errors = BatchErrors(data=[])
        batch.errors.data.append(
            BatchError(code="batch_failed", message=message))
        async with self._lock:
            await self._save_metadata()

    async def _finalize_batch(
        self,
        batch: BatchObject,
        outputs: list[BatchRequestOutput],
        error_outputs: list[BatchRequestOutput],
        status: str,
    ) -> None:
        now = int(time.time())
        batch.finalizing_at = now

        # Write output file
        if outputs:
            output_lines = "\n".join(
                o.model_dump_json() for o in outputs) + "\n"
            output_file = await self._serving_files.upload_file(
                output_lines.encode(),
                f"{batch.id}_output.jsonl",
                "batch_output",
            )
            batch.output_file_id = output_file.id

        # Write error file
        if error_outputs:
            error_lines = "\n".join(
                o.model_dump_json() for o in error_outputs) + "\n"
            error_file = await self._serving_files.upload_file(
                error_lines.encode(),
                f"{batch.id}_errors.jsonl",
                "batch_error",
            )
            batch.error_file_id = error_file.id

        batch.status = status
        if status == "completed":
            batch.completed_at = now
        elif status == "cancelled":
            batch.cancelled_at = now

        # Update expires_at based on completion time
        if self._retention_hours > 0:
            batch.expires_at = now + self._retention_hours * 3600

        async with self._lock:
            await self._save_metadata()

        logger.info("Batch %s finalized with status=%s "
                     "(completed=%d, failed=%d)",
                     batch.id, status,
                     batch.request_counts.completed,
                     batch.request_counts.failed)

    async def _cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(15 * 60)  # Every 15 minutes
                await self._cleanup_expired()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Error in batch cleanup loop")

    async def _cleanup_expired(self) -> None:
        now = int(time.time())
        expired_ids = [
            b.id for b in self._batches.values()
            if b.expires_at is not None
            and b.expires_at <= now
            and b.status in ("completed", "failed", "cancelled")
        ]
        for batch_id in expired_ids:
            batch = self._batches[batch_id]
            for file_id in (batch.input_file_id,
                            batch.output_file_id,
                            batch.error_file_id):
                if file_id:
                    await self._serving_files.delete_file(file_id)

            async with self._lock:
                del self._batches[batch_id]
                await self._save_metadata()

            logger.info("Cleaned up expired batch %s", batch_id)

    async def wait_for_batch(
        self, batch_id: str, timeout: float = 60
    ) -> None:
        """Wait for a batch task to complete. Used in tests."""
        task = self._tasks.get(batch_id)
        if task is not None:
            await asyncio.wait_for(task, timeout=timeout)

    async def shutdown(self) -> None:
        """Graceful shutdown: cancel tasks, persist state."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        for cancel_event in list(self._cancel_events.values()):
            cancel_event.set()

        for task in list(self._tasks.values()):
            try:
                await asyncio.wait_for(task, timeout=30)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

        now = int(time.time())
        for batch in self._batches.values():
            if batch.status in ("validating", "in_progress", "cancelling"):
                batch.status = "cancelled"
                batch.cancelled_at = now

        async with self._lock:
            await self._save_metadata()

        logger.info("Batch serving shut down")
