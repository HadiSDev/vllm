# Online Batch API for vLLM

**Date:** 2026-04-07
**Status:** Draft

## Overview

Add a full OpenAI-compatible Batch API to vLLM's existing API server. This enables users to upload JSONL batch files via HTTP, create batch jobs that process requests at low priority using the running engine, and retrieve results — all without managing separate batch processes.

## Motivation

Currently vLLM supports batch processing only via a standalone CLI tool (`python -m vllm.entrypoints.openai.run_batch`). This requires direct filesystem access and cannot be used by clients that interact with vLLM through its HTTP API. Cloud providers like OpenAI offer a Batch API that lets users submit large workloads at lower priority and cost. This design brings the same capability to vLLM's API server.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Integration approach | Integrated into existing API server | Follows codebase patterns, shares engine, no extra deployment complexity |
| Execution model | Same engine, lower priority | Batch fills idle capacity; online requests always take precedence |
| Storage | Local filesystem | Survives restarts, simple, configurable directory |
| Limits | None by default | Self-hosted operators decide what's reasonable |
| Completion window | Accepted but ignored | API compatibility without arbitrary timeouts |
| Cleanup | Auto-delete after configurable TTL | Prevents unbounded disk usage |

## API Endpoints

### Files API (`/v1/files`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/files` | POST | Upload a JSONL file (multipart form: `file` + `purpose` field) |
| `/v1/files` | GET | List uploaded files (supports `purpose` filter) |
| `/v1/files/{file_id}` | GET | Retrieve file metadata |
| `/v1/files/{file_id}` | DELETE | Delete a file (409 if tied to active batch) |
| `/v1/files/{file_id}/content` | GET | Download file content |

### Batches API (`/v1/batches`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/batches` | POST | Create a batch (`input_file_id`, `endpoint`, `completion_window`) |
| `/v1/batches` | GET | List batches (`after` + `limit` pagination) |
| `/v1/batches/{batch_id}` | GET | Retrieve batch status and metadata |
| `/v1/batches/{batch_id}/cancel` | POST | Cancel an in-progress batch |

## Batch Lifecycle

```
validating --> in_progress --> completed
                           --> failed
                           --> cancelling --> cancelled
```

### States

- **validating**: JSONL file is being parsed and validated. If any line is malformed or references an unsupported endpoint, the batch transitions to `failed`.
- **in_progress**: Requests are being processed by the engine at low priority.
- **completed**: All requests finished. Output file and (optionally) error file are available.
- **failed**: Validation failed or an unrecoverable error occurred.
- **cancelled**: User cancelled the batch. Partially completed results are available in the output file.

## Architecture

### New Files

| File | Class | Purpose |
|------|-------|---------|
| `vllm/entrypoints/openai/serving_files.py` | `OpenAIServingFiles` | File upload, listing, retrieval, deletion |
| `vllm/entrypoints/openai/serving_batches.py` | `OpenAIServingBatches` | Batch creation, status, cancellation, background processing |

`OpenAIServingBatches` holds references to the existing serving handlers (chat, embedding, score) and the engine client. It does NOT inherit from `OpenAIServing` since it delegates to existing handlers rather than interacting with the engine directly.

`OpenAIServingFiles` is a standalone class — it has no need for an engine client, model config, or tokenizer. It only manages file CRUD on the filesystem.

### Protocol Models (added to `protocol.py`)

```python
class FileObject(OpenAIBaseModel):
    id: str                    # "file-{uuid}"
    object: str = "file"
    bytes: int
    created_at: int            # Unix timestamp
    filename: str
    purpose: str               # "batch", "batch_output", "batch_error"

class FileListResponse(OpenAIBaseModel):
    object: str = "list"
    data: list[FileObject]

class BatchObject(OpenAIBaseModel):
    id: str                    # "batch-{uuid}"
    object: str = "batch"
    endpoint: str              # "/v1/chat/completions", "/v1/embeddings", "/v1/score"
    input_file_id: str
    output_file_id: Optional[str]
    error_file_id: Optional[str]
    status: str                # "validating", "in_progress", "completed", "failed", "cancelling", "cancelled"
    completion_window: str     # Accepted but not enforced
    created_at: int
    in_progress_at: Optional[int]
    finalizing_at: Optional[int]
    completed_at: Optional[int]
    failed_at: Optional[int]
    cancelling_at: Optional[int]
    cancelled_at: Optional[int]
    expires_at: Optional[int]  # When the batch will be cleaned up (created_at + retention)
    request_counts: BatchRequestCounts
    errors: Optional[BatchErrors]
    metadata: Optional[dict[str, str]]  # Up to 16 key-value pairs, for client tracking

class BatchRequestCounts(OpenAIBaseModel):
    total: int
    completed: int
    failed: int

class BatchErrors(OpenAIBaseModel):
    object: str = "list"
    data: list[BatchError]

class BatchError(OpenAIBaseModel):
    code: str
    message: str
    param: Optional[str]
    line: Optional[int]

class BatchListResponse(OpenAIBaseModel):
    object: str = "list"
    data: list[BatchObject]
    has_more: bool
    first_id: Optional[str]
    last_id: Optional[str]
```

### Batch Processing Flow

```
POST /v1/files (purpose="batch")
    --> File saved to {storage_dir}/files/file-{uuid}.jsonl
    --> FileObject created and persisted to metadata

POST /v1/batches (input_file_id=...)
    --> BatchObject created (status="validating")
    --> Background asyncio.Task spawned:
        1. Read and parse JSONL file
           - Validate each line as BatchRequestInput
           - If invalid: status="failed", write error file, return
        2. status="in_progress"
        3. For each request:
           - Route to appropriate serving handler based on URL
             (/v1/chat/completions -> OpenAIServingChat, etc.)
           - Call handler with priority=LOW (higher number than default 0)
           - Collect BatchRequestOutput (success or per-request error)
           - Update request_counts in real-time
        4. On completion:
           - Write output JSONL to {storage_dir}/files/file-{uuid}.jsonl
           - Create output FileObject (purpose="batch_output")
           - If any errors: write error JSONL, create error FileObject
           - status="completed", set completed_at timestamp
```

### Priority Integration

vLLM's engine supports a `priority` parameter on `generate()` and `encode()`. Online requests use the default priority `0`. Batch requests use a higher number (= lower priority, default `100`, configurable via `--batch-priority`).

**Mechanism**: Before dispatching each request to the serving handler, `OpenAIServingBatches` mutates `request.body.priority` to the configured batch priority value. The existing serving handlers (`create_chat_completion`, etc.) read `request.priority` and pass it to the engine. This is the simplest approach — no handler signature changes needed.

**`raw_request` handling**: Serving handlers accept `raw_request: Optional[Request]` for HTTP-specific features like disconnect detection. In batch mode, `raw_request=None` is passed (same pattern used by the existing `run_batch.py`). Handlers already handle this gracefully.

### Storage Layout

```
{storage_dir}/                          # Default: ~/.vllm/batches/
  files/
    file-abc123.jsonl                   # Uploaded input files
    file-def456.jsonl                   # Output result files
    file-ghi789.jsonl                   # Error files
  metadata/
    files.json                          # FileObject index
    batches.json                        # BatchObject index
```

### State Management

- `OpenAIServingFiles` holds an in-memory `dict[str, FileObject]`, synced to `metadata/files.json` on every mutation.
- `OpenAIServingBatches` holds an in-memory `dict[str, BatchObject]`, synced to `metadata/batches.json` on every mutation.
- **Thread safety**: All metadata writes are guarded by an `asyncio.Lock` per metadata file. This prevents corruption when multiple concurrent batch tasks update `request_counts` or status simultaneously.
- On server startup, metadata is loaded from disk to restore state.
- Batches that were `in_progress` or `cancelling` during a prior server crash are marked `failed` with an error message indicating server restart. No partial output file is available for crash-failed batches (unlike cancelled batches which complete their in-flight requests and write partial output).

## Concurrency & Cancellation

- Each batch runs as a single `asyncio.Task`.
- Requests are submitted with **bounded concurrency** using an `asyncio.Semaphore` (default limit: 256 concurrent requests per batch). This prevents memory exhaustion for large batch files with thousands of requests. Requests are read and dispatched in a loop, with the semaphore gating how many are in-flight simultaneously.
- A `dict[str, asyncio.Task]` in `OpenAIServingBatches` tracks active batch tasks.
- Cancellation: `POST /v1/batches/{id}/cancel` transitions to `status="cancelling"`, sets `cancelling_at`, and sets a cancellation event. The batch task checks this event between request dispatches, stops submitting new requests, waits for in-flight requests to complete, writes partial results to the output file, then transitions to `status="cancelled"` with `cancelled_at` set.

## Error Handling

- **Per-request errors**: Isolated. A failed request is recorded in the error output file; other requests continue. Matches OpenAI behavior.
- **Batch-level errors**: Invalid JSONL or unrecognized endpoint URLs fail the entire batch during the validation phase.
- **Server crash recovery**: On startup, any batch with `status="in_progress"` from a prior run is marked `status="failed"`.

## Cleanup & Retention

- When a batch reaches a terminal state (`completed`, `failed`, `cancelled`), a TTL countdown begins.
- After the TTL expires (default: 24 hours), the batch metadata, input file, output file, and error file are all deleted.
- A periodic `asyncio.Task` runs every 15 minutes, scanning for expired batches and cleaning them up.
- Configurable via `--batch-retention-hours` (default `24`). Set to `0` to disable automatic cleanup.
- Files tied to an active (non-terminal) batch cannot be manually deleted (returns 409).
- Expired batches are fully removed — `GET /v1/batches/{id}` returns 404.

## Server Integration

### Changes to `api_server.py`

1. New dependency functions: `files(request)` and `batches(request)` returning from `app.state`.
2. Eight new route handlers added to the router.
3. In the lifespan handler: create `OpenAIServingFiles` and `OpenAIServingBatches` alongside existing serving classes.
4. On shutdown: cancel running batch tasks gracefully, persist final state.

### Changes to `cli_args.py`

Three new flags:
- `--batch-storage-dir` (default: `~/.vllm/batches/`) — where files and metadata are stored.
- `--batch-retention-hours` (default: `24`) — how long completed batches are retained before cleanup.
- `--batch-priority` (default: `100`) — priority value for batch requests (higher = lower priority; online requests use `0`).

### Changes to `protocol.py`

Add `FileObject`, `FileListResponse`, `BatchObject`, `BatchRequestCounts`, `BatchErrors`, `BatchError`, `BatchListResponse` Pydantic models.

## Supported Batch Endpoints

Matching the existing `run_batch.py` support:
- `/v1/chat/completions`
- `/v1/embeddings`
- `/v1/score`

Other endpoints return a validation error when the batch is created.

## Shutdown Behavior

On graceful server shutdown:
1. Running batch tasks receive a cancellation signal.
2. In-flight requests are allowed to complete (up to a 30-second grace period).
3. Partial results are written to output files.
4. Batches are marked `status="cancelled"` with `cancelled_at` set.
5. Final metadata is persisted to disk.

On hard kill (SIGKILL / crash): no cleanup occurs. On next startup, in-progress batches are marked `failed` per the crash recovery logic.

## Pagination

`GET /v1/batches` returns batches sorted by `created_at` descending (newest first). The `after` parameter is exclusive (returns batches created after the given batch ID). Default `limit` is 20, maximum 100.

`GET /v1/files` returns files sorted by `created_at` descending. Supports `purpose` query parameter for filtering.

## Out of Scope

- Rate limiting or queue depth limits
- Completion window enforcement (field accepted for compatibility)
- Webhook/callback notifications
- Multi-node batch coordination
- Changes to the existing `run_batch.py` CLI tool
- `/v1/completions` (text completions) as a batch endpoint — can be added later
- File upload validation beyond JSONL parsing (size limits, content-type checks) — can be added later
