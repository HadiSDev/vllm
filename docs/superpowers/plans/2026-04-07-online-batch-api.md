# Online Batch API Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full OpenAI-compatible Files API and Batches API to vLLM's API server, enabling HTTP-based batch processing with lower-priority scheduling.

**Architecture:** Two new standalone serving classes (`OpenAIServingFiles`, `OpenAIServingBatches`) integrated into the existing FastAPI server. Files are stored on local disk. Batch processing runs as background asyncio tasks feeding requests into the existing engine at low priority. Metadata is persisted to JSON files with asyncio.Lock guards.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, asyncio, vLLM engine (AsyncLLMEngine/AsyncLLM)

**Spec:** `docs/superpowers/specs/2026-04-07-online-batch-api-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `vllm/entrypoints/openai/protocol.py` | Modify | Add FileObject, BatchObject, and related Pydantic models |
| `vllm/entrypoints/openai/serving_files.py` | Create | File CRUD: upload, list, get, delete, content download |
| `vllm/entrypoints/openai/serving_batches.py` | Create | Batch lifecycle: create, list, get, cancel, background processing |
| `vllm/entrypoints/openai/api_server.py` | Modify | Add routes and dependency functions for files/batches |
| `vllm/entrypoints/openai/cli_args.py` | Modify | Add --batch-storage-dir, --batch-retention-hours, --batch-priority flags |
| `tests/entrypoints/openai/test_serving_files.py` | Create | Unit tests for OpenAIServingFiles |
| `tests/entrypoints/openai/test_serving_batches.py` | Create | Unit tests for OpenAIServingBatches |
| `tests/entrypoints/openai/test_batch_api_integration.py` | Create | Integration tests for the full batch workflow via HTTP |

---

## Chunk 1: Protocol Models & CLI Args

### Task 1: Add Protocol Models to `protocol.py`

**Files:**
- Modify: `vllm/entrypoints/openai/protocol.py` (append after line ~1375, after `BatchRequestOutput`)
- Test: `tests/entrypoints/openai/test_batch_protocol.py`

- [ ] **Step 1: Write tests for the new protocol models**

Create `tests/entrypoints/openai/test_batch_protocol.py`:

```python
"""Tests for Batch API protocol models."""
import time
from vllm.entrypoints.openai.protocol import (
    BatchError,
    BatchErrors,
    BatchObject,
    BatchRequestCounts,
    FileObject,
    FileListResponse,
    BatchListResponse,
)


def test_file_object_creation():
    f = FileObject(
        id="file-abc123",
        bytes=1024,
        created_at=int(time.time()),
        filename="batch_input.jsonl",
        purpose="batch",
    )
    assert f.object == "file"
    assert f.id == "file-abc123"
    assert f.purpose == "batch"


def test_file_object_allows_batch_purposes():
    for purpose in ("batch", "batch_output", "batch_error"):
        f = FileObject(
            id="file-x",
            bytes=0,
            created_at=0,
            filename="f.jsonl",
            purpose=purpose,
        )
        assert f.purpose == purpose


def test_batch_request_counts_defaults():
    c = BatchRequestCounts(total=10, completed=0, failed=0)
    assert c.total == 10


def test_batch_error_model():
    e = BatchError(code="invalid_request", message="bad", param=None, line=3)
    assert e.code == "invalid_request"
    assert e.line == 3


def test_batch_errors_model():
    errs = BatchErrors(data=[
        BatchError(code="err", message="msg", param=None, line=1),
    ])
    assert errs.object == "list"
    assert len(errs.data) == 1


def test_batch_object_creation():
    now = int(time.time())
    b = BatchObject(
        id="batch-abc",
        endpoint="/v1/chat/completions",
        input_file_id="file-123",
        completion_window="24h",
        status="validating",
        created_at=now,
        request_counts=BatchRequestCounts(total=5, completed=0, failed=0),
    )
    assert b.object == "batch"
    assert b.status == "validating"
    assert b.output_file_id is None
    assert b.error_file_id is None
    assert b.metadata is None
    assert b.expires_at is None


def test_batch_object_with_metadata():
    b = BatchObject(
        id="batch-abc",
        endpoint="/v1/chat/completions",
        input_file_id="file-123",
        completion_window="24h",
        status="validating",
        created_at=0,
        request_counts=BatchRequestCounts(total=1, completed=0, failed=0),
        metadata={"user": "test", "job": "nightly"},
    )
    assert b.metadata == {"user": "test", "job": "nightly"}


def test_batch_list_response():
    r = BatchListResponse(
        data=[],
        has_more=False,
        first_id=None,
        last_id=None,
    )
    assert r.object == "list"
    assert r.data == []


def test_file_list_response():
    r = FileListResponse(data=[])
    assert r.object == "list"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/entrypoints/openai/test_batch_protocol.py -v`
Expected: ImportError — models don't exist yet.

- [ ] **Step 3: Implement the protocol models**

Add to `vllm/entrypoints/openai/protocol.py` after the existing `BatchRequestOutput` class (around line 1375):

```python
class FileObject(OpenAIBaseModel):
    """Represents an uploaded file."""
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str  # "batch", "batch_output", "batch_error"


class FileListResponse(OpenAIBaseModel):
    object: str = "list"
    data: list[FileObject]


class BatchRequestCounts(OpenAIBaseModel):
    total: int
    completed: int
    failed: int


class BatchError(OpenAIBaseModel):
    code: str
    message: str
    param: Optional[str] = None
    line: Optional[int] = None


class BatchErrors(OpenAIBaseModel):
    object: str = "list"
    data: list[BatchError]


class BatchObject(OpenAIBaseModel):
    """Represents a batch processing job."""
    id: str
    object: str = "batch"
    endpoint: str
    input_file_id: str
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    status: str  # validating, in_progress, completed, failed, cancelling, cancelled
    completion_window: str
    created_at: int
    in_progress_at: Optional[int] = None
    finalizing_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    expires_at: Optional[int] = None
    request_counts: BatchRequestCounts
    errors: Optional[BatchErrors] = None
    metadata: Optional[dict[str, str]] = None


class BatchListResponse(OpenAIBaseModel):
    object: str = "list"
    data: list[BatchObject]
    has_more: bool
    first_id: Optional[str] = None
    last_id: Optional[str] = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/entrypoints/openai/test_batch_protocol.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/entrypoints/openai/protocol.py tests/entrypoints/openai/test_batch_protocol.py
git commit -m "feat: add protocol models for Files and Batches API"
```

### Task 2: Add CLI Arguments

**Files:**
- Modify: `vllm/entrypoints/openai/cli_args.py` (around line 240, before `AsyncEngineArgs.add_cli_args`)

- [ ] **Step 1: Add batch CLI arguments**

First, add `import os` to the imports at the top of `cli_args.py` if not already present.

Then add the following arguments to `make_arg_parser()` in `cli_args.py`, before the `AsyncEngineArgs.add_cli_args(parser)` call (around line 241):

```python
    parser.add_argument(
        "--batch-storage-dir",
        type=str,
        default=os.path.expanduser("~/.vllm/batches"),
        help="Directory for storing batch files and metadata. "
        "Default: ~/.vllm/batches")
    parser.add_argument(
        "--batch-retention-hours",
        type=int,
        default=24,
        help="Hours to retain completed batches before cleanup. "
        "Set to 0 to disable automatic cleanup. Default: 24")
    parser.add_argument(
        "--batch-priority",
        type=int,
        default=0,
        help="Priority value for batch requests. Higher values = lower "
        "priority. Online requests use 0. Set to >0 only if the model "
        "is served with priority scheduling enabled. Default: 0")
```

Also add `import os` at the top of the file if not already imported.

- [ ] **Step 2: Verify the args parse correctly**

Run: `python -c "from vllm.entrypoints.openai.cli_args import make_arg_parser; p = make_arg_parser(); a = p.parse_args(['--model', 'x', '--batch-storage-dir', '/tmp/test', '--batch-retention-hours', '48', '--batch-priority', '50']); print(a.batch_storage_dir, a.batch_retention_hours, a.batch_priority)"`
Expected: `/tmp/test 48 50`

- [ ] **Step 3: Commit**

```bash
git add vllm/entrypoints/openai/cli_args.py
git commit -m "feat: add batch API CLI arguments"
```

---

## Chunk 2: OpenAIServingFiles

### Task 3: Implement `OpenAIServingFiles`

**Files:**
- Create: `vllm/entrypoints/openai/serving_files.py`
- Create: `tests/entrypoints/openai/test_serving_files.py`

- [ ] **Step 1: Write tests for OpenAIServingFiles**

Create `tests/entrypoints/openai/test_serving_files.py`:

```python
"""Tests for OpenAIServingFiles."""
import json
import os
import tempfile
import pytest
import asyncio

from vllm.entrypoints.openai.serving_files import OpenAIServingFiles
from vllm.entrypoints.openai.protocol import FileObject


@pytest.fixture
def storage_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def serving_files(storage_dir):
    return OpenAIServingFiles(storage_dir=storage_dir)


@pytest.mark.asyncio
async def test_upload_file(serving_files, storage_dir):
    content = b'{"custom_id": "r1", "method": "POST", "url": "/v1/chat/completions", "body": {}}\n'
    result = await serving_files.upload_file(
        content=content,
        filename="batch.jsonl",
        purpose="batch",
    )
    assert isinstance(result, FileObject)
    assert result.id.startswith("file-")
    assert result.filename == "batch.jsonl"
    assert result.purpose == "batch"
    assert result.bytes == len(content)
    # File should exist on disk
    assert os.path.exists(os.path.join(storage_dir, "files", result.id + ".jsonl"))


@pytest.mark.asyncio
async def test_list_files(serving_files):
    await serving_files.upload_file(b"line1\n", "a.jsonl", "batch")
    await serving_files.upload_file(b"line2\n", "b.jsonl", "batch_output")
    all_files = await serving_files.list_files()
    assert len(all_files) == 2
    batch_only = await serving_files.list_files(purpose="batch")
    assert len(batch_only) == 1
    assert batch_only[0].filename == "a.jsonl"


@pytest.mark.asyncio
async def test_get_file(serving_files):
    uploaded = await serving_files.upload_file(b"data\n", "f.jsonl", "batch")
    retrieved = await serving_files.get_file(uploaded.id)
    assert retrieved is not None
    assert retrieved.id == uploaded.id


@pytest.mark.asyncio
async def test_get_file_not_found(serving_files):
    result = await serving_files.get_file("file-nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_get_file_content(serving_files):
    content = b"hello world\n"
    uploaded = await serving_files.upload_file(content, "f.jsonl", "batch")
    retrieved_content = await serving_files.get_file_content(uploaded.id)
    assert retrieved_content == content


@pytest.mark.asyncio
async def test_get_file_content_not_found(serving_files):
    result = await serving_files.get_file_content("file-nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_delete_file(serving_files, storage_dir):
    uploaded = await serving_files.upload_file(b"data\n", "f.jsonl", "batch")
    success = await serving_files.delete_file(uploaded.id)
    assert success is True
    assert await serving_files.get_file(uploaded.id) is None
    # File should be removed from disk
    assert not os.path.exists(
        os.path.join(storage_dir, "files", uploaded.id + ".jsonl"))


@pytest.mark.asyncio
async def test_delete_file_not_found(serving_files):
    success = await serving_files.delete_file("file-nonexistent")
    assert success is False


@pytest.mark.asyncio
async def test_metadata_persistence(storage_dir):
    """Files survive re-instantiation (loaded from disk)."""
    sf1 = OpenAIServingFiles(storage_dir=storage_dir)
    uploaded = await sf1.upload_file(b"data\n", "f.jsonl", "batch")

    sf2 = OpenAIServingFiles(storage_dir=storage_dir)
    retrieved = await sf2.get_file(uploaded.id)
    assert retrieved is not None
    assert retrieved.id == uploaded.id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/entrypoints/openai/test_serving_files.py -v`
Expected: ImportError — `serving_files` module doesn't exist yet.

- [ ] **Step 3: Implement OpenAIServingFiles**

Create `vllm/entrypoints/openai/serving_files.py`:

```python
"""File management for the OpenAI-compatible Batch API."""
import asyncio
import json
import os
import time
from typing import Optional

from vllm.entrypoints.openai.protocol import FileObject
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)


class OpenAIServingFiles:
    """Manages file uploads and retrieval for the Batch API.

    Standalone class — does not inherit from OpenAIServing since it has
    no need for an engine client, model config, or tokenizer.
    """

    def __init__(self, storage_dir: str) -> None:
        self.storage_dir = storage_dir
        self.files_dir = os.path.join(storage_dir, "files")
        self.metadata_dir = os.path.join(storage_dir, "metadata")
        os.makedirs(self.files_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        self._files: dict[str, FileObject] = {}
        self._lock = asyncio.Lock()
        self._metadata_path = os.path.join(self.metadata_dir, "files.json")

        self._load_metadata()

    def _load_metadata(self) -> None:
        if os.path.exists(self._metadata_path):
            with open(self._metadata_path, "r") as f:
                data = json.load(f)
            for item in data:
                fo = FileObject.model_validate(item)
                self._files[fo.id] = fo
            logger.info("Loaded %d files from metadata", len(self._files))

    async def _save_metadata(self) -> None:
        data = [fo.model_dump() for fo in self._files.values()]
        with open(self._metadata_path, "w") as f:
            json.dump(data, f)

    async def upload_file(
        self,
        content: bytes,
        filename: str,
        purpose: str,
    ) -> FileObject:
        file_id = f"file-{random_uuid()}"
        file_path = os.path.join(self.files_dir, f"{file_id}.jsonl")

        with open(file_path, "wb") as f:
            f.write(content)

        file_obj = FileObject(
            id=file_id,
            bytes=len(content),
            created_at=int(time.time()),
            filename=filename,
            purpose=purpose,
        )

        async with self._lock:
            self._files[file_id] = file_obj
            await self._save_metadata()

        logger.info("Uploaded file %s (%s, %d bytes)", file_id, filename,
                     len(content))
        return file_obj

    async def list_files(
        self,
        purpose: Optional[str] = None,
    ) -> list[FileObject]:
        files = list(self._files.values())
        if purpose is not None:
            files = [f for f in files if f.purpose == purpose]
        files.sort(key=lambda f: f.created_at, reverse=True)
        return files

    async def get_file(self, file_id: str) -> Optional[FileObject]:
        return self._files.get(file_id)

    async def get_file_content(self, file_id: str) -> Optional[bytes]:
        if file_id not in self._files:
            return None
        file_path = os.path.join(self.files_dir, f"{file_id}.jsonl")
        if not os.path.exists(file_path):
            return None
        with open(file_path, "rb") as f:
            return f.read()

    async def delete_file(
        self,
        file_id: str,
        force: bool = False,
    ) -> bool:
        """Delete a file. If force=False, callers should check batch
        associations before calling this."""
        if file_id not in self._files:
            return False

        file_path = os.path.join(self.files_dir, f"{file_id}.jsonl")
        if os.path.exists(file_path):
            os.remove(file_path)

        async with self._lock:
            del self._files[file_id]
            await self._save_metadata()

        logger.info("Deleted file %s", file_id)
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/entrypoints/openai/test_serving_files.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/entrypoints/openai/serving_files.py tests/entrypoints/openai/test_serving_files.py
git commit -m "feat: add OpenAIServingFiles for Batch API file management"
```

---

## Chunk 3: OpenAIServingBatches

### Task 4: Implement `OpenAIServingBatches` — Core Lifecycle

**Files:**
- Create: `vllm/entrypoints/openai/serving_batches.py`
- Create: `tests/entrypoints/openai/test_serving_batches.py`

- [ ] **Step 1: Write tests for batch creation, listing, retrieval, and cancellation**

Create `tests/entrypoints/openai/test_serving_batches.py`:

```python
"""Tests for OpenAIServingBatches."""
import asyncio
import json
import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.entrypoints.openai.protocol import (
    BatchObject,
    BatchRequestCounts,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ErrorResponse,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_batches import OpenAIServingBatches
from vllm.entrypoints.openai.serving_files import OpenAIServingFiles


@pytest.fixture
def storage_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def serving_files(storage_dir):
    return OpenAIServingFiles(storage_dir=storage_dir)


def _make_chat_response():
    """Helper to create a minimal ChatCompletionResponse."""
    return ChatCompletionResponse(
        id="chatcmpl-test",
        created=int(time.time()),
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


def _make_mock_chat_handler():
    handler = AsyncMock()
    handler.create_chat_completion = AsyncMock(
        return_value=_make_chat_response())
    return handler


@pytest.fixture
def serving_batches(storage_dir, serving_files):
    return OpenAIServingBatches(
        storage_dir=storage_dir,
        serving_files=serving_files,
        serving_chat=_make_mock_chat_handler(),
        serving_embedding=None,
        serving_score=None,
        batch_priority=100,
        retention_hours=24,
    )


def _make_batch_jsonl(n=2):
    """Create JSONL content with n chat completion requests."""
    lines = []
    for i in range(n):
        lines.append(json.dumps({
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "test-model",
                "messages": [{"role": "user", "content": f"Hello {i}"}],
                "max_tokens": 10,
            },
        }))
    return "\n".join(lines).encode()


@pytest.mark.asyncio
async def test_create_batch(serving_batches, serving_files):
    content = _make_batch_jsonl(2)
    file_obj = await serving_files.upload_file(content, "in.jsonl", "batch")

    batch = await serving_batches.create_batch(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    assert isinstance(batch, BatchObject)
    assert batch.id.startswith("batch-")
    assert batch.status == "validating"
    assert batch.input_file_id == file_obj.id
    assert batch.request_counts.total == 0  # Set during async processing


@pytest.mark.asyncio
async def test_create_batch_invalid_file(serving_batches):
    result = await serving_batches.create_batch(
        input_file_id="file-nonexistent",
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_create_batch_invalid_endpoint(serving_batches, serving_files):
    content = _make_batch_jsonl(1)
    file_obj = await serving_files.upload_file(content, "in.jsonl", "batch")
    result = await serving_batches.create_batch(
        input_file_id=file_obj.id,
        endpoint="/v1/invalid",
        completion_window="24h",
    )
    assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_get_batch(serving_batches, serving_files):
    content = _make_batch_jsonl(1)
    file_obj = await serving_files.upload_file(content, "in.jsonl", "batch")
    batch = await serving_batches.create_batch(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    retrieved = await serving_batches.get_batch(batch.id)
    assert retrieved is not None
    assert retrieved.id == batch.id


@pytest.mark.asyncio
async def test_get_batch_not_found(serving_batches):
    result = await serving_batches.get_batch("batch-nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_list_batches(serving_batches, serving_files):
    content = _make_batch_jsonl(1)
    f1 = await serving_files.upload_file(content, "a.jsonl", "batch")
    f2 = await serving_files.upload_file(content, "b.jsonl", "batch")
    await serving_batches.create_batch(f1.id, "/v1/chat/completions", "24h")
    await serving_batches.create_batch(f2.id, "/v1/chat/completions", "24h")
    batches, has_more = await serving_batches.list_batches(limit=20)
    assert len(batches) == 2
    assert not has_more


@pytest.mark.asyncio
async def test_list_batches_pagination(serving_batches, serving_files):
    content = _make_batch_jsonl(1)
    ids = []
    for i in range(3):
        f = await serving_files.upload_file(content, f"{i}.jsonl", "batch")
        b = await serving_batches.create_batch(
            f.id, "/v1/chat/completions", "24h")
        ids.append(b.id)

    batches, has_more = await serving_batches.list_batches(limit=2)
    assert len(batches) == 2
    assert has_more

    batches2, has_more2 = await serving_batches.list_batches(
        limit=2, after=batches[-1].id)
    assert len(batches2) == 1
    assert not has_more2


@pytest.mark.asyncio
async def test_batch_processes_to_completion(serving_batches, serving_files):
    content = _make_batch_jsonl(2)
    file_obj = await serving_files.upload_file(content, "in.jsonl", "batch")
    batch = await serving_batches.create_batch(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    # Wait for background processing to complete
    await serving_batches.wait_for_batch(batch.id, timeout=10)

    updated = await serving_batches.get_batch(batch.id)
    assert updated.status == "completed"
    assert updated.request_counts.completed == 2
    assert updated.request_counts.failed == 0
    assert updated.output_file_id is not None
    assert updated.completed_at is not None

    # Output file should exist and contain valid JSONL
    output_content = await serving_files.get_file_content(
        updated.output_file_id)
    assert output_content is not None
    lines = output_content.decode().strip().split("\n")
    assert len(lines) == 2


@pytest.mark.asyncio
async def test_cancel_batch(serving_batches, serving_files):
    # Use a slow handler to give time to cancel
    slow_handler = AsyncMock()

    async def slow_response(*args, **kwargs):
        await asyncio.sleep(10)
        return _make_chat_response()

    slow_handler.create_chat_completion = AsyncMock(side_effect=slow_response)
    serving_batches._serving_chat = slow_handler

    content = _make_batch_jsonl(50)
    file_obj = await serving_files.upload_file(content, "in.jsonl", "batch")
    batch = await serving_batches.create_batch(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    # Let it start processing
    await asyncio.sleep(0.1)

    result = await serving_batches.cancel_batch(batch.id)
    assert result is not None
    assert result.status in ("cancelling", "cancelled")

    # Wait for cancellation to finalize
    await serving_batches.wait_for_batch(batch.id, timeout=10)
    updated = await serving_batches.get_batch(batch.id)
    assert updated.status == "cancelled"
    assert updated.cancelled_at is not None


@pytest.mark.asyncio
async def test_metadata_persistence(storage_dir, serving_files):
    sb1 = OpenAIServingBatches(
        storage_dir=storage_dir,
        serving_files=serving_files,
        serving_chat=_make_mock_chat_handler(),
        serving_embedding=None,
        serving_score=None,
        batch_priority=100,
        retention_hours=24,
    )
    content = _make_batch_jsonl(1)
    f = await serving_files.upload_file(content, "in.jsonl", "batch")
    batch = await sb1.create_batch(f.id, "/v1/chat/completions", "24h")
    await sb1.wait_for_batch(batch.id, timeout=10)
    await sb1.shutdown()

    # Re-instantiate — should load from disk
    sb2 = OpenAIServingBatches(
        storage_dir=storage_dir,
        serving_files=serving_files,
        serving_chat=_make_mock_chat_handler(),
        serving_embedding=None,
        serving_score=None,
        batch_priority=100,
        retention_hours=24,
    )
    retrieved = await sb2.get_batch(batch.id)
    assert retrieved is not None
    assert retrieved.status == "completed"
    await sb2.shutdown()


@pytest.mark.asyncio
async def test_crash_recovery_marks_in_progress_as_failed(
        storage_dir, serving_files):
    """Simulate a crash by writing metadata with in_progress batch, then reload."""
    metadata_dir = os.path.join(storage_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    batch_data = [{
        "id": "batch-crashed",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-x",
        "status": "in_progress",
        "completion_window": "24h",
        "created_at": int(time.time()),
        "request_counts": {"total": 10, "completed": 3, "failed": 0},
    }]
    with open(os.path.join(metadata_dir, "batches.json"), "w") as f:
        json.dump(batch_data, f)

    sb = OpenAIServingBatches(
        storage_dir=storage_dir,
        serving_files=serving_files,
        serving_chat=None,
        serving_embedding=None,
        serving_score=None,
        batch_priority=100,
        retention_hours=24,
    )
    recovered = await sb.get_batch("batch-crashed")
    assert recovered.status == "failed"
    assert recovered.failed_at is not None
    await sb.shutdown()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/entrypoints/openai/test_serving_batches.py -v`
Expected: ImportError — `serving_batches` module doesn't exist yet.

- [ ] **Step 3: Implement OpenAIServingBatches**

Create `vllm/entrypoints/openai/serving_batches.py`:

```python
"""Batch processing for the OpenAI-compatible Batch API."""
import asyncio
import json
import os
import time
from typing import Any, Callable, Optional

from vllm.entrypoints.openai.protocol import (
    BatchError,
    BatchErrors,
    BatchObject,
    BatchRequestCounts,
    BatchRequestInput,
    BatchRequestOutput,
    BatchResponseData,
    ChatCompletionResponse,
    EmbeddingResponse,
    ErrorResponse,
    FileObject,
    ScoreResponse,
)
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
        batch_priority: int = 100,
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
            self._cleanup_task = asyncio.ensure_future(
                self._cleanup_loop())

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
        # Persist recovery changes synchronously at startup
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
        # Validate file exists
        file_obj = await self._serving_files.get_file(input_file_id)
        if file_obj is None:
            return ErrorResponse(
                message=f"File {input_file_id} not found",
                type="invalid_request_error",
                code=404,
            )

        # Validate endpoint
        if endpoint not in SUPPORTED_ENDPOINTS:
            return ErrorResponse(
                message=f"Endpoint {endpoint} is not supported. "
                f"Supported: {', '.join(sorted(SUPPORTED_ENDPOINTS))}",
                type="invalid_request_error",
                code=400,
            )

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
        task = asyncio.ensure_future(
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
            return ErrorResponse(
                message=f"Cannot cancel batch with status '{batch.status}'",
                type="invalid_request_error",
                code=400,
            )

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
                    # Periodically save progress
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
            # Force non-streaming and set low priority for batch
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
                    error=ErrorResponse(
                        message=f"No handler for {request.url}",
                        type="invalid_request_error",
                        code=400,
                    ),
                )

            # Call handler with request body only (raw_request=None,
            # same pattern as run_batch.py)
            response = await handler_fn(request.body)

            if isinstance(response, (ChatCompletionResponse,
                                     EmbeddingResponse, ScoreResponse)):
                return BatchRequestOutput(
                    id=f"vllm-{random_uuid()}",
                    custom_id=request.custom_id,
                    response=BatchResponseData(
                        body=response,
                        request_id=f"vllm-batch-{random_uuid()}"),
                    error=None,
                )
            elif isinstance(response, ErrorResponse):
                return BatchRequestOutput(
                    id=f"vllm-{random_uuid()}",
                    custom_id=request.custom_id,
                    response=BatchResponseData(
                        status_code=response.code,
                        request_id=f"vllm-batch-{random_uuid()}"),
                    error=response,
                )
            else:
                return BatchRequestOutput(
                    id=f"vllm-{random_uuid()}",
                    custom_id=request.custom_id,
                    response=None,
                    error=ErrorResponse(
                        message="Unexpected response type",
                        type="server_error",
                        code=500,
                    ),
                )
        except Exception as e:
            return BatchRequestOutput(
                id=f"vllm-{random_uuid()}",
                custom_id=request.custom_id,
                response=None,
                error=ErrorResponse(
                    message=str(e),
                    type="server_error",
                    code=500,
                ),
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
            # Delete associated files
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

        for batch_id, cancel_event in list(self._cancel_events.items()):
            cancel_event.set()

        for task in list(self._tasks.values()):
            try:
                await asyncio.wait_for(task, timeout=30)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

        # Mark any remaining in-progress batches
        now = int(time.time())
        for batch in self._batches.values():
            if batch.status in ("validating", "in_progress", "cancelling"):
                batch.status = "cancelled"
                batch.cancelled_at = now

        async with self._lock:
            await self._save_metadata()

        logger.info("Batch serving shut down")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/entrypoints/openai/test_serving_batches.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/entrypoints/openai/serving_batches.py tests/entrypoints/openai/test_serving_batches.py
git commit -m "feat: add OpenAIServingBatches for batch lifecycle management"
```

---

## Chunk 4: API Server Integration

### Task 5: Add Routes and Dependencies to `api_server.py`

**Files:**
- Modify: `vllm/entrypoints/openai/api_server.py`

- [ ] **Step 1: Add imports**

Add to the imports section of `api_server.py` (around line 70, with other serving imports):

```python
from vllm.entrypoints.openai.serving_files import OpenAIServingFiles
from vllm.entrypoints.openai.serving_batches import OpenAIServingBatches
```

Add to protocol imports (around line 50):

```python
from vllm.entrypoints.openai.protocol import (
    # ... existing imports ...
    BatchObject,
    BatchListResponse,
    FileObject,
    FileListResponse,
)
```

- [ ] **Step 2: Add dependency functions**

Add after the existing dependency functions (around line 387):

```python
def files(request: Request) -> Optional[OpenAIServingFiles]:
    return request.app.state.openai_serving_files

def batches(request: Request) -> Optional[OpenAIServingBatches]:
    return request.app.state.openai_serving_batches
```

- [ ] **Step 3: Add Files API routes**

Add new route handlers after the existing endpoints:

```python
@router.post("/v1/files")
async def upload_file(raw_request: Request):
    handler = files(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="Batch API is not enabled")

    form = await raw_request.form()
    file_field = form.get("file")
    purpose = form.get("purpose", "batch")

    if file_field is None:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "No file provided",
                               "type": "invalid_request_error"}})

    content = await file_field.read()
    result = await handler.upload_file(
        content=content,
        filename=file_field.filename or "upload.jsonl",
        purpose=purpose,
    )
    return JSONResponse(content=result.model_dump())


@router.get("/v1/files")
async def list_files(raw_request: Request, purpose: Optional[str] = None):
    handler = files(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="Batch API is not enabled")
    file_list = await handler.list_files(purpose=purpose)
    response = FileListResponse(data=file_list)
    return JSONResponse(content=response.model_dump())


@router.get("/v1/files/{file_id}")
async def get_file(file_id: str, raw_request: Request):
    handler = files(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="Batch API is not enabled")
    result = await handler.get_file(file_id)
    if result is None:
        return JSONResponse(status_code=404,
                            content={"error": {"message": "File not found",
                                               "type": "invalid_request_error"}})
    return JSONResponse(content=result.model_dump())


@router.delete("/v1/files/{file_id}")
async def delete_file(file_id: str, raw_request: Request):
    handler = files(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="Batch API is not enabled")
    # Check if file is tied to an active batch
    batch_handler = batches(raw_request)
    if batch_handler and batch_handler.is_file_in_active_batch(file_id):
        return JSONResponse(
            status_code=409,
            content={"error": {
                "message": "File is referenced by an active batch",
                "type": "invalid_request_error"}})
    success = await handler.delete_file(file_id)
    if not success:
        return JSONResponse(status_code=404,
                            content={"error": {"message": "File not found",
                                               "type": "invalid_request_error"}})
    return JSONResponse(content={"id": file_id, "object": "file",
                                 "deleted": True})


@router.get("/v1/files/{file_id}/content")
async def get_file_content(file_id: str, raw_request: Request):
    handler = files(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="Batch API is not enabled")
    content = await handler.get_file_content(file_id)
    if content is None:
        return JSONResponse(status_code=404,
                            content={"error": {"message": "File not found",
                                               "type": "invalid_request_error"}})
    from starlette.responses import Response
    return Response(content=content, media_type="application/octet-stream")
```

- [ ] **Step 4: Add Batches API routes**

```python
@router.post("/v1/batches",
             dependencies=[Depends(validate_json_request)])
async def create_batch(raw_request: Request):
    handler = batches(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="Batch API is not enabled")
    data = await raw_request.json()
    result = await handler.create_batch(
        input_file_id=data.get("input_file_id", ""),
        endpoint=data.get("endpoint", ""),
        completion_window=data.get("completion_window", "24h"),
        metadata=data.get("metadata"),
    )
    if isinstance(result, ErrorResponse):
        return JSONResponse(status_code=result.code,
                            content=result.model_dump())
    return JSONResponse(content=result.model_dump())


@router.get("/v1/batches")
async def list_batches(raw_request: Request,
                       after: Optional[str] = None,
                       limit: int = 20):
    handler = batches(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="Batch API is not enabled")
    batch_list, has_more = await handler.list_batches(
        limit=limit, after=after)
    response = BatchListResponse(
        data=batch_list,
        has_more=has_more,
        first_id=batch_list[0].id if batch_list else None,
        last_id=batch_list[-1].id if batch_list else None,
    )
    return JSONResponse(content=response.model_dump())


@router.get("/v1/batches/{batch_id}")
async def get_batch(batch_id: str, raw_request: Request):
    handler = batches(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="Batch API is not enabled")
    result = await handler.get_batch(batch_id)
    if result is None:
        return JSONResponse(status_code=404,
                            content={"error": {"message": "Batch not found",
                                               "type": "invalid_request_error"}})
    return JSONResponse(content=result.model_dump())


@router.post("/v1/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str, raw_request: Request):
    handler = batches(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="Batch API is not enabled")
    result = await handler.cancel_batch(batch_id)
    if result is None:
        return JSONResponse(status_code=404,
                            content={"error": {"message": "Batch not found",
                                               "type": "invalid_request_error"}})
    if isinstance(result, ErrorResponse):
        return JSONResponse(status_code=result.code,
                            content=result.model_dump())
    return JSONResponse(content=result.model_dump())
```

- [ ] **Step 5: Initialize serving classes in `init_app_state`**

In `init_app_state()` (around line 995), add after all existing serving class initialization:

```python
    # Batch API serving classes
    serving_files_instance = OpenAIServingFiles(
        storage_dir=args.batch_storage_dir,
    )
    state.openai_serving_files = serving_files_instance
    state.openai_serving_batches = OpenAIServingBatches(
        storage_dir=args.batch_storage_dir,
        serving_files=serving_files_instance,
        serving_chat=state.openai_serving_chat,
        serving_embedding=state.openai_serving_embedding,
        serving_score=state.openai_serving_scores,
        batch_priority=args.batch_priority,
        retention_hours=args.batch_retention_hours,
    )
```

- [ ] **Step 6: Add shutdown hook in lifespan**

In the lifespan `finally` block (around line 133), add before `del app.state`:

```python
        if hasattr(state, 'openai_serving_batches') and state.openai_serving_batches:
            await state.openai_serving_batches.shutdown()
```

- [ ] **Step 7: Commit**

```bash
git add vllm/entrypoints/openai/api_server.py
git commit -m "feat: add Files and Batches API routes to API server"
```

---

## Chunk 5: Integration Tests

### Task 6: Write Integration Tests

**Files:**
- Create: `tests/entrypoints/openai/test_batch_api_integration.py`

- [ ] **Step 1: Write integration test for the full batch workflow**

Create `tests/entrypoints/openai/test_batch_api_integration.py`:

```python
"""Integration tests for the Online Batch API.

These tests start a vLLM server and exercise the full batch workflow
via HTTP: upload file -> create batch -> poll status -> download results.

Requires a model to be available. Uses a small model for speed.
"""
import json
import time

import openai
import pytest
import requests

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "facebook/opt-125m"


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype", "float16",
        "--max-model-len", "256",
        "--batch-storage-dir", "/tmp/vllm-test-batches",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def base_url(server):
    return server.url_for("")


def test_full_batch_workflow(base_url):
    """Test: upload -> create batch -> poll -> get results."""

    # 1. Upload input file
    batch_input = "\n".join([
        json.dumps({
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": f"Say hello {i}"}],
                "max_tokens": 10,
            },
        })
        for i in range(3)
    ])

    upload_resp = requests.post(
        f"{base_url}/v1/files",
        files={"file": ("batch.jsonl", batch_input.encode())},
        data={"purpose": "batch"},
    )
    assert upload_resp.status_code == 200
    file_obj = upload_resp.json()
    assert file_obj["id"].startswith("file-")

    # 2. Create batch
    create_resp = requests.post(
        f"{base_url}/v1/batches",
        json={
            "input_file_id": file_obj["id"],
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    assert create_resp.status_code == 200
    batch_obj = create_resp.json()
    assert batch_obj["id"].startswith("batch-")
    assert batch_obj["status"] in ("validating", "in_progress")

    # 3. Poll until complete
    batch_id = batch_obj["id"]
    for _ in range(60):
        status_resp = requests.get(f"{base_url}/v1/batches/{batch_id}")
        assert status_resp.status_code == 200
        batch_obj = status_resp.json()
        if batch_obj["status"] in ("completed", "failed"):
            break
        time.sleep(1)

    assert batch_obj["status"] == "completed"
    assert batch_obj["request_counts"]["completed"] == 3
    assert batch_obj["request_counts"]["failed"] == 0
    assert batch_obj["output_file_id"] is not None

    # 4. Download results
    output_resp = requests.get(
        f"{base_url}/v1/files/{batch_obj['output_file_id']}/content")
    assert output_resp.status_code == 200
    lines = output_resp.text.strip().split("\n")
    assert len(lines) == 3

    for line in lines:
        output = json.loads(line)
        assert output["response"]["status_code"] == 200
        assert output["response"]["body"]["choices"][0]["message"]["content"]


def test_list_batches(base_url):
    resp = requests.get(f"{base_url}/v1/batches")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


def test_list_files(base_url):
    resp = requests.get(f"{base_url}/v1/files")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"


def test_get_nonexistent_batch(base_url):
    resp = requests.get(f"{base_url}/v1/batches/batch-nonexistent")
    assert resp.status_code == 404


def test_get_nonexistent_file(base_url):
    resp = requests.get(f"{base_url}/v1/files/file-nonexistent")
    assert resp.status_code == 404


def test_create_batch_invalid_file(base_url):
    resp = requests.post(
        f"{base_url}/v1/batches",
        json={
            "input_file_id": "file-nonexistent",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    assert resp.status_code == 404


def test_create_batch_invalid_endpoint(base_url):
    # First upload a file
    upload_resp = requests.post(
        f"{base_url}/v1/files",
        files={"file": ("batch.jsonl", b'{"custom_id":"r1","method":"POST","url":"/v1/chat/completions","body":{"model":"x","messages":[]}}\n')},
        data={"purpose": "batch"},
    )
    file_id = upload_resp.json()["id"]

    resp = requests.post(
        f"{base_url}/v1/batches",
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/invalid",
            "completion_window": "24h",
        },
    )
    assert resp.status_code == 400


def test_delete_file(base_url):
    upload_resp = requests.post(
        f"{base_url}/v1/files",
        files={"file": ("delete_me.jsonl", b"test\n")},
        data={"purpose": "batch"},
    )
    file_id = upload_resp.json()["id"]

    del_resp = requests.delete(f"{base_url}/v1/files/{file_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # Verify it's gone
    get_resp = requests.get(f"{base_url}/v1/files/{file_id}")
    assert get_resp.status_code == 404
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/entrypoints/openai/test_batch_api_integration.py -v --timeout=120`
Expected: All tests PASS. Note: this requires GPU and model download, so may be slow on first run.

- [ ] **Step 3: Commit**

```bash
git add tests/entrypoints/openai/test_batch_api_integration.py
git commit -m "test: add integration tests for Online Batch API"
```

### Task 7: Final Verification

- [ ] **Step 1: Run all batch-related tests together**

Run: `python -m pytest tests/entrypoints/openai/test_batch_protocol.py tests/entrypoints/openai/test_serving_files.py tests/entrypoints/openai/test_serving_batches.py -v`
Expected: All unit tests PASS.

- [ ] **Step 2: Run existing batch tests to verify no regressions**

Run: `python -m pytest tests/entrypoints/openai/test_run_batch.py -v`
Expected: All existing tests still PASS.

- [ ] **Step 3: Final commit with all files**

```bash
git add -A
git status
git commit -m "feat: Online Batch API - OpenAI-compatible Files and Batches endpoints

Adds /v1/files and /v1/batches endpoints to vLLM's API server, enabling
HTTP-based batch processing. Batch requests run at lower priority than
online requests, using the existing engine's scheduler.

Key features:
- Full OpenAI Batch API compatibility (Files + Batches CRUD)
- Background asyncio task processing with bounded concurrency
- Local filesystem persistence with crash recovery
- Auto-cleanup of expired batches (configurable TTL)
- Priority-based scheduling (batch requests yield to online traffic)"
```
