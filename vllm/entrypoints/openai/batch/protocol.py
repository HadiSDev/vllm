# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol definitions for the Online Batch API."""
from typing import Any, Optional, Union

from pydantic import TypeAdapter, field_validator
from pydantic_core import ValidationInfo

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel


class BatchRequestInput(OpenAIBaseModel):
    """
    The per-line object of the batch input file.
    """

    # A developer-provided per-request id that will be used to match outputs to
    # inputs. Must be unique for each request in a batch.
    custom_id: str

    # The HTTP method to be used for the request. Currently only POST is
    # supported.
    method: str

    # The OpenAI API relative URL to be used for the request.
    url: str

    # The parameters of the request.
    body: Any

    @field_validator('body', mode='plain')
    @classmethod
    def check_type_for_url(cls, value: Any, info: ValidationInfo):
        url = info.data['url']
        if url == "/v1/chat/completions":
            from vllm.entrypoints.openai.chat_completion.protocol import (
                ChatCompletionRequest,
            )
            return ChatCompletionRequest.model_validate(value)
        if url == "/v1/embeddings":
            from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest
            return TypeAdapter(EmbeddingRequest).validate_python(value)
        if url == "/v1/score":
            from vllm.entrypoints.pooling.scoring.protocol import ScoreRequest
            return ScoreRequest.model_validate(value)
        return value


class BatchResponseData(OpenAIBaseModel):
    # HTTP status code of the response.
    status_code: int = 200

    # An unique identifier for the API request.
    request_id: str

    # The body of the response.
    body: Optional[Any] = None


class BatchRequestOutput(OpenAIBaseModel):
    """
    The per-line object of the batch output and error files
    """

    id: str

    # A developer-provided per-request id that will be used to match outputs to
    # inputs.
    custom_id: str

    response: Optional[BatchResponseData]

    # For requests that failed with a non-HTTP error, this will contain more
    # information on the cause of the failure.
    error: Optional[Any]


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
    status: str
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
