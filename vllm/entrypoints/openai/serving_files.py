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

    async def delete_file(self, file_id: str) -> bool:
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
