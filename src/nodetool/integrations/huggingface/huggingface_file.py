"""Helpers for retrieving information about individual Hugging Face files."""

import asyncio

from huggingface_hub import HfFileSystem
from pydantic import BaseModel


class HFFileInfo(BaseModel):
    size: int
    repo_id: str
    path: str


class HFFileRequest(BaseModel):
    repo_id: str
    path: str


def get_huggingface_file_infos(requests: list[HFFileRequest]) -> list[HFFileInfo]:
    """Return file metadata for a list of ``repo_id``/``path`` pairs.

    Parameters
    ----------
    requests:
        A list of :class:`HFFileRequest` describing the files to query.

    Returns
    -------
    list[HFFileInfo]
        Metadata for each requested file, including its size in bytes.
    """

    fs = HfFileSystem()
    file_infos = []

    for request in requests:
        file_info = fs.info(f"{request.repo_id}/{request.path}")
        file_infos.append(
            HFFileInfo(
                size=file_info["size"],
                repo_id=request.repo_id,
                path=request.path,
            )
        )

    return file_infos


async def get_huggingface_file_infos_async(
    requests: list[HFFileRequest],
) -> list[HFFileInfo]:
    """Async wrapper that retrieves file infos without blocking the event loop.

    Uses ``asyncio.to_thread`` to call the synchronous HfFileSystem.info for each
    request, running them concurrently.

    Parameters
    ----------
    requests:
        A list of :class:`HFFileRequest` describing the files to query.

    Returns
    -------
    list[HFFileInfo]
        Metadata for each requested file, including its size in bytes.
    """

    fs = HfFileSystem()

    async def fetch(req: HFFileRequest) -> HFFileInfo:
        info = await asyncio.to_thread(fs.info, f"{req.repo_id}/{req.path}")
        return HFFileInfo(size=info["size"], repo_id=req.repo_id, path=req.path)

    results = await asyncio.gather(*(fetch(r) for r in requests))
    return list(results)
