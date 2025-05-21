"""Helpers for retrieving information about individual Hugging Face files."""

from pydantic import BaseModel
from huggingface_hub import HfFileSystem


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
