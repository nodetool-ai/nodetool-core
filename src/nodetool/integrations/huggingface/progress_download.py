# progress_download.py

import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import huggingface_hub.file_download as _fd
from huggingface_hub.utils import build_hf_headers, validate_hf_hub_args

from huggingface_hub import constants

logger = _fd.logger

ProgressCallback = Callable[[int, Optional[int]], None]
import threading

from tqdm.auto import tqdm

_thread_local = threading.local()


class _CallbackTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        # Force disable display so it doesn't print to stderr
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)

        # Get callback from thread local storage
        self._progress_callback = getattr(_thread_local, "progress_callback", None)

        # Report initial progress if resuming
        if self.n > 0 and self._progress_callback is not None:
            self._progress_callback(self.n, self.total)

    def update(self, n: int | float = 1):
        # Update internal state
        super().update(n)
        # Forward to callback
        if n and self._progress_callback is not None:
            self._progress_callback(int(n), self.total)


def _download_to_tmp_and_move_with_progress(
    incomplete_path: Path,
    destination_path: Path,
    url_to_download: str,
    proxies: Optional[Dict],
    headers: Dict[str, str],
    expected_size: Optional[int],
    filename: str,
    force_download: bool,
    etag: Optional[str],
    xet_file_data,
    progress_callback: Optional[ProgressCallback],
) -> None:
    """
    Variant of `huggingface_hub.file_download._download_to_tmp_and_move` that
    reports byte deltas through `progress_callback`.
    """

    # Fallback to upstream implementation if no callback is requested.
    if progress_callback is None:
        return _fd._download_to_tmp_and_move(
            incomplete_path=incomplete_path,
            destination_path=destination_path,
            url_to_download=url_to_download,
            proxies=proxies,
            headers=headers,
            expected_size=expected_size,
            filename=filename,
            force_download=force_download,
            etag=etag,
            xet_file_data=xet_file_data,
        )

    if destination_path.exists() and not force_download:
        # Do nothing if already exists (except if force_download=True)
        return

    if incomplete_path.exists() and (force_download or (constants.HF_HUB_ENABLE_HF_TRANSFER and not proxies)):
        message = f"Removing incomplete file '{incomplete_path}'"
        if force_download:
            message += " (force_download=True)"
        elif constants.HF_HUB_ENABLE_HF_TRANSFER and not proxies:
            message += " (hf_transfer=True)"
        logger.info(message)
        incomplete_path.unlink(missing_ok=True)

    with incomplete_path.open("ab") as f:
        resume_size = f.tell()
        message = f"Downloading '{filename}' to '{incomplete_path}'"
        if resume_size > 0 and expected_size is not None:
            message += f" (resume from {resume_size}/{expected_size})"
        logger.info(message)

        if expected_size is not None:
            _fd._check_disk_space(expected_size, incomplete_path.parent)
            _fd._check_disk_space(expected_size, destination_path.parent)

        # Set callback in thread local storage
        _thread_local.progress_callback = progress_callback
        try:
            if xet_file_data is not None and _fd.is_xet_available():
                logger.debug("Xet storage is enabled for this repo. Downloading file from Xet storage.")
                _fd.xet_get(
                    incomplete_path=incomplete_path,
                    xet_file_data=xet_file_data,
                    headers=headers,
                    expected_size=expected_size,
                    displayed_filename=filename,
                    _tqdm_bar=_CallbackTqdm,
                )
            else:
                if xet_file_data is not None and not constants.HF_HUB_DISABLE_XET:
                    logger.warning(
                        "Xet storage is enabled for this repo, but the 'hf_xet' package is not installed. "
                        "Falling back to regular HTTP download. "
                        "For better performance, install the package with: "
                        "`pip install huggingface_hub[hf_xet]` or `pip install hf_xet`"
                    )

                _fd.http_get(
                    url=url_to_download,
                    temp_file=f,
                    proxies=proxies,
                    resume_size=resume_size,
                    headers=headers,
                    expected_size=expected_size,
                    _tqdm_bar=_CallbackTqdm,
                )
        finally:
            # Clean up thread local storage
            if hasattr(_thread_local, "progress_callback"):
                del _thread_local.progress_callback

    logger.info(f"Download complete. Moving file to {destination_path}")
    _fd._chmod_and_move(incomplete_path, destination_path)


def _hf_hub_download_to_cache_dir_with_progress(
    *,
    cache_dir: str,
    repo_id: str,
    filename: str,
    repo_type: str,
    revision: str,
    endpoint: Optional[str],
    etag_timeout: float,
    headers: Dict[str, str],
    proxies: Optional[Dict],
    token: Optional[bool | str],
    local_files_only: bool,
    force_download: bool,
    progress_callback: Optional[ProgressCallback],
) -> str:
    """
    Copy of `_hf_hub_download_to_cache_dir` that routes the actual transfer
    through `_download_to_tmp_and_move_with_progress`.
    """
    # Fast path: no progress requested, use upstream implementation as is.
    if progress_callback is None:
        return _fd._hf_hub_download_to_cache_dir(
            cache_dir=cache_dir,
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            endpoint=endpoint,
            etag_timeout=etag_timeout,
            headers=headers,
            proxies=proxies,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
        )

    locks_dir = os.path.join(cache_dir, ".locks")
    storage_folder = os.path.join(cache_dir, _fd.repo_folder_name(repo_id=repo_id, repo_type=repo_type))

    relative_filename = os.path.join(*filename.split("/"))
    if os.name == "nt":
        if relative_filename.startswith("..\\") or "\\..\\" in relative_filename:
            raise ValueError(
                f"Invalid filename: cannot handle filename '{relative_filename}' on Windows. "
                "Ask the repository owner to rename this file."
            )

    if _fd.REGEX_COMMIT_HASH.match(revision):
        pointer_path = _fd._get_pointer_path(storage_folder, revision, relative_filename)
        if os.path.exists(pointer_path) and not force_download:
            return pointer_path

    (
        url_to_download,
        etag,
        commit_hash,
        expected_size,
        xet_file_data,
        head_call_error,
    ) = _fd._get_metadata_or_catch_error(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        endpoint=endpoint,
        proxies=proxies,
        etag_timeout=etag_timeout,
        headers=headers,
        token=token,
        local_files_only=local_files_only,
        storage_folder=storage_folder,
        relative_filename=relative_filename,
    )

    if head_call_error is not None:
        if not force_download:
            commit_hash = None
            if _fd.REGEX_COMMIT_HASH.match(revision):
                commit_hash = revision
            else:
                ref_path = os.path.join(storage_folder, "refs", revision)
                if os.path.isfile(ref_path):
                    with open(ref_path) as f:
                        commit_hash = f.read()
            if commit_hash is not None:
                pointer_path = _fd._get_pointer_path(storage_folder, commit_hash, relative_filename)
                if os.path.exists(pointer_path) and not force_download:
                    return pointer_path

        _fd._raise_on_head_call_error(head_call_error, force_download, local_files_only)

    assert etag is not None
    assert commit_hash is not None
    assert url_to_download is not None
    assert expected_size is not None

    blob_path = os.path.join(storage_folder, "blobs", etag)
    pointer_path = _fd._get_pointer_path(storage_folder, commit_hash, relative_filename)

    os.makedirs(os.path.dirname(blob_path), exist_ok=True)
    os.makedirs(os.path.dirname(pointer_path), exist_ok=True)

    _fd._cache_commit_hash_for_specific_revision(storage_folder, revision, commit_hash)

    lock_path = os.path.join(
        locks_dir,
        _fd.repo_folder_name(repo_id=repo_id, repo_type=repo_type),
        f"{etag}.lock",
    )

    if (
        os.name == "nt"
        and len(os.path.abspath(lock_path)) > 255
        and not os.path.abspath(lock_path).startswith("\\\\?\\")
    ):
        lock_path = "\\\\?\\" + os.path.abspath(lock_path)

    if (
        os.name == "nt"
        and len(os.path.abspath(blob_path)) > 255
        and not os.path.abspath(blob_path).startswith("\\\\?\\")
    ):
        blob_path = "\\\\?\\" + os.path.abspath(blob_path)

    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)

    if not force_download and os.path.exists(pointer_path):
        return pointer_path

    if not force_download and os.path.exists(blob_path):
        with _fd.WeakFileLock(lock_path):
            if not os.path.exists(pointer_path):
                _fd._create_symlink(blob_path, pointer_path, new_blob=False)
            return pointer_path

    with _fd.WeakFileLock(lock_path):
        _download_to_tmp_and_move_with_progress(
            incomplete_path=Path(blob_path + ".incomplete"),
            destination_path=Path(blob_path),
            url_to_download=url_to_download,
            proxies=proxies,
            headers=headers,
            expected_size=expected_size,
            filename=filename,
            force_download=force_download,
            etag=etag,
            xet_file_data=xet_file_data,
            progress_callback=progress_callback,
        )
        if not os.path.exists(pointer_path):
            _fd._create_symlink(blob_path, pointer_path, new_blob=True)

    return pointer_path


def _hf_hub_download_to_local_dir_with_progress(
    *,
    local_dir: str | Path,
    repo_id: str,
    repo_type: str,
    filename: str,
    revision: str,
    endpoint: Optional[str],
    etag_timeout: float,
    headers: Dict[str, str],
    proxies: Optional[Dict],
    token: bool | str | None,
    cache_dir: str,
    force_download: bool,
    local_files_only: bool,
    progress_callback: Optional[ProgressCallback],
) -> str:
    """
    Copy of `_hf_hub_download_to_local_dir` that routes transfers through
    `_download_to_tmp_and_move_with_progress`.
    """
    if progress_callback is None:
        return _fd._hf_hub_download_to_local_dir(
            local_dir=local_dir,
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            revision=revision,
            endpoint=endpoint,
            etag_timeout=etag_timeout,
            headers=headers,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
        )

    if os.name == "nt" and len(os.path.abspath(local_dir)) > 255:
        local_dir = "\\\\?\\" + os.path.abspath(local_dir)
    local_dir = Path(local_dir)
    paths = _fd.get_local_download_paths(local_dir=local_dir, filename=filename)
    local_metadata = _fd.read_download_metadata(local_dir=local_dir, filename=filename)

    if (
        not force_download
        and _fd.REGEX_COMMIT_HASH.match(revision)
        and paths.file_path.is_file()
        and local_metadata is not None
        and local_metadata.commit_hash == revision
    ):
        return str(paths.file_path)

    (
        url_to_download,
        etag,
        commit_hash,
        expected_size,
        xet_file_data,
        head_call_error,
    ) = _fd._get_metadata_or_catch_error(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        endpoint=endpoint,
        proxies=proxies,
        etag_timeout=etag_timeout,
        headers=headers,
        token=token,
        local_files_only=local_files_only,
    )

    if head_call_error is not None:
        if not force_download and paths.file_path.is_file():
            logger.warning(
                "Couldn't access the Hub to check for update but local file already exists. "
                f"Defaulting to existing file. (error: {head_call_error})"
            )
            return str(paths.file_path)
        _fd._raise_on_head_call_error(head_call_error, force_download, local_files_only)

    assert etag is not None
    assert commit_hash is not None
    assert url_to_download is not None
    assert expected_size is not None

    if not force_download and paths.file_path.is_file():
        if local_metadata is not None and local_metadata.etag == etag:
            _fd.write_download_metadata(local_dir=local_dir, filename=filename, commit_hash=commit_hash, etag=etag)
            return str(paths.file_path)

        if local_metadata is None and _fd.REGEX_SHA256.match(etag) is not None:
            with open(paths.file_path, "rb") as f:
                file_hash = _fd.sha_fileobj(f).hex()
            if file_hash == etag:
                _fd.write_download_metadata(local_dir=local_dir, filename=filename, commit_hash=commit_hash, etag=etag)
                return str(paths.file_path)

    if not force_download:
        cached_path = _fd.try_to_load_from_cache(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            revision=commit_hash,
            repo_type=repo_type,
        )
        if isinstance(cached_path, str):
            with _fd.WeakFileLock(paths.lock_path):
                paths.file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached_path, paths.file_path)
            _fd.write_download_metadata(local_dir=local_dir, filename=filename, commit_hash=commit_hash, etag=etag)
            return str(paths.file_path)

    with _fd.WeakFileLock(paths.lock_path):
        paths.file_path.unlink(missing_ok=True)
        _download_to_tmp_and_move_with_progress(
            incomplete_path=paths.incomplete_path(etag),
            destination_path=paths.file_path,
            url_to_download=url_to_download,
            proxies=proxies,
            headers=headers,
            expected_size=expected_size,
            filename=filename,
            force_download=force_download,
            etag=etag,
            xet_file_data=xet_file_data,
            progress_callback=progress_callback,
        )

    _fd.write_download_metadata(local_dir=local_dir, filename=filename, commit_hash=commit_hash, etag=etag)
    return str(paths.file_path)


@validate_hf_hub_args
def hf_hub_download_with_progress(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    user_agent: Dict | str | None = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
    token: bool | str | None = None,
    local_files_only: bool = False,
    headers: Optional[Dict[str, str]] = None,
    endpoint: Optional[str] = None,
    resume_download: Optional[bool] = None,
    force_filename: Optional[str] = None,
    local_dir_use_symlinks: bool | str = "auto",
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """
    Replacement for `huggingface_hub.hf_hub_download` that exposes
    `progress_callback(delta_bytes, total_bytes)`.

    If `progress_callback` is None, it simply forwards to the upstream function.
    """
    from huggingface_hub import hf_hub_download as _orig_hf_hub_download

    if progress_callback is None:
        return _orig_hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            repo_type=repo_type,
            revision=revision,
            library_name=library_name,
            library_version=library_version,
            cache_dir=cache_dir,
            local_dir=local_dir,
            user_agent=user_agent,
            force_download=force_download,
            proxies=proxies,
            etag_timeout=etag_timeout,
            token=token,
            local_files_only=local_files_only,
            headers=headers,
            endpoint=endpoint,
            resume_download=resume_download,
            force_filename=force_filename,
            local_dir_use_symlinks=local_dir_use_symlinks,
        )

    if constants.HF_HUB_ETAG_TIMEOUT != constants.DEFAULT_ETAG_TIMEOUT:
        etag_timeout = constants.HF_HUB_ETAG_TIMEOUT

    if force_filename is not None:
        import warnings

        warnings.warn(
            "The `force_filename` parameter is deprecated as a new caching system, "
            "which keeps the filenames as they are on the Hub, is now in place.",
            FutureWarning,
            stacklevel=2,
        )
    if resume_download is not None:
        import warnings

        warnings.warn(
            "`resume_download` is deprecated and will be removed in version 1.0.0. "
            "Downloads always resume when possible. "
            "If you want to force a new download, use `force_download=True`.",
            FutureWarning,
            stacklevel=2,
        )

    if cache_dir is None:
        cache_dir = constants.HF_HUB_CACHE
    if revision is None:
        revision = constants.DEFAULT_REVISION
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if isinstance(local_dir, Path):
        local_dir = str(local_dir)

    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if repo_type is None:
        repo_type = "model"
    if repo_type not in constants.REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}. Accepted repo types are: {str(constants.REPO_TYPES)}")

    hf_headers = build_hf_headers(
        token=token,
        library_name=library_name,
        library_version=library_version,
        user_agent=user_agent,
        headers=headers,
    )

    if local_dir is not None:
        if local_dir_use_symlinks != "auto":
            import warnings

            warnings.warn(
                "`local_dir_use_symlinks` parameter is deprecated and will be ignored. "
                "The process to download files to a local folder has been updated and does "
                "not rely on symlinks anymore.",
                stacklevel=2,
            )

        return _hf_hub_download_to_local_dir_with_progress(
            local_dir=local_dir,
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            revision=revision,
            endpoint=endpoint,
            etag_timeout=etag_timeout,
            headers=hf_headers,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            progress_callback=progress_callback,
        )
    else:
        return _hf_hub_download_to_cache_dir_with_progress(
            cache_dir=cache_dir,
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            endpoint=endpoint,
            etag_timeout=etag_timeout,
            headers=hf_headers,
            proxies=proxies,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            progress_callback=progress_callback,
        )
