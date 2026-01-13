import asyncio
import os

import aiofiles


def get_files(path: str, extensions: list[str] | None = None):
    """
    Recursively retrieves all files with specified extensions in the given path.

    Args:
        path (str): The path to search for files.
        extensions (list[str], optional): List of file extensions to include.

    Returns:
        list[str]: A list of file paths matching the specified extensions.
    """
    extensions = extensions or [".py", ".js", ".ts", ".jsx", ".tsx", ".md"]
    ext = os.path.splitext(path)[1]
    if os.path.isfile(path) and ext in extensions:
        return [path]
    files = []
    if os.path.isdir(path):
        for file in os.listdir(path):
            files += get_files(os.path.join(path, file), extensions)
    return files


async def get_files_async(path: str, extensions: list[str] | None = None) -> list[str]:
    """
    Recursively retrieves all files with specified extensions in the given path (async version).

    Args:
        path (str): The path to search for files.
        extensions (list[str], optional): List of file extensions to include.

    Returns:
        list[str]: A list of file paths matching the specified extensions.
    """
    # Offload the blocking get_files operation to a thread
    return await asyncio.to_thread(get_files, path, extensions)


def get_content(
    paths: list[str],
    extensions: list[str] | None = None,
):
    """
    Retrieves the content of files with specified extensions in the given paths.

    Args:
        paths (list[str]): A list of paths to search for files.
        extensions (list[str], optional): A list of file extensions to include. Defaults to [".py"].

    Returns:
        str: The concatenated content of all the files found.
    """
    extensions = extensions or [".py", ".js", ".ts", ".jsx", ".tsx", ".md"]
    content = ""
    for path in paths:
        for file in get_files(path, extensions):
            content += "\n\n"
            content += f"## {file}\n\n"
            with open(file, encoding="utf-8") as f:
                content += f.read()
    return content


async def get_content_async(
    paths: list[str],
    extensions: list[str] | None = None,
) -> str:
    """
    Retrieves the content of files with specified extensions in the given paths (async version).

    Args:
        paths (list[str]): A list of paths to search for files.
        extensions (list[str], optional): A list of file extensions to include. Defaults to [".py"].

    Returns:
        str: The concatenated content of all the files found.
    """
    extensions = extensions or [".py", ".js", ".ts", ".jsx", ".tsx", ".md"]
    content = ""
    for path in paths:
        files = await get_files_async(path, extensions)
        for file in files:
            content += "\n\n"
            content += f"## {file}\n\n"
            async with aiofiles.open(file, encoding="utf-8") as f:
                content += await f.read()
    return content
