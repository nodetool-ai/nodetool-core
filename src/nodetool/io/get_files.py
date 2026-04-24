import os


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
    # ⚡ Bolt Optimization: Use a set for O(1) extension lookups
    ext_set = set(extensions)

    if os.path.isfile(path):
        ext = os.path.splitext(path)[1]
        if ext in ext_set:
            return [path]
        return []

    files = []
    if os.path.isdir(path):
        # ⚡ Bolt Optimization: Use os.walk instead of custom recursion with os.listdir + os.path.isdir.
        # os.walk uses os.scandir internally which avoids the performance overhead of repeated stat system calls.
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                ext = os.path.splitext(filename)[1]
                if ext in ext_set:
                    files.append(os.path.join(root, filename))
    return files


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
