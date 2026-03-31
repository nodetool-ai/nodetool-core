def get_content_optimized(
    paths: list[str],
    extensions: list[str] | None = None,
):
    from src.nodetool.io.get_files import get_files
    extensions = extensions or [".py", ".js", ".ts", ".jsx", ".tsx", ".md"]
    content_parts = []
    for path in paths:
        for file in get_files(path, extensions):
            content_parts.append("\n\n")
            content_parts.append(f"## {file}\n\n")
            with open(file, encoding="utf-8") as f:
                content_parts.append(f.read())
    return "".join(content_parts)

from src.nodetool.io.get_files import get_content
old = get_content(["src/nodetool/io/get_files.py"])
new = get_content_optimized(["src/nodetool/io/get_files.py"])

print("Same result:", old == new)
