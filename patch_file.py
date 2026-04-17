import re

with open("src/nodetool/api/file.py", "r") as f:
    content = f.read()

# Fix 1: Add check in list_files
list_files_search = """        # Validate and normalize path
        abs_path = path
        exists = await asyncio.to_thread(os.path.exists, abs_path)
        if not exists:"""
list_files_replace = """        # Validate and normalize path
        abs_path = path
        if not _is_safe_path(abs_path):
            raise HTTPException(status_code=403, detail="Access to this path is forbidden")

        exists = await asyncio.to_thread(os.path.exists, abs_path)
        if not exists:"""
content = content.replace(list_files_search, list_files_replace)

# Fix 2: Add check in get_file (info)
get_file_search = """    try:
        exists = await asyncio.to_thread(os.path.exists, path)"""
get_file_replace = """    try:
        if not _is_safe_path(path):
            raise HTTPException(status_code=403, detail="Access to this path is forbidden")
        exists = await asyncio.to_thread(os.path.exists, path)"""
content = content.replace(get_file_search, get_file_replace)

# Fix 3: Fix the whitelist bypass
is_safe_search = """            # Check for hidden files or directories (starting with .)
            parts = p.split(os.sep)
            if any(part.startswith(".") for part in parts if part):
                return False

        return True"""
is_safe_replace = """            # Check for hidden files or directories (starting with .)
            parts = p.split(os.sep)
            if any(part.startswith(".") for part in parts if part):
                return False

        return is_in_safe_root"""
content = content.replace(is_safe_search, is_safe_replace)

with open("src/nodetool/api/file.py", "w") as f:
    f.write(content)
