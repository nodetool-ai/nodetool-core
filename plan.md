1. Modify `src/nodetool/worker/comfy_handler.py` to strip the `Authorization` header case-insensitively when crossing origins, instead of relying on `headers.pop("Authorization", None)` which is case-sensitive and might fail if the token was somehow inserted with different capitalization, or if there's an `authorization` key.

Actually, the issue is that standard python dicts are case sensitive. If `Authorization` is added as `authorization`, it wouldn't be removed.

Wait! The memory says:
"When manually stripping HTTP headers (like `Authorization`) from standard Python dictionaries (e.g., during cross-origin manual redirect handling), ensure case-insensitive removal (e.g., `[k for k in headers if k.lower() == 'authorization']`) because standard Python dicts are case-sensitive, unlike specialized HTTP header objects."

Let's check if this is exactly the issue.
