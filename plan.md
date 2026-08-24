## Proposed Plan
1. **Target the SSRF/Credential Leakage Vulnerability**: In `src/nodetool/workflows/processing_context.py` inside `_http_request_with_retries`, `aiohttp` follows manual redirects safely checking `is_ip_private` using `_validate_http_target`. However, it leaks `Authorization` headers to the redirected domains.
2. **Implement Security Fix**: Modify `_http_request_with_retries` in `src/nodetool/workflows/processing_context.py` to drop the `Authorization` header when a redirect leads to a different domain. Use `urllib.parse.urlparse` to compare hostnames.
3. **Pre-commit Checks**: Run `pre_commit_instructions` and format, lint, and test.
4. **Submit**: Create PR.
