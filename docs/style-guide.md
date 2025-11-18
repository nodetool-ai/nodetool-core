[← Back to Docs Index](index.md)

# Docs Style Guide

**Audience:** Contributors writing or updating documentation.  
**What you will learn:** Formatting, casing, and example conventions.

- **Voice and casing:** Use “NodeTool” consistently; avoid mixed casing. Keep tone concise and actionable.
- **Headings:** Start pages with **Audience** and **What you will learn**. Use sentence case for headings and avoid deep nesting when possible.
- **Code fences:** Always add a language tag (`bash`, `python`, `json`, `yaml`, `mermaid`). Commands should be copy-pastable without leading prompts.
- **Paths and ports:** Default to `http://127.0.0.1:8000` for API examples unless a different host/port is required, and state when it changes.
- **Authentication:** Show `Authorization: Bearer ...` headers in examples when auth can be enforced. Note when `AUTH_PROVIDER` is `local`/`none` for unauthenticated environments.
- **Links:** Avoid file-and-line references; link to sections/anchors instead.
- **Diagrams:** Use fenced `mermaid` blocks for diagrams; keep them small and labeled.
- **No secrets:** Use obvious placeholders (`YOUR_TOKEN`, `YOUR_ENDPOINT_ID`) instead of real or fake keys.
