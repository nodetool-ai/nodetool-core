# Message Parsing Regex Caching

**Insight**: The message parsing module was compiling regex patterns on every function call, causing unnecessary CPU overhead in performance-critical agent planning and execution code paths.

**File**: `src/nodetool/utils/message_parsing.py`

**Original Code**:
```python
def remove_think_tags(text_content: Optional[str]) -> Optional[str]:
    return re.sub(r"```.*?```", "", text_content, flags=re.DOTALL).strip()

def lenient_json_parse(text: str) -> Optional[dict[str, Any]]:
    py_text = re.sub(r"\btrue\b", "True", py_text)
    py_text = re.sub(r"\bfalse\b", "False", py_text)
    py_text = re.sub(r"\bnull\b", "None", py_text)

def extract_json_from_message(message: Optional[Message]) -> Optional[dict]:
    json_fence_pattern = r"```(?:json)?\s*\n(.*?)\n```"
    matches = re.findall(json_fence_pattern, cleaned_content, re.DOTALL)
```

**Solution**: Pre-compile all regex patterns at module level:
```python
# Pre-compiled regex patterns for performance (avoid recompilation on every call)
# These patterns are used in agent planning and execution, which are performance-critical paths
_THINK_TAGS_RE = re.compile(r"```.*?```", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)
_TRUE_RE = re.compile(r"\btrue\b")
_FALSE_RE = re.compile(r"\bfalse\b")
_NULL_RE = re.compile(r"\bnull\b")

def remove_think_tags(text_content: Optional[str]) -> Optional[str]:
    return _THINK_TAGS_RE.sub("", text_content).strip()

def lenient_json_parse(text: str) -> Optional[dict[str, Any]]:
    py_text = _TRUE_RE.sub("True", py_text)
    py_text = _FALSE_RE.sub("False", py_text)
    py_text = _NULL_RE.sub("None", py_text)

def extract_json_from_message(message: Optional[Message]) -> Optional[dict]:
    matches = _JSON_FENCE_RE.findall(cleaned_content)
```

**Impact**: Eliminates regex compilation overhead on every call to these functions. These functions are used in agent task planning and execution paths, which are called frequently during agent workflows. For agent workflows that process hundreds of messages, this reduces regex operations from thousands to just 5 initial compilations.

**Date**: 2026-03-20
