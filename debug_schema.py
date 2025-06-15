#!/usr/bin/env python3

import json

# Test schema from the example
test_schema = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "problem": {"type": "string"},
                    "solution": {"type": "string"},
                    "comments": {"type": "array", "items": {"type": "string"}},
                    "url": {"type": "string", "format": "uri"},
                },
                "required": ["title", "problem", "solution", "comments", "url"],
            },
        }
    },
}

print("Original schema:")
print(test_schema)
print()

print("json.dumps(schema):")
json_str = json.dumps(test_schema)
print(json_str)
print(f"Length: {len(json_str)}")
print()

print("Attempting json.loads() on the dumped string:")
try:
    parsed = json.loads(json_str)
    print("SUCCESS: Parsed back to dict")
    print(parsed)
except json.JSONDecodeError as e:
    print(f"ERROR: {e}")
print()

# Test if there might be extra characters
print("Testing with extra characters:")
malformed = json_str + "}'"
print(f"Malformed: {malformed}")
try:
    parsed = json.loads(malformed)
    print("SUCCESS: This shouldn't happen")
except json.JSONDecodeError as e:
    print(f"ERROR: {e}")