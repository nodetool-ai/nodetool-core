#!/usr/bin/env python3

import json

def robust_json_parse(json_str, context="test"):
    """Test the robust JSON parsing logic"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as json_err:
        print(f"{context}: Initial JSON parsing failed: {json_err}")
        
        # Try to find the end of valid JSON by counting braces
        brace_count = 0
        valid_end = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(json_str):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        valid_end = i + 1
                        break
        
        if valid_end > 0:
            truncated_schema = json_str[:valid_end]
            print(f"{context}: Attempting to parse truncated schema: '{truncated_schema}'")
            try:
                result = json.loads(truncated_schema)
                print(f"{context}: Successfully recovered schema after truncation")
                return result
            except json.JSONDecodeError:
                # If truncation doesn't work, raise the original error
                raise json_err
        else:
            # If we can't find valid JSON, raise the original error
            raise json_err

# Test cases
test_cases = [
    '{"type": "string"}',  # Valid JSON
    '{"type": "string"}extra',  # Valid JSON with extra text
    '{"type": "string"}}\'',  # Valid JSON with extra braces and quote
    '{"nested": {"key": "value"}}garbage',  # Nested valid JSON with garbage
    '{"broken": "}',  # Broken JSON
]

for i, test_case in enumerate(test_cases):
    print(f"\nTest {i+1}: {test_case}")
    try:
        result = robust_json_parse(test_case, f"Test{i+1}")
        print(f"Success: {result}")
    except Exception as e:
        print(f"Failed: {e}")