import sys
try:
    import nodetool.api
    print("Found nodetool.api")
except Exception as e:
    print(f"Error: {e}")
