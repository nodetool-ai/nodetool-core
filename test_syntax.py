import ast
import sys

def check_syntax(filepath):
    try:
        with open(filepath, "r") as f:
            ast.parse(f.read(), filename=filepath)
        return True
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return False

files = [
    "src/nodetool/agents/serp_providers/apify_provider.py",
    "src/nodetool/agents/serp_providers/data_for_seo_provider.py"
]

all_good = True
for f in files:
    if not check_syntax(f):
        all_good = False

if all_good:
    print("All files have valid syntax.")
else:
    sys.exit(1)
