import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.main import get_workspace_tree

print("\n--- Testing Workspace Directory Tree Indexing ---")

tree = get_workspace_tree(max_depth=2, max_files=10)
print(tree)
print("--- End Tree ---")

assert len(tree) > 0, "Tree should not be empty"
assert ".git" not in tree, "Tree should exclude .git"
assert ".venv" not in tree, "Tree should exclude .venv"
assert "__pycache__" not in tree, "Tree should exclude __pycache__"

print("Workspace Directory Tree Indexing: PASS\n")
