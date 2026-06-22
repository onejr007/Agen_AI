import sys
import os

from app.agent import apply_unified_diff
import app.main as main_module
from app.main import get_git_workspace_context, summarize_session_history, preprocess_tool_call

def test_git_context():
    print("\n--- Testing Git Context Integration ---")
    ctx = get_git_workspace_context()
    print(f"Git Context output:\n{ctx}")
    # Git should be initialized in this project
    assert "Active Git Workspace Status" in ctx or ctx == ""
    print("Git Context Integration: PASS")

def test_diff_engine():
    print("\n--- Testing SEARCH-REPLACE Diff Engine ---")
    original = "line 1\nline 2: database = mysql\nline 3\n"
    diff = """
<<<<<<< SEARCH
line 2: database = mysql
=======
line 2: database = mysql_xampp
>>>>>>> REPLACE
"""
    updated = apply_unified_diff(original, diff)
    print(f"Original:\n{original}\nDiff:\n{diff}\nUpdated:\n{updated}")
    assert "mysql_xampp" in updated
    assert "database = mysql\n" not in updated
    print("SEARCH-REPLACE Diff Engine: PASS")

def test_preprocess_tool_call():
    print("\n--- Testing Tool Call Diff Preprocessing ---")
    
    workspace_dir = "/app_host" if os.path.exists("/app_host") else "."
    dummy_file_name = "dummy_test_diff.py"
    dummy_file_path = os.path.join(workspace_dir, dummy_file_name)
    
    # Write a dummy file to the workspace
    with open(dummy_file_path, "w", encoding="utf-8") as f:
        f.write("a = 10\nb = 20\nc = 30\n")
        
    tool_call = {
        "name": "replace_file_content",
        "arguments": {
            "path": dummy_file_name,
            "content": "<<<<<<< SEARCH\nb = 20\n=======\nb = 200\n>>>>>>> REPLACE"
        }
    }
    
    processed = preprocess_tool_call(tool_call)
    print(f"Processed Tool Call:\n{processed}")
    
    # Assert it was converted to write_to_file and applied
    assert processed["name"] == "write_to_file"
    assert "b = 200" in processed["arguments"]["content"]
    assert "b = 20\n" not in processed["arguments"]["content"]
    
    # Clean up dummy file
    if os.path.exists(dummy_file_path):
        os.remove(dummy_file_path)
        
    print("Tool Call Diff Preprocessing: PASS")

def test_session_compression():
    print("\n--- Testing Session Context Compression ---")
    # Small test conversation
    test_msgs = [
        {"role": "user", "content": "Hello agent, let's start a php program."},
        {"role": "assistant", "content": "Sure, I can help you with PHP programming. What is your goal?"},
        {"role": "user", "content": "I want to create a PDO class to handle database query transactions."},
        {"role": "assistant", "content": "Great choice. I will set up the db class with retry logic."}
    ]
    summary = summarize_session_history(test_msgs)
    print(f"Generated Summary:\n{summary}")
    assert len(summary) > 5
    print("Session Context Compression: PASS")


def test_session_compression_fallback():
    print("\n--- Testing Session Compression Fallback ---")

    test_msgs = [
        {"role": "user", "content": "Tolong perbaiki validasi bearer token di app/main.py agar token kosong ditolak."},
        {"role": "assistant", "content": "Saya akan memperketat parser Authorization dan menambah test regresi."},
        {"role": "user", "content": "Tambahkan juga fallback agar ringkasan sesi tidak gagal total saat Ollama down."}
    ]

    import requests
    original_post = requests.post
    try:
        def raising_post(*args, **kwargs):
            raise requests.ConnectionError("simulated Ollama outage")

        requests.post = raising_post
        summary = summarize_session_history(test_msgs)
    finally:
        requests.post = original_post

    print(f"Fallback Summary:\n{summary}")
    assert "fallback" in summary.lower()
    assert "bearer token" in summary.lower()
    assert "app/main.py" in summary
    print("Session Compression Fallback: PASS")

if __name__ == "__main__":
    test_git_context()
    test_diff_engine()
    test_preprocess_tool_call()
    test_session_compression()
    test_session_compression_fallback()
    print("\nAll Phase III Validation tests PASSED!")
