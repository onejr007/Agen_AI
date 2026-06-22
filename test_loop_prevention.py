import sys
import os

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.main import detect_tool_call_loop

print("\n--- Testing Infinite Tool Call Loop Prevention ---")

# Mock messages simulating a loop
messages = [
    {"role": "user", "content": "Please read config.json"},
    {"role": "assistant", "tool_calls": [{"name": "read_file", "arguments": {"path": "config.json"}}]},
    {"role": "user", "content": "File not found"},
    {"role": "assistant", "tool_calls": [{"name": "read_file", "arguments": {"path": "config.json"}}]},
    {"role": "user", "content": "File not found"},
    {"role": "assistant", "tool_calls": [{"name": "read_file", "arguments": {"path": "config.json"}}]},
]

is_loop = detect_tool_call_loop(messages)
print(f"Loop detected: {is_loop} (Expected: True)")
assert is_loop is True, "Failed to detect loop"

# Mock messages simulating NO loop
messages_no_loop = [
    {"role": "user", "content": "Please read config.json"},
    {"role": "assistant", "tool_calls": [{"name": "read_file", "arguments": {"path": "config.json"}}]},
    {"role": "user", "content": "File not found"},
    {"role": "assistant", "tool_calls": [{"name": "read_file", "arguments": {"path": "config.yaml"}}]},
    {"role": "user", "content": "File not found"},
    {"role": "assistant", "tool_calls": [{"name": "read_file", "arguments": {"path": "settings.json"}}]},
]

is_loop_false = detect_tool_call_loop(messages_no_loop)
print(f"Loop detected: {is_loop_false} (Expected: False)")
assert is_loop_false is False, "False positive in loop detection"

print("Infinite Tool Call Loop Prevention: PASS\n")
