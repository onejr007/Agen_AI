import os
import sys
import json
import requests
import time

HISTORY_FILE = "classroom_history.json"
TEACHER_PROMPT_FILE = "teacher_prompt.txt"

def parse_api_key(filepath="./api_key.txt"):
    """Reads the API key from the api_key.txt file."""
    if not os.path.exists(filepath):
        # Check parent directory or workspace root
        if os.path.exists("../api_key.txt"):
            filepath = "../api_key.txt"
        else:
            return "local_developer_secret_key"
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("AGENT_API_KEY="):
                    return line.split("=", 1)[1].strip()
                elif line.strip() and "=" not in line:
                    return line.strip()
    except Exception as e:
        print(f"Error reading api_key.txt: {e}")
    return "local_developer_secret_key"

def get_headers(api_key):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

def reset_session():
    """Resets the classroom session history."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    print("Classroom session history reset successfully.")

def load_history():
    """Loads the conversation history."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading history file: {e}")
        return []

def save_history(history):
    """Saves the conversation history."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving history file: {e}")

def send_prompt(api_key):
    """Reads teacher_prompt.txt, sends it to the Student API, and records the reply."""
    if not os.path.exists(TEACHER_PROMPT_FILE):
        print(f"Error: Prompt file {TEACHER_PROMPT_FILE} not found. Please create it first.")
        return False
        
    with open(TEACHER_PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_content = f.read().strip()
        
    if not prompt_content:
        print("Error: teacher_prompt.txt is empty. Write a prompt first.")
        return False
        
    history = load_history()
    
    # Append Teacher turn
    history.append({"role": "user", "content": prompt_content})
    
    print("\n" + "="*50)
    print(" [TEACHER PROMPT SENT]")
    print("-"*50)
    print(prompt_content)
    print("="*50 + "\n")
    
    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": "qwen2.5-coder:1.5b",
        "messages": history,
        "stream": False,
        "user": "teacher-session",
        "max_completion_tokens": 300
    }
    
    print("Connecting to Student Agent API at http://localhost:8000...")
    try:
        response = requests.post(url, headers=get_headers(api_key), json=payload, timeout=600)
        if response.status_code == 200:
            result = response.json()
            student_reply = result["choices"][0]["message"]["content"]
            
            print("\n" + "="*50)
            print(" [STUDENT AGENT REPLY]")
            print("-"*50)
            print(student_reply)
            print("="*50 + "\n")
            
            # Append Student reply
            history.append({"role": "assistant", "content": student_reply})
            save_history(history)
            return True
        else:
            print(f"Error from Student API ({response.status_code}): {response.text}")
            return False
    except Exception as e:
        print(f"Connection to Student API failed: {e}")
        return False

def teach_rag(api_key, title, content, tags):
    """Sends knowledge items directly to the Student's RAG DB."""
    url = "http://localhost:8000/v1/knowledge"
    payload = {
        "title": title,
        "content": content,
        "tags": tags
    }
    try:
        response = requests.post(url, headers=get_headers(api_key), json=payload, timeout=30)
        if response.status_code == 200:
            print(f"Successfully indexed RAG knowledge: '{title}'")
            return True
        else:
            print(f"Failed to index RAG knowledge ({response.status_code}): {response.text}")
            return False
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

def show_status():
    """Displays current session history."""
    history = load_history()
    if not history:
        print("No active classroom session history found.")
        return
    print("\n" + "="*60)
    print(" CURRENT CLASSROOM SESSION LOG")
    print("="*60)
    for idx, turn in enumerate(history):
        role = "TEACHER" if turn["role"] == "user" else "STUDENT"
        print(f"\nTurn {idx+1} - [{role}]:")
        print(turn["content"])
        print("-"*40)
    print("="*60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classroom_talk.py [reset|send|status|teach-rag]")
        sys.exit(1)
        
    action = sys.argv[1].lower()
    api_key = parse_api_key()
    
    if action == "reset":
        reset_session()
    elif action == "send":
        send_prompt(api_key)
    elif action == "status":
        show_status()
    elif action == "teach-rag":
        if len(sys.argv) < 5:
            print("Usage: python classroom_talk.py teach-rag \"<title>\" \"<content>\" \"<tags>\"")
            sys.exit(1)
        teach_rag(api_key, sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
