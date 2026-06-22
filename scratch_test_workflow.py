import requests
import json
import time

def test_workflow():
    url = "http://localhost:8000/v1/chat/completions"
    api_key = "local_developer_secret_key"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 1. Test Casual Mode
    print("=== Testing Casual Mode (Message: 'test') ===")
    payload_casual = {
        "messages": [
            {"role": "user", "content": "test"}
        ],
        "stream": False,
        "user": "test-user-session"
    }
    
    start_time = time.time()
    try:
        r = requests.post(url, headers=headers, json=payload_casual, timeout=90)
        duration = time.time() - start_time
        print(f"Status: {r.status_code}")
        print(f"Duration: {duration:.2f} seconds")
        if r.status_code == 200:
            print("Response:")
            print(r.json()["choices"][0]["message"]["content"])
        else:
            print(r.text)
    except Exception as e:
        print(f"Error: {e}")

    # 2. Test Engineering Mode
    print("\n=== Testing Engineering Mode (Message: 'Write Roblox Luau function to print Player Name.') ===")
    payload_eng = {
        "messages": [
            {"role": "user", "content": "Write Roblox Luau function to print Player Name."}
        ],
        "stream": False,
        "user": "test-user-session"
    }
    
    start_time = time.time()
    try:
        # Give it a higher timeout just in case prompt evaluation takes time
        r = requests.post(url, headers=headers, json=payload_eng, timeout=120)
        duration = time.time() - start_time
        print(f"Status: {r.status_code}")
        print(f"Duration: {duration:.2f} seconds")
        if r.status_code == 200:
            print("Response:")
            print(r.json()["choices"][0]["message"]["content"])
        else:
            print(r.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_workflow()
