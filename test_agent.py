import requests
import json
import time

def test_gateway():
    base_url = "http://localhost:8000"
    api_key = "local_developer_secret_key"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 1. Health check
    print("=== Testing Health Check ===")
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        print(f"Status: {r.status_code}")
        print(f"Response: {json.dumps(r.json(), indent=2)}")
    except Exception as e:
        print(f"Failed to connect to gateway: {e}")
        return

    # 2. Non-Streaming Chat Completion
    print("\n=== Testing Chat Completion (Non-Streaming) ===")
    payload = {
        "messages": [
            {"role": "user", "content": "Write a Roblox Luau function to print Player Name."}
        ],
        "stream": False,
        "user": "test-user-session"
    }
    
    start_time = time.time()
    try:
        r = requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=payload, timeout=180)
        duration = time.time() - start_time
        print(f"Status: {r.status_code}")
        print(f"Duration: {duration:.2f} seconds")
        if r.status_code == 200:
            print("Response content:")
            print(r.json()["choices"][0]["message"]["content"])
        else:
            print(r.text)
    except Exception as e:
        print(f"Error during request: {e}")

    # 3. Streaming Chat Completion with Web Search
    print("\n=== Testing Chat Completion (Streaming + Search Trigger) ===")
    payload_stream = {
        "messages": [
            {"role": "user", "content": "what is the latest updates on Roblox Luau syntax search on internet?"}
        ],
        "stream": True,
        "user": "test-user-session"
    }

    try:
        r = requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=payload_stream, stream=True, timeout=180)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("Streaming chunks:")
            for line in r.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            print("\n[STREAM COMPLETE]")
                            break
                        try:
                            data_json = json.loads(data_str)
                            content = data_json["choices"][0]["delta"].get("content", "")
                            print(content, end="", flush=True)
                        except Exception as e:
                            print(f"\n[JSON ERROR: {e} for data: {data_str}]")
        else:
            print(r.text)
    except Exception as e:
        print(f"Error during stream request: {e}")

if __name__ == "__main__":
    print("Waiting 3 seconds before starting test...")
    time.sleep(3)
    test_gateway()
