import sys
import requests
import json

def query_local(prompt: str, stream: bool = True):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Authorization": "Bearer local_developer_secret_key",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        if response.status_code == 200:
            if stream:
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("data: "):
                            data_str = decoded[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                content = data_json["choices"][0]["delta"].get("content", "")
                                sys.stdout.write(content)
                                sys.stdout.flush()
                            except Exception:
                                pass
                print()
            else:
                result = response.json()
                print(result["choices"][0]["message"]["content"])
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_local_agent.py <prompt>")
        sys.exit(1)
        
    prompt_arg = sys.argv[1]
    query_local(prompt_arg, stream=True)
