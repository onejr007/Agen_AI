import os
import sys
import re
import json
import time
import subprocess
import threading
import urllib.request
import urllib.error
from urllib.parse import parse_qs, urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler

# Ensure standard system environment variables are set (crucial for subprocess execution on Windows)
if sys.platform == "win32":
    if "SystemRoot" not in os.environ:
        os.environ["SystemRoot"] = "C:\\Windows"
    sys_path = "C:\\Windows\\System32"
    if sys_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + sys_path

# Load settings from .env file or environment
def load_env_val(name: str, default: str = "") -> str:
    val = os.getenv(name)
    if val:
        return val.strip()
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith(f"{name}="):
                        # Extract value after '=' and strip quotes
                        parts = line.split("=", 1)[1].strip()
                        if (parts.startswith('"') and parts.endswith('"')) or (parts.startswith("'") and parts.endswith("'")):
                            parts = parts[1:-1]
                        return parts
        except Exception:
            pass
    return default

# Initialize configuration
TELEGRAM_BOT_TOKEN = load_env_val("TELEGRAM_BOT_TOKEN")
AGENT_API_KEY = load_env_val("AGENT_API_KEY", "local_developer_secret_key")
DATABASE_URL = load_env_val("DATABASE_URL", "mysql+pymysql://root:@host.docker.internal:3306/agent_db")
OLLAMA_MODEL = load_env_val("OLLAMA_MODEL", "qwen2.5-coder:1.5b")

session_expiry = 0.0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_LOG_PATH = os.path.join(SCRIPT_DIR, "host_executor_debug.log")
TELEGRAM_LOG_PATH = os.path.join(SCRIPT_DIR, "host_telegram_bot.log")

def log_telegram(msg: str):
    """Logs background Telegram bot events with timestamps to a file and prints to stdout."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {msg}"
    print(log_line, flush=True)
    try:
        with open(TELEGRAM_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except Exception:
        pass

# Zero-dependency HTTP helpers
def requests_post_json(url: str, json_data: dict, headers: dict = None, timeout: int = 30) -> tuple[int, str]:
    """Zero-dependency HTTP POST sending JSON."""
    data = json.dumps(json_data).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
            
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            err_body = str(e)
        return e.code, err_body
    except urllib.error.URLError as e:
        return 0, str(e.reason)
    except Exception as e:
        return 0, str(e)

# In-memory history fallback
mem_history = {}
active_sessions = {}
sessions_lock = threading.Lock()

# ============ Zero-dependency Redis helpers (raw TCP socket) ============
def redis_command(*args, host="localhost", port=6379, timeout=2):
    """Execute a Redis command via raw TCP socket. Zero-dependency."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, int(port)))
        # Build RESP protocol
        cmd = f"*{len(args)}\r\n"
        for arg in args:
            arg_str = str(arg)
            cmd += f"${len(arg_str)}\r\n{arg_str}\r\n"
        s.send(cmd.encode())
        resp = s.recv(4096).decode("utf-8", errors="ignore")
        s.close()
        return resp
    except Exception:
        return None

def redis_set(key, value, ex=None):
    """Redis SET via socket."""
    if ex:
        return redis_command("SET", key, value, "EX", str(ex))
    return redis_command("SET", key, value)

def redis_get(key):
    """Redis GET via socket. Returns string value or None."""
    resp = redis_command("GET", key)
    if resp and not resp.startswith("$-1"):
        lines = resp.split("\r\n")
        if len(lines) >= 2 and lines[0].startswith("$"):
            return lines[1]
    return None

def redis_delete(key):
    """Redis DEL via socket."""
    return redis_command("DEL", key)

def get_host_chat_history(chat_id: str, limit: int = 15) -> list:
    """Attempts to retrieve chat history from MySQL, falling back to memory if DB driver is missing or offline."""
    global mem_history
    try:
        import mysql.connector
        
        # Parse DATABASE_URL
        clean_url = DATABASE_URL.split("://", 1)[1]
        user_pass, host_db = clean_url.split("@", 1)
        user = user_pass.split(":", 1)[0]
        password = user_pass.split(":", 1)[1] if ":" in user_pass else ""
        host_port, db_name = host_db.split("/", 1)
        host = host_port.split(":", 1)[0]
        if host == "host.docker.internal":
            host = "localhost" # Run on host machine
        port = int(host_port.split(":", 1)[1]) if ":" in host_port else 3306
        
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_name
        )
        try:
            cursor = conn.cursor(dictionary=True)
            sql = "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY created_at ASC"
            cursor.execute(sql, (chat_id,))
            rows = cursor.fetchall()
            history = [{"role": r["role"], "content": r["content"]} for r in rows if r["role"] in ("user", "assistant")]
            cursor.close()
            return history[-limit:]
        finally:
            conn.close()
    except Exception as e:
        print(f"[Host Telegram Bot] DB history load warning: {str(e)}", flush=True)
        if chat_id not in mem_history:
            mem_history[chat_id] = []
        return mem_history[chat_id][-limit:]

def save_host_chat_message(chat_id: str, role: str, content: str):
    """Attempts to save chat message to MySQL or in-memory fallback."""
    global mem_history
    
    # Save to memory
    if chat_id not in mem_history:
        mem_history[chat_id] = []
    mem_history[chat_id].append({"role": role, "content": content})
    
    try:
        import mysql.connector
        
        clean_url = DATABASE_URL.split("://", 1)[1]
        user_pass, host_db = clean_url.split("@", 1)
        user = user_pass.split(":", 1)[0]
        password = user_pass.split(":", 1)[1] if ":" in user_pass else ""
        host_port, db_name = host_db.split("/", 1)
        host = host_port.split(":", 1)[0]
        if host == "host.docker.internal":
            host = "localhost"
        port = int(host_port.split(":", 1)[1]) if ":" in host_port else 3306
        
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_name
        )
        try:
            cursor = conn.cursor()
            
            # Check if chat session exists
            cursor.execute("SELECT id FROM chats WHERE id = %s", (chat_id,))
            if not cursor.fetchone():
                cursor.execute("INSERT INTO chats (id, title) VALUES (%s, %s)", (chat_id, "Telegram Host Conversation"))
                conn.commit()
                
            # Insert message
            sql = "INSERT INTO messages (chat_id, role, content) VALUES (%s, %s, %s)"
            cursor.execute(sql, (chat_id, role, content))
            conn.commit()
            cursor.close()
        finally:
            conn.close()
    except Exception as e:
        print(f"[Host Telegram Bot] DB message save warning: {str(e)}", flush=True)


def save_telegram_feedback(chat_id: int, message_id: int, action: str, username: str):
    """Saves user feedback (like/unlike) to MySQL or logs it."""
    try:
        import mysql.connector
        
        clean_url = DATABASE_URL.split("://", 1)[1]
        user_pass, host_db = clean_url.split("@", 1)
        user = user_pass.split(":", 1)[0]
        password = user_pass.split(":", 1)[1] if ":" in user_pass else ""
        host_port, db_name = host_db.split("/", 1)
        host = host_port.split(":", 1)[0]
        if host == "host.docker.internal":
            host = "localhost"
        port = int(host_port.split(":", 1)[1]) if ":" in host_port else 3306
        
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_name
        )
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telegram_feedback (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    chat_id VARCHAR(50),
                    message_id INT,
                    feedback VARCHAR(10),
                    username VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            
            cursor.execute(
                "INSERT INTO telegram_feedback (chat_id, message_id, feedback, username) VALUES (%s, %s, %s, %s)",
                (str(chat_id), message_id, action, username)
            )
            conn.commit()
            cursor.close()
        finally:
            conn.close()
    except Exception as e:
        print(f"[Host Telegram Bot] DB feedback save warning: {str(e)}", flush=True)

def parse_suggested_questions(text: str) -> list:
    """Parses bullet points ending with ? from LLM response text."""
    lines = text.split("\n")
    questions = []
    for line in lines:
        line_strip = line.strip()
        # Look for bullet points: "- Question?", "* Question?", etc.
        match = re.match(r"^[-*•]\s*(.*\?)$", line_strip)
        if match:
            q = match.group(1).strip()
            # Remove basic markdown bold/italic tags
            q_clean = re.sub(r"[*_`]", "", q)
            # Limit button text length to 45 chars to be safe and readable
            if 5 < len(q_clean) <= 45:
                questions.append(q_clean)
    return questions[:3] # Limit to top 3 questions

def clear_host_chat_history(chat_id: str):
    """Deletes all messages for a chat session from MySQL and in-memory fallback."""
    global mem_history
    if chat_id in mem_history:
        mem_history[chat_id] = []
        
    try:
        import mysql.connector
        
        clean_url = DATABASE_URL.split("://", 1)[1]
        user_pass, host_db = clean_url.split("@", 1)
        user = user_pass.split(":", 1)[0]
        password = user_pass.split(":", 1)[1] if ":" in user_pass else ""
        host_port, db_name = host_db.split("/", 1)
        host = host_port.split(":", 1)[0]
        if host == "host.docker.internal":
            host = "localhost"
        port = int(host_port.split(":", 1)[1]) if ":" in host_port else 3306
        
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_name
        )
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE chat_id = %s", (chat_id,))
            conn.commit()
            cursor.close()
        finally:
            conn.close()
    except Exception as e:
        print(f"[Host Telegram Bot] DB clear warning: {str(e)}", flush=True)

# Monologue & Tool Parsers
def format_instinct_for_telegram(text: str) -> str:
    """Detects <instinct>...</instinct> and formats it nicely for Telegram."""
    pattern = r"<instinct>(.*?)</instinct>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        instinct_content = match.group(1).strip()
        cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return f"💭 *Naluri & Nalar:*\n_{instinct_content}_\n\n{cleaned_text}"
        
    # Check if only opening tag exists
    if "<instinct>" in text:
        parts = text.split("<instinct>", 1)
        cleaned_before = parts[0].strip()
        remaining = parts[1].strip()
        if remaining:
            return f"💭 *Naluri & Nalar:*\n_{remaining}_\n\n{cleaned_before}"
            
    return text

def parse_developer_tool_call(text: str) -> tuple[str, dict]:
    """Parses a tool call from LLM output. Returns (tool_name, arguments)."""
    blocks = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    for b in blocks:
        try:
            data = json.loads(b)
            if "tool" in data and "arguments" in data:
                return data["tool"], data["arguments"]
        except Exception:
            pass
            
    # Try parsing any substring JSON object
    matches = re.findall(r"(\{.*?\})", text, re.DOTALL)
    for m in matches:
        try:
            data = json.loads(m)
            if "tool" in data and "arguments" in data:
                return data["tool"], data["arguments"]
        except Exception:
            pass
            
    return None, None

def execute_host_tool_locally(tool_name: str, arguments: dict) -> str:
    """Runs developer tools natively on the host machine."""
    try:
        base_dir = SCRIPT_DIR
        
        if tool_name == "run_command":
            command = arguments.get("CommandLine")
            if not command:
                return "Error: Missing 'CommandLine' argument."
            
            res = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                timeout=90
            )
            return f"Exit Code: {res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            
        elif tool_name == "view_file":
            path = arguments.get("AbsolutePath")
            if not path:
                return "Error: Missing 'AbsolutePath' argument."
            full_path = os.path.join(base_dir, path.lstrip("/"))
            if not os.path.exists(full_path):
                return f"Error: File '{path}' does not exist."
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(20000)
            return content
            
        elif tool_name == "write_to_file":
            path = arguments.get("TargetFile")
            content = arguments.get("CodeContent")
            if not path or content is None:
                return "Error: Missing 'TargetFile' or 'CodeContent' arguments."
            full_path = os.path.join(base_dir, path.lstrip("/"))
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Success: File '{path}' written successfully."
            
        elif tool_name == "replace_file_content":
            path = arguments.get("TargetFile")
            target = arguments.get("TargetContent")
            replacement = arguments.get("ReplacementContent")
            if not path or target is None or replacement is None:
                return "Error: Missing 'TargetFile', 'TargetContent', or 'ReplacementContent' arguments."
            full_path = os.path.join(base_dir, path.lstrip("/"))
            if not os.path.exists(full_path):
                return f"Error: File '{path}' does not exist."
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                file_content = f.read()
            if target not in file_content:
                return f"Error: TargetContent not found in '{path}'."
            new_content = file_content.replace(target, replacement, 1)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"Success: Replaced content in '{path}' successfully."
            
        elif tool_name == "list_dir":
            path = arguments.get("DirectoryPath", ".")
            full_path = os.path.join(base_dir, path.lstrip("/"))
            if not os.path.exists(full_path):
                return f"Error: Directory '{path}' does not exist."
            entries = os.listdir(full_path)
            res = []
            for entry in entries:
                entry_path = os.path.join(full_path, entry)
                is_dir = os.path.isdir(entry_path)
                res.append(f"{'[DIR] ' if is_dir else ''}{entry}")
            return "\n".join(res)
            
        else:
            return f"Error: Unknown tool '{tool_name}'."
    except Exception as ex:
        return f"Exception executing local tool: {str(ex)}"

# Timing & Typing threads
def send_typing_periodically(token: str, chat_id: int, stop_event: threading.Event):
    url_action = f"https://api.telegram.org/bot{token}/sendChatAction"
    while not stop_event.is_set():
        try:
            requests_post_json(url_action, {"chat_id": chat_id, "action": "typing"}, timeout=5)
        except Exception:
            pass
        for _ in range(40):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def update_placeholder_timer(token: str, chat_id: int, message_id: int, stop_event: threading.Event):
    url_edit = f"https://api.telegram.org/bot{token}/editMessageText"
    start_time = time.time()
    while not stop_event.is_set():
        for _ in range(30):
            if stop_event.is_set():
                break
            time.sleep(0.1)
        if stop_event.is_set():
            break
        elapsed = int(time.time() - start_time)
        text = f"Thinking... 🧠 (Elapsed: {elapsed}s)\n\n[Ketik pesan baru untuk membatalkan/mengganti]"
        try:
            requests_post_json(url_edit, {"chat_id": chat_id, "message_id": message_id, "text": text}, timeout=5)
        except Exception:
            pass

def send_telegram_reply(token: str, chat_id: int, reply: str, message_id: int = 0, reply_markup: dict = None):
    url_send = f"https://api.telegram.org/bot{token}/sendMessage"
    url_edit = f"https://api.telegram.org/bot{token}/editMessageText"
    
    payload = {
        "chat_id": chat_id,
        "text": reply,
        "parse_mode": "Markdown"
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
        
    if message_id:
        edit_payload = dict(payload)
        edit_payload["message_id"] = message_id
        status, body = requests_post_json(url_edit, edit_payload, timeout=10)
        if status == 200:
            return True
        edit_payload_plain = dict(edit_payload)
        if "parse_mode" in edit_payload_plain:
            del edit_payload_plain["parse_mode"]
        status, body = requests_post_json(url_edit, edit_payload_plain, timeout=10)
        if status == 200:
            return True

    status, body = requests_post_json(url_send, payload, timeout=10)
    if status == 200:
        return True
        
    payload_plain = dict(payload)
    if "parse_mode" in payload_plain:
        del payload_plain["parse_mode"]
    status, body = requests_post_json(url_send, payload_plain, timeout=10)
    return status == 200

# ============ Real-time Data Enrichment Engine ============
def fetch_realtime_enrichment(text: str) -> str:
    """Detect if user asks about real-time data and fetch from free public APIs.
    Returns compact enrichment string to inject into user message."""
    text_lower = text.lower()
    enrichments = []
    
    # --- Currency / Exchange Rate ---
    currency_keywords = ["kurs", "dolar", "dollar", "mata uang", "currency", "exchange",
                         "usd", "eur", "gbp", "jpy", "idr", "rupiah", "euro", "yen", "pound",
                         "nilai tukar"]
    if any(kw in text_lower for kw in currency_keywords):
        try:
            req = urllib.request.Request("https://open.er-api.com/v6/latest/USD", method="GET")
            req.add_header("User-Agent", "AgentAI/1.0")
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
                if data.get("result") == "success":
                    rates = data.get("rates", {})
                    parts = ["[DATA KURS] 1 USD ="]
                    for code in ["IDR", "EUR", "GBP", "JPY", "SGD", "MYR"]:
                        if code in rates:
                            r = rates[code]
                            if code == "IDR":
                                parts.append(f"IDR Rp{r:,.0f}")
                            elif code in ("EUR", "GBP"):
                                parts.append(f"{code} {r:.4f}")
                            else:
                                parts.append(f"{code} {r:,.1f}")
                    enrichments.append(" | ".join(parts))
        except Exception as e:
            log_telegram(f"[Enrichment] Currency API error: {str(e)}")
    
    # --- Weather ---
    weather_keywords = ["cuaca", "weather", "hujan", "panas", "suhu", "temperatur",
                        "temperature", "ramalan", "forecast"]
    if any(kw in text_lower for kw in weather_keywords):
        cities = {
            "jakarta": "Jakarta", "surabaya": "Surabaya", "bandung": "Bandung",
            "medan": "Medan", "semarang": "Semarang", "yogya": "Yogyakarta",
            "bali": "Bali", "makassar": "Makassar", "palembang": "Palembang",
            "tokyo": "Tokyo", "singapore": "Singapore", "london": "London",
            "new york": "New+York", "paris": "Paris", "bogor": "Bogor",
            "depok": "Depok", "tangerang": "Tangerang", "bekasi": "Bekasi",
            "malang": "Malang", "solo": "Solo", "denpasar": "Denpasar"
        }
        city = "Jakarta"
        for key, val in cities.items():
            if key in text_lower:
                city = val
                break
        
        try:
            url = f"https://wttr.in/{city}?format=j1"
            req = urllib.request.Request(url, method="GET")
            req.add_header("User-Agent", "AgentAI/1.0")
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
                cur = data.get("current_condition", [{}])[0]
                desc = cur.get("weatherDesc", [{}])[0].get("value", "?")
                temp = cur.get("temp_C", "?")
                feels = cur.get("FeelsLikeC", "?")
                hum = cur.get("humidity", "?")
                wind = cur.get("windspeedKmph", "?")
                
                # Also get forecast for today
                forecast_parts = []
                weather_list = data.get("weather", [])
                if weather_list:
                    today = weather_list[0]
                    max_t = today.get("maxtempC", "?")
                    min_t = today.get("mintempC", "?")
                    forecast_parts.append(f"Min/Max: {min_t}°C/{max_t}°C")
                    hourly = today.get("hourly", [])
                    for h in hourly:
                        time_val = h.get("time", "")
                        h_desc = h.get("weatherDesc", [{}])[0].get("value", "")
                        h_temp = h.get("tempC", "")
                        h_rain = h.get("chanceofrain", "0")
                        if time_val in ("600", "900", "1200", "1500", "1800", "2100"):
                            hr = int(time_val) // 100
                            forecast_parts.append(f"{hr}:00={h_desc},{h_temp}°C,hujan {h_rain}%")
                
                fc_str = " | ".join(forecast_parts) if forecast_parts else ""
                enrichments.append(
                    f"[CUACA {city.replace('+', ' ')}] Sekarang: {desc}, {temp}°C (terasa {feels}°C), "
                    f"Kelembapan {hum}%, Angin {wind}km/h. {fc_str}"
                )
        except Exception as e:
            log_telegram(f"[Enrichment] Weather API error: {str(e)}")
    
    if enrichments:
        return "\n\n[DATA REAL-TIME - format ulang data ini dengan visual cantik pakai emoji]\n" + "\n".join(enrichments)
    return ""

# Main agent thread logic
def process_host_telegram_message(token: str, api_key: str, chat_id: int, text: str, placeholder_msg_id: int, stop_event: threading.Event, telegram_user: dict):
    typing_thread = threading.Thread(
        target=send_typing_periodically,
        args=(token, chat_id, stop_event),
        daemon=True
    )
    typing_thread.start()
    
    if placeholder_msg_id:
        timer_thread = threading.Thread(
            target=update_placeholder_timer,
            args=(token, chat_id, placeholder_msg_id, stop_event),
            daemon=True
        )
        timer_thread.start()
        
    db_chat_id = f"telegram-{chat_id}"
    history = get_host_chat_history(db_chat_id, limit=8)
    
    is_developer = telegram_user.get("username") == "BagasJr"
    
    # Build compact system prompt for DIRECT Ollama (bypass FastAPI for maximum speed)
    import datetime
    now = datetime.datetime.now()
    local_time_str = now.strftime("%A, %d %B %Y, %H:%M WIB")
    
    user_name = telegram_user.get("first_name", "User")
    username_str = telegram_user.get("username", "")
    
    # Fetch real-time data BEFORE building messages (runs in parallel with prompt building)
    enrichment = fetch_realtime_enrichment(text)
    if enrichment:
        log_telegram(f"[Enrichment] Data fetched for query: {text[:30]}")
    
    # System prompt MUST be ultra-compact for 1.5B model performance
    if is_developer:
        compact_system = (
            f"Kamu AgentAI. Developer: {user_name} (@{username_str}). "
            f"Waktu: {local_time_str}. "
            "Jawab natural pakai emoji, bold, bullet. JANGAN output raw JSON/code kecuali diminta. "
            "Untuk command sistem: ```json\n{\"tool\":\"run_command\",\"arguments\":{\"CommandLine\":\"<cmd>\"}}\n```"
        )
    else:
        compact_system = (
            f"Kamu AgentAI, asisten cerdas. User: {user_name}. "
            f"Waktu: {local_time_str}. "
            "Jawab natural pakai emoji, bold, bullet. JANGAN output raw JSON/code. "
            "Gunakan bahasa yang sama dengan user."
        )
    
    # Inject enrichment data directly into user message (not system prompt!)
    # This ensures the small model sees the data RIGHT NEXT TO the question
    user_message = text
    if enrichment:
        user_message = text + enrichment
    
    history.append({"role": "user", "content": user_message})
    
    ollama_messages = [{"role": "system", "content": compact_system}] + list(history)
    ollama_url = "http://localhost:11434/api/chat"
    
    max_iterations = 5 if is_developer else 1
    iteration = 0
    executed_calls = []
    reply = ""
    is_fallback = False
    
    while iteration < max_iterations:
        if stop_event.is_set():
            break
            
        # PRIMARY: Direct Ollama call (bypasses all FastAPI middleware for maximum speed)
        ollama_payload = {
            "model": OLLAMA_MODEL,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "num_predict": 512 if is_developer else 384,
                "temperature": 0.4,
                "num_ctx": 2048
            }
        }
        
        try:
            # Set Redis status to prevent background workers from competing for Ollama
            redis_set("system_status", "USER_PRIORITY", ex=120)
            log_telegram(f"[DIRECT OLLAMA] Sending request (Iteration {iteration}, {len(ollama_messages)} msgs)")
            status, res_body = requests_post_json(ollama_url, ollama_payload, timeout=120)
            if status == 200:
                data = json.loads(res_body)
                reply = data.get("message", {}).get("content", "")
                is_fallback = False
                eval_dur = data.get("eval_duration", 0) / 1e9  # nanoseconds to seconds
                total_dur = data.get("total_duration", 0) / 1e9
                log_telegram(f"[DIRECT OLLAMA] Reply received (eval={eval_dur:.1f}s, total={total_dur:.1f}s)")
            else:
                raise Exception(f"Direct Ollama returned status {status}")
        except Exception as e:
            # FALLBACK: Try FastAPI if direct Ollama is unavailable
            log_telegram(f"[DIRECT OLLAMA] Failed ({str(e)}). Falling back to FastAPI...")
            is_fallback = True
            
            fastapi_url = "http://localhost:8000/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            chat_payload = {
                "model": OLLAMA_MODEL,
                "messages": history,
                "stream": False,
                "user": db_chat_id,
                "telegram_user": telegram_user
            }
            
            try:
                log_telegram(f"[FALLBACK] Sending request to FastAPI: {fastapi_url}")
                status, res_body = requests_post_json(fastapi_url, chat_payload, headers=headers, timeout=240)
                if status == 200:
                    data = json.loads(res_body)
                    reply = data["choices"][0]["message"]["content"]
                    log_telegram(f"[FALLBACK] FastAPI reply received.")
                else:
                    reply = f"⚠️ Error: Semua engine sedang offline (status {status})."
            except Exception as fallback_e:
                reply = f"⚠️ Kedua engine sedang offline. Coba lagi nanti ya!"
                log_telegram(f"Both engines failed: {str(fallback_e)}")
            break
                
        if is_fallback:
            break
            
        if not is_developer:
            break
            
        tool_name, tool_args = parse_developer_tool_call(reply)
        if not tool_name:
            break
            
        # Prevent infinite loops executing the exact same tool and arguments
        call_signature = (tool_name, json.dumps(tool_args, sort_keys=True) if isinstance(tool_args, dict) else str(tool_args))
        if call_signature in executed_calls:
            log_telegram(f"Warning: Model is repeating tool call {tool_name} with same arguments. Breaking loop.")
            # Inject a warning and get final answer directly from Ollama
            warning_msg = {"role": "user", "content": f"[System Warning]: You already executed {tool_name}. Provide your final answer NOW."}
            ollama_messages.append(warning_msg)
            try:
                status, res_body = requests_post_json(ollama_url, {
                    "model": OLLAMA_MODEL,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {"num_predict": 256, "temperature": 0.3, "num_ctx": 2048}
                }, timeout=120)
                if status == 200:
                    reply = json.loads(res_body).get("message", {}).get("content", "")
            except Exception:
                pass
            break
            
        executed_calls.append(call_signature)
        
        # Tool call detected
        iteration += 1
        log_telegram(f"Developer tool call detected: {tool_name} with {tool_args}")
        
        if placeholder_msg_id and not stop_event.is_set():
            exec_text = f"Executing tool: `{tool_name}`... ⚙️\n\n[Ketik pesan baru untuk membatalkan]"
            requests_post_json(f"https://api.telegram.org/bot{token}/editMessageText", {
                "chat_id": chat_id,
                "message_id": placeholder_msg_id,
                "text": exec_text
            }, timeout=5)
            
        # Run locally since we are already on host!
        tool_output = execute_host_tool_locally(tool_name, tool_args)
        log_telegram(f"Local tool output: {tool_output[:200]}...")
        
        # Append to both history (for DB save) and ollama_messages (for next iteration)
        history.append({"role": "assistant", "content": reply})
        history.append({"role": "user", "content": f"[Tool Output for '{tool_name}']:\n{tool_output}"})
        ollama_messages.append({"role": "assistant", "content": reply})
        ollama_messages.append({"role": "user", "content": f"[Tool Output for '{tool_name}']:\n{tool_output}"})
        
    if not stop_event.is_set() and reply:
        stop_event.set()
        
        log_telegram(f"Sending final reply to Telegram chat {chat_id}.")
        # Save messages: Direct Ollama (primary) needs us to save; FastAPI (fallback) saves internally
        if not is_fallback:
            save_host_chat_message(db_chat_id, "user", text)
            save_host_chat_message(db_chat_id, "assistant", reply)
        else:
            # FastAPI already saves to MySQL; just update local memory backup
            global mem_history
            if db_chat_id not in mem_history:
                mem_history[db_chat_id] = []
            mem_history[db_chat_id].append({"role": "user", "content": text})
            mem_history[db_chat_id].append({"role": "assistant", "content": reply})
        
        formatted_reply = format_instinct_for_telegram(reply)
        if is_fallback:
            formatted_reply = "⚠️ *[Sistem Cadangan - FastAPI]*\n\n" + formatted_reply
            
        # Parse suggested follow-up questions
        suggestions = parse_suggested_questions(reply)
        
        # Build reply markup (Like/Unlike row and Suggestion rows)
        inline_keyboard = []
        feedback_id = placeholder_msg_id or int(time.time())
        inline_keyboard.append([
            {"text": "👍 Like", "callback_data": f"like_{feedback_id}"},
            {"text": "👎 Unlike", "callback_data": f"unlike_{feedback_id}"}
        ])
        
        for q in suggestions:
            inline_keyboard.append([
                {"text": f"💬 {q}", "callback_data": f"suggest_{q}"}
            ])
            
        reply_markup = {"inline_keyboard": inline_keyboard} if inline_keyboard else None
        
        send_telegram_reply(token, chat_id, formatted_reply, placeholder_msg_id, reply_markup=reply_markup)
        
    with sessions_lock:
        if chat_id in active_sessions and active_sessions[chat_id]["stop_event"] == stop_event:
            del active_sessions[chat_id]

# Long polling thread
def run_host_telegram_polling():
    token = TELEGRAM_BOT_TOKEN
    if not token:
        log_telegram("[Host Telegram Bot] TELEGRAM_BOT_TOKEN is not set. Host polling disabled.")
        return
        
    log_telegram(f"[Host Telegram Bot] Initializing host-side polling for token: {token[:12]}...")
    url_get = f"https://api.telegram.org/bot{token}/getUpdates"
    url_send = f"https://api.telegram.org/bot{token}/sendMessage"
    url_edit = f"https://api.telegram.org/bot{token}/editMessageText"
    
    offset = 0
    while True:
        try:
            payload = {"offset": offset, "timeout": 15}
            status, res_body = requests_post_json(url_get, payload, timeout=20)
            if status == 200:
                data = json.loads(res_body)
                if data.get("ok"):
                    updates = data.get("result", [])
                    for update in updates:
                        update_id = update.get("update_id")
                        offset = update_id + 1
                        
                        message = update.get("message")
                        callback_query = update.get("callback_query")
                        
                        if message:
                          try:
                            chat = message.get("chat", {})
                            chat_id = chat.get("id")
                            text = message.get("text", "")
                            
                            if not text or not chat_id:
                                continue
                                
                            db_chat_id = f"telegram-{chat_id}"
                            text_clean = text.strip().lower()
                            if text_clean in ("/clear", "clear", "/reset", "reset"):
                                # Interrupt previous session if still running
                                with sessions_lock:
                                    if chat_id in active_sessions:
                                        log_telegram(f"[Host Telegram Bot] Interrupting active session for chat {chat_id} due to reset command")
                                        active_sessions[chat_id]["stop_event"].set()
                                        del active_sessions[chat_id]
                                clear_host_chat_history(db_chat_id)
                                try:
                                    requests_post_json(url_send, {
                                        "chat_id": chat_id,
                                        "text": "🧹 *Sesi obrolan berhasil di-reset!* Riwayat chat Anda telah dihapus. Silakan mulai percakapan baru.",
                                        "parse_mode": "Markdown"
                                    }, timeout=5)
                                except Exception as clear_err:
                                    log_telegram(f"Error sending clear confirmation: {str(clear_err)}")
                                continue
                            
                            # === /selflearn command (developer only) ===
                            telegram_user = message.get("from", {})
                            is_dev = telegram_user.get("username") == "BagasJr"
                            
                            if text_clean in ("/selflearn", "self learn", "selflearn", "/self-learn") and is_dev:
                                log_telegram(f"[Host Telegram Bot] Developer triggered self-learning from Telegram")
                                # Send initial progress message
                                progress_msg_id = None
                                try:
                                    status_s, res_s = requests_post_json(url_send, {
                                        "chat_id": chat_id,
                                        "text": "🧠 *Self-Learning Triggered!*\n━━━━━━━━━━━━━━━━━━━━\n⏳ Step 0/4: Initializing...\n⬜ Step 1/4: Middleware Upgrade\n⬜ Step 2/4: Topic Discovery\n⬜ Step 3/4: Self-Learning\n⬜ Step 4/4: Self-Rebuild Check\n━━━━━━━━━━━━━━━━━━━━\n⏱ Elapsed: 0s",
                                        "parse_mode": "Markdown"
                                    }, timeout=10)
                                    if status_s == 200:
                                        progress_msg_id = json.loads(res_s).get("result", {}).get("message_id")
                                except Exception:
                                    pass
                                
                                # Trigger self-learning via FastAPI
                                try:
                                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                                    requests_post_json("http://localhost:8000/self-learning/trigger", {}, headers=headers, timeout=10)
                                except Exception as trig_err:
                                    log_telegram(f"Self-learning trigger failed: {str(trig_err)}")
                                
                                # Start progress monitor thread
                                def monitor_selflearn_progress(chat_id, msg_id, token, api_key):
                                    """Poll Redis for self-learning progress and update Telegram message."""
                                    import time as _time
                                    url_edit_local = f"https://api.telegram.org/bot{token}/editMessageText"
                                    start_time = _time.time()
                                    step_icons = {"completed": "✅", "running": "⏳", "interrupted": "🛑", "error": "❌", "idle": "⬜"}
                                    last_text = ""
                                    
                                    for _ in range(120):  # Max 10 minutes (120 * 5s)
                                        _time.sleep(5)
                                        elapsed = int(_time.time() - start_time)
                                        
                                        # Get progress from Redis
                                        progress_raw = redis_get("self_learning_progress")
                                        if progress_raw:
                                            try:
                                                progress = json.loads(progress_raw)
                                            except Exception:
                                                progress = {"step": 0, "total": 4, "label": "Unknown", "status": "running", "detail": ""}
                                        else:
                                            progress = {"step": 0, "total": 4, "label": "Waiting", "status": "running", "detail": "Menunggu respons..."}
                                        
                                        current_step = progress.get("step", 0)
                                        current_status = progress.get("status", "running")
                                        detail = progress.get("detail", "")
                                        
                                        steps = [
                                            (1, "Middleware Upgrade"),
                                            (2, "Topic Discovery"),
                                            (3, "Self-Learning"),
                                            (4, "Self-Rebuild Check")
                                        ]
                                        
                                        lines = ["🧠 *Self-Learning Progress*", "━━━━━━━━━━━━━━━━━━━━"]
                                        for step_num, step_label in steps:
                                            if step_num < current_step:
                                                icon = "✅"
                                            elif step_num == current_step:
                                                icon = step_icons.get(current_status, "⏳")
                                            else:
                                                icon = "⬜"
                                            lines.append(f"{icon} Step {step_num}/4: {step_label}")
                                        
                                        lines.append("━━━━━━━━━━━━━━━━━━━━")
                                        if detail:
                                            lines.append(f"📋 {detail}")
                                        lines.append(f"⏱ Elapsed: {elapsed}s")
                                        
                                        new_text = "\n".join(lines)
                                        if new_text != last_text and msg_id:
                                            try:
                                                requests_post_json(url_edit_local, {
                                                    "chat_id": chat_id,
                                                    "message_id": msg_id,
                                                    "text": new_text,
                                                    "parse_mode": "Markdown"
                                                }, timeout=5)
                                                last_text = new_text
                                            except Exception:
                                                pass
                                        
                                        # Stop if completed, errored, or interrupted
                                        if current_status in ("completed", "error", "interrupted") and current_step >= 4:
                                            break
                                        if current_status in ("error", "interrupted"):
                                            break
                                    
                                    # Final message
                                    elapsed = int(_time.time() - start_time)
                                    if msg_id:
                                        try:
                                            final_status = "✅ Selesai!" if current_status == "completed" else f"🛑 {current_status.title()}"
                                            requests_post_json(url_edit_local, {
                                                "chat_id": chat_id,
                                                "message_id": msg_id,
                                                "text": f"{last_text}\n\n*Status: {final_status}*\n⏱ Total: {elapsed}s",
                                                "parse_mode": "Markdown"
                                            }, timeout=5)
                                        except Exception:
                                            pass
                                
                                monitor_thread = threading.Thread(
                                    target=monitor_selflearn_progress,
                                    args=(chat_id, progress_msg_id, token, api_key),
                                    daemon=True
                                )
                                monitor_thread.start()
                                continue
                                
                            log_telegram(f"[Host Telegram Bot] Received message from chat {chat_id}: '{text[:40]}'")
                            
                            # Interrupt previous session if still running
                            with sessions_lock:
                                if chat_id in active_sessions:
                                    log_telegram(f"[Host Telegram Bot] Interrupting active session for chat {chat_id}")
                                    active_sessions[chat_id]["stop_event"].set()
                                    
                                    old_msg_id = active_sessions[chat_id]["message_id"]
                                    try:
                                        requests_post_json(url_edit, {
                                            "chat_id": chat_id,
                                            "message_id": old_msg_id,
                                            "text": "🛑 Pertanyaan sebelumnya dibatalkan. Memproses pertanyaan baru... ⏳"
                                        }, timeout=5)
                                    except Exception:
                                        pass
                                        
                                    del active_sessions[chat_id]
                                    
                            # Send placeholder
                            placeholder_msg_id = None
                            send_payload = {
                                "chat_id": chat_id,
                                "text": "Thinking... 🧠 (Elapsed: 0s)\n\n[Ketik pesan baru untuk membatalkan/mengganti]"
                            }
                            try:
                                status_send, res_send = requests_post_json(url_send, send_payload, timeout=10)
                                if status_send == 200:
                                    placeholder_msg_id = json.loads(res_send).get("result", {}).get("message_id")
                            except Exception as send_err:
                                log_telegram(f"[Host Telegram Bot] Error sending placeholder: {str(send_err)}")
                                
                            if not placeholder_msg_id:
                                placeholder_msg_id = 0
                                
                            from_user = message.get("from", {})
                            telegram_user = {
                                "first_name": from_user.get("first_name", "User"),
                                "last_name": from_user.get("last_name", ""),
                                "username": from_user.get("username", "")
                            }
                            
                            stop_event = threading.Event()
                            with sessions_lock:
                                active_sessions[chat_id] = {
                                    "stop_event": stop_event,
                                    "message_id": placeholder_msg_id
                                }
                                
                            t = threading.Thread(
                                target=process_host_telegram_message,
                                args=(token, AGENT_API_KEY, chat_id, text, placeholder_msg_id, stop_event, telegram_user),
                                daemon=True
                            )
                            t.start()
                            
                          except Exception as msg_err:
                            log_telegram(f"[Host Telegram Bot] ERROR processing message: {str(msg_err)}")
                            import traceback
                            log_telegram(traceback.format_exc())

                        elif callback_query:
                            cq_id = callback_query.get("id")
                            cq_data = callback_query.get("data", "")
                            cq_message = callback_query.get("message", {})
                            cq_chat_id = cq_message.get("chat", {}).get("id")
                            cq_msg_id = cq_message.get("message_id")
                            cq_from = callback_query.get("from", {})
                            cq_user = {
                                "first_name": cq_from.get("first_name", "User"),
                                "last_name": cq_from.get("last_name", ""),
                                "username": cq_from.get("username", "")
                            }
                            
                            # Acknowledge callback query
                            url_answer = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
                            requests_post_json(url_answer, {"callback_query_id": cq_id}, timeout=5)
                            
                            if cq_data.startswith("like_") or cq_data.startswith("unlike_"):
                                action = "like" if cq_data.startswith("like_") else "unlike"
                                log_telegram(f"[Host Telegram Bot] User {cq_user.get('username')} clicked feedback {action} on msg {cq_msg_id}")
                                
                                try:
                                    save_telegram_feedback(cq_chat_id, cq_msg_id, action, cq_user.get("username"))
                                except Exception as e:
                                    log_telegram(f"Error saving feedback: {str(e)}")
                                    
                                status_text = "Liked 👍" if action == "like" else "Unliked 👎"
                                requests_post_json(f"https://api.telegram.org/bot{token}/editMessageReplyMarkup", {
                                    "chat_id": cq_chat_id,
                                    "message_id": cq_msg_id,
                                    "reply_markup": {
                                        "inline_keyboard": [[
                                            {"text": status_text, "callback_data": "done"}
                                        ]]
                                    }
                                }, timeout=5)
                                
                            elif cq_data.startswith("suggest_"):
                                question = cq_data[len("suggest_"):].strip()
                                log_telegram(f"[Host Telegram Bot] Suggestion clicked: '{question}' in chat {cq_chat_id}")
                                
                                placeholder_msg_id = 0
                                try:
                                    status_s, s_body = requests_post_json(url_send, {
                                        "chat_id": cq_chat_id,
                                        "text": "Thinking... 🧠 (Elapsed: 0s)\n\n[Ketik pesan baru untuk membatalkan/mengganti]"
                                    }, timeout=10)
                                    if status_s == 200:
                                        placeholder_msg_id = json.loads(s_body).get("result", {}).get("message_id", 0)
                                except Exception as s_err:
                                    log_telegram(f"Error sending suggestion placeholder: {str(s_err)}")
                                    
                                stop_event = threading.Event()
                                with sessions_lock:
                                    if cq_chat_id in active_sessions:
                                        active_sessions[cq_chat_id]["stop_event"].set()
                                    active_sessions[cq_chat_id] = {
                                        "stop_event": stop_event,
                                        "message_id": placeholder_msg_id
                                    }
                                    
                                t = threading.Thread(
                                    target=process_host_telegram_message,
                                    args=(token, AGENT_API_KEY, cq_chat_id, question, placeholder_msg_id, stop_event, cq_user),
                                    daemon=True
                                )
                                t.start()
                else:
                    log_telegram(f"[Host Telegram Bot] getUpdates 'ok' was False: {res_body}")
                    time.sleep(5)
            elif status == 401:
                log_telegram("[Host Telegram Bot] Unauthorized token. Verify TELEGRAM_BOT_TOKEN.")
                time.sleep(30)
            else:
                log_telegram(f"[Host Telegram Bot] HTTP {status} on getUpdates: {res_body}")
                time.sleep(5)
        except Exception as e:
            log_telegram(f"[Host Telegram Bot] Exception in main polling loop: {str(e)}")
            time.sleep(5)

# Standard HTTP Gateway Handlers
class HostExecutorHandler(BaseHTTPRequestHandler):
    def send_json(self, status_code: int, data: dict):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def check_auth(self) -> bool:
        auth_header = self.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return False
        key_provided = auth_header.split(" ", 1)[1].strip()
        return key_provided == AGENT_API_KEY

    def do_POST(self):
        global session_expiry
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if not self.check_auth():
            self.send_json(401, {"error": "Unauthorized. Invalid AGENT_API_KEY."})
            return

        if path == "/session":
            query_params = parse_qs(parsed_url.query)
            duration_str = query_params.get("duration", ["300"])[0]
            try:
                duration = int(duration_str)
            except ValueError:
                duration = 300

            duration = min(3600, max(10, duration))
            session_expiry = time.time() + duration
            expiry_dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_expiry))
            
            self.send_json(200, {
                "message": f"Host execution session active for {duration} seconds.",
                "expires_at": expiry_dt,
                "remaining_seconds": int(session_expiry - time.time())
            })
            return

        if path == "/execute":
            now = time.time()
            if now > session_expiry:
                self.send_json(403, {
                    "error": "Forbidden. No active host execution session or session has expired.",
                    "suggested": "Request a new host session using /session first."
                })
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body_data = self.rfile.read(content_length) if content_length > 0 else b""
            try:
                payload = json.loads(body_data.decode("utf-8")) if body_data else {}
            except Exception:
                payload = {}

            command = payload.get("command", "").strip()
            if not command:
                self.send_json(400, {"error": "Bad Request. Command field is required."})
                return

            print(f"Executing command on Windows host: {command}", flush=True)
            
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as dbg:
                dbg.write(f"--- Request Received at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                dbg.write(f"Command: {command}\n")

            try:
                res = subprocess.run(
                    command,
                    shell=True,
                    text=True,
                    capture_output=True,
                    timeout=60
                )
                self.send_json(200, {
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                    "exit_code": res.returncode,
                    "session_remaining_seconds": int(session_expiry - time.time())
                })
            except FileNotFoundError as fnf:
                try:
                    cmd_path = os.environ.get("COMSPEC", "C:\\Windows\\System32\\cmd.exe")
                    res = subprocess.run(
                        [cmd_path, "/c", command],
                        text=True,
                        capture_output=True,
                        timeout=60
                    )
                    self.send_json(200, {
                        "stdout": res.stdout,
                        "stderr": res.stderr,
                        "exit_code": res.returncode,
                        "session_remaining_seconds": int(session_expiry - time.time())
                    })
                except Exception as fallback_err:
                    self.send_json(500, {
                        "error": f"Command execution failed. Fallback failed: {str(fallback_err)}",
                        "session_remaining_seconds": int(session_expiry - time.time())
                    })
            except subprocess.TimeoutExpired:
                self.send_json(408, {
                    "error": "Request Timeout. Host command execution exceeded 60 seconds.",
                    "session_remaining_seconds": int(session_expiry - time.time())
                })
            except Exception as e:
                self.send_json(500, {
                    "error": f"Internal Server Error: {str(e)}",
                    "session_remaining_seconds": int(session_expiry - time.time())
                })
            return

        self.send_json(404, {"error": "Not Found."})

def run(port=5015):
    print(f"==========================================================")
    print(f" AgentAI Windows Host Executor Gateway Active")
    print(f" Listening on http://localhost:{port}")
    print(f" Security: Secured via AGENT_API_KEY")
    print(f"==========================================================")
    
    if TELEGRAM_BOT_TOKEN:
        t = threading.Thread(target=run_host_telegram_polling, daemon=True)
        t.start()
        print("[Host Telegram Bot] Background polling thread started on Windows host.", flush=True)

    server_address = ("", port)
    httpd = HTTPServer(server_address, HostExecutorHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Host Executor...")
        httpd.server_close()

if __name__ == "__main__":
    run()
