import os
import re
import time
import requests
import logging
import threading
from app.config import settings
from app.database import SessionLocal
from app.models import Message, ChatSession

logger = logging.getLogger("agent.telegram")

def format_instinct_for_telegram(text: str) -> str:
    """Detects <instinct>...</instinct> and formats it nicely for Telegram.
    
    If the markdown parser fails, Telegram will fall back to plain text, so
    we format it in a clean markdown block.
    """
    pattern = r"<instinct>(.*?)</instinct>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        instinct_content = match.group(1).strip()
        cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        formatted_instinct = f"💭 *Naluri & Nalar:*\n_{instinct_content}_\n\n"
        return formatted_instinct + cleaned_text
        
    # Check if only opening tag exists
    if "<instinct>" in text:
        parts = text.split("<instinct>", 1)
        cleaned_before = parts[0].strip()
        remaining = parts[1].strip()
        if remaining:
            formatted_instinct = f"💭 *Naluri & Nalar:*\n_{remaining}_\n\n"
            return (cleaned_before + "\n\n" + formatted_instinct).strip()
            
    return text


# Global dict to track active sessions per chat_id
# Format: { chat_id: { "stop_event": Event, "message_id": int } }
active_sessions = {}
sessions_lock = threading.Lock()

def get_chat_history(chat_id: str, limit: int = 15) -> list:
    """Retrieves the last few messages for a chat session from MySQL database."""
    db = SessionLocal()
    try:
        session_exists = db.query(ChatSession).filter(ChatSession.id == chat_id).first()
        if not session_exists:
            session = ChatSession(id=chat_id, title="Telegram Conversation")
            db.add(session)
            db.commit()
            
        messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at.asc()).all()
        
        history = []
        for msg in messages:
            if msg.role in ("user", "assistant"):
                history.append({
                    "role": msg.role,
                    "content": msg.content
                })
        return history[-limit:]
    except Exception as e:
        logger.error(f"Telegram Bot: Error loading history from DB: {str(e)}")
        return []
    finally:
        db.close()

def send_typing_periodically(token: str, chat_id: int, stop_event: threading.Event):
    """Sends 'typing' chat action every 4 seconds to show typing status in Telegram."""
    url_action = f"https://api.telegram.org/bot{token}/sendChatAction"
    while not stop_event.is_set():
        try:
            requests.post(url_action, json={"chat_id": chat_id, "action": "typing"}, timeout=5)
        except Exception:
            pass
        # Wait 4 seconds, checking stop_event frequently
        for _ in range(40):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def update_placeholder_timer(token: str, chat_id: int, message_id: int, stop_event: threading.Event):
    """Periodically updates the placeholder message text with a live elapsed timer."""
    url_edit = f"https://api.telegram.org/bot{token}/editMessageText"
    start_time = time.time()
    while not stop_event.is_set():
        for _ in range(30): # check every 100ms, update every 3 seconds
            if stop_event.is_set():
                break
            time.sleep(0.1)
        if stop_event.is_set():
            break
        elapsed = int(time.time() - start_time)
        text = f"Thinking... 🧠 (Elapsed: {elapsed}s)\n\n[Ketik pesan baru untuk membatalkan/mengganti]"
        try:
            requests.post(url_edit, json={"chat_id": chat_id, "message_id": message_id, "text": text}, timeout=5)
        except Exception:
            pass

def send_telegram_reply(token: str, chat_id: int, reply: str, message_id: int = 0):
    """Sends or edits a Telegram message, falling back to plain text if Markdown fails."""
    url_send = f"https://api.telegram.org/bot{token}/sendMessage"
    url_edit = f"https://api.telegram.org/bot{token}/editMessageText"
    
    if message_id:
        try:
            # Try editing with Markdown
            r = requests.post(url_edit, json={
                "chat_id": chat_id,
                "message_id": message_id,
                "text": reply,
                "parse_mode": "Markdown"
            }, timeout=10)
            if r.status_code == 200:
                return True
            # Retry editing without Markdown
            logger.warning(f"Telegram Bot: Markdown edit failed ({r.status_code}: {r.text}). Retrying plain text...")
            r = requests.post(url_edit, json={
                "chat_id": chat_id,
                "message_id": message_id,
                "text": reply
            }, timeout=10)
            if r.status_code == 200:
                return True
        except Exception as e:
            logger.error(f"Telegram Bot: Exception editing message: {str(e)}")

    # Fallback to sending a new message
    try:
        # Try sending with Markdown
        r = requests.post(url_send, json={
            "chat_id": chat_id,
            "text": reply,
            "parse_mode": "Markdown"
        }, timeout=10)
        if r.status_code == 200:
            return True
        # Retry sending without Markdown
        logger.warning(f"Telegram Bot: Markdown send failed ({r.status_code}: {r.text}). Retrying plain text...")
        r = requests.post(url_send, json={
            "chat_id": chat_id,
            "text": reply
        }, timeout=10)
        return r.status_code == 200
    except Exception as e:
        logger.error(f"Telegram Bot: Exception sending message: {str(e)}")
        return False

def process_message_thread(api_key: str, token: str, chat_id: int, text: str, placeholder_msg_id: int, stop_event: threading.Event, telegram_user: dict = None):
    """Thread function to process a single user request and update Telegram once finished."""
    completions_url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 1. Start typing action thread
    typing_thread = threading.Thread(
        target=send_typing_periodically,
        args=(token, chat_id, stop_event),
        daemon=True
    )
    typing_thread.start()
    
    # 2. Start timer updater thread
    if placeholder_msg_id:
        timer_thread = threading.Thread(
            target=update_placeholder_timer,
            args=(token, chat_id, placeholder_msg_id, stop_event),
            daemon=True
        )
        timer_thread.start()
        
    # 3. Load chat history from MySQL
    history = get_chat_history(chat_id=f"telegram-{chat_id}")
    history.append({"role": "user", "content": text})
    
    chat_payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": history,
        "stream": False,
        "user": f"telegram-{chat_id}"
    }
    if telegram_user:
        chat_payload["telegram_user"] = telegram_user
        
    reply = ""
    # Check stop event before starting API request
    if not stop_event.is_set():
        try:
            comp_res = requests.post(completions_url, headers=headers, json=chat_payload, timeout=300)
            if comp_res.status_code == 200:
                reply = comp_res.json()["choices"][0]["message"]["content"]
            else:
                reply = f"Error: Received status {comp_res.status_code} from completions server."
                logger.error(f"Telegram Bot: API error: {comp_res.text}")
        except Exception as api_err:
            reply = f"Error: Failed to communicate with completions engine."
            logger.error(f"Telegram Bot: Connection error: {str(api_err)}")
            
    # Check stop event again after API request
    if not stop_event.is_set() and reply:
        # Stop typing and timer threads
        stop_event.set()
        
        # Format monologue beautifully
        formatted_reply = format_instinct_for_telegram(reply)
        
        # Send/edit final reply to Telegram using robust fallback helper
        send_telegram_reply(token, chat_id, formatted_reply, placeholder_msg_id)
            
    # Clean up global session registry
    with sessions_lock:
        if chat_id in active_sessions and active_sessions[chat_id]["stop_event"] == stop_event:
            del active_sessions[chat_id]

def run_telegram_bot(api_key: str):
    """Long polling loop for Telegram Bot."""
    token = settings.TELEGRAM_BOT_TOKEN
    if not token:
        logger.info("Telegram Bot: TELEGRAM_BOT_TOKEN environment variable not set. Bot is disabled.")
        return

    logger.info("Telegram Bot: Initializing long-polling connection...")
    
    url_get = f"https://api.telegram.org/bot{token}/getUpdates"
    url_send = f"https://api.telegram.org/bot{token}/sendMessage"
    url_edit = f"https://api.telegram.org/bot{token}/editMessageText"

    offset = 0
    
    while True:
        try:
            payload = {"offset": offset, "timeout": 15}
            r = requests.post(url_get, json=payload, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if data.get("ok"):
                    updates = data.get("result", [])
                    for update in updates:
                        update_id = update.get("update_id")
                        offset = update_id + 1
                        
                        message = update.get("message")
                        if not message:
                            continue
                            
                        chat = message.get("chat", {})
                        chat_id = chat.get("id")
                        text = message.get("text", "")
                        
                        if not text or not chat_id:
                            continue
                            
                        logger.info(f"Telegram Bot: Received text from chat {chat_id}: '{text[:40]}'")
                        
                        # 1. Handle user interrupt/cancel previous active thread
                        with sessions_lock:
                            if chat_id in active_sessions:
                                logger.info(f"Telegram Bot: Interrupting active session for chat {chat_id}")
                                active_sessions[chat_id]["stop_event"].set()
                                
                                old_msg_id = active_sessions[chat_id]["message_id"]
                                try:
                                    requests.post(url_edit, json={
                                        "chat_id": chat_id,
                                        "message_id": old_msg_id,
                                        "text": "🛑 Pertanyaan sebelumnya dibatalkan. Memproses pertanyaan baru... ⏳"
                                    }, timeout=5)
                                except Exception:
                                    pass
                                    
                                del active_sessions[chat_id]
                        
                        # 2. Send initial placeholder message
                        placeholder_msg_id = None
                        send_payload = {
                            "chat_id": chat_id,
                            "text": "Thinking... 🧠 (Elapsed: 0s)\n\n[Ketik pesan baru untuk membatalkan/mengganti]"
                        }
                        try:
                            send_res = requests.post(url_send, json=send_payload, timeout=10)
                            if send_res.status_code == 200:
                                placeholder_msg_id = send_res.json().get("result", {}).get("message_id")
                        except Exception as send_err:
                            logger.error(f"Telegram Bot: Network error sending placeholder: {str(send_err)}")

                        if not placeholder_msg_id:
                            placeholder_msg_id = 0
                            
                        # 3. Extract user metadata
                        from_user = message.get("from", {})
                        telegram_user = {
                            "first_name": from_user.get("first_name", "User"),
                            "last_name": from_user.get("last_name", ""),
                            "username": from_user.get("username", "")
                        }

                        # 4. Register new session and spawn processor thread
                        stop_event = threading.Event()
                        with sessions_lock:
                            active_sessions[chat_id] = {
                                "stop_event": stop_event,
                                "message_id": placeholder_msg_id
                            }
                            
                        t = threading.Thread(
                            target=process_message_thread, 
                            args=(api_key, token, chat_id, text, placeholder_msg_id, stop_event, telegram_user),
                            daemon=True
                        )
                        t.start()
                else:
                    logger.error(f"Telegram Bot: getUpdates 'ok' was False: {r.text}")
                    time.sleep(5)
            elif r.status_code == 401:
                logger.error("Telegram Bot: Unauthorized token. Verify TELEGRAM_BOT_TOKEN.")
                time.sleep(30)
            else:
                logger.error(f"Telegram Bot: HTTP {r.status_code} on getUpdates: {r.text}")
                time.sleep(5)
        except Exception as e:
            logger.error(f"Telegram Bot: Exception in main polling loop: {str(e)}")
            time.sleep(5)

def start_telegram_bot(api_key: str):
    """Starts the Telegram Bot in a background thread."""
    t = threading.Thread(target=run_telegram_bot, args=(api_key,), daemon=True)
    t.start()
    logger.info("Telegram Bot background thread spawned.")
