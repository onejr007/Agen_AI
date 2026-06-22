import sys
import os
import json

os.environ["OLLAMA_HOST"] = "http://localhost:11434"
os.environ["DATABASE_URL"] = "sqlite:///./test_cache.db"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.agent import get_embedding
from app.database import SessionLocal, init_db_with_retry
from app.models import Message
from app.main import chat_completions

print("\n--- Testing Semantic Query Caching ---")

init_db_with_retry()
db = SessionLocal()

# Get embedding for a specific question
question = "How to configure JWT authentication in FastAPI?"
vector = get_embedding(question)

chat_id = "test-cache-chat-999"

try:
    # Insert user message
    user_msg = Message(
        chat_id=chat_id,
        role="user",
        content=question,
        embedding=json.dumps(vector) if vector else None
    )
    db.add(user_msg)
    db.commit()

    # Insert assistant message
    cached_answer = "To configure JWT in FastAPI, use PyJWT and fastapi.security.OAuth2PasswordBearer."
    assistant_msg = Message(
        chat_id=chat_id,
        role="assistant",
        content=cached_answer,
        embedding=None
    )
    db.add(assistant_msg)
    db.commit()

    payload = {
        "model": "qwen2.5-coder:1.5b",
        "messages": [
            {"role": "user", "content": question}
        ],
        "stream": False
    }

    # Second user comes along and asks the same exact question
    response = chat_completions(payload, db=db)
    content = response.get("choices", [])[0].get("message", {}).get("content", "")
    print(f"Original Question: {question}")
    print(f"Cached Response: {content}")
    assert content == cached_answer, f"Response did not match cached answer! Got: {content}"
    print("Semantic Query Caching: PASS\n")

except Exception as e:
    print(f"Semantic Query Caching: FAIL ({str(e)})\n")
finally:
    # Cleanup
    db.query(Message).filter(Message.chat_id == chat_id).delete()
    db.commit()
    db.close()
