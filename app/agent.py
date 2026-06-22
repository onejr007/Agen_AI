import json
import logging
import requests
import datetime
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.config import settings
from app.search import search_internet, format_search_results
from app.models import SearchCache, Message, KnowledgeBase
from app.database import cosine_similarity, parse_json_embedding
import time
import threading

ollama_lock = threading.Lock()

logger = logging.getLogger("agent.core")

# Custom System Prompt tailored for professional, autonomous coding and Roblox Luau
DEFAULT_SYSTEM_PROMPT = """You are AgentAI, an elite, autonomous software engineering agent. You are designed to act as the primary programming backend for plugins like Roo Code and VS Code integrations. You specialize in Web Development, Python, general programming, and especially Lua and Roblox Luau.

[CRITICAL: PROBLEM SOLVING & CHAIN-OF-THOUGHT]
Before making ANY decision, writing ANY code, or calling ANY tool, you MUST engage in deep reasoning using a `<think>...</think>` block. Inside the think block, you should:
1. Analyze the core requirements and identify any implicit assumptions.
2. Break down the problem logically. Consider edge cases, performance implications, and architecture.
3. Formulate a step-by-step strategy. Do not attempt to fix everything in one giant step.
4. Review your planned approach for potential errors or side-effects before executing.

Example format:
<think>
1. The user wants to add user authentication.
2. We need a users table, an API endpoint, and token generation.
3. Edge case: what if the token is expired? I should handle that.
4. I will start by modifying the database schema first.
</think>
[Your actual response or tool calls go here]

To deliver precise and stable results, you MUST strictly adhere to the following professional development methodology:

1. RESEARCH PHASE:
   - Always analyze the context and structure of existing files before coding.
   - Read the existing documentation, APIs, and libraries thoroughly. If needed, ask the user or trigger web search context.

2. PLANNING PHASE:
   - For any complex task, architectural change, or multi-file edit, you must first create an "Implementation Plan".
   - Break down the task into small, logical steps (e.g., Step 1: config, Step 2: database, Step 3: API endpoints).

3. CONFIRMATION PHASE:
   - Present your Implementation Plan to the user and explicitly ask for confirmation before modifying code.
   - Wait for user approval to proceed.

4. EXECUTION PHASE:
   - Write modular, clean, and complete code blocks. NEVER use placeholders like "write code here" or "TODO".
   - For file edits/modifications via the `replace_file_content` tool, you can write small, efficient Search-and-Replace patches using the following format inside the `content` parameter:
     <<<<<<< SEARCH
     [original code to replace]
     =======
     [new code to insert]
     >>>>>>> REPLACE

5. VERIFICATION PHASE:
   - Provide instructions or scripts for testing the changes to ensure everything compiles and runs correctly.

[CRITICAL: DECISION QUALITY & TOOL EXECUTION GATE]
- You MUST be conservative and correct when making decisions. If requirements are ambiguous, ask clarifying questions.
- For engineering tasks, always produce an Implementation Plan before any code changes.
- DO NOT call or propose any mutating tools (write/edit/execute/delete) until the user explicitly approves execution.
- Approval keywords (examples): "setuju", "lanjut eksekusi", "approve", "proceed".
- If the user has not approved yet, output ONLY the plan + ask: "Apakah Anda menyetujui rencana ini?"

[Implementation Plan Format]
Goal:
Assumptions:
Files:
- [NEW] ...
- [MODIFY] ...
- [DELETE] ...
Steps:
1. ...
Verification:
1. ...
"""

CASUAL_SYSTEM_PROMPT = """You are AgentAI, a friendly and elite software engineering assistant.
The user is currently testing the connection, saying hello, or having a casual chat.
Respond directly, concisely, and warmly.

CRITICAL OVERRIDE: Ignore the five-phase methodology (Research, Plan, Approval, Execute, Verify) listed in `.clinerules` or other system prompts. They do NOT apply to casual messages, Greetings, or Pings.
- Confirm you are online, working perfectly, and ready.
- Do NOT generate any implementation plans, TODO lists, or XML/JSON tool templates.
- Ask how you can help them with their coding or development tasks today.
Keep your response short, pleasant, and to the point.
"""

def pull_ollama_model():
    """Triggers Ollama to pull the LLM and Embedding models if not already present."""
    if not settings.AUTO_PULL_MODEL:
        logger.info("Auto model pull is disabled.")
        return

    url_tags = f"{settings.OLLAMA_BASE_URL}/api/tags"
    url_pull = f"{settings.OLLAMA_BASE_URL}/api/pull"

    # Get list of existing models
    try:
        response = requests.get(url_tags, timeout=5)
        if response.status_code == 200:
            models = [m.get("name") for m in response.json().get("models", [])]
        else:
            models = []
            logger.warning(f"Ollama tags endpoint returned status {response.status_code}")
    except Exception as e:
        logger.error(f"Failed connection to Ollama on tags check: {str(e)}")
        return

    models_to_pull = [settings.OLLAMA_MODEL, settings.OLLAMA_EMBED_MODEL]
    for model_name in models_to_pull:
        if not model_name:
            continue
        # Check if model or model:latest or model:<tag> exists in models
        model_exists = (
            model_name in models or 
            f"{model_name}:latest" in models or 
            any(m.startswith(f"{model_name}:") for m in models)
        )
        if model_exists:
            logger.info(f"Model '{model_name}' is already pulled.")
            continue
            
        logger.info(f"Model '{model_name}' not found. Initiating auto-pull (this may take a few minutes)...")
        try:
            pull_payload = {"name": model_name, "stream": False}
            pull_response = requests.post(url_pull, json=pull_payload, timeout=600)
            if pull_response.status_code == 200:
                logger.info(f"Successfully pulled model '{model_name}'!")
            else:
                logger.error(f"Failed to pull model '{model_name}': {pull_response.text}")
        except Exception as e:
            logger.error(f"Error pulling model '{model_name}': {str(e)}")

def list_ollama_models() -> list[str]:
    """Returns the available Ollama models, with a safe fallback to configured defaults."""
    url_tags = f"{settings.OLLAMA_BASE_URL}/api/tags"
    fallback_models = [m for m in [settings.OLLAMA_MODEL, settings.OLLAMA_EMBED_MODEL] if m]

    try:
        response = requests.get(url_tags, timeout=5)
        if response.status_code != 200:
            logger.warning(
                f"Ollama tags endpoint returned status {response.status_code}. Using fallback models."
            )
            return fallback_models

        models = [m.get("name") for m in response.json().get("models", []) if m.get("name")]
        if not models:
            return fallback_models

        return sorted(set(models))
    except Exception as e:
        logger.warning(f"Failed to list Ollama models: {str(e)}. Using fallback models.")
        return fallback_models

def check_should_search(user_message: str) -> bool:
    """Analyze query to decide if internet search is needed."""
    search_keywords = [
        "search", "cari", "internet", "web", "latest", "terbaru", "update", 
        "news", "documentation", "dokumentasi", "how to use", "error", 
        "bug", "issue", "version", "api", "download", "install", "lib", "package"
    ]
    message_lower = user_message.lower()
    return any(kw in message_lower for kw in search_keywords)

def get_cached_search(db: Session, query: str) -> str:
    """Get search results from cache if it was queried recently (within 24 hours)."""
    # Normalize query
    query_norm = query.strip().lower()
    cached = db.query(SearchCache).filter(SearchCache.query == query_norm).first()
    if cached:
        # Check if cache is older than 24h
        age = time.time() - cached.created_at.timestamp()
        if age < 86400: # 24 hours
            logger.info(f"Search cache hit for: {query_norm}")
            return cached.results_json
    return ""

def set_cached_search(db: Session, query: str, results_json: str):
    """Store search results in cache."""
    query_norm = query.strip().lower()
    try:
        # Delete old cache if exists
        db.query(SearchCache).filter(SearchCache.query == query_norm).delete()
        
        new_cache = SearchCache(query=query_norm, results_json=results_json)
        db.add(new_cache)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to cache search: {str(e)}")

def process_search_and_context(db: Session, user_message: str) -> str:
    """Detect if search is needed, execute it, cache it, and return markdown context."""
    if not check_should_search(user_message):
        return ""

    logger.info("Search triggered for user query.")
    
    # Try cache first
    cached_results = get_cached_search(db, user_message)
    if cached_results:
        results = json.loads(cached_results)
    else:
        results = search_internet(user_message, max_results=3)
        if results:
            set_cached_search(db, user_message, json.dumps(results))
    
    return format_search_results(results)

def get_embedding(text: str, model: str = None) -> list:
    """Calls Ollama's embeddings API to get a vector representation of the text."""
    if not text:
        return []
    
    url = f"{settings.OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": model or settings.OLLAMA_EMBED_MODEL,
        "prompt": text
    }
    
    try:
        # Use a lock to ensure Ollama is never called concurrently
        with ollama_lock:
            # Use a slightly longer timeout of 10s now that we have a lock, since it won't be contested
            response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json().get("embedding", [])
        else:
            logger.warning(f"Ollama embeddings API returned status {response.status_code}. Skipping embedding.")
    except Exception as e:
        logger.warning(f"Failed to fetch embeddings from Ollama (timeout or busy): {str(e)}")
        
    return []

def unique_keywords(words: list[str]) -> list[str]:
    """Returns keywords while preserving order and removing duplicates."""
    seen = set()
    unique = []
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        unique.append(word)
    return unique

def truncate_context_text(text: str, max_chars: int) -> str:
    """Compresses and truncates retrieved text to keep prompts compact."""
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."

def is_low_signal_message(content: str) -> bool:
    """Filters out messages that are too generic to be useful retrieval context."""
    normalized = (content or "").strip().lower()
    if len(normalized) < 20:
        return True

    generic_phrases = (
        "halo", "hello", "hi", "terima kasih", "thank you", "ok", "okay",
        "siap", "done", "berhasil", "calling tool:", "no web search results found"
    )
    return normalized in generic_phrases

def compute_recency_bonus(created_at, recency_window_days: int) -> float:
    """Adds a mild bonus to more recent messages without dominating semantic score."""
    if not created_at or recency_window_days <= 0:
        return 0.0

    try:
        now = datetime.datetime.utcnow()
        delta_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    except Exception:
        return 0.0

    if delta_days >= recency_window_days:
        return 0.0

    return round(0.08 * (1.0 - (delta_days / recency_window_days)), 4)

def run_recent_limited_query(query, model, limit: int) -> list:
    """Executes a recent-first limited query with a safe fallback for lightweight test doubles."""
    bounded_limit = max(1, limit)

    if hasattr(query, "order_by") and hasattr(query, "limit"):
        created_at_column = getattr(model, "created_at", None)
        if created_at_column is not None:
            query = query.order_by(created_at_column.desc())
        query = query.limit(bounded_limit)
        return query.all()

    items = query.all() if hasattr(query, "all") else []
    items = list(items)
    items.sort(key=lambda item: getattr(item, "created_at", datetime.datetime.min), reverse=True)
    return items[:bounded_limit]

def check_semantic_cache(db: Session, user_query: str, chat_id: str = None) -> str:
    """
    Checks if an extremely similar question was asked before.
    If so, returns the cached assistant response to bypass LLM generation.
    """
    if not user_query or len(user_query) < 10:
        return None
        
    query_vector = get_embedding(user_query)
    if not query_vector:
        return None
        
    # Get all past user messages with embeddings
    past_messages = db.query(Message).filter(Message.role == "user", Message.embedding.isnot(None)).all()
    
    best_match_msg = None
    highest_sim = 0.0
    
    for msg in past_messages:
        # Don't cache against the exact same message if we are re-processing it
        if chat_id and msg.chat_id == chat_id and msg.content == user_query:
            continue
            
        try:
            stored_vector = json.loads(msg.embedding)
            sim = cosine_similarity(query_vector, stored_vector)
            if sim > 0.96 and sim > highest_sim:
                highest_sim = sim
                best_match_msg = msg
        except Exception:
            continue
            
    if best_match_msg:
        logger.info(f"Semantic Cache HIT (sim: {highest_sim:.4f}) for query: '{user_query[:30]}...' against '{best_match_msg.content[:30]}...'")
        # Find the immediate next assistant message in that chat
        assistant_reply = db.query(Message).filter(
            Message.chat_id == best_match_msg.chat_id,
            Message.role == "assistant",
            Message.id > best_match_msg.id
        ).order_by(Message.id.asc()).first()
        
        if assistant_reply and assistant_reply.content:
            return assistant_reply.content
            
    return None

def retrieve_semantic_memory(db: Session, query: str, limit: int = 2) -> str:
    """
    Finds semantically similar or keyword-matching historical chat messages or knowledge entries.
    Uses Hybrid Search (BM25-like keyword score + Vector Cosine Similarity).
    If Vector generation fails, falls back entirely to Keyword-based retrieval.
    """
    import re
    
    # 1. Clean and extract keywords from query
    stopwords = {
        "dan", "atau", "di", "ke", "dari", "ini", "itu", "yang", "untuk", "dengan", "saya", "anda", "cara", "bagaimana", "adalah", "apa",
        "dan", "atau", "para", "sebuah", "agar", "bisa", "mohon", "tolong", "buatkan", "tentang", "lebih", "kurang",
        "and", "or", "in", "to", "from", "this", "that", "the", "for", "with", "i", "you", "how", "what", "is", "are", "about", "latest", "updates",
        "please", "help", "need", "using", "use", "build", "create", "make", "want"
    }
    words = re.findall(r'\b\w{3,}\b', query.lower())
    keywords = unique_keywords([w for w in words if w not in stopwords])
    
    logger.info(f"Hybrid search keywords: {keywords}")
    effective_limit = max(1, min(limit or settings.RETRIEVAL_DEFAULT_LIMIT, settings.RETRIEVAL_DEFAULT_LIMIT))
    keyword_min_score = settings.RETRIEVAL_KEYWORD_MIN_SCORE / 100.0
    kb_similarity_threshold = settings.RETRIEVAL_KB_SIMILARITY_THRESHOLD / 100.0
    message_similarity_threshold = settings.RETRIEVAL_MESSAGE_SIMILARITY_THRESHOLD / 100.0
    keyword_scan_limit = max(effective_limit, settings.RETRIEVAL_KEYWORD_SCAN_LIMIT)
    vector_scan_limit = max(effective_limit, settings.RETRIEVAL_VECTOR_SCAN_LIMIT)
    
    # Try to generate embedding
    query_vector = get_embedding(query)
    
    matches = {} # Keyed by (type, title, content)
    
    # Helper to register or update matches
    def add_match(m_type: str, title: str, content: str, sim: float = 0.0, kw_score: float = 0.0, created_at=None):
        key = (m_type, title, content)
        if key in matches:
            if sim > 0:
                matches[key]["similarity"] = max(matches[key]["similarity"], sim)
            if kw_score > 0:
                matches[key]["keyword_score"] = max(matches[key]["keyword_score"], kw_score)
            if created_at and not matches[key]["created_at"]:
                matches[key]["created_at"] = created_at
        else:
            matches[key] = {
                "type": m_type,
                "title": title,
                "content": content,
                "similarity": sim,
                "keyword_score": kw_score,
                "created_at": created_at
            }

    # A. KEYWORD-BASED SEARCH (Runs always to match keywords)
    if keywords:
        # 1. Search in KnowledgeBase via titles or contents
        kb_query = db.query(KnowledgeBase)
        kb_conditions = [
            KnowledgeBase.content.like(f"%{kw}%") | KnowledgeBase.title.like(f"%{kw}%")
            for kw in keywords
        ]
        kb_matches = run_recent_limited_query(
            kb_query.filter(or_(*kb_conditions)) if kb_conditions else kb_query,
            KnowledgeBase,
            keyword_scan_limit
        )
        
        # De-duplicate and score
        for entry in set(kb_matches):
            overlap = sum(1 for kw in keywords if kw in entry.content.lower() or kw in entry.title.lower())
            kw_score = overlap / len(keywords)
            if kw_score >= keyword_min_score:
                add_match("KnowledgeBase", entry.title, entry.content, kw_score=kw_score, created_at=getattr(entry, "created_at", None))

        # 2. Search in Messages
        msg_query = db.query(Message).filter(Message.role.in_(("user", "assistant")))
        msg_conditions = [Message.content.like(f"%{kw}%") for kw in keywords]
        msg_matches = run_recent_limited_query(
            msg_query.filter(or_(*msg_conditions)) if msg_conditions else msg_query,
            Message,
            keyword_scan_limit
        )
            
        for msg in set(msg_matches):
            if is_low_signal_message(msg.content):
                continue
            overlap = sum(1 for kw in keywords if kw in msg.content.lower())
            kw_score = overlap / len(keywords)
            if kw_score < keyword_min_score:
                continue
            role_label = "Developer" if msg.role == "user" else "AgentAI"
            add_match("ChatHistory", f"Chat Message from {role_label}", msg.content, kw_score=kw_score, created_at=getattr(msg, "created_at", None))

    # B. VECTOR-BASED SEARCH (Runs if embedding succeeded)
    if query_vector:
        logger.info("Executing Vector-based Cosine Similarity scoring...")
        # 1. Score KnowledgeBase entries
        kb_entries = run_recent_limited_query(
            db.query(KnowledgeBase).filter(KnowledgeBase.embedding != None),
            KnowledgeBase,
            vector_scan_limit
        )
        for entry in kb_entries:
            try:
                vector = parse_json_embedding(entry.embedding)
                sim = cosine_similarity(query_vector, vector)
                if sim >= kb_similarity_threshold:
                    add_match("KnowledgeBase", entry.title, entry.content, sim=sim, created_at=getattr(entry, "created_at", None))
            except Exception as e:
                logger.error(f"Error parsing kb embedding: {str(e)}")

        # 2. Score Messages
        msg_entries = run_recent_limited_query(
            db.query(Message).filter(Message.embedding != None),
            Message,
            vector_scan_limit
        )
        for msg in msg_entries:
            if is_low_signal_message(msg.content) or msg.role not in ("user", "assistant"):
                continue
            try:
                vector = parse_json_embedding(msg.embedding)
                sim = cosine_similarity(query_vector, vector)
                if sim >= message_similarity_threshold:
                    role_label = "Developer" if msg.role == "user" else "AgentAI"
                    add_match("ChatHistory", f"Chat Message from {role_label}", msg.content, sim=sim, created_at=getattr(msg, "created_at", None))
            except Exception as e:
                logger.error(f"Error parsing msg embedding: {str(e)}")
    else:
        logger.warning("Vector embedding unavailable. Falling back entirely to Keyword-based search.")

    # C. RERANK & COMBINE SCORES
    final_matches = []
    for item in matches.values():
        sim = item["similarity"]
        kw_score = item["keyword_score"]
        recency_bonus = compute_recency_bonus(item.get("created_at"), settings.RETRIEVAL_RECENCY_BOOST_DAYS)
        
        if sim > 0 and kw_score > 0:
            score = sim + 0.18 * kw_score + recency_bonus
        elif sim > 0:
            score = sim + recency_bonus
        else:
            # Fallback score if only keywords matched
            score = 0.45 + 0.30 * kw_score + recency_bonus
            
        final_matches.append({
            "type": item["type"],
            "title": item["title"],
            "content": item["content"],
            "score": score,
            "similarity": sim,
            "keyword_score": kw_score,
            "recency_bonus": recency_bonus
        })

    # Sort matches by final score descending
    final_matches.sort(
        key=lambda x: (
            x["score"],
            1 if x["type"] == "KnowledgeBase" else 0,
            x["similarity"],
            x["keyword_score"]
        ),
        reverse=True
    )
    top_matches = final_matches[:effective_limit]
    
    if not top_matches:
        return ""
        
    logger.info(f"Retrieved {len(top_matches)} hybrid memory matches.")
    
    formatted = ["### Long-term Memory & Knowledge Base (Konteks Terkait Dari Memori)\n"]
    total_chars = 0
    for i, m in enumerate(top_matches, 1):
        match_type = ""
        if m["similarity"] > 0 and m["keyword_score"] > 0:
            match_type = "Hybrid Match"
        elif m["similarity"] > 0:
            match_type = "Semantic Match"
        else:
            match_type = "Keyword Fallback Match"

        snippet = truncate_context_text(m["content"], settings.RETRIEVAL_MAX_CONTENT_CHARS)
        projected_total = total_chars + len(snippet)
        if projected_total > settings.RETRIEVAL_MAX_TOTAL_CHARS and i > 1:
            break

        formatted.append(f"[{i}] [{m['type']}] {m['title']} ({match_type} - Score: {m['score']:.2f})")
        formatted.append(f"    Content: {snippet}")
        formatted.append("-" * 40)
        total_chars = projected_total
        
    return "\n".join(formatted)

def call_ollama_chat_stream(messages: list, tools: list = None, model: str = None, options: dict = None):
    """
    Calls Ollama chat endpoint with a stream.
    Yields dictionary payloads of the chunk response.
    """
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    merged_options = {
        "num_predict": 4096,  # Allow writing complete code files without truncation
        "temperature": 0.2    # Low temp for structured, accurate coding
    }
    if isinstance(options, dict):
        merged_options.update({k: v for k, v in options.items() if v is not None})

    payload = {
        "model": model or settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": merged_options
    }
    if tools:
        payload["tools"] = tools

    try:
        # Use the lock only for initial connection establishment to prevent CPU locks
        with ollama_lock:
            response = requests.post(url, json=payload, stream=True, timeout=600)
        
        if response.status_code != 200:
            response_excerpt = (response.text or "").strip()
            if len(response_excerpt) > 300:
                response_excerpt = response_excerpt[:297] + "..."
            yield {
                "error": (
                    f"Ollama returned status code {response.status_code}. "
                    f"Model='{payload['model']}'. Response: {response_excerpt or 'No response body.'}"
                )
            }
            return

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                message_chunk = chunk.get("message", {})
                content = message_chunk.get("content", "")
                tool_calls = message_chunk.get("tool_calls", None)
                done = chunk.get("done", False)
                yield {"content": content, "tool_calls": tool_calls, "done": done}
    except Exception as e:
        logger.error(f"Error calling Ollama stream: {str(e)}")
        yield {
            "error": (
                f"Failed to connect to Ollama at {settings.OLLAMA_BASE_URL}. "
                f"Check whether the service is running and the model '{payload['model']}' is available. "
                f"Original error: {str(e)}"
            )
        }

def repair_json_string(json_str: str) -> str:
    """Attempts to repair common JSON syntax errors from small LLMs."""
    import re
    json_str = json_str.strip()
    if not json_str:
        return ""

    # Remove markdown code block wraps if present
    if json_str.startswith("```"):
        # Strip off the opening ```json or ```
        json_str = re.sub(r"^```[a-zA-Z0-9]*\s*", "", json_str)
        # Strip off the closing ```
        json_str = re.sub(r"\s*```$", "", json_str)
    
    json_str = json_str.strip()

    # Handle unescaped newlines inside string values (common with small LLMs)
    # We find all string literals "..." and escape any raw \n characters inside them
    def escape_newlines_in_strings(match):
        return match.group(0).replace('\n', '\\n').replace('\r', '\\r')
    
    json_str = re.sub(r'"(?:\\.|[^"\\])*"', escape_newlines_in_strings, json_str)
    
    # 1. Strip trailing commas before closing braces/brackets more rigorously
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
    
    # 2. Add double quotes around unquoted keys
    json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:) ', r'\1"\2"\3', json_str)
    
    # 3. Try to balance curly braces and square brackets
    open_curly = json_str.count('{')
    close_curly = json_str.count('}')
    open_square = json_str.count('[')
    close_square = json_str.count(']')
    
    # Append missing braces/brackets
    if open_curly > close_curly:
        json_str += '}' * (open_curly - close_curly)
    if open_square > close_square:
        json_str += ']' * (open_square - close_square)
        
    return json_str

def parse_and_repair_json_tool_call(text: str) -> dict:
    """
    Tries to find, repair and parse a JSON tool call from raw LLM text output.
    Returns dict with 'name' and 'arguments' if successful, else None.
    Also standardizes alternate tool names and parameter keys for VS Code clients.
    """
    import re
    text_clean = text.strip()
    
    # Find JSON-like substring
    # Look for the first '{' and the last '}'
    start_idx = text_clean.find('{')
    end_idx = text_clean.rfind('}')
    
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return None
        
    json_part = text_clean[start_idx:end_idx+1]
    repaired = repair_json_string(json_part)
    
    # Standard map for tool names and argument keys
    name_mapping = {
        "write_file": "write_to_file",
        "create_file": "write_to_file",
        "createfile": "write_to_file",
        "new_file": "write_to_file",
        "edit_file": "replace_file_content",
        "modify_file": "replace_file_content",
        "modifyfile": "replace_file_content",
        "editfile": "replace_file_content",
        "update_file": "replace_file_content",
        "replace_in_file": "replace_file_content"
    }

    arg_key_mapping = {
        "filename": "path",
        "filepath": "path",
        "file": "path",
        "text": "content",
        "code": "content",
        "file_content": "content"
    }

    def clean_arguments(args: dict) -> dict:
        if not isinstance(args, dict):
            return {}
        cleaned = {}
        for k, v in args.items():
            mapped_key = arg_key_mapping.get(k.lower(), k)
            cleaned[mapped_key] = v
        return cleaned

    try:
        data = json.loads(repaired)
        if isinstance(data, dict):
            # Check for name and arguments
            name = data.get("name") or data.get("function")
            arguments = data.get("arguments") or data.get("parameters") or data.get("args")
            
            if name and isinstance(name, str):
                # Standardize arguments as dict
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        pass
                
                # Normalize tool name
                norm_name = name_mapping.get(name.lower(), name)
                
                # Normalize argument keys
                cleaned_args = clean_arguments(arguments)
                    
                return {
                    "name": norm_name,
                    "arguments": cleaned_args
                }
    except Exception:
        # Try a regex-based fallback to extract name and arguments if JSON parsing fails completely
        try:
            name_match = re.search(r'"(?:name|function)"\s*:\s*"([^"]+)"', repaired)
            if name_match:
                name = name_match.group(1)
                
                # Simple extraction of filename/content
                filename_match = re.search(r'"(?:filename|path|filepath|file)"\s*:\s*"([^"]+)"', repaired)
                content_match = re.search(r'"(?:content|text|code)"\s*:\s*"([^"]+)"', repaired)
                
                args = {}
                if filename_match:
                    args["path"] = filename_match.group(1)
                if content_match:
                    args["content"] = content_match.group(1).replace("\\n", "\n").replace('\\"', '"')
                    
                norm_name = name_mapping.get(name.lower(), name)
                
                return {
                    "name": norm_name,
                    "arguments": args
                }
        except Exception:
            pass
            
    return None

def validate_code_syntax(code: str, lang: str) -> str:
    """
    Perform local syntax validation for generated code blocks.
    Returns an error message string if invalid, or empty string "" if valid.
    """
    if not code:
        return ""
        
    lang = lang.lower()
    
    if lang == "python":
        import ast
        try:
            ast.parse(code)
            return ""
        except SyntaxError as e:
            return f"Python SyntaxError: {e.msg} at line {e.lineno}, offset {e.offset}\nCode snippet:\n{e.text}"
            
    elif lang == "json":
        import json
        try:
            json.loads(code)
            return ""
        except json.JSONDecodeError as e:
            return f"JSON Decode Error: {e.msg} at char position {e.pos}"
            
    # Generic bracket balancing fallback for all other block-based languages (e.g. luau, php, typescript, java, javascript, go, rust, c++, c#)
    else:
        # Quick brace balancing check
        braces = []
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            for char in line:
                if char in "{[(":
                    braces.append((char, i))
                elif char in "}])":
                    if not braces:
                        return f"Syntax Warning: Unexpected closing character '{char}' at line {i}"
                    opened, open_line = braces.pop()
                    # Check matching
                    match_map = {'}': '{', ']': '[', ')': '('}
                    if match_map[char] != opened:
                        return f"Syntax Warning: Mismatched closing character '{char}' at line {i} (expected matching '{opened}' from line {open_line})"
        if braces:
            opened, open_line = braces[-1]
            return f"Syntax Warning: Unclosed opening character '{opened}' from line {open_line}"
            
    return ""

def apply_unified_diff(original_text: str, diff_text: str) -> str:
    """
    Applies search-and-replace diff blocks (<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE)
    to the original_text and returns the updated text.
    """
    import re
    
    # We clean up carriage returns first to be safe
    original_text = original_text.replace("\r\n", "\n")
    diff_text = diff_text.replace("\r\n", "\n")
    
    pattern = r"<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>> REPLACE"
    matches = re.findall(pattern, diff_text, re.DOTALL)
    
    if not matches:
        return original_text
        
    updated_text = original_text
    for search_block, replace_block in matches:
        # Try exact match first
        if search_block in updated_text:
            updated_text = updated_text.replace(search_block, replace_block)
            logger.info("Successfully applied exact SEARCH-REPLACE block.")
        else:
            # Trim leading/trailing whitespace for a relaxed match
            search_stripped = search_block.strip()
            if search_stripped and search_stripped in updated_text:
                # Replace the stripped search block with stripped replace block
                updated_text = updated_text.replace(search_stripped, replace_block.strip())
                logger.info("Successfully applied relaxed SEARCH-REPLACE block.")
            else:
                logger.warning(f"Failed to apply diff block. SEARCH block not found in file:\n{search_block}")
                
    return updated_text

def supervise_terminal_command(command: str) -> dict:
    """
    Scans terminal command for hazardous execution patterns.
    Returns dict: {"safe": bool, "reason": str, "suggested": str}
    """
    if not command:
        return {"safe": True, "reason": "", "suggested": ""}
        
    cmd_clean = command.strip().lower()
    
    # 1. Block dangerous deletions
    # e.g., rm -rf / or rm -rf without safe directory constraints
    import re
    if "rm " in cmd_clean and ("-rf" in cmd_clean or "-f" in cmd_clean):
        # Allow removing specific dummy files or local tmp files if constrained, but block root or recursive parent deletes
        if re.search(r'rm\s+-rf\s+(?:/|\.\./|\*|~)', cmd_clean) or len(cmd_clean.split()) < 3:
            return {
                "safe": False,
                "reason": "Dangerous recursive deletion command detected.",
                "suggested": "Use a safer, specific file deletion command without recursive root flags."
            }
            
    # 2. Block pipe execution of remote unverified scripts
    # e.g., curl ... | sh, wget ... | bash
    if any(p in cmd_clean for p in ("| sh", "| bash", "| zsh", "| ksh", "| cmd", "| powershell", "| pwsh")):
        if any(downloader in cmd_clean for downloader in ("curl ", "wget ", "fetch ")):
            return {
                "safe": False,
                "reason": "Direct execution of remote scripts via pipe is unsafe.",
                "suggested": "Download the script first, inspect it, then execute it locally."
            }
            
    # 3. Block insecure unrestricted permissions
    # e.g., chmod 777 or chmod -R 777
    if "chmod " in cmd_clean and "777" in cmd_clean:
        return {
            "safe": False,
            "reason": "Setting 777 permissions creates extreme security vulnerabilities.",
            "suggested": "Use more restrictive permissions (e.g., chmod 755 or chmod 644)."
        }
        
    return {"safe": True, "reason": "", "suggested": ""}

def lint_code_style(code: str, lang: str) -> str:
    """
    Performs style checks on generated code according to conventions.
    Returns warning message if styling guidelines are violated, or "" if passed.
    """
    if not code:
        return ""
        
    lang = lang.lower()
    warnings = []
    lines = code.split("\n")
    
    if lang == "python":
        import re
        for i, line in enumerate(lines, 1):
            # Check line length (PEP 8 recommends max 79 chars)
            if len(line) > 100:
                warnings.append(f"Line {i} exceeds 100 characters ({len(line)} chars). PEP 8 recommends max 79.")
            # Check space after comma
            if re.search(r',\w', line):
                warnings.append(f"Line {i} is missing a space after comma.")
            # Check naming conventions for functions (snake_case)
            func_match = re.search(r'def\s+([a-zA-Z0-9_]+)\(', line)
            if func_match:
                func_name = func_match.group(1)
                if not func_name.islower() and "_" not in func_name and not func_name.startswith("__"):
                    warnings.append(f"Function name '{func_name}' at line {i} is not snake_case.")
                    
        # Check for docstrings in python functions
        if "def " in code and '"""' not in code and "'''" not in code:
            warnings.append("Python functions are missing Google-style docstrings (PEP 257).")
            
    elif lang == "php":
        # Check PSR-12 strict types declaration
        if "<?php" in code and "declare(strict_types=1);" not in code:
            warnings.append("PHP file is missing PSR-12 strict types declaration: 'declare(strict_types=1);'.")
        # Check PSR-12 closing tag omission
        if "?>" in code:
            warnings.append("PSR-12 recommendation: omit closing '?>' tag at the end of pure PHP files.")
            
    elif lang in ("typescript", "javascript"):
        # Check var usage
        for i, line in enumerate(lines, 1):
            if "var " in line and not line.strip().startswith("//") and not line.strip().startswith("/*"):
                if any(x in line for x in ("= ", "=json", " =")):
                    warnings.append(f"Line {i} uses 'var'. Use 'const' or 'let' instead.")
                    
    if warnings:
        return "Style Warning:\n- " + "\n- ".join(warnings)
        
    return ""
