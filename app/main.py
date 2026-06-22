import datetime
import uuid
import json
import logging
import re
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.database import Base, engine, get_db, init_db_with_retry, SessionLocal
from app.models import APIKey, ChatSession, Message, KnowledgeBase, LanguageGuideline
from app.agent import DEFAULT_SYSTEM_PROMPT, CASUAL_SYSTEM_PROMPT, pull_ollama_model, process_search_and_context, call_ollama_chat_stream, retrieve_semantic_memory, get_embedding, parse_and_repair_json_tool_call, validate_code_syntax, apply_unified_diff, supervise_terminal_command, lint_code_style, list_ollama_models, repair_json_string, check_semantic_cache
import os
import threading
import time
import queue
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("agent.main")

# Create a thread-safe queue for processing embeddings sequentially
embedding_queue = queue.Queue()

def embedding_worker():
    """Worker thread that processes embedding tasks sequentially from the queue."""
    logger.info("Starting background sequential embedding worker...")
    while True:
        try:
            # Block until a task is available
            task_type, task_id, content = embedding_queue.get()
            
            # Simple cooling sleep to prevent CPU locks
            time.sleep(1.0)
            
            db = SessionLocal()
            try:
                if task_type == "knowledge":
                    logger.info(f"Worker: Generating embedding for knowledge base ID {task_id}...")
                    vector = get_embedding(content)
                    if vector:
                        db.execute(
                            text("UPDATE knowledge_base SET embedding = :embedding WHERE id = :id"),
                            {"embedding": json.dumps(vector), "id": task_id}
                        )
                        db.commit()
                        logger.info(f"Worker: Successfully updated embedding for knowledge base ID {task_id}")
                    else:
                        logger.warning(f"Worker: Failed to get embedding for knowledge base ID {task_id}")
                
                elif task_type == "message":
                    logger.info(f"Worker: Generating embedding for message ID {task_id}...")
                    vector = get_embedding(content)
                    if vector:
                        db.execute(
                            text("UPDATE messages SET embedding = :embedding WHERE id = :id"),
                            {"embedding": json.dumps(vector), "id": task_id}
                        )
                        db.commit()
                        logger.info(f"Worker: Successfully updated embedding for message ID {task_id}")
                    else:
                        logger.warning(f"Worker: Failed to get embedding for message ID {task_id}")
            except Exception as e:
                logger.error(f"Worker: Failed to process embedding task for {task_type} ID {task_id}: {str(e)}")
            finally:
                db.close()
                embedding_queue.task_done()
        except Exception as e:
            logger.error(f"Worker loop error: {str(e)}")
            time.sleep(2)

# Start the background worker thread on import
worker_thread = threading.Thread(target=embedding_worker, daemon=True)
worker_thread.start()

def generate_chat_pair_embeddings_async(user_msg_id: int, user_content: str, assistant_msg_id: int, assistant_content: str):
    """Queues embedding tasks for the user and assistant messages."""
    embedding_queue.put(("message", user_msg_id, user_content))
    embedding_queue.put(("message", assistant_msg_id, assistant_content))

def generate_knowledge_embedding_async(kb_id: int, content: str):
    """Queues embedding task for a knowledge base entry."""
    embedding_queue.put(("knowledge", kb_id, content))

# Programming Language Strict Guidelines
LUAU_INSTRUCTIONS = """
[Strict Coding Standards: Roblox Luau]
- Use strictly typed Luau (e.g. `local player: Player = ...`).
- Use PascalCase for Roblox services (e.g. `game:GetService("Players")`).
- Use camelCase for local variable and function names.
- Always check if instances are not nil before manipulating them, e.g. `if player.Character then ...`.
- Write clean, modular, Roblox scheduler-safe threads using `task.defer` or `task.delay` instead of `spawn` or `delay`.
"""

PYTHON_INSTRUCTIONS = """
[Strict Coding Standards: Python]
- Strictly adhere to PEP 8 standards.
- Always include clear type hinting for function parameters and return values (e.g., `def calculate(a: int) -> float:`).
- Document classes and functions with descriptive Google-style docstrings.
- Handle exceptions safely using `try...except` blocks and specific exception types.
"""

WEB_INSTRUCTIONS = """
[Strict Coding Standards: Web Development]
- Use modern HTML5 semantic elements (e.g., `<header>`, `<main>`, `<footer>`).
- Write clean, scoped, responsive CSS using variables, flexbox, or grid layouts. Avoid inline styles.
- Use strict TypeScript/JavaScript conventions: ES6+ syntax, const/let, proper promise handling with async/await.
"""

PHP_INSTRUCTIONS = """
[Strict Coding Standards: PHP]
- Follow PSR-12 coding standard guidelines.
- Always declare strict types at the very top of file: `declare(strict_types=1);`.
- Use explicit type hints for function arguments and return types.
- Ensure proper exception handling using try-catch blocks and log failures.
- Prevent SQL injection by using PDO with prepared statements and parameterized queries.
"""

MYSQL_INSTRUCTIONS = """
[Strict Coding Standards: MySQL / SQL]
- Write SQL keywords in uppercase (e.g. `SELECT`, `INSERT`, `UPDATE`, `DELETE`, `JOIN`, `WHERE`).
- Use descriptive, snake_case names for tables, columns, and database schemas.
- Ensure all queries are optimized, indexing is used where appropriate, and table joins use explicit `INNER JOIN` or `LEFT JOIN` syntax.
- Avoid using `SELECT *`; always specify the exact columns required.
- Use parameterized queries / placeholders in code integrations to prevent injection vulnerabilities.
"""

TS_INSTRUCTIONS = """
[Strict Coding Standards: TypeScript]
- Define strong types and interfaces for all objects, function signatures, and class members. Avoid using `any` type.
- Adhere to ES6+ conventions: use `const` and `let` instead of `var`.
- Use modern async/await patterns for promise handling, and wrap them in try-catch.
- Follow clean coding naming conventions: camelCase for variables/functions, PascalCase for classes/interfaces.
"""

JAVA_INSTRUCTIONS = """
[Strict Coding Standards: Java]
- Strictly follow standard Java coding conventions (camelCase for variables, PascalCase for classes, UPPERCASE for constants).
- Always specify access modifiers (`private`, `protected`, `public`) for class fields and methods.
- Write descriptive Javadoc comments for all public classes and APIs.
- Handle exceptions safely; avoid catching general `Exception` or throwing raw exceptions where specific ones apply.
- Use modern Java features safely (e.g., Try-with-resources for streams/connections, Stream API).
"""

SELF_HEALING_PROMPT_TEMPLATE = """
[CRITICAL ALERT: PREVIOUS ATTEMPT FAILED]
The previous action returned the following error or failure feedback:
"{error_snippet}"

Analyze the error above immediately. You MUST:
1. Use a `<think>...</think>` block to explicitly diagnose the root cause of the error.
2. Formulate a hypothesis on how to fix it and state your new approach inside the think block.
3. Once the analysis is complete, emit the corrected tool calls (such as writing or modifying files) to resolve the error.
4. If you realize the current path is completely blocked, explain why and propose an alternative strategy.
"""

def seed_new_language_guideline_if_missing(db: Session, lang: str):
    """Dynamically seeds a new programming language style guide into the database if not already present."""
    try:
        lang_lower = lang.lower().strip()
        if not lang_lower or lang_lower == "general":
            return
            
        existing = db.query(LanguageGuideline).filter(LanguageGuideline.language_name == lang_lower).first()
        if not existing:
            logger.info(f"Dynamic Seeding: Adding new language guidelines for '{lang_lower}' to DB...")
            # Formulate clean standard instructions based on language name
            instructions = (
                f"\n[Strict Coding Standards: {lang_lower.upper()}]\n"
                f"- Write clean, idiomatic, and modern {lang_lower} code conforming to standard conventions.\n"
                f"- Ensure robust exception and error handling using try-catch or language-specific idioms.\n"
                f"- Include type hinting, clear definitions, and documentation for major functions and structures.\n"
            )
            
            new_guideline = LanguageGuideline(
                language_name=lang_lower,
                keywords=f"{lang_lower},{lang_lower}lang",
                instructions=instructions,
                is_active=True
            )
            db.add(new_guideline)
            db.commit()
            logger.info(f"Dynamic Seeding: Successfully saved guidelines for '{lang_lower}' in DB.")
    except Exception as e:
        logger.error(f"Failed to dynamically seed language guideline for {lang}: {str(e)}")

def classify_programming_language(last_message: str, semantic_context: str, db: Session) -> str:
    """Classifies the target programming language based on query and workspace files using database guidelines."""
    msg_lower = last_message.lower()
    
    try:
        # Fetch active guidelines from database
        guidelines = db.query(LanguageGuideline).filter(LanguageGuideline.is_active == True).all()
    except Exception as e:
        logger.error(f"Error fetching language guidelines for classification: {str(e)}")
        guidelines = []

    # Check for explicit keywords in query matching database rules
    for g in guidelines:
        if g.keywords:
            keywords_list = [k.strip().lower() for k in g.keywords.split(",") if k.strip()]
            if any(k in msg_lower for k in keywords_list):
                # Dynamically seed standard guidelines if not present (safety check)
                seed_new_language_guideline_if_missing(db, g.language_name)
                return g.language_name
                
    # Dynamic fallback: check if any language in the workspace profile is mentioned in the query
    for lang in WORKSPACE_PROFILE:
        variations = [lang]
        if lang == "csharp":
            variations.append("c#")
        elif lang == "cpp":
            variations.append("c++")
            
        if any(v in msg_lower for v in variations):
            seed_new_language_guideline_if_missing(db, lang)
            return lang
                
    # Check RAG workspace context tags
    if semantic_context:
        ctx_lower = semantic_context.lower()
        counts = {}
        for g in guidelines:
            ext_mappings = {
                "luau": [".lua", ".luau"],
                "python": [".py"],
                "php": [".php"],
                "mysql": [".sql"],
                "typescript": [".ts", ".tsx"],
                "javascript": [".js", ".jsx"],
                "java": [".java"],
                "web": [".html", ".css"]
            }
            exts = ext_mappings.get(g.language_name.lower(), [f".{g.language_name.lower()}"])
            counts[g.language_name] = sum(ctx_lower.count(ext) for ext in exts)
            
        if counts:
            max_lang = max(counts, key=counts.get)
            if counts[max_lang] > 0:
                seed_new_language_guideline_if_missing(db, max_lang)
                return max_lang
            
    return "general"

def classify_workflow_mode(user_message: str) -> str:
    """
    Classifies whether the user message requires a complex software engineering workflow (Research/Planning)
    or is just a simple conversational query/test/greeting.
    """
    msg_clean = user_message.strip().lower()
    if not msg_clean:
        return "casual"
        
    # Casual/test keywords
    casual_keywords = {
        "test", "testing", "tes", "coba", "hello", "hi", "halo", "hey", "ping", "p", 
        "ready", "online", "aktif", "sudah aktif", "ok", "okay", "siap", "siap bos",
        "apa kabar", "how are you", "who are you", "siapa kamu", "info", "help", "bantuan"
    }
    
    # If the message matches any casual keyword exactly
    if msg_clean in casual_keywords:
        return "casual"
        
    # If the message is very short and has no strong coding action verbs
    coding_verbs = {
        "write", "create", "make", "implement", "fix", "debug", "modify", "update", "change", "add", "remove", "delete", "run", "execute",
        "buat", "tulis", "bikin", "tambah", "ubah", "hapus", "perbaiki", "jalankan", "buatkan"
    }
    
    if len(msg_clean) < 25:
        # Check if there are any coding verbs
        words = msg_clean.split()
        if not any(w in coding_verbs for w in words):
            return "casual"
            
    return "engineering"

def classify_user_intent(user_message: str) -> str:
    """Classifies the primary intent of the user's message (question, statement, command, greeting)."""
    msg_clean = user_message.strip().lower()
    if not msg_clean:
        return "statement"
        
    # Heuristics for question
    question_words = {"what", "how", "why", "when", "where", "who", "which", "is", "are", "do", "does", "can", "could", "would", "should", "apa", "bagaimana", "kenapa", "mengapa", "kapan", "dimana", "siapa", "apakah", "bisa"}
    
    if "?" in msg_clean:
        return "question"
        
    words = msg_clean.split()
    if words and words[0] in question_words:
        return "question"
        
    # Heuristics for command / task
    command_verbs = {"write", "create", "make", "implement", "fix", "debug", "modify", "update", "change", "add", "remove", "delete", "run", "execute", "buat", "tulis", "bikin", "tambah", "ubah", "hapus", "perbaiki", "jalankan", "buatkan", "tolong"}
    if any(verb in msg_clean for verb in command_verbs):
        return "command"
        
    # Greetings
    greeting_words = {"hello", "hi", "halo", "hey", "ping", "p", "morning", "pagi", "siang", "sore", "malam"}
    if len(words) <= 3 and any(greet in words for greet in greeting_words):
        return "greeting"
        
    return "statement"

def detect_response_language(user_message: str) -> str:
    """Detects a lightweight response language for short/static replies."""
    msg_lower = user_message.lower()

    indonesian_markers = [
        "halo", "hai", "tes", "coba", "siapa kamu", "apa kabar", "bantuan",
        "tolong", "buat", "bikin", "ubah", "hapus", "jalankan", "saya", "anda"
    ]
    english_markers = [
        "hello", "hi", "test", "testing", "help", "how are you", "who are you",
        "please", "create", "update", "delete", "run", "i", "you"
    ]

    id_score = sum(1 for marker in indonesian_markers if marker in msg_lower)
    en_score = sum(1 for marker in english_markers if marker in msg_lower)

    return "id" if id_score >= en_score else "en"

def check_static_response(user_message: str) -> str:
    """
    Checks if a user query is a simple greeting or test and returns a quick static response 
    to bypass LLM processing and guarantee a fast, direct answer.
    """
    msg_clean = user_message.strip().lower()
    # Strip common punctuation
    msg_clean = msg_clean.rstrip("?!.")

    response_language = detect_response_language(msg_clean)

    if msg_clean in ("test", "testing", "tes", "coba", "ping"):
        if response_language == "id":
            return "Halo! AgentAI sedang online, berjalan baik, dan siap membantu. Tugas coding apa yang ingin Anda kerjakan?"
        return "Hello! AgentAI is online, working perfectly, and ready to assist you. What coding task can I help you with today?"
    if msg_clean in ("hello", "hi", "halo", "hey", "p"):
        if response_language == "id":
            return "Halo! AgentAI sedang online dan siap membantu. Ada tugas coding atau development apa yang ingin Anda kerjakan?"
        return "Hello! AgentAI is online and ready to assist you. How can I help you with your coding tasks today?"
    return ""

def detect_previous_errors(messages: list) -> str:
    """Scans the last few messages in the history for errors."""
    error_keywords = [
        "error", "failed", "exception", "syntaxerror", "exit code 1", "stderr", 
        "cannot find", "not recognized", "crash", "invalid syntax", "unexpected token"
    ]
    
    # Scan last 3 messages
    for msg in reversed(messages[-3:]):
        content = msg.get("content", "")
        if not content:
            continue
        
        content_lower = content.lower()
        if any(kw in content_lower for kw in error_keywords):
            return content[:300] + ("..." if len(content) > 300 else "")
            
def detect_tool_call_loop(messages: list) -> bool:
    """
    Detects if the AI is caught in an infinite loop repeating the exact same tool calls
    in its recent history without making progress.
    """
    recent_assistant_msgs = [m for m in messages if m.get("role") == "assistant"][-4:]
    
    if len(recent_assistant_msgs) < 3:
        return False
        
    def hash_tool_calls(tool_calls):
        if not tool_calls:
            return None
        import json
        try:
            # Sort keys to handle minor dictionary ordering differences
            return json.dumps(tool_calls, sort_keys=True)
        except Exception:
            return str(tool_calls)

    # Check if the last 3 assistant messages have the exact same tool calls
    hashes = [hash_tool_calls(msg.get("tool_calls")) for msg in recent_assistant_msgs[-3:]]
    
    # If all 3 hashes are identical and they are not None, it's a loop
    if hashes[0] is not None and hashes[0] == hashes[1] and hashes[1] == hashes[2]:
        return True
        
    return False
            
WORKSPACE_PROFILE = {}

def get_workspace_directory() -> str:
    """Returns the workspace directory based on container mounts."""
    return "/app_host" if os.path.exists("/app_host") else "."

def analyze_workspace_languages():
    """Scans the workspace directory to profile file extensions and languages dynamically."""
    global WORKSPACE_PROFILE
    workspace_dir = get_workspace_directory()
    logger.info(f"Analyzing workspace languages in directory: {workspace_dir}")
    
    lang_counts = {}
    exclude_dirs = {".git", "node_modules", "venv", ".venv", "__pycache__", ".gemini", ".agents", "build", "dist"}
    
    ext_map = {
        ".lua": "luau",
        ".luau": "luau",
        ".py": "python",
        ".php": "php",
        ".sql": "mysql",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".java": "java",
        ".html": "web",
        ".css": "web",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".cs": "csharp",
        ".rb": "ruby",
        ".kt": "kotlin",
        ".swift": "swift",
        ".sh": "bash",
        ".ps1": "powershell",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json"
    }
    
    try:
        for root, dirs, files in os.walk(workspace_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if not ext:
                    continue
                lang = ext_map.get(ext, ext[1:]) # fallback to extension name
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
                    
        WORKSPACE_PROFILE = {k: v for k, v in lang_counts.items() if v > 0}
        logger.info(f"Workspace language profile complete: {WORKSPACE_PROFILE}")
    except Exception as e:
        logger.error(f"Error profiling workspace languages: {str(e)}")

def get_workspace_tree(max_depth: int = 3, max_files: int = 50) -> str:
    """Generates an ASCII directory tree map of the workspace."""
    workspace_dir = get_workspace_directory()
    exclude_dirs = {".git", "node_modules", "venv", ".venv", "__pycache__", ".gemini", ".agents", "build", "dist"}
    
    tree_lines = []
    file_count = 0
    
    try:
        # Get relative path length to calculate depth
        start_depth = workspace_dir.rstrip(os.path.sep).count(os.path.sep)
        
        for root, dirs, files in os.walk(workspace_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            # Calculate current depth
            current_depth = root.rstrip(os.path.sep).count(os.path.sep) - start_depth
            
            if current_depth > max_depth:
                dirs[:] = []  # Stop descending further
                continue
                
            if file_count >= max_files:
                tree_lines.append(f"... (truncated after {max_files} files)")
                break
                
            indent = "  " * current_depth
            folder_name = os.path.basename(root)
            if current_depth == 0:
                folder_name = "."
            
            if current_depth > 0:
                tree_lines.append(f"{indent}|-- {folder_name}/")
            else:
                tree_lines.append(f"{folder_name}/")
                
            # Add files in this directory
            for f in files:
                if f.endswith(".pyc") or f == ".DS_Store":
                    continue
                file_count += 1
                if file_count > max_files:
                    break
                tree_lines.append(f"{indent}    |-- {f}")
                
        if not tree_lines:
            return ""
            
        return "\n".join(tree_lines)
    except Exception as e:
        logger.error(f"Error generating workspace tree: {str(e)}")
        return ""

def get_git_workspace_context() -> str:
    """Runs git commands to retrieve current branch, status, and lightweight diff stats."""
    import subprocess
    workspace_dir = get_workspace_directory()
    
    # Check if git directory exists
    if not os.path.exists(os.path.join(workspace_dir, ".git")):
        return ""
        
    try:
        # Get active branch name
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
            cwd=workspace_dir, 
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        
        # Get git status
        status = subprocess.check_output(
            ["git", "status", "-s"], 
            cwd=workspace_dir, 
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        
        # Get git diff stats
        diff = subprocess.check_output(
            ["git", "diff", "--stat"], 
            cwd=workspace_dir, 
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        
        context_str = f"[Active Git Workspace Status]\nBranch: {branch}\n"
        if status:
            context_str += f"Modified Files:\n{status}\n"
        if diff:
            context_str += f"Diff Stats:\n{diff}\n"
            
        return context_str
    except Exception as e:
        logger.warning(f"Failed to retrieve git context: {str(e)}")
        return ""

def summarize_session_history(messages_to_summarize: list) -> str:
    """Calls Ollama internally to generate a concise summary of the older chat history."""
    if not messages_to_summarize:
        return "No prior session history to summarize."

    url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    
    # Format the dialogue to be summarized
    formatted_dialogue = []
    for msg in messages_to_summarize:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        formatted_dialogue.append(f"{role}: {content}")
        
    dialogue_str = "\n".join(formatted_dialogue)
    
    summary_prompt = f"Summarize the following chat dialogue concisely. Focus on key decisions, technical choices, completed files, and active requests. Do not include pleasantries. Keep the summary under 300 words:\n\n{dialogue_str}"
    
    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a professional system assistant designed to summarize developer dialogue history. Output only the raw summary text."},
            {"role": "user", "content": summary_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    def compact_summary_text(text: str, max_len: int = 180) -> str:
        """Normalizes and trims summary fragments to a compact size."""
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[: max_len - 3].rstrip() + "..."

    def fallback_summary(messages: list) -> str:
        """Builds a deterministic fallback summary without relying on Ollama."""
        recent_user_requests = []
        recent_assistant_actions = []
        mentioned_files = []
        seen_files = set()

        file_pattern = re.compile(r"[\w./\\-]+\.[A-Za-z0-9]{1,8}")

        for msg in messages[-12:]:
            role = (msg.get("role") or "user").lower()
            content = compact_summary_text(msg.get("content", ""), max_len=220)
            if not content:
                continue

            for match in file_pattern.findall(content):
                if match not in seen_files:
                    seen_files.add(match)
                    mentioned_files.append(match)
                if len(mentioned_files) >= 5:
                    break

            lowered = content.lower()
            if role == "user":
                if content not in recent_user_requests:
                    recent_user_requests.append(content)
            elif role == "assistant":
                if "calling tool:" in lowered:
                    continue
                if content not in recent_assistant_actions:
                    recent_assistant_actions.append(content)

        lines = ["Conversation summary (fallback):"]
        if recent_user_requests:
            lines.append("Recent user requests:")
            for item in recent_user_requests[-3:]:
                lines.append(f"- {item}")
        if recent_assistant_actions:
            lines.append("Recent assistant outputs:")
            for item in recent_assistant_actions[-3:]:
                lines.append(f"- {item}")
        if mentioned_files:
            lines.append(f"Mentioned files: {', '.join(mentioned_files[:5])}")

        summary = "\n".join(lines).strip()
        return summary[:900].rstrip()
    
    try:
        # Import requests locally to be safe
        import requests
        from app.agent import ollama_lock
        with ollama_lock:
            response = requests.post(url, json=payload, timeout=180)
        if response.status_code == 200:
            summary_text = response.json().get("message", {}).get("content", "").strip()
            if summary_text:
                return summary_text
            logger.warning("Ollama returned an empty session summary. Using fallback summarizer.")
        else:
            logger.warning(f"Ollama summary request returned status {response.status_code}. Using fallback summarizer.")
    except Exception as e:
        logger.warning(f"Failed to generate history summary: {str(e)}")

    return fallback_summary(messages_to_summarize)

def preprocess_tool_call(tool_call: dict) -> dict:
    """
    1. If tool call is replace_file_content and uses SEARCH-REPLACE diff format,
       automatically reads the original file, applies the diff, and converts it
       to a standard full write block or full edit argument.
    2. If tool call is execute_command, supervises the shell command and blocks
       it securely if it is hazardous.
    """
    if not tool_call:
        return tool_call
        
    name = tool_call.get("name")
    
    # A. Supervise terminal execution commands
    if name == "execute_command":
        args = tool_call.get("arguments", {})
        command = args.get("command")
        if command:
            guard = supervise_terminal_command(command)
            if not guard["safe"]:
                logger.warning(f"Terminal Command blocked by supervisor: '{command}'. Reason: {guard['reason']}")
                err_msg = f"Error: [Terminal Supervisor Blocked this command because: {guard['reason']} Suggested alternative: {guard['suggested']}]"
                # Safe print statement
                tool_call["arguments"]["command"] = f'echo "{err_msg}"'
                
        return tool_call

    # B. Apply unified search-and-replace diff patches
    if name != "replace_file_content":
        return tool_call
        
    args = tool_call.get("arguments", {})
    path = args.get("path")
    diff_content = args.get("content")
    
    if not path or not diff_content or "<<<<<<< SEARCH" not in diff_content:
        return tool_call
        
    # Determine path prefix
    workspace_dir = get_workspace_directory()
    full_path = os.path.join(workspace_dir, path)
    
    if not os.path.exists(full_path):
        logger.warning(f"File not found for diff application: {full_path}")
        return tool_call
        
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            original_text = f.read()
            
        updated_text = apply_unified_diff(original_text, diff_content)
        
        # Standardize arguments by overriding content with full updated text
        # and map name to write_to_file so Cline/Roo Code can directly save it!
        tool_call["name"] = "write_to_file"
        tool_call["arguments"]["content"] = updated_text
        logger.info(f"Diff successfully pre-applied. Converted tool call to write_to_file for path: {path}")
    except Exception as e:
        logger.error(f"Failed to pre-apply diff to file {path}: {str(e)}")
        
    return tool_call

def normalize_openai_tools(tools: list) -> list:
    """
    Normalizes incoming OpenAI-compatible tool definitions into a clean structure
    accepted more reliably by Ollama and small local models.
    """
    if not isinstance(tools, list):
        return []

    normalized_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") != "function":
            continue

        function_block = tool.get("function", {})
        if not isinstance(function_block, dict):
            continue

        name = function_block.get("name")
        if not name:
            continue

        parameters = function_block.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        else:
            parameters.setdefault("type", "object")
            parameters.setdefault("properties", {})

        normalized_function = {
            "name": name,
            "description": function_block.get("description", ""),
            "parameters": parameters
        }

        if isinstance(function_block.get("strict"), bool):
            normalized_function["strict"] = function_block["strict"]

        normalized_tools.append({
            "type": "function",
            "function": normalized_function
        })

    return normalized_tools

def normalize_message_content(content):
    """Converts OpenAI-style content blocks or other content shapes into plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if content.get("type") == "text" and "text" in content:
            return str(content["text"])
        if "text" in content:
            return str(content["text"])
        return json.dumps(content, ensure_ascii=False)

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    text_parts.append(str(part["text"]))
                elif "text" in part:
                    text_parts.append(str(part["text"]))
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)

    if content is None:
        return ""

    return str(content)

def clamp_number(value, minimum: float = None, maximum: float = None, default=None):
    """Safely casts and clamps numeric request parameters."""
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default

    if minimum is not None:
        numeric = max(minimum, numeric)
    if maximum is not None:
        numeric = min(maximum, numeric)
    return numeric

def build_ollama_options(body: dict) -> dict:
    """Builds Ollama generation options from OpenAI-compatible request fields."""
    body = body or {}
    options = {}

    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    num_predict = clamp_number(max_tokens, minimum=1, maximum=32768, default=None)
    if num_predict is not None:
        options["num_predict"] = int(num_predict)

    temperature = clamp_number(body.get("temperature"), minimum=0.0, maximum=2.0, default=None)
    if temperature is not None:
        options["temperature"] = temperature

    top_p = clamp_number(body.get("top_p"), minimum=0.0, maximum=1.0, default=None)
    if top_p is not None:
        options["top_p"] = top_p

    seed = clamp_number(body.get("seed"), minimum=0, maximum=2147483647, default=None)
    if seed is not None:
        options["seed"] = int(seed)

    stop = body.get("stop")
    if isinstance(stop, str) and stop:
        options["stop"] = [stop]
    elif isinstance(stop, list):
        stop_values = [str(item) for item in stop if isinstance(item, (str, int, float)) and str(item)]
        if stop_values:
            options["stop"] = stop_values[:8]

    return options

def response_format_requires_json(body: dict) -> bool:
    """Checks whether the caller requested JSON-shaped output."""
    response_format = body.get("response_format")
    if not isinstance(response_format, dict):
        return False

    format_type = response_format.get("type")
    return format_type in ("json_object", "json_schema")

def apply_response_format_instructions(messages: list, body: dict) -> list:
    """Injects lightweight formatting guidance into the last user message."""
    if not response_format_requires_json(body):
        return messages

    response_format = body.get("response_format", {})
    format_type = response_format.get("type")
    json_instruction = "\n\n[Output Format Requirement]\nReturn valid JSON only. Do not include markdown fences, explanations, or extra text outside the JSON object."

    if format_type == "json_schema":
        json_schema = response_format.get("json_schema", {})
        schema_name = json_schema.get("name", "response")
        schema_definition = json_schema.get("schema")
        json_instruction += f"\nThe JSON must follow the schema named '{schema_name}'."
        if schema_definition:
            json_instruction += f"\nSchema:\n{json.dumps(schema_definition, ensure_ascii=False)}"

    for msg in reversed(messages):
        if msg.get("role") == "user":
            msg["content"] = f"{msg.get('content', '')}{json_instruction}"
            break

    return messages

app = FastAPI(title="AgentAI Gateway", version="1.0.0")

# Add CORS Middleware to allow client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False)

ALLOWED_CHAT_ROLES = {"system", "user", "assistant", "tool", "developer"}
MUTATING_TOOL_NAMES = {
    "write_to_file",
    "replace_file_content",
    "execute_command",
    "delete_file",
    "write_file",
    "create_file",
    "edit_file",
    "update_file",
    "modify_file",
}

def normalize_approval_keywords(raw_keywords: str) -> list[str]:
    """Parses a comma-separated list of approval keywords."""
    keywords = []
    for item in (raw_keywords or "").split(","):
        cleaned = item.strip().lower()
        if cleaned:
            keywords.append(cleaned)
    return keywords

APPROVAL_KEYWORDS = normalize_approval_keywords(settings.EXECUTION_APPROVAL_KEYWORDS)

def is_execution_approved(user_message: str) -> bool:
    """Checks if the user explicitly approved execution (for mutating tool calls)."""
    msg = (user_message or "").strip().lower()
    if not msg:
        return False

    # Strict exact matches
    if msg in APPROVAL_KEYWORDS:
        return True

    # Common patterns
    approval_patterns = (
        "setuju",
        "lanjut eksekusi",
        "approve",
        "approved",
        "proceed",
        "jalankan",
        "eksekusi",
    )
    if any(p in msg for p in approval_patterns):
        return True

    return False

def is_mutating_tool_name(name: str) -> bool:
    """Detects whether a tool name is likely to mutate workspace or execute commands."""
    if not name:
        return False
    lowered = str(name).strip().lower()
    if lowered in MUTATING_TOOL_NAMES:
        return True
    return any(token in lowered for token in ("write", "replace", "edit", "modify", "delete", "execute", "run_command"))

def build_approval_gate_message() -> str:
    """Returns a short message instructing the user how to approve execution."""
    examples = ", ".join([f"'{kw}'" for kw in (APPROVAL_KEYWORDS[:4] or ["setuju"])])
    return (
        "\n\n[Approval Gate]\n"
        "Saya belum akan melakukan eksekusi (menulis/mengubah file atau menjalankan command) sebelum Anda memberi persetujuan eksplisit.\n"
        f"Balas dengan salah satu: {examples}.\n"
    )

def filter_tool_calls(tool_calls: list, allow_mutations: bool) -> tuple[list, bool]:
    """
    Filters tool calls based on whether mutations are allowed.
    Returns: (filtered_tool_calls, suppressed_mutation_attempted)
    """
    if not tool_calls:
        return [], False

    filtered = []
    suppressed = False
    for tc in tool_calls:
        func = tc.get("function", {}) if isinstance(tc, dict) else {}
        tool_name = func.get("name")
        if not allow_mutations and is_mutating_tool_name(tool_name):
            suppressed = True
            continue
        filtered.append(tc)
    return filtered, suppressed

def ensure_json_object(body: Any, endpoint_name: str) -> dict:
    """Ensures request bodies are JSON objects before further validation."""
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{endpoint_name} request body must be a JSON object."
        )
    return body

def normalize_session_identifier(raw_value: Any) -> str:
    """Normalizes the chat session identifier to a safe bounded string."""
    if raw_value is None:
        return "default-session"
    normalized = str(raw_value).strip()
    return normalized[:255] or "default-session"

def validate_chat_messages(req_messages: Any) -> list[dict]:
    """Validates chat messages shape, roles, and size before processing."""
    if not isinstance(req_messages, list) or not req_messages:
        raise HTTPException(status_code=400, detail="Messages list is empty or invalid.")

    if len(req_messages) > settings.MAX_CHAT_MESSAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Messages list exceeds the limit of {settings.MAX_CHAT_MESSAGES} items."
        )

    sanitized_messages = []
    has_non_empty_user_message = False

    for index, msg in enumerate(req_messages):
        if not isinstance(msg, dict):
            raise HTTPException(status_code=400, detail=f"Message at index {index} must be an object.")

        role = str(msg.get("role", "")).strip().lower()
        if role not in ALLOWED_CHAT_ROLES:
            raise HTTPException(
                status_code=400,
                detail=f"Message role at index {index} is invalid. Allowed roles: {sorted(ALLOWED_CHAT_ROLES)}."
            )

        normalized_content = normalize_message_content(msg.get("content"))
        if len(normalized_content) > settings.MAX_MESSAGE_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"Message at index {index} exceeds the limit of {settings.MAX_MESSAGE_CHARS} characters."
            )

        normalized_message = dict(msg)
        normalized_message["role"] = role
        normalized_message["content"] = normalized_content
        sanitized_messages.append(normalized_message)

        if role == "user" and normalized_content.strip():
            has_non_empty_user_message = True

    if not has_non_empty_user_message:
        raise HTTPException(status_code=400, detail="At least one non-empty user message is required.")

    return sanitized_messages

def validate_embedding_input(input_payload: Any) -> list[str]:
    """Validates and normalizes embedding inputs into a list of plain strings."""
    if input_payload is None:
        raise HTTPException(status_code=400, detail="Input is required.")

    if isinstance(input_payload, list):
        normalized_inputs = [normalize_message_content(item) for item in input_payload]
    else:
        normalized_inputs = [normalize_message_content(input_payload)]

    if not normalized_inputs:
        raise HTTPException(status_code=400, detail="Input must contain at least one text item.")

    if len(normalized_inputs) > settings.MAX_EMBEDDING_ITEMS:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding input exceeds the limit of {settings.MAX_EMBEDDING_ITEMS} items."
        )

    for index, item in enumerate(normalized_inputs):
        if not item.strip():
            raise HTTPException(status_code=400, detail=f"Embedding input at index {index} is empty.")
        if len(item) > settings.MAX_EMBEDDING_INPUT_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"Embedding input at index {index} exceeds the limit of {settings.MAX_EMBEDDING_INPUT_CHARS} characters."
            )

    return normalized_inputs

def get_api_key(api_key_header: str = Depends(API_KEY_HEADER), db: Session = Depends(get_db)):
    """Verifies that the provided API key (Bearer token format) is valid in the database."""
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Header. Use Bearer token format."
        )
    
    header_value = api_key_header.strip()

    if header_value.lower() == "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API token is empty."
        )

    if header_value.lower().startswith("bearer "):
        token = header_value[7:].strip()
    elif " " in header_value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Use 'Bearer <token>'."
        )
    else:
        token = header_value

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API token is empty."
        )

    # Query key in database
    db_key = db.query(APIKey).filter(APIKey.key_value == token, APIKey.is_active == True).first()
    if not db_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API Key."
        )
    
    # Update last used time
    db_key.last_used_at = datetime.datetime.utcnow()
    db.commit()
    return db_key

def sync_database_schema(db: Session):
    """Ensures existing tables have the required columns for vector embeddings."""
    try:
        # Check messages table
        result = db.execute(text("SHOW COLUMNS FROM messages LIKE 'embedding'")).fetchone()
        if not result:
            logger.info("Adding 'embedding' column to 'messages' table...")
            db.execute(text("ALTER TABLE messages ADD COLUMN embedding LONGTEXT NULL"))
            db.commit()

        # Check knowledge_base table
        result = db.execute(text("SHOW COLUMNS FROM knowledge_base LIKE 'embedding'")).fetchone()
        if not result:
            logger.info("Adding 'embedding' column to 'knowledge_base' table...")
            db.execute(text("ALTER TABLE knowledge_base ADD COLUMN embedding LONGTEXT NULL"))
            db.commit()
    except Exception as e:
        logger.error(f"Error during schema sync: {str(e)}")

def extract_and_index_database_schema(db: Session):
    """
    Queries MySQL database schema (tables, columns, types) 
    and indexes them into the RAG knowledge base.
    """
    try:
        logger.info("Extracting database schema for RAG...")
        # Get list of all tables
        tables_res = db.execute(text("SHOW TABLES")).fetchall()
        tables = [row[0] for row in tables_res]
        
        schema_descriptions = []
        for table in tables:
            # Skip session or logs if too large/sensitive
            create_res = db.execute(text(f"SHOW CREATE TABLE `{table}`")).fetchone()
            if create_res:
                create_stmt = create_res[1]
                schema_descriptions.append(f"Table: {table}\nSQL Create Statement:\n{create_stmt}\n")
                
        if schema_descriptions:
            full_schema_doc = "\n".join(schema_descriptions)
            title = f"Database Schema for agent_db"
            content = f"Here is the database schema structure of the MySQL database 'agent_db' for references in SQL query generation:\n\n{full_schema_doc}"
            
            # Check if this knowledge entry already exists
            existing = db.query(KnowledgeBase).filter(KnowledgeBase.title == title).first()
            if existing:
                existing.content = content
                db.commit()
                # Trigger async embedding update
                generate_knowledge_embedding_async(existing.id, content)
                logger.info("Updated existing database schema in RAG.")
            else:
                new_kb = KnowledgeBase(
                    title=title,
                    content=content,
                    tags="mysql-schema,database-schema",
                    embedding=None
                )
                db.add(new_kb)
                db.commit()
                # Trigger async embedding generation
                generate_knowledge_embedding_async(new_kb.id, content)
                logger.info("Indexed new database schema into RAG.")
    except Exception as e:
        logger.error(f"Failed to extract database schema for RAG: {str(e)}")

@app.on_event("startup")
def startup_event():
    """Run database setup and pull Ollama model on startup."""
    logger.info("Waiting for database connection...")
    init_db_with_retry()
    logger.info("Initializing Database tables...")
    Base.metadata.create_all(bind=engine)

    # Sync schema for existing tables (adds embedding column if missing)
    db = next(get_db())
    try:
        sync_database_schema(db)
        
        # Extract and index database schema for RAG queries
        extract_and_index_database_schema(db)
        
        # Insert default API Key from config if no keys exist
        existing_keys = db.query(APIKey).count()
        if existing_keys == 0:
            logger.info(f"Generating default API Key from environment: {settings.AGENT_API_KEY}")
            default_key = APIKey(
                key_value=settings.AGENT_API_KEY,
                name="Default Local Dev Key"
            )
            db.add(default_key)
            db.commit()
        
        if settings.uses_default_api_key:
            logger.warning("AgentAI is running with the default development API key. Change AGENT_API_KEY before broader use.")

        # Write active API Key to a file in workspace so user can grab it easily
        if settings.WRITE_API_KEY_FILE:
            api_key_file = "/app_host/api_key.txt"
            if not os.path.exists("/app_host"):
                api_key_file = "./api_key.txt"

            with open(api_key_file, "w", encoding="utf-8") as f:
                f.write(f"AGENT_API_KEY={settings.AGENT_API_KEY}\n")
                f.write("Use the API Key above in VS Code configurations (Continue / Cline / Roo Code).\n")
                f.write("API Base URL: http://localhost:8000/v1\n")
            logger.info(f"API Key written to workspace directory at {api_key_file}")
        else:
            logger.info("WRITE_API_KEY_FILE disabled. Skipping api_key.txt export.")

        # Seed default language guidelines if the table is empty
        existing_guidelines = db.query(LanguageGuideline).count()
        if existing_guidelines == 0:
            logger.info("Seeding default language guidelines into the database...")
            default_guidelines = [
                LanguageGuideline(
                    language_name="luau",
                    keywords="lua,luau,roblox,rbx",
                    instructions=LUAU_INSTRUCTIONS
                ),
                LanguageGuideline(
                    language_name="python",
                    keywords="python,pep8,pip,django,flask,fastapi",
                    instructions=PYTHON_INSTRUCTIONS
                ),
                LanguageGuideline(
                    language_name="web",
                    keywords="html,css,web,react,nextjs,vue,tailwind",
                    instructions=WEB_INSTRUCTIONS
                ),
                LanguageGuideline(
                    language_name="php",
                    keywords="php,composer,laravel,symfony,wordpress",
                    instructions=PHP_INSTRUCTIONS
                ),
                LanguageGuideline(
                    language_name="mysql",
                    keywords="sql,mysql,database,query,table,schema",
                    instructions=MYSQL_INSTRUCTIONS
                ),
                LanguageGuideline(
                    language_name="typescript",
                    keywords="typescript,ts,tsx",
                    instructions=TS_INSTRUCTIONS
                ),
                LanguageGuideline(
                    language_name="javascript",
                    keywords="javascript,js,jsx,node,npm",
                    instructions=WEB_INSTRUCTIONS
                ),
                LanguageGuideline(
                    language_name="java",
                    keywords="java,maven,gradle,spring,jdk",
                    instructions=JAVA_INSTRUCTIONS
                )
            ]
            db.bulk_save_objects(default_guidelines)
            db.commit()
            logger.info("Successfully seeded default language guidelines.")
    except Exception as e:
        logger.error(f"Error during database startup sync: {str(e)}")
    finally:
        db.close()

    # Pull Ollama model
    pull_ollama_model()

    # Analyze workspace language profiling
    analyze_workspace_languages()

@app.get("/health", tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """Simple status check endpoint."""
    try:
        # Check DB connection
        db.execute(text("SELECT 1"))
        db_status = "Healthy"
    except Exception as e:
        logger.error(f"Health check DB error: {str(e)}")
        db_status = "Unhealthy"

    return {
        "status": "online",
        "database": db_status,
        "model_configured": settings.OLLAMA_MODEL,
        "ollama_url": settings.OLLAMA_BASE_URL
    }

@app.get("/v1/models", tags=["OpenAI Models"])
def openai_list_models(
    api_key: APIKey = Depends(get_api_key)
):
    """OpenAI-compatible model list endpoint for editor and SDK integrations."""
    created_timestamp = int(datetime.datetime.utcnow().timestamp())
    models = list_ollama_models()

    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": created_timestamp,
                "owned_by": "agentai"
            }
            for model_name in models
        ]
    }

@app.get("/v1/models/{model_id}", tags=["OpenAI Models"])
def openai_get_model(
    model_id: str,
    api_key: APIKey = Depends(get_api_key)
):
    """OpenAI-compatible single model metadata endpoint."""
    models = list_ollama_models()
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' was not found.")

    return {
        "id": model_id,
        "object": "model",
        "created": int(datetime.datetime.utcnow().timestamp()),
        "owned_by": "agentai"
    }

@app.post("/v1/embeddings", tags=["OpenAI Embeddings"])
def openai_embeddings(
    body: dict,
    api_key: APIKey = Depends(get_api_key)
):
    """OpenAI-compatible embeddings endpoint backed by Ollama embeddings."""
    body = ensure_json_object(body, "Embeddings")
    model = body.get("model") if isinstance(body.get("model"), str) and body.get("model").strip() else settings.OLLAMA_EMBED_MODEL
    normalized_inputs = validate_embedding_input(body.get("input"))

    data = []
    total_tokens_estimate = 0

    for index, item in enumerate(normalized_inputs):
        embedding = get_embedding(item, model=model)
        if not embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding from Ollama.")

        total_tokens_estimate += max(1, len(item) // 4) if item else 1
        data.append({
            "object": "embedding",
            "index": index,
            "embedding": embedding
        })

    return {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": total_tokens_estimate,
            "total_tokens": total_tokens_estimate
        }
    }

@app.post("/v1/knowledge", tags=["Knowledge Base"])
def add_knowledge(
    body: dict,
    api_key: APIKey = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    """
    Add custom documentation or coding reference to the RAG knowledge base.
    Generates semantic embedding asynchronously.
    """
    title = body.get("title")
    content = body.get("content")
    tags = body.get("tags", "")

    if not title or not content:
        raise HTTPException(status_code=400, detail="Title and content are required.")

    try:
        new_knowledge = KnowledgeBase(
            title=title,
            content=content,
            tags=tags,
            embedding=None
        )
        db.add(new_knowledge)
        db.commit()
        
        # Trigger async embedding generation
        generate_knowledge_embedding_async(new_knowledge.id, content)
        
        logger.info(f"Successfully added knowledge entry: {title} (embedding queued)")
        return {"status": "success", "message": f"Knowledge '{title}' added successfully. Embedding generation started in background."}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save knowledge to DB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database save error: {str(e)}")

@app.post("/v1/chat/completions", tags=["OpenAI Chat"])
def chat_completions(
    body: dict,
    api_key: APIKey = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    """
    OpenAI-compatible chat completions endpoint.
    Supports streaming and non-streaming, with vector memory RAG.
    """
    body = ensure_json_object(body, "Chat completions")
    req_messages = validate_chat_messages(body.get("messages", []))
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise HTTPException(status_code=400, detail="'stream' must be a boolean value.")

    raw_tools = body.get("tools", None)
    if raw_tools is not None and not isinstance(raw_tools, list):
        raise HTTPException(status_code=400, detail="'tools' must be a list when provided.")
    tools = normalize_openai_tools(raw_tools)
    requested_model = body.get("model") if isinstance(body.get("model"), str) and body.get("model").strip() else settings.OLLAMA_MODEL
    request_options = build_ollama_options(body)
    json_output_required = response_format_requires_json(body)

    # Get last user query
    last_user_message = ""
    for msg in reversed(req_messages):
        if msg.get("role") == "user":
            last_user_message = msg.get("content", "")
            break

    # 1. Start or retrieve Chat Session (generate a session ID if not provided)
    chat_id = normalize_session_identifier(body.get("user", "default-session"))
    chat_session = db.query(ChatSession).filter(ChatSession.id == chat_id).first()
    if not chat_session:
        chat_session = ChatSession(id=chat_id, title=last_user_message[:50] or "Conversation")
        db.add(chat_session)
        db.commit()

    # Check for instant static response (bypass LLM/RAG entirely for connection pings & greetings)
    static_reply = "" if json_output_required else check_static_response(last_user_message)
    
    if not static_reply and not json_output_required:
        cached_response = check_semantic_cache(db, last_user_message, chat_id)
        if cached_response:
            static_reply = cached_response
            logger.info("Semantic Cache: Intercepted message with cached response.")
            
    if static_reply:
        if not static_reply.startswith(cached_response if "cached_response" in locals() and cached_response else "---"):
            logger.info("Static Router: Intercepted casual message with static reply.")
        
        # Save user message to DB
        user_msg_db = Message(
            chat_id=chat_id,
            role="user",
            content=last_user_message,
            embedding=None
        )
        db.add(user_msg_db)
        db.commit()

        # Save assistant message to DB
        assistant_msg_db = Message(
            chat_id=chat_id,
            role="assistant",
            content=static_reply,
            embedding=None
        )
        db.add(assistant_msg_db)
        db.commit()

        if stream:
            def static_event_generator():
                completion_id = f"chatcmpl-{uuid.uuid4()}"
                created_timestamp = int(datetime.datetime.utcnow().timestamp())
                
                # Send the content chunk
                chunk_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_timestamp,
                    "model": requested_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": static_reply},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # Send final done chunk
                final_chunk_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_timestamp,
                    "model": requested_model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk_data)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(static_event_generator(), media_type="text/event-stream")
        else:
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(datetime.datetime.utcnow().timestamp()),
                "model": requested_model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": static_reply
                    },
                    "finish_reason": "stop"
                }]
            }

    # 2. Determine workflow mode (engineering or casual)
    workflow_mode = classify_workflow_mode(last_user_message)
    logger.info(f"Workflow Classifier: Mode={workflow_mode}")

    execution_approved = (workflow_mode != "engineering") or (not settings.REQUIRE_APPROVAL_FOR_MUTATIONS) or is_execution_approved(last_user_message)

    if workflow_mode == "casual":
        search_context = ""
        semantic_context = ""
        lang = "general"
        detected_error = ""
    else:
        # Trigger Search & Semantic Retrieval Modules
        search_context = process_search_and_context(db, last_user_message)
        semantic_context = retrieve_semantic_memory(db, last_user_message)

        # Classify programming language and check for previous errors in history
        lang = classify_programming_language(last_user_message, semantic_context, db)
        detected_error = detect_previous_errors(req_messages)
        logger.info(f"Dynamic Classifier: Lang={lang}, ErrorDetected={True if detected_error else False}")

    # 3. Save User Message to MySQL (embeddings will be generated sequentially after chat completion)
    user_msg_db = Message(
        chat_id=chat_id,
        role="user",
        content=last_user_message,
        search_results=search_context if search_context else None,
        embedding=None
    )
    db.add(user_msg_db)
    db.commit()
    user_msg_id = user_msg_db.id

    # 4. Build prompt list for Ollama, preserving incoming client messages (which contain tool call data/definitions)
    # Clone the messages dictionary fully to preserve all fields (e.g. tool_calls, tool_call_id)
    import copy
    ollama_messages = [copy.deepcopy(msg) for msg in req_messages]
    ollama_messages = apply_response_format_instructions(ollama_messages, body)

    # Normalize tool_calls in message history for Ollama (it expects function arguments to be JSON objects/dicts, not stringified JSON strings)
    for msg in ollama_messages:
        tool_calls = msg.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            new_tool_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tc_copy = copy.deepcopy(tc)
                    func = tc_copy.get("function")
                    if func and "arguments" in func:
                        args = func.get("arguments")
                        if isinstance(args, str):
                            try:
                                func["arguments"] = json.loads(args)
                            except Exception:
                                pass
                    new_tool_calls.append(tc_copy)
            msg["tool_calls"] = new_tool_calls

    # Calculate estimated tokens of history (1 token ~ 4 chars)
    total_chars = sum(len(msg.get("content", "")) for msg in ollama_messages if msg.get("content"))
    estimated_tokens = total_chars / 4.0
    
    if estimated_tokens > 2000 and len(ollama_messages) > 6:
        logger.info(f"Session history size ({estimated_tokens:.0f} tokens) exceeds 2000. Compressing...")
        
        # Check if index 0 is system message
        has_system = ollama_messages[0].get("role") == "system"
        start_idx = 1 if has_system else 0
        end_idx = len(ollama_messages) - 3 # Keep last 3 messages (user, assistant, user)
        
        if end_idx > start_idx + 2:
            to_compress = ollama_messages[start_idx:end_idx]
            summary = summarize_session_history(to_compress)
            logger.info("Successfully generated session history summary.")
            
            summary_msg = {
                "role": "system",
                "content": f"[Summary of Past Session History]\n{summary}\n[End of Summary]"
            }
            
            # Reconstruct message list
            new_messages = []
            if has_system:
                new_messages.append(ollama_messages[0])
            new_messages.append(summary_msg)
            new_messages.extend(ollama_messages[end_idx:])
            ollama_messages = new_messages

    # Find the system message to inject RAG and search context
    system_msg = None
    for msg in ollama_messages:
        if msg.get("role") == "system":
            system_msg = msg
            break

    # TOKEN BUDGET CONTROLLER
    # We must ensure the final prompt leaves enough room for the model to reply without OOM/truncation.
    MAX_CONTEXT_TOKENS = 3500
    current_tokens = sum(len(msg.get("content", "")) for msg in ollama_messages if msg.get("content")) / 4.0
    
    context_content = ""
    
    # Critical Context (Always injected if available)
    if detected_error:
        context_content += f"\n\n{SELF_HEALING_PROMPT_TEMPLATE.format(error_snippet=detected_error)}\n"

    if lang and lang != "general":
        try:
            db_guideline = db.query(LanguageGuideline).filter(
                LanguageGuideline.language_name == lang,
                LanguageGuideline.is_active == True
            ).first()
            if db_guideline and db_guideline.instructions:
                context_content += f"\n\n{db_guideline.instructions}\n"
        except Exception as e:
            logger.error(f"Error querying language instructions from database: {str(e)}")

    if WORKSPACE_PROFILE:
        profile_str = ", ".join([f"{k}: {v} files" for k, v in WORKSPACE_PROFILE.items()])
        context_content += f"\n\n[Active Workspace Profile]\nThe current project contains files in the following languages: {profile_str}. Always ensure edits match this workspace environment."

    git_context = get_git_workspace_context()
    if git_context:
        context_content += f"\n\n{git_context}\n"

    # Secondary Context (Hierarchically dropped if budget is tight)
    if semantic_context:
        mem_tokens = len(semantic_context) / 4.0
        if current_tokens + (len(context_content)/4.0) + mem_tokens < MAX_CONTEXT_TOKENS:
            context_content += f"\n\n{semantic_context}\n"
        else:
            logger.warning("Token Budget Controller: Dropped semantic_context to prevent OOM.")

    if search_context:
        search_tokens = len(search_context) / 4.0
        if current_tokens + (len(context_content)/4.0) + search_tokens < MAX_CONTEXT_TOKENS:
            context_content += f"\n\nHere is relevant real-time internet context for the user query:\n{search_context}\n"
        else:
            logger.warning("Token Budget Controller: Dropped search_context to prevent OOM.")

    workspace_tree = get_workspace_tree()
    if workspace_tree:
        tree_tokens = len(workspace_tree) / 4.0
        if current_tokens + (len(context_content)/4.0) + tree_tokens < MAX_CONTEXT_TOKENS:
            context_content += f"\n\n[Workspace Directory Tree Map]\n```text\n{workspace_tree}\n```\nUse this map to correctly locate files without guessing paths."
        else:
            logger.warning("Token Budget Controller: Dropped workspace tree map to prevent OOM.")

    # Inject intent classification logic
    user_intent = classify_user_intent(last_user_message)
    intent_instructions = {
        "question": "The user is asking a question. Analyze the request and provide a clear, factual answer. Do not execute commands or edit files unless explicitly requested.",
        "statement": "The user is providing a statement, information, or context. Acknowledge the context and integrate it into your knowledge. Respond appropriately.",
        "command": "The user is issuing a command or requesting a task. Formulate a plan, analyze constraints, and generate the necessary tool calls to complete the task.",
        "greeting": "The user is greeting you. Respond politely, briefly, and ask how you can help with their software engineering tasks."
    }
    context_content += f"\n\n[User Intent Analysis]: Detected intent is '{user_intent}'. {intent_instructions.get(user_intent, '')}\n"

    # Apply Dynamic Temperature Scaling based on user intent (Overrides client setting to protect small local LLMs)
    if user_intent == "command":
        request_options["temperature"] = 0.1 # Strict deterministic mode for coding/tools
        logger.info("Dynamic Temperature: Overridden to 0.1 (Strict Mode for Command)")
    elif user_intent == "question":
        request_options["temperature"] = 0.3 # Slightly more variance for explanations
        logger.info("Dynamic Temperature: Overridden to 0.3 (Explanation Mode for Question)")
    elif user_intent == "statement":
        request_options["temperature"] = 0.4 # Balanced
        logger.info("Dynamic Temperature: Overridden to 0.4 (Balanced Mode for Statement)")
    elif user_intent == "greeting" or workflow_mode == "casual":
        request_options["temperature"] = 0.7 # Creative conversational mode
        logger.info("Dynamic Temperature: Overridden to 0.7 (Creative Mode for Casual/Greeting)")

    # Inject Tool Call Loop Prevention Warning
    if detect_tool_call_loop(req_messages):
        context_content += "\n\n[CRITICAL SYSTEM WARNING]: You are caught in an INFINITE LOOP repeating the exact same tool calls. STOP doing this immediately! Change your approach, read a different file, or ask the user for help."
        logger.warning("Tool Call Loop Detected! Injected critical system warning into context.")

    # Select appropriate system prompt based on workflow mode
    active_system_prompt = DEFAULT_SYSTEM_PROMPT if workflow_mode == "engineering" else CASUAL_SYSTEM_PROMPT

    if system_msg:
        # Append active_system_prompt to client's system prompt if not present
        # Keep this static to maximize Ollama KV caching efficiency
        if "AgentAI" not in system_msg["content"]:
            system_msg["content"] = f"{system_msg['content']}\n\n{active_system_prompt}"
    else:
        # Create system message
        system_content = f"{active_system_prompt}"
        ollama_messages.insert(0, {"role": "system", "content": system_content})

    # Append all dynamic context (RAG, search, git, profiles, instructions, etc.) to the last user message.
    # This keeps the system message and history static, enabling Ollama's KV Cache to speed up responses dramatically.
    if context_content and len(ollama_messages) > 0:
        for msg in reversed(ollama_messages):
            if msg.get("role") == "user":
                msg["content"] = f"{msg.get('content', '')}\n\n[Dynamic Context]:\n{context_content}"
                break

    # 5. Handle Streaming Response
    if stream:
        def event_generator():
            completion_id = f"chatcmpl-{uuid.uuid4()}"
            created_timestamp = int(datetime.datetime.utcnow().timestamp())
            accumulated_content = []
            has_tool_calls = False
            suppressed_mutation_attempted = False

            import queue
            import threading

            chunk_queue = queue.Queue()

            def fetch_chunks():
                try:
                    # Attempt primary model
                    for chunk in call_ollama_chat_stream(ollama_messages, tools=tools, model=requested_model, options=request_options):
                        chunk_queue.put(chunk)
                except Exception as e:
                    logger.warning(f"Primary model '{requested_model}' failed: {str(e)}. Attempting fallback routing...")
                    try:
                        available_models = list_ollama_models()
                        fallback_model = None
                        
                        # Prioritize coder, qwen, or llama models
                        for m in available_models:
                            if m != requested_model and any(kw in m.lower() for kw in ("coder", "qwen", "llama")):
                                fallback_model = m
                                break
                                
                        # Fallback to anything available if specific architectures aren't found
                        if not fallback_model and available_models:
                            fallback_model = [m for m in available_models if m != requested_model]
                            fallback_model = fallback_model[0] if fallback_model else None
                            
                        if fallback_model:
                            logger.info(f"Fallback Routing: Switching to model '{fallback_model}'")
                            # Inform the new model of its fallback duty
                            fallback_messages = list(ollama_messages)
                            fallback_messages.append({"role": "system", "content": "[System Note: The primary model failed. You are running as the fallback model. Fulfill the user's request securely and accurately.]"})
                            for chunk in call_ollama_chat_stream(fallback_messages, tools=tools, model=fallback_model, options=request_options):
                                chunk_queue.put(chunk)
                        else:
                            chunk_queue.put({"error": f"Primary model failed and no fallback models available. Original error: {str(e)}"})
                    except Exception as fallback_e:
                        chunk_queue.put({"error": f"Both primary and fallback models failed. Original error: {str(e)}. Fallback error: {str(fallback_e)}"})
                finally:
                    chunk_queue.put(None)

            t = threading.Thread(target=fetch_chunks, daemon=True)
            t.start()

            try:
                while True:
                    try:
                        chunk = chunk_queue.get(timeout=5.0)
                        if chunk is None:
                            break
                        if "error" in chunk:
                            yield f"event: error\ndata: {json.dumps({'error': chunk['error']})}\n\n"
                            break
                        
                        content = chunk.get("content", "")
                        tool_calls = chunk.get("tool_calls", None)
                        done = chunk.get("done", False)

                        if content or tool_calls:
                            chunk_data = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_timestamp,
                                "model": requested_model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": None
                                }]
                            }
                            if content:
                                accumulated_content.append(content)
                                chunk_data["choices"][0]["delta"]["content"] = content
                            if tool_calls:
                                filtered_calls, suppressed = filter_tool_calls(tool_calls, allow_mutations=execution_approved)
                                suppressed_mutation_attempted = suppressed_mutation_attempted or suppressed
                                if filtered_calls:
                                    has_tool_calls = True
                                openai_tool_calls = []
                                for idx, tc in enumerate(filtered_calls):
                                    func = tc.get("function", {})
                                    openai_tool_calls.append({
                                        "index": idx,
                                        "id": f"call_{uuid.uuid4().hex[:12]}",
                                        "type": "function",
                                        "function": {
                                            "name": func.get("name"),
                                            "arguments": json.dumps(func.get("arguments")) if isinstance(func.get("arguments"), dict) else func.get("arguments", "{}")
                                        }
                                    })
                                if openai_tool_calls:
                                    chunk_data["choices"][0]["delta"]["tool_calls"] = openai_tool_calls
                            
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                    except queue.Empty:
                        yield ": keep-alive\n\n"
            finally:
                # Save full answer and generate embedding asynchronously after stream completes or cancels
                full_answer = "".join(accumulated_content)
                
                # Verify code blocks syntax if any
                if full_answer and "[Gateway Code Quality Warning]" not in full_answer:
                    import re
                    code_blocks = re.findall(r'```([a-zA-Z0-9_\-+]*)\n(.*?)\n```', full_answer, re.DOTALL)
                    syntax_errors = []
                    for c_lang, c_code in code_blocks:
                        c_lang_norm = c_lang.lower()
                        val_lang = ""
                        if c_lang_norm in ("py", "python"):
                            val_lang = "python"
                        elif c_lang_norm in ("php",):
                            val_lang = "php"
                        elif c_lang_norm in ("ts", "typescript"):
                            val_lang = "typescript"
                        elif c_lang_norm in ("js", "javascript"):
                            val_lang = "javascript"
                        elif c_lang_norm in ("java",):
                            val_lang = "java"
                        elif c_lang_norm in ("lua", "luau"):
                            val_lang = "luau"
                        elif c_lang_norm in ("json",):
                            val_lang = "json"
                            
                        if val_lang:
                            err = validate_code_syntax(c_code, val_lang)
                            if err:
                                syntax_errors.append(f"[{c_lang_norm.upper()} Block]: {err}")
                            else:
                                lint_warn = lint_code_style(c_code, val_lang)
                                if lint_warn:
                                    syntax_errors.append(f"[{c_lang_norm.upper()} Block]: {lint_warn}")
                                
                    if syntax_errors:
                        syntax_err_msg = "\n".join(syntax_errors)
                        logger.warning(f"Syntax validation failed for generated code:\n{syntax_err_msg}")
                        full_answer += f"\n\n[Gateway Code Quality Warning]:\n{syntax_err_msg}"

                # Check if the text output is actually a text-based JSON tool call
                repaired_tool = parse_and_repair_json_tool_call(full_answer)
                repaired_tool = preprocess_tool_call(repaired_tool)
                if repaired_tool and (execution_approved or (not is_mutating_tool_name(repaired_tool.get("name")))):
                    has_tool_calls = True
                    try:
                        # Send the final tool call delta chunk if we can still yield
                        tool_chunk_data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_timestamp,
                            "model": requested_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": 0,
                                        "id": f"call_{uuid.uuid4().hex[:12]}",
                                        "type": "function",
                                        "function": {
                                            "name": repaired_tool["name"],
                                            "arguments": json.dumps(repaired_tool["arguments"])
                                        }
                                    }]
                                },
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(tool_chunk_data)}\n\n"
                    except GeneratorExit:
                        pass
                    
                    # Clean up the DB storage for this message
                    full_answer = f"Calling tool: {repaired_tool['name']}({json.dumps(repaired_tool['arguments'])})"
                elif repaired_tool and (not execution_approved) and is_mutating_tool_name(repaired_tool.get("name")):
                    suppressed_mutation_attempted = True

                if suppressed_mutation_attempted:
                    full_answer = f"{full_answer}{build_approval_gate_message()}"

                if full_answer:
                    try:
                        # Direct database write using SessionLocal to prevent connection context leaks
                        db_session = SessionLocal()
                        db_assistant_msg = Message(
                            chat_id=chat_id,
                            role="assistant",
                            content=full_answer,
                            embedding=None
                        )
                        db_session.add(db_assistant_msg)
                        db_session.commit()
                        
                        # Trigger async embedding generation
                        generate_chat_pair_embeddings_async(
                            user_msg_id, last_user_message,
                            db_assistant_msg.id, full_answer
                        )
                        db_session.close()
                    except Exception as e:
                        logger.error(f"Failed to save assistant message or trigger embedding in finally: {str(e)}")

                try:
                    # Send final done indicators
                    final_chunk_data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_timestamp,
                        "model": requested_model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls" if has_tool_calls else "stop"
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk_data)}\n\n"
                    yield "data: [DONE]\n\n"
                except GeneratorExit:
                    pass

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # 6. Handle Standard JSON Response (non-streaming)
    else:
        accumulated_content = []
        accumulated_tool_calls = []
        for chunk in call_ollama_chat_stream(ollama_messages, tools=tools, model=requested_model, options=request_options):
            if "error" in chunk:
                raise HTTPException(status_code=500, detail=chunk["error"])
            content = chunk.get("content", "")
            tool_calls = chunk.get("tool_calls", None)
            if content:
                accumulated_content.append(content)
            if tool_calls:
                accumulated_tool_calls.extend(tool_calls)

        full_answer = "".join(accumulated_content)

        # Verify code blocks syntax if any
        if full_answer and "[Gateway Code Quality Warning]" not in full_answer:
            import re
            code_blocks = re.findall(r'```([a-zA-Z0-9_\-+]*)\n(.*?)\n```', full_answer, re.DOTALL)
            syntax_errors = []
            for c_lang, c_code in code_blocks:
                c_lang_norm = c_lang.lower()
                val_lang = ""
                if c_lang_norm in ("py", "python"):
                    val_lang = "python"
                elif c_lang_norm in ("php",):
                    val_lang = "php"
                elif c_lang_norm in ("ts", "typescript"):
                    val_lang = "typescript"
                elif c_lang_norm in ("js", "javascript"):
                    val_lang = "javascript"
                elif c_lang_norm in ("java",):
                    val_lang = "java"
                elif c_lang_norm in ("lua", "luau"):
                    val_lang = "luau"
                elif c_lang_norm in ("json",):
                    val_lang = "json"
                    
                if val_lang:
                    err = validate_code_syntax(c_code, val_lang)
                    if err:
                        syntax_errors.append(f"[{c_lang_norm.upper()} Block]: {err}")
                    else:
                        lint_warn = lint_code_style(c_code, val_lang)
                        if lint_warn:
                            syntax_errors.append(f"[{c_lang_norm.upper()} Block]: {lint_warn}")
                        
            if syntax_errors:
                syntax_err_msg = "\n".join(syntax_errors)
                logger.warning(f"Syntax validation failed for generated code:\n{syntax_err_msg}")
                full_answer += f"\n\n[Gateway Code Quality Warning]:\n{syntax_err_msg}"

        # Check if the text output is actually a text-based JSON tool call
        repaired_tool = parse_and_repair_json_tool_call(full_answer)
        repaired_tool = preprocess_tool_call(repaired_tool)
        suppressed_mutation_attempted = False
        if repaired_tool:
            # Convert text tool call to accumulated tool calls
            if execution_approved or (not is_mutating_tool_name(repaired_tool.get("name"))):
                accumulated_tool_calls = [{"function": {
                    "name": repaired_tool["name"],
                    "arguments": repaired_tool["arguments"]
                }}]
                full_answer = f"Calling tool: {repaired_tool['name']}({json.dumps(repaired_tool['arguments'])})"
            else:
                suppressed_mutation_attempted = True

        filtered_calls, suppressed = filter_tool_calls(accumulated_tool_calls, allow_mutations=execution_approved)
        suppressed_mutation_attempted = suppressed_mutation_attempted or suppressed
        accumulated_tool_calls = filtered_calls

        if suppressed_mutation_attempted:
            full_answer = f"{full_answer}{build_approval_gate_message()}"

        # Save assistant message to MySQL database and generate embedding for both messages sequentially in background
        if full_answer:
            try:
                assistant_msg_db = Message(
                    chat_id=chat_id,
                    role="assistant",
                    content=full_answer,
                    embedding=None
                )
                db.add(assistant_msg_db)
                db.commit()
                
                # Trigger async embedding generation
                generate_chat_pair_embeddings_async(
                    user_msg_db.id, last_user_message,
                    assistant_msg_db.id, full_answer
                )
            except Exception as e:
                logger.error(f"Failed to save assistant message or trigger embedding: {str(e)}")

        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": full_answer
            },
            "finish_reason": "stop"
        }
        if accumulated_tool_calls:
            openai_tool_calls = []
            for idx, tc in enumerate(accumulated_tool_calls):
                func = tc.get("function", {})
                openai_tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": func.get("name"),
                        "arguments": json.dumps(func.get("arguments")) if isinstance(func.get("arguments"), dict) else func.get("arguments", "{}")
                    }
                })
            choice["message"]["tool_calls"] = openai_tool_calls
            choice["finish_reason"] = "tool_calls"

        # Build OpenAI compatible output
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(datetime.datetime.utcnow().timestamp()),
            "model": requested_model,
            "choices": [choice],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1
            }
        }
