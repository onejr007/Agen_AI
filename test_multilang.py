import sys
import os
import datetime

import app.agent as agent_module
from app.agent import parse_and_repair_json_tool_call, retrieve_semantic_memory
import app.main
from app.main import classify_programming_language, analyze_workspace_languages
from app.models import KnowledgeBase, Message
from app.database import parse_json_embedding


class DummyGuideline:
    def __init__(self, language_name: str, keywords: str):
        self.language_name = language_name
        self.keywords = keywords


class DummyQuery:
    def __init__(self, items):
        self.items = items
        self._ordered = False
        self._limit = None

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        self._ordered = True
        return self

    def limit(self, value):
        self._limit = value
        return self

    def order_by(self, *args):
        self._ordered = True
        return self

    def first(self):
        res = self.all()
        return res[0] if res else None

    def all(self):
        items = list(self.items)
        if self._ordered:
            items.sort(key=lambda item: getattr(item, "id", None) or 0)
        if self._limit is not None:
            items = items[:self._limit]
        return items


class DummyRetrievalEntry:
    def __init__(self, title: str = None, content: str = None, embedding=None, created_at=None, role=None, id=None, chat_id=None):
        self.title = title
        self.content = content
        self.embedding = embedding
        self.created_at = created_at or datetime.datetime.utcnow()
        self.role = role
        self.id = id
        self.chat_id = chat_id


class DummyDB:
    def __init__(self):
        self.guidelines = [
            DummyGuideline("luau", "lua,luau,roblox,rbx"),
            DummyGuideline("python", "python,pep8,pip,django,flask,fastapi,pandas"),
            DummyGuideline("web", "html,css,web,react,nextjs,vue,tailwind"),
            DummyGuideline("php", "php,composer,laravel,symfony,wordpress,pdo"),
            DummyGuideline("mysql", "sql,mysql,database,query,table,schema,select"),
            DummyGuideline("typescript", "typescript,ts,tsx,interface"),
            DummyGuideline("javascript", "javascript,js,jsx,node,npm,es6"),
            DummyGuideline("java", "java,maven,gradle,spring,jdk")
        ]
        self.knowledge_entries = []
        self.message_entries = []

    def query(self, model):
        if model == KnowledgeBase:
            return DummyQuery(self.knowledge_entries)
        if model == Message:
            return DummyQuery(self.message_entries)
        return DummyQuery(self.guidelines)

def test_language_classification():
    print("\n--- Testing Language Classification ---")
    mock_db = DummyDB()
    tests = [
        ("Bagaimana cara membuat PDO connection di PHP?", "php"),
        ("Tolong optimasi query SELECT * FROM users WHERE age > 18", "mysql"),
        ("Buat interface User dengan type definition di TypeScript", "typescript"),
        ("Bagaimana cara run class Java Spring Boot?", "java"),
        ("Tulis script Roblox Luau print player name", "luau"),
        ("Buat script python menggunakan pandas", "python"),
        ("Tulis code javascript modern ES6", "javascript")
    ]
    
    for query, expected in tests:
        detected = classify_programming_language(query, "", mock_db)
        print(f"Query: '{query}' -> Detected: {detected} (Expected: {expected})")
        assert detected == expected or (expected == "javascript" and detected == "javascript"), f"Failed for {query}"
    print("Language Classification Tests: PASS")

def test_tool_call_repair():
    print("\n--- Testing Tool Call Repair & Normalization ---")
    
    # 1. Test write_file mapping to write_to_file
    tc1 = '{"name": "write_file", "arguments": {"filename": "app.py", "code": "print(123)"}}'
    repaired1 = parse_and_repair_json_tool_call(tc1)
    print(f"Original: {tc1}\nRepaired: {repaired1}")
    assert repaired1["name"] == "write_to_file"
    assert repaired1["arguments"]["path"] == "app.py"
    assert repaired1["arguments"]["content"] == "print(123)"

    # 2. Test edit_file mapping to replace_file_content
    tc2 = '{"function": "edit_file", "parameters": {"file": "index.php", "text": "<?php echo 123;"}}'
    repaired2 = parse_and_repair_json_tool_call(tc2)
    print(f"Original: {tc2}\nRepaired: {repaired2}")
    assert repaired2["name"] == "replace_file_content"
    assert repaired2["arguments"]["path"] == "index.php"
    assert repaired2["arguments"]["content"] == "<?php echo 123;"

    print("Tool Call Repair Tests: PASS")

def test_workspace_profiler():
    print("\n--- Testing Workspace Profiler ---")
    analyze_workspace_languages()
    print(f"Profiled workspace counts: {app.main.WORKSPACE_PROFILE}")
    assert len(app.main.WORKSPACE_PROFILE) > 0, "Workspace profile is empty!"
    print("Workspace Profiler Tests: PASS")


def test_retrieval_memory_ranking():
    print("\n--- Testing Retrieval Memory Ranking ---")

    mock_db = DummyDB()
    mock_db.knowledge_entries = [
        DummyRetrievalEntry(
            title="FastAPI Auth Guide",
            content="Panduan autentikasi FastAPI dengan bearer token, validasi header authorization, dan keamanan API.",
            embedding=None,
            created_at=datetime.datetime.utcnow() - datetime.timedelta(days=5)
        ),
        DummyRetrievalEntry(
            title="Irrelevant Note",
            content="Catatan umum tentang berkebun dan pupuk organik di halaman rumah.",
            embedding=None,
            created_at=datetime.datetime.utcnow() - datetime.timedelta(days=1)
        ),
    ]
    mock_db.message_entries = [
        DummyRetrievalEntry(
            title="",
            content="Kita perlu memperbaiki validasi bearer token agar request tanpa token ditolak.",
            embedding=None,
            created_at=datetime.datetime.utcnow() - datetime.timedelta(days=1),
            role="assistant"
        ),
        DummyRetrievalEntry(
            title="",
            content="halo",
            embedding=None,
            created_at=datetime.datetime.utcnow(),
            role="user"
        ),
    ]

    original_get_embedding = agent_module.get_embedding
    try:
        agent_module.get_embedding = lambda query, model=None: []
        context = retrieve_semantic_memory(mock_db, "bagaimana validasi bearer token fastapi authorization header", limit=3)
    finally:
        agent_module.get_embedding = original_get_embedding

    print(f"Retrieved context:\n{context}")
    assert "FastAPI Auth Guide" in context
    assert "bearer token" in context.lower()
    assert "berkebun" not in context.lower()
    assert "Content: halo" not in context
    print("Retrieval Memory Ranking: PASS")


def test_retrieval_context_truncation():
    print("\n--- Testing Retrieval Context Truncation ---")

    mock_db = DummyDB()
    very_long_content = "FastAPI authorization token " * 80
    mock_db.knowledge_entries = [
        DummyRetrievalEntry(
            title="Long Auth Note",
            content=very_long_content,
            embedding=None,
            created_at=datetime.datetime.utcnow()
        )
    ]

    original_get_embedding = agent_module.get_embedding
    try:
        agent_module.get_embedding = lambda query, model=None: []
        context = retrieve_semantic_memory(mock_db, "fastapi authorization token", limit=1)
    finally:
        agent_module.get_embedding = original_get_embedding

    print(f"Truncated context:\n{context}")
    assert len(context) < len(very_long_content)
    assert "..." in context
    print("Retrieval Context Truncation: PASS")


def test_embedding_parser_guard():
    print("\n--- Testing Embedding Parser Guard ---")

    assert parse_json_embedding("[1, 2, 3]") == [1.0, 2.0, 3.0]
    assert parse_json_embedding('{"bad": true}') == []
    assert parse_json_embedding("[1, \"x\", 3]") == []
    assert parse_json_embedding("not json") == []

    print("Embedding Parser Guard: PASS")


def test_retrieval_handles_invalid_embeddings():
    print("\n--- Testing Retrieval Handles Invalid Embeddings ---")

    mock_db = DummyDB()
    now = datetime.datetime.utcnow()
    mock_db.knowledge_entries = [
        DummyRetrievalEntry(
            title="Valid Vector Entry",
            content="Panduan token authorization untuk FastAPI.",
            embedding='[1, 0]',
            created_at=now
        ),
        DummyRetrievalEntry(
            title="Broken Vector Entry",
            content="Konten rusak yang embedding-nya bukan JSON valid.",
            embedding='{"oops": true}',
            created_at=now - datetime.timedelta(minutes=1)
        ),
    ]
    mock_db.message_entries = []

    original_get_embedding = agent_module.get_embedding
    try:
        agent_module.get_embedding = lambda query, model=None: [1.0, 0.0]
        context = retrieve_semantic_memory(mock_db, "fastapi authorization token", limit=2)
    finally:
        agent_module.get_embedding = original_get_embedding

    print(f"Vector retrieval context:\n{context}")
    assert "Valid Vector Entry" in context
    assert "Broken Vector Entry" not in context
    print("Retrieval Handles Invalid Embeddings: PASS")


def test_semantic_cache():
    print("\n--- Testing Semantic Query Caching ---")
    import json
    from app.agent import check_semantic_cache, get_embedding
    
    mock_db = DummyDB()
    question = "How do I setup routing in Vue?"
    mock_db.message_entries = [
        DummyRetrievalEntry(
            id=1, chat_id="chat-x", role="user", content=question,
            embedding=json.dumps([1, 0, 0])
        ),
        DummyRetrievalEntry(
            id=2, chat_id="chat-x", role="assistant", content="Use vue-router with createRouter.",
            embedding=None
        )
    ]
    
    class MockDBWithMessages:
        def query(self, model):
            if model == Message:
                class MsgQuery:
                    def filter(self, *args, **kwargs):
                        return self
                    def order_by(self, *args, **kwargs):
                        return self
                    def first(self):
                        return mock_db.message_entries[1]
                    def all(self):
                        return mock_db.message_entries
                return MsgQuery()
            return mock_db.query(model)
            
    mock_db_with_messages = MockDBWithMessages()
    
    # We monkey-patch get_embedding to return exactly [1, 0, 0] so cosine sim is 1.0
    import app.agent as agent_module
    original_get_embedding = getattr(agent_module, "get_embedding", None)
    try:
        agent_module.get_embedding = lambda query, model=None: [1, 0, 0]
        
        cached = check_semantic_cache(mock_db_with_messages, question)
        assert cached == "Use vue-router with createRouter.", f"Cache miss or wrong answer: {cached}"
        
        # Now test with same chat_id (should NOT cache if it's the current identical message)
        cached_same_chat = check_semantic_cache(mock_db_with_messages, question, chat_id="chat-x")
        assert cached_same_chat is None, "Should skip cache if it matches exact message in same chat"
        
    finally:
        agent_module.get_embedding = original_get_embedding

    print("Semantic Query Caching: PASS")


if __name__ == "__main__":
    test_language_classification()
    test_tool_call_repair()
    test_workspace_profiler()
    test_retrieval_memory_ranking()
    test_retrieval_context_truncation()
    test_embedding_parser_guard()
    test_retrieval_handles_invalid_embeddings()
    test_semantic_cache()
    
    print("\nAll Multilingual, Tool Repair, and Cache tests PASSED!")
