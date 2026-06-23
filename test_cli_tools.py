import os

import cli_utils
import knowledge
import search_memory


def test_resolve_local_url():
    print("\n--- Testing Local URL Resolution ---")

    resolved_ollama = cli_utils.resolve_local_url("http://ollama:11434", cli_utils.DEFAULT_OLLAMA_BASE_URL)
    resolved_db_host = cli_utils.normalize_service_hostname("host.docker.internal")

    print(f"Resolved Ollama URL: {resolved_ollama}")
    print(f"Resolved DB host: {resolved_db_host}")

    assert resolved_ollama == "http://localhost:11434"
    assert resolved_db_host == "localhost"

    print("Local URL Resolution: PASS")


def test_chunk_text():
    print("\n--- Testing Chunk Text ---")

    content = "\n".join([f"line {index}" for index in range(1, 40)])
    chunks = cli_utils.chunk_text(content, chunk_size=60, max_chunks=3)

    print(f"Generated chunks: {chunks}")
    assert 1 <= len(chunks) <= 3
    assert all(chunk.strip() for chunk in chunks)

    print("Chunk Text: PASS")


def test_load_api_key_from_file():
    print("\n--- Testing API Key Loading ---")

    temp_file = "temp_api_key_test.txt"
    original_env = os.environ.get("AGENT_API_KEY")
    try:
        if "AGENT_API_KEY" in os.environ:
            del os.environ["AGENT_API_KEY"]

        with open(temp_file, "w", encoding="utf-8") as file:
            file.write("AGENT_API_KEY=test_cli_key\n")

        loaded_key = cli_utils.load_api_key(temp_file)
        print(f"Loaded key: {loaded_key}")
        assert loaded_key == "test_cli_key"
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if original_env is not None:
            os.environ["AGENT_API_KEY"] = original_env

    print("API Key Loading: PASS")


def test_knowledge_parser():
    print("\n--- Testing Knowledge CLI Parser ---")

    parser = knowledge.build_parser()
    args = parser.parse_args([
        "index-workspace",
        "--dir", ".",
        "--extensions", ".py,.md",
        "--chunk-size", "1200",
        "--max-chunks-per-file", "4",
    ])

    print(f"Parsed args: {args}")
    assert args.command == "index-workspace"
    assert args.chunk_size == 1200
    assert args.max_chunks_per_file == 4

    print("Knowledge CLI Parser: PASS")


def test_search_memory_parser():
    print("\n--- Testing Search Memory CLI Parser ---")

    parser = search_memory.build_parser()
    args = parser.parse_args(["bearer token", "--limit", "3", "--snippet", "180"])

    print(f"Parsed args: {args}")
    assert args.query == "bearer token"
    assert args.limit == 3
    assert args.snippet == 180

    print("Search Memory CLI Parser: PASS")


def test_cli_improvements():
    print("\n--- Testing CLI Improvements & Integration Tools ---")

    # 1. Cosine similarity function
    v1 = [1.0, 0.0]
    v2 = [1.0, 0.0]
    v3 = [0.0, 1.0]

    sim_ident = knowledge.calculate_cosine_similarity(v1, v2)
    sim_ortho = knowledge.calculate_cosine_similarity(v1, v3)

    print(f"Cosine Similarity (identical): {sim_ident}")
    print(f"Cosine Similarity (orthogonal): {sim_ortho}")

    assert abs(sim_ident - 1.0) < 1e-5
    assert abs(sim_ortho - 0.0) < 1e-5

    # 2. Database tests with SQLite
    db_file = "test_cli_temp.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    os.environ["DATABASE_URL"] = f"sqlite:///./{db_file}"

    # Set up tables
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            tags TEXT,
            embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            role TEXT,
            content TEXT,
            search_results TEXT,
            embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS language_guidelines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            language_name TEXT,
            keywords TEXT,
            instructions TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

    try:
        import deduplicate
        import sync_clinerules

        # Seeding guidelines for sync test
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO language_guidelines (language_name, instructions, is_active) VALUES (?, ?, ?)",
            ("python", "Python instructions here", 1)
        )
        cursor.execute(
            "INSERT INTO language_guidelines (language_name, instructions, is_active) VALUES (?, ?, ?)",
            ("luau", "Luau instructions here", 1)
        )
        conn.commit()
        conn.close()

        # Test 1: Add and list knowledge
        print("Testing add_knowledge...")
        original_get_embedding = knowledge.get_embedding
        knowledge.get_embedding = lambda x: [0.5, 0.5] if "python" in x.lower() else [0.1, 0.9]

        knowledge.add_knowledge("Python Guidelines", "Always use snake_case for python variables.", "python,guidelines")
        knowledge.add_knowledge("Luau Guidelines", "Avoid using spawn function in Luau code.", "luau,guidelines")

        # Test 2: Search knowledge (hybrid)
        print("Testing search_knowledge...")
        knowledge.search_knowledge("python", mode="hybrid", limit=2)

        # Test 3: Deduplicate RAG
        print("Testing deduplicate_rag...")
        # Add another identical Python entry
        knowledge.add_knowledge("Python Guidelines Duplicate", "Always use snake_case for python variables.", "python,guidelines")

        # Run deduplicator with 0.95 similarity threshold
        deduplicate.deduplicate_rag(threshold=0.95, dry_run=False)

        # Verify one of them is gone
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM knowledge_base WHERE tags LIKE '%python%'")
        count = cursor.fetchone()[0]
        print(f"Number of python entries after deduplication: {count}")
        assert count == 1, f"Expected 1 python entry after deduplication, got {count}"

        # Test 4: Prune chat history
        print("Testing prune_chat_history...")
        cursor.execute("INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)",
                       ("chat1", "Old Chat", "2020-01-01 00:00:00"))
        cursor.execute("INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)",
                       ("chat2", "New Chat", datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        deduplicate.prune_chat_history(prune_days=10, dry_run=False)

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM chats")
        chats = cursor.fetchall()
        print(f"Remaining chats after pruning: {chats}")
        assert len(chats) == 1
        assert chats[0][0] == "chat2"
        conn.close()

        # Test 5: sync_clinerules
        print("Testing sync_rules...")
        # Write dummy files to workspace
        with open("dummy_test_sync.py", "w") as f:
            f.write("# Python script")

        original_analyze = sync_clinerules.analyze_workspace
        sync_clinerules.analyze_workspace = lambda: {"python": 1}

        sync_clinerules.sync_rules()

        assert os.path.exists(".clinerules")
        with open(".clinerules", "r", encoding="utf-8") as f:
            clinerules_content = f.read()
        print(f".clinerules content:\n{clinerules_content}")
        assert "Python instructions here" in clinerules_content
        assert "Luau instructions here" not in clinerules_content

        # Restore mocks and cleanup
        sync_clinerules.analyze_workspace = original_analyze
        knowledge.get_embedding = original_get_embedding

        if os.path.exists("dummy_test_sync.py"):
            os.remove("dummy_test_sync.py")
        if os.path.exists(".clinerules"):
            os.remove(".clinerules")

    finally:
        if os.path.exists(db_file):
            os.remove(db_file)

    print("CLI Improvements & Integration Tests: PASS")


def test_dotnet_middleware_bridge():
    print("\n--- Testing .NET Core Middleware Bridge ---")

    # 1. Verify .NET Core SDK installation
    try:
        res = subprocess.run(["dotnet", "--version"], capture_output=True, text=True, timeout=10)
        print(f"Dotnet SDK version: {res.stdout.strip()}")
        assert res.returncode == 0
        print(".NET SDK Installation Check: PASS")
    except Exception as e:
        print(f"Peringatan: .NET SDK tidak ditemukan di host ({e}). Melewati build test.")

    # 2. Test DotNetBridgeMiddleware interception logic
    class MockRequest:
        def __init__(self, method, path, headers, body_bytes):
            self.method = method
            self.url = self
            self.path = path
            self.headers = headers
            self.scope = {"headers": []}
            self._body_bytes = body_bytes

        async def body(self):
            return self._body_bytes

    original_post = requests.post

    def mock_post_allow(url, json=None, timeout=None):
        class MockResponse:
            status_code = 200
            def json(self):
                return {
                    "action": "allow",
                    "modifiedHeaders": {"X-Processed-By-DotNet-Bridge": "true"}
                }
        return MockResponse()

    def mock_post_block(url, json=None, timeout=None):
        class MockResponse:
            status_code = 200
            def json(self):
                return {
                    "action": "block",
                    "statusCode": 403,
                    "detail": "Blocked by .NET Bridge"
                }
        return MockResponse()

    from app.main import DotNetBridgeMiddleware

    middleware = DotNetBridgeMiddleware(app=None)

    # Test Allow Scenario
    requests.post = mock_post_allow
    mock_req = MockRequest("GET", "/index.html", {"Host": "localhost"}, b"hello")

    async def mock_call_next(req):
        class MockHTTPResponse:
            pass
        return MockHTTPResponse()

    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(middleware.dispatch(mock_req, mock_call_next))
    print(f"Allow check completed. Scope headers: {mock_req.scope['headers']}")
    assert (b"x-processed-by-dotnet-bridge", b"true") in mock_req.scope["headers"]
    print("Allow Interception: PASS")

    # Test Block Scenario
    requests.post = mock_post_block
    mock_req_block = MockRequest("GET", "/admin", {"Host": "localhost"}, b"")
    res_block = loop.run_until_complete(middleware.dispatch(mock_req_block, mock_call_next))
    print(f"Block check completed. Response status: {res_block.status_code}, content: {res_block.body}")
    assert res_block.status_code == 403
    assert res_block.body == b"Blocked by .NET Bridge"
    print("Block Interception: PASS")

    # 3. Test perform_autonomous_middleware_upgrade loop
    print("Testing perform_autonomous_middleware_upgrade loop...")
    calls = []
    def mock_post_full_evolution(url, json=None, timeout=None):
        calls.append(url)
        class MockResponse:
            status_code = 200
            def json(self):
                if "/api/chat" in url:
                    return {
                        "message": {
                            "content": '{"filePath": "app/main.py", "action": "patch", "searchContent": "logger.info(\\"Starting background\\")", "content": "logger.info(\\"Starting background worker...\\")", "triggerRebuild": false}'
                        }
                    }
                elif "/apply-upgrade" in url:
                    return {
                        "success": True
                    }
                return {}
        return MockResponse()

    requests.post = mock_post_full_evolution
    from app.main import perform_autonomous_middleware_upgrade
    perform_autonomous_middleware_upgrade()

    print(f"Post calls recorded: {calls}")
    assert any("11434/api/chat" in c for c in calls)
    assert any("5000/apply-upgrade" in c for c in calls)
    print("Ollama to .NET Core self-evolution loop: PASS")

    # Restore requests.post
    requests.post = original_post
    print(".NET Core Middleware Bridge Interception Tests: PASS")


if __name__ == "__main__":
    import sqlite3
    import datetime
    import subprocess
    import requests

    test_resolve_local_url()
    test_chunk_text()
    test_load_api_key_from_file()
    test_knowledge_parser()
    test_search_memory_parser()
    test_cli_improvements()
    test_dotnet_middleware_bridge()
    print("\nAll CLI tooling tests PASSED!")
