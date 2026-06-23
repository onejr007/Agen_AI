import app.main as main

class DummyAPIKeyRecord:
    def __init__(self, key_value: str, is_active: bool = True):
        self.key_value = key_value
        self.is_active = is_active
        self.last_used_at = None


class DummyAPIKeyQuery:
    def __init__(self, records):
        self.records = records

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        for record in self.records:
            if record.is_active:
                return record
        return None


class DummyAPIKeyDB:
    def __init__(self, records):
        self.records = records
        self.commit_called = False

    def query(self, model):
        return DummyAPIKeyQuery(self.records)

    def commit(self):
        self.commit_called = True


def test_static_response_language():
    print("\n--- Testing Static Response Language ---")

    response_id = main.check_static_response("halo")
    response_en = main.check_static_response("hello")

    print(f"ID response: {response_id}")
    print(f"EN response: {response_en}")

    assert response_id.startswith("Halo!")
    assert response_en.startswith("Hello!")

    print("Static Response Language: PASS")


def test_openai_models_endpoints():
    print("\n--- Testing OpenAI-Compatible Models Endpoints ---")

    original = main.list_ollama_models
    try:
        main.list_ollama_models = lambda: ["qwen2.5-coder:1.5b", "nomic-embed-text"]

        models_payload = main.openai_list_models(api_key=None)
        print(f"List models payload: {models_payload}")
        assert models_payload["object"] == "list"
        assert len(models_payload["data"]) == 2
        assert models_payload["data"][0]["object"] == "model"

        model_payload = main.openai_get_model("qwen2.5-coder:1.5b", api_key=None)
        print(f"Single model payload: {model_payload}")
        assert model_payload["id"] == "qwen2.5-coder:1.5b"
        assert model_payload["object"] == "model"
    finally:
        main.list_ollama_models = original

    print("OpenAI-Compatible Models Endpoints: PASS")

def test_tool_normalization():
    print("\n--- Testing Tool Normalization ---")

    incoming_tools = [
        {
            "type": "function",
            "function": {
                "name": "write_to_file",
                "description": "Menulis file",
                "parameters": {
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    }
                }
            }
        },
        {
            "type": "web_search_preview",
            "function": {
                "name": "ignored_tool"
            }
        }
    ]

    normalized = main.normalize_openai_tools(incoming_tools)
    print(f"Normalized tools: {normalized}")

    assert len(normalized) == 1
    assert normalized[0]["type"] == "function"
    assert normalized[0]["function"]["name"] == "write_to_file"
    assert normalized[0]["function"]["parameters"]["type"] == "object"

    print("Tool Normalization: PASS")


def test_openai_embeddings_endpoint():
    print("\n--- Testing OpenAI-Compatible Embeddings Endpoint ---")

    original = main.get_embedding
    try:
        main.get_embedding = lambda text, model=None: [0.11, 0.22, float(len(text or ""))]

        payload = main.openai_embeddings(
            {
                "model": "nomic-embed-text",
                "input": ["halo dunia", {"type": "text", "text": "agent lokal"}]
            },
            api_key=None
        )
        print(f"Embeddings payload: {payload}")

        assert payload["object"] == "list"
        assert payload["model"] == "nomic-embed-text"
        assert len(payload["data"]) == 2
        assert payload["data"][0]["object"] == "embedding"
        assert payload["usage"]["total_tokens"] >= 2
    finally:
        main.get_embedding = original

    print("OpenAI-Compatible Embeddings Endpoint: PASS")


def test_authorization_header_validation():
    print("\n--- Testing Authorization Header Validation ---")

    db = DummyAPIKeyDB([DummyAPIKeyRecord("local_developer_secret_key")])
    auth_record = main.get_api_key("Bearer local_developer_secret_key", db=db)
    assert auth_record.key_value == "local_developer_secret_key"
    assert db.commit_called is True

    try:
        main.get_api_key("Token abc", db=db)
        raise AssertionError("Malformed Authorization header should fail.")
    except main.HTTPException as exc:
        print(f"Malformed auth header rejected: {exc.detail}")
        assert exc.status_code == 401

    try:
        main.get_api_key("Bearer   ", db=db)
        raise AssertionError("Empty bearer token should fail.")
    except main.HTTPException as exc:
        print(f"Empty bearer token rejected: {exc.detail}")
        assert exc.status_code == 401

    print("Authorization Header Validation: PASS")


def test_chat_request_validation():
    print("\n--- Testing Chat Request Validation ---")

    valid_messages = main.validate_chat_messages([
        {"role": "user", "content": [{"type": "text", "text": "Halo agen"}]}
    ])
    assert valid_messages[0]["content"] == "Halo agen"

    try:
        main.validate_chat_messages([{"role": "invalid", "content": "test"}])
        raise AssertionError("Invalid chat role should fail.")
    except main.HTTPException as exc:
        print(f"Invalid role rejected: {exc.detail}")
        assert exc.status_code == 400

    try:
        main.validate_chat_messages([{"role": "user", "content": ""}])
        raise AssertionError("Empty user message should fail.")
    except main.HTTPException as exc:
        print(f"Empty user content rejected: {exc.detail}")
        assert exc.status_code == 400

    print("Chat Request Validation: PASS")


def test_embedding_input_validation():
    print("\n--- Testing Embedding Input Validation ---")

    normalized = main.validate_embedding_input(["satu", {"type": "text", "text": "dua"}])
    assert normalized == ["satu", "dua"]

    try:
        main.validate_embedding_input([""])
        raise AssertionError("Empty embedding input should fail.")
    except main.HTTPException as exc:
        print(f"Empty embedding rejected: {exc.detail}")
        assert exc.status_code == 400

    original_limit = main.settings.MAX_EMBEDDING_ITEMS
    try:
        main.settings.MAX_EMBEDDING_ITEMS = 1
        main.validate_embedding_input(["a", "b"])
        raise AssertionError("Embedding input beyond limit should fail.")
    except main.HTTPException as exc:
        print(f"Too many embedding items rejected: {exc.detail}")
        assert exc.status_code == 400
    finally:
        main.settings.MAX_EMBEDDING_ITEMS = original_limit

    print("Embedding Input Validation: PASS")


def test_execution_approval_gate_helpers():
    print("\n--- Testing Execution Approval Gate Helpers ---")

    assert main.is_execution_approved("setuju") is True
    assert main.is_execution_approved("lanjut eksekusi") is True
    assert main.is_execution_approved("approve") is True
    assert main.is_execution_approved("tidak") is False

    assert main.is_mutating_tool_name("write_to_file") is True
    assert main.is_mutating_tool_name("replace_file_content") is True
    assert main.is_mutating_tool_name("execute_command") is True
    assert main.is_mutating_tool_name("read_file") is False

    tool_calls = [
        {"function": {"name": "read_file", "arguments": {"path": "a.txt"}}},
        {"function": {"name": "write_to_file", "arguments": {"path": "b.txt", "content": "x"}}},
    ]
    filtered, suppressed = main.filter_tool_calls(tool_calls, allow_mutations=False)
    print(f"Filtered calls: {filtered}, suppressed={suppressed}")
    assert suppressed is True
    assert len(filtered) == 1
    assert filtered[0]["function"]["name"] == "read_file"

    filtered2, suppressed2 = main.filter_tool_calls(tool_calls, allow_mutations=True)
    assert suppressed2 is False
    assert len(filtered2) == 2

    print("Execution Approval Gate Helpers: PASS")


def test_token_counting():
    print("\n--- Testing Token Counting Helpers ---")
    
    # 1. Test count_tokens
    t1 = main.count_tokens("Hello, world!")
    print(f"Token count for 'Hello, world!': {t1}")
    assert t1 > 0
    
    t2 = main.count_tokens("")
    assert t2 == 0
    
    # 2. Test count_messages_tokens
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    t3 = main.count_messages_tokens(messages)
    print(f"Token count for message history: {t3}")
    assert t3 > 0
    
    print("Token Counting Helpers: PASS")



def test_sql_formatting_and_validation():
    print("\n--- Testing SQL Formatting and Validation ---")
    
    # 1. Formatting
    raw_markdown = "Here is a query:\n```sql\nselect * from users where id = '1'\n```"
    formatted = main.format_sql_blocks(raw_markdown)
    print(f"Formatted markdown: {repr(formatted)}")
    assert "SELECT" in formatted
    assert "FROM" in formatted
    assert "WHERE" in formatted
    
    # 2. Valid Syntax
    err1 = main.validate_code_syntax("SELECT * FROM users WHERE id = 1;", "sql")
    print(f"Valid SQL syntax check result: {repr(err1)}")
    assert err1 == ""
    
    # 3. Invalid Syntax (unbalanced quotes)
    err2 = main.validate_code_syntax("SELECT * FROM 'table_name;", "sql")
    print(f"Unbalanced quote check result: {repr(err2)}")
    assert "SQL Syntax Error" in err2 or "Syntax Warning" in err2
    
    # 4. Invalid Syntax (unclosed parenthesis)
    err3 = main.validate_code_syntax("SELECT * FROM table WHERE (id = 1", "sql")
    print(f"Unclosed parenthesis check result: {repr(err3)}")
    assert "SQL Syntax Error" in err3 or "Syntax Warning" in err3
    
    print("SQL Formatting and Validation: PASS")

def test_jsonschema_validation_and_repair():
    print("\n--- Testing JSONSchema Validation and Repair ---")
    
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "lines": {"type": "integer"},
            "overwrite": {"type": "boolean", "default": False}
        },
        "required": ["path", "lines"]
    }
    
    # 1. String to integer/boolean conversion
    args = {"path": "test.txt", "lines": "123", "overwrite": "true"}
    repaired = main.validate_and_repair_arguments(args, schema)
    print(f"Repaired arguments: {repaired}")
    assert repaired["lines"] == 123
    assert repaired["overwrite"] is True
    
    # 2. Missing required field default injection
    args2 = {"path": "test.txt"}
    repaired2 = main.validate_and_repair_arguments(args2, schema)
    print(f"Repaired missing field: {repaired2}")
    assert "lines" in repaired2
    assert repaired2["lines"] == 0
    
    print("JSONSchema Validation and Repair: PASS")

def test_native_tool_call_preprocessing_and_protection():
    print("\n--- Testing Native Tool Call Preprocessing and Protection ---")
    
    client_tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_command",
                "description": "Run shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        }
    ]
    
    # 1. Safe command
    native_calls = [
        {
            "type": "function",
            "function": {
                "name": "execute_command",
                "arguments": {"command": "git status"}
            }
        }
    ]
    preprocessed = main.preprocess_native_tool_calls(native_calls, client_tools=client_tools)
    print(f"Safe native call preprocessed: {preprocessed}")
    assert len(preprocessed) == 1
    assert preprocessed[0]["function"]["name"] == "execute_command"
    assert preprocessed[0]["function"]["arguments"]["command"] == "git status"
    
    # 2. Hazardous command
    hazardous_calls = [
        {
            "type": "function",
            "function": {
                "name": "execute_command",
                "arguments": {"command": "rm -rf /"}
            }
        }
    ]
    preprocessed_haz = main.preprocess_native_tool_calls(hazardous_calls, client_tools=client_tools)
    print(f"Hazardous native call preprocessed: {preprocessed_haz}")
    assert len(preprocessed_haz) == 1
    command_run = preprocessed_haz[0]["function"]["arguments"]["command"]
    assert "echo" in command_run
    assert "Blocked" in command_run
    
    print("Native Tool Call Preprocessing and Protection: PASS")


if __name__ == "__main__":
    test_static_response_language()
    test_openai_models_endpoints()
    test_tool_normalization()
    test_openai_embeddings_endpoint()
    test_authorization_header_validation()
    test_chat_request_validation()
    test_embedding_input_validation()
    test_execution_approval_gate_helpers()
    test_token_counting()
    test_sql_formatting_and_validation()
    test_jsonschema_validation_and_repair()
    test_native_tool_call_preprocessing_and_protection()
    print("\nAPI compatibility tests PASSED!")
