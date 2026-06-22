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


if __name__ == "__main__":
    test_resolve_local_url()
    test_chunk_text()
    test_load_api_key_from_file()
    test_knowledge_parser()
    test_search_memory_parser()
    print("\nAll CLI tooling tests PASSED!")
