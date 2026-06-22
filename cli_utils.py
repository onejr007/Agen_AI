import os
from urllib.parse import urlparse


DEFAULT_DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "agent_db",
    "port": 3306,
}

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_GATEWAY_BASE_URL = "http://localhost:8000/v1"
DEFAULT_AGENT_API_KEY = "local_developer_secret_key"


def load_env_file(env_path: str = ".env") -> dict[str, str]:
    """Loads a lightweight .env file into a dictionary without mutating os.environ."""
    env_values: dict[str, str] = {}
    if not os.path.exists(env_path):
        return env_values

    try:
        with open(env_path, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                env_values[key.strip()] = value.strip().strip('"').strip("'")
    except Exception:
        return env_values

    return env_values


def get_config_value(key: str, default: str = "") -> str:
    """Reads a configuration value from environment variables or the local .env file."""
    env_value = os.getenv(key)
    if env_value is not None and str(env_value).strip():
        return str(env_value).strip()
    return load_env_file().get(key, default)


def normalize_service_hostname(hostname: str) -> str:
    """Converts container-oriented hostnames to localhost for host-side CLI tools."""
    if not hostname:
        return "localhost"
    normalized = hostname.strip().lower()
    if normalized in {"db", "ollama", "host.docker.internal"}:
        return "localhost"
    return hostname


def resolve_local_url(url: str, default_url: str) -> str:
    """Normalizes service URLs so CLI scripts can run from the local host."""
    candidate = (url or "").strip() or default_url
    parsed = urlparse(candidate)
    if not parsed.scheme or not parsed.netloc:
        return default_url

    hostname = normalize_service_hostname(parsed.hostname or "localhost")
    port = parsed.port
    auth_prefix = ""
    if parsed.username:
        auth_prefix = parsed.username
        if parsed.password:
            auth_prefix += f":{parsed.password}"
        auth_prefix += "@"

    host_port = hostname
    if port:
        host_port = f"{hostname}:{port}"

    return parsed._replace(netloc=f"{auth_prefix}{host_port}").geturl()


def get_mysql_connection_config() -> dict:
    """Builds host-side MySQL connection settings from DATABASE_URL when possible."""
    database_url = get_config_value("DATABASE_URL", "")
    if not database_url.startswith("mysql"):
        return dict(DEFAULT_DB_CONFIG)

    parsed = urlparse(database_url)
    config = dict(DEFAULT_DB_CONFIG)
    config["host"] = normalize_service_hostname(parsed.hostname or config["host"])
    config["user"] = parsed.username or config["user"]
    config["password"] = parsed.password or config["password"]
    config["port"] = parsed.port or config["port"]

    db_name = parsed.path.lstrip("/")
    if db_name:
        config["database"] = db_name

    return config


def get_ollama_base_url() -> str:
    """Returns the host-side Ollama base URL."""
    configured = get_config_value("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    return resolve_local_url(configured, DEFAULT_OLLAMA_BASE_URL)


def get_ollama_embed_model() -> str:
    """Returns the embedding model configured for local CLI tools."""
    return get_config_value("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)


def get_gateway_base_url() -> str:
    """Returns the OpenAI-compatible gateway base URL for local tooling."""
    configured = get_config_value("AGENT_API_BASE_URL", DEFAULT_GATEWAY_BASE_URL)
    return resolve_local_url(configured, DEFAULT_GATEWAY_BASE_URL)


def load_api_key(api_key_file: str = "api_key.txt") -> str:
    """Loads the agent API key from environment or the exported api_key.txt file."""
    env_key = os.getenv("AGENT_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    if os.path.exists(api_key_file):
        try:
            with open(api_key_file, "r", encoding="utf-8") as file:
                for line in file:
                    if line.startswith("AGENT_API_KEY="):
                        return line.split("=", 1)[1].strip()
        except Exception:
            pass

    env_file_key = load_env_file().get("AGENT_API_KEY", "").strip()
    if env_file_key:
        return env_file_key

    return DEFAULT_AGENT_API_KEY


def build_gateway_headers(api_key: str | None = None) -> dict[str, str]:
    """Builds standard JSON headers for the local gateway."""
    resolved_key = api_key or load_api_key()
    return {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }


def normalize_csv_list(raw_value: str | None, default_values: list[str]) -> list[str]:
    """Converts a comma-separated list argument into a cleaned list."""
    if not raw_value:
        return list(default_values)
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    return values or list(default_values)


def trim_snippet(text: str, max_chars: int) -> str:
    """Trims text into a single compact snippet."""
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def chunk_text(text: str, chunk_size: int = 1500, max_chunks: int | None = None) -> list[str]:
    """Splits text into bounded chunks while preserving line boundaries when possible."""
    if not text.strip():
        return []

    bounded_chunk_size = max(200, chunk_size)
    chunks: list[str] = []
    current_lines: list[str] = []
    current_length = 0

    for line in text.splitlines():
        normalized_line = line.rstrip()
        line_length = len(normalized_line) + 1

        if current_lines and current_length + line_length > bounded_chunk_size:
            chunks.append("\n".join(current_lines).strip())
            current_lines = [normalized_line]
            current_length = line_length
        else:
            current_lines.append(normalized_line)
            current_length += line_length

        if max_chunks and len(chunks) >= max_chunks:
            break

    if current_lines and (not max_chunks or len(chunks) < max_chunks):
        chunks.append("\n".join(current_lines).strip())

    return [chunk for chunk in chunks if chunk]
