import os
from pydantic_settings import BaseSettings


def _parse_bool_env(name: str, default: str) -> bool:
    """Parses boolean-like environment variables safely."""
    return os.getenv(name, default).strip().lower() in ("true", "1", "yes", "on")


def _parse_int_env(name: str, default: str) -> int:
    """Parses integer environment variables with a sane fallback."""
    try:
        return int(os.getenv(name, default).strip())
    except (AttributeError, TypeError, ValueError):
        return int(default)

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "mysql+pymysql://agent_user:agent_password_123@db/agent_db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    AGENT_API_KEY: str = os.getenv("AGENT_API_KEY", "local_developer_secret_key")
    AUTO_PULL_MODEL: bool = _parse_bool_env("AUTO_PULL_MODEL", "True")
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    WHOOGLE_URL: str = os.getenv("WHOOGLE_URL", "")
    CORS_ALLOWED_ORIGINS: str = os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost,http://127.0.0.1,http://localhost:3000,http://127.0.0.1:3000"
    )
    CORS_ALLOW_CREDENTIALS: bool = _parse_bool_env("CORS_ALLOW_CREDENTIALS", "False")
    WRITE_API_KEY_FILE: bool = _parse_bool_env("WRITE_API_KEY_FILE", "True")
    MAX_CHAT_MESSAGES: int = _parse_int_env("MAX_CHAT_MESSAGES", "100")
    MAX_MESSAGE_CHARS: int = _parse_int_env("MAX_MESSAGE_CHARS", "24000")
    MAX_EMBEDDING_ITEMS: int = _parse_int_env("MAX_EMBEDDING_ITEMS", "32")
    MAX_EMBEDDING_INPUT_CHARS: int = _parse_int_env("MAX_EMBEDDING_INPUT_CHARS", "12000")
    RETRIEVAL_DEFAULT_LIMIT: int = _parse_int_env("RETRIEVAL_DEFAULT_LIMIT", "3")
    RETRIEVAL_MAX_CONTENT_CHARS: int = _parse_int_env("RETRIEVAL_MAX_CONTENT_CHARS", "500")
    RETRIEVAL_MAX_TOTAL_CHARS: int = _parse_int_env("RETRIEVAL_MAX_TOTAL_CHARS", "1800")
    RETRIEVAL_KB_SIMILARITY_THRESHOLD: int = _parse_int_env("RETRIEVAL_KB_SIMILARITY_THRESHOLD", "65")
    RETRIEVAL_MESSAGE_SIMILARITY_THRESHOLD: int = _parse_int_env("RETRIEVAL_MESSAGE_SIMILARITY_THRESHOLD", "70")
    RETRIEVAL_KEYWORD_MIN_SCORE: int = _parse_int_env("RETRIEVAL_KEYWORD_MIN_SCORE", "20")
    RETRIEVAL_RECENCY_BOOST_DAYS: int = _parse_int_env("RETRIEVAL_RECENCY_BOOST_DAYS", "14")
    RETRIEVAL_KEYWORD_SCAN_LIMIT: int = _parse_int_env("RETRIEVAL_KEYWORD_SCAN_LIMIT", "250")
    RETRIEVAL_VECTOR_SCAN_LIMIT: int = _parse_int_env("RETRIEVAL_VECTOR_SCAN_LIMIT", "400")

    # Agent execution safety gate (especially important for Roo Code / Cline tool execution)
    REQUIRE_APPROVAL_FOR_MUTATIONS: bool = _parse_bool_env("REQUIRE_APPROVAL_FOR_MUTATIONS", "True")
    EXECUTION_APPROVAL_KEYWORDS: str = os.getenv(
        "EXECUTION_APPROVAL_KEYWORDS",
        "setuju, lanjut, lanjut eksekusi, ok, oke, gas, jalankan, eksekusi, approve, approved, proceed"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        """Returns normalized CORS origins from a comma-separated environment variable."""
        origins = [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]
        return origins or ["http://localhost"]

    @property
    def uses_default_api_key(self) -> bool:
        """Indicates whether the well-known default development key is still in use."""
        return self.AGENT_API_KEY.strip() == "local_developer_secret_key"

settings = Settings()
