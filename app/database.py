from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings
import time
import logging
import math
import json

logger = logging.getLogger("agent.db")

# Create engine with connection pooling and pre-ping to ensure MySQL connection stays healthy
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_recycle=1800,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def create_database_if_not_exists():
    """Extracts MySQL credentials from DATABASE_URL and creates the database if it doesn't exist on host XAMPP."""
    db_url = settings.DATABASE_URL
    if not db_url.startswith("mysql"):
        return

    try:
        prefix, rest = db_url.split("://", 1)
        user_pass_host, db_name = rest.rsplit("/", 1)
        if "?" in db_name:
            db_name = db_name.split("?", 1)[0]
            
        base_url = f"{prefix}://{user_pass_host}/"
        
        # Create a temporary engine without database name
        temp_engine = create_engine(base_url, pool_pre_ping=True)
        with temp_engine.connect() as conn:
            # Create database if it doesn't exist
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"))
            logger.info(f"Database '{db_name}' checked/created successfully on XAMPP MySQL.")
        temp_engine.dispose()
    except Exception as e:
        logger.error(f"Failed to check/create database on XAMPP MySQL: {str(e)}")

def init_db_with_retry(max_retries=15, delay=3):
    """Attempts to connect to the database with retries to handle container startup lag."""
    # Ensure database exists first
    create_database_if_not_exists()
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Database connection attempt {attempt}/{max_retries}...")
            # Try to establish connection and run a simple query
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully.")
            return True
        except Exception as e:
            logger.warning(f"Database connection failed: {str(e)}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise ConnectionError("Failed to connect to the database after multiple retries.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def cosine_similarity(v1: list, v2: list) -> float:
    """Calculates cosine similarity between two float vectors in pure Python."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

def parse_json_embedding(raw_embedding) -> list[float]:
    """Safely parses a JSON-encoded embedding vector into a float list."""
    if raw_embedding is None:
        return []

    try:
        parsed = json.loads(raw_embedding) if isinstance(raw_embedding, str) else raw_embedding
    except (TypeError, ValueError, json.JSONDecodeError):
        return []

    if not isinstance(parsed, list):
        return []

    vector = []
    for value in parsed:
        try:
            vector.append(float(value))
        except (TypeError, ValueError):
            return []

    return vector
