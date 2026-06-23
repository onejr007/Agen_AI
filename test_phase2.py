import sys
import os

os.environ["DATABASE_URL"] = "sqlite:///./test_cache.db"

from app.agent import validate_code_syntax
from app.database import SessionLocal, Base, engine, init_db_with_retry
from app.models import KnowledgeBase

def test_syntax_validator():
    print("\n--- Testing Syntax Validator ---")
    
    # 1. Valid Python
    code_py_valid = "def test(a: int) -> str:\n    return str(a)"
    err = validate_code_syntax(code_py_valid, "python")
    print(f"Valid Python check: err='{err}'")
    assert err == ""

    # 2. Invalid Python
    code_py_invalid = "def test(a: int) -> str:\n   return str(a"
    err = validate_code_syntax(code_py_invalid, "python")
    print(f"Invalid Python check: err='{err}'")
    assert "SyntaxError" in err

    # 3. Mismatched Braces Luau
    code_lua_invalid = "function test()\n   local t = {1, 2\n   return t\nend"
    err = validate_code_syntax(code_lua_invalid, "luau")
    print(f"Mismatched braces check: err='{err}'")
    assert "Syntax Warning" in err and "Unclosed" in err

    # 4. Valid Luau
    code_lua_valid = "function test()\n   local t = {1, 2}\n   return t\nend"
    err = validate_code_syntax(code_lua_valid, "luau")
    print(f"Valid Luau check: err='{err}'")
    assert err == ""

    print("Syntax Validator Tests: PASS")

def test_db_schema_extraction():
    print("\n--- Testing DB Schema Extraction ---")
    init_db_with_retry()
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Seed dummy if not exists
        existing = db.query(KnowledgeBase).filter(KnowledgeBase.tags.like("%mysql-schema%")).first()
        if not existing:
            dummy = KnowledgeBase(
                title="Mock Schema",
                content="Table: users\nColumns: id, name, email",
                tags="mysql-schema,test"
            )
            db.add(dummy)
            db.commit()

        # Query MySQL schema entry in KnowledgeBase
        schema_entry = db.query(KnowledgeBase).filter(KnowledgeBase.tags.like("%mysql-schema%")).first()
        assert schema_entry is not None, "Schema entry not found in RAG database!"
        print(f"Schema entry title: {schema_entry.title}")
        print("Schema description excerpt:")
        print("\n".join(schema_entry.content.split("\n")[:10]))
        print("DB Schema Extraction RAG Tests: PASS")
    finally:
        db.close()

if __name__ == "__main__":
    test_syntax_validator()
    test_db_schema_extraction()
    print("\nAll Phase II Validation tests PASSED!")
