import sys
import os
import time
import datetime
import unittest
from unittest.mock import patch, MagicMock

# Force SQLite for unit tests
os.environ["DATABASE_URL"] = "sqlite:///./test_cache.db"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal, Base, engine, init_db_with_retry
from app.models import LanguageDocumentation
import app.main as main_module

class TestSelfLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db_with_retry()
        Base.metadata.create_all(bind=engine)

    def setUp(self):
        self.db = SessionLocal()
        self.db.query(LanguageDocumentation).delete()
        self.db.commit()

    def tearDown(self):
        self.db.query(LanguageDocumentation).delete()
        self.db.commit()
        self.db.close()

    @patch("app.main.search_internet")
    @patch("requests.post")
    def test_cache_hit_and_expiration(self, mock_post, mock_search):
        # 1. Setup mock search results and Ollama response
        mock_search.return_value = [
            {"title": "PHP Tutorial", "href": "http://php.net", "body": "PHP is a scripting language."}
        ]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Synthesized PHP Documentation Guide"}
        }
        mock_post.return_value = mock_response

        # 2. Call the function first time (Cache MISS)
        doc = main_module.get_or_fetch_language_documentation(self.db, "php")
        self.assertEqual(doc, "Synthesized PHP Documentation Guide")
        self.assertEqual(mock_search.call_count, 1)

        # 3. Call the function second time (Cache HIT)
        doc2 = main_module.get_or_fetch_language_documentation(self.db, "php")
        self.assertEqual(doc2, "Synthesized PHP Documentation Guide")
        self.assertEqual(mock_search.call_count, 1)

        # 4. Simulate cache expiration (updated_at > 7 days ago)
        cached_entry = self.db.query(LanguageDocumentation).filter(
            LanguageDocumentation.language_name == "php"
        ).first()
        self.assertIsNotNone(cached_entry)
        
        # Set updated_at to 8 days ago
        cached_entry.updated_at = datetime.datetime.utcnow() - datetime.timedelta(days=8)
        self.db.commit()

        # 5. Call the function third time (Cache EXPIRED)
        doc3 = main_module.get_or_fetch_language_documentation(self.db, "php")
        self.assertEqual(doc3, "Synthesized PHP Documentation Guide")
        self.assertEqual(mock_search.call_count, 2)

    @patch("app.main.search_internet")
    @patch("requests.post")
    @patch("app.main.redis_client", None)  # Isolate from live Redis — use time-based check only
    def test_self_learning_idle_and_interruption(self, mock_post, mock_search):
        mock_search.return_value = [
            {"title": "Go Lang", "href": "http://golang.org", "body": "Go is statically typed."}
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Synthesized Go Documentation"}
        }
        mock_post.return_value = mock_response

        # 1. Reset last_request_time to simulate 10 minutes ago (Idle)
        main_module.last_request_time = time.time() - 600
        self.assertFalse(main_module.is_learning_interrupted())


        # 2. Trigger perform_self_learning
        main_module.perform_self_learning()

        # Check that it successfully learned Go (first lang in candidates, 'luau')
        cached_luau = self.db.query(LanguageDocumentation).filter(
            LanguageDocumentation.language_name == "luau"
        ).first()
        self.assertIsNotNone(cached_luau)
        self.assertEqual(cached_luau.documentation_content, "Synthesized Go Documentation")

        # 3. Clean up database to test interruption
        self.db.query(LanguageDocumentation).delete()
        self.db.commit()

        # 4. Simulate request coming in immediately after starting learning (Interrupted)
        main_module.last_request_time = time.time() - 600
        
        original_is_interrupted = main_module.is_learning_interrupted
        try:
            # First check returns False (idle), second check returns True (incoming request)
            call_times = []
            def side_effect():
                call_times.append(time.time())
                if len(call_times) > 1:
                    return True
                return False
            
            main_module.is_learning_interrupted = side_effect
            main_module.perform_self_learning()

            # The DB should NOT have Luau since it got interrupted mid-process
            cached_luau_check = self.db.query(LanguageDocumentation).filter(
                LanguageDocumentation.language_name == "luau"
            ).first()
            self.assertIsNone(cached_luau_check)
            print("Self-learning interruption successfully verified!")
        finally:
            main_module.is_learning_interrupted = original_is_interrupted

if __name__ == "__main__":
    unittest.main()
