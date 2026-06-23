import sys
import os
import time
import unittest
from unittest.mock import patch, MagicMock

# Force SQLite for unit tests
os.environ["DATABASE_URL"] = "sqlite:///./test_cache.db"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal, Base, engine, init_db_with_retry, cosine_similarity
from app.models import KnowledgeBase, LanguageDocumentation
import app.main as main_module
import app.search as search_module

class TestExternalBrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db_with_retry()
        Base.metadata.create_all(bind=engine)

    def setUp(self):
        self.db = SessionLocal()
        self.db.query(KnowledgeBase).delete()
        self.db.query(LanguageDocumentation).delete()
        self.db.commit()

    def tearDown(self):
        self.db.query(KnowledgeBase).delete()
        self.db.query(LanguageDocumentation).delete()
        self.db.commit()
        self.db.close()

    def test_numpy_cosine_similarity(self):
        print("\n--- Testing NumPy Cosine Similarity ---")
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)

        v3 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0)

        # Opposite vectors
        v4 = [-1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v4), -1.0)
        print("NumPy Cosine Similarity: PASS")

    @patch("requests.get")
    def test_markdownify_html_scraping(self, mock_get):
        print("\n--- Testing HTML to Markdown Scraping ---")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = b"<html><body><header>Nav</header><main><h1>Title</h1><p>This is a <b>bold</b> paragraph.</p><pre><code>a = 10</code></pre></main></body></html>"
        mock_get.return_value = mock_response

        scraped = search_module.scrape_url("http://example.com")
        print(f"Scraped content: {repr(scraped)}")
        self.assertTrue("# Title" in scraped or "Title\n=====" in scraped)
        self.assertIn("This is a **bold** paragraph.", scraped)
        self.assertIn("a = 10", scraped)
        print("HTML to Markdown Scraping: PASS")

    @patch("app.main.generate_knowledge_embedding_async")
    def test_documentation_rag_sync(self, mock_async_embed):
        print("\n--- Testing Documentation RAG Sync ---")
        lang = "rust"
        content = "Rust coding standard guidelines."
        
        main_module.save_or_update_knowledge_base_documentation(self.db, lang, content)
        
        # Check KnowledgeBase entry
        kb = self.db.query(KnowledgeBase).filter(KnowledgeBase.title.like("%Rust%")).first()
        self.assertIsNotNone(kb)
        self.assertEqual(kb.content, content)
        self.assertIn("rust", kb.tags)
        self.assertTrue(mock_async_embed.called)
        print("Documentation RAG Sync: PASS")

    @patch("requests.post")
    def test_conversation_knowledge_auto_extraction(self, mock_post):
        print("\n--- Testing Conversation Knowledge Auto-Extraction ---")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": '{"worthy": true, "title": "Mock JWT Setup in Python", "content": "FastAPI JWT setup python explanation."}'
            }
        }
        mock_post.return_value = mock_response

        # Execute worker directly
        main_module.extract_knowledge_from_conversation_worker(
            "How to set up JWT in FastAPI?",
            "Use FastAPI and PyJWT. Complete solution code: ..."
        )

        kb = self.db.query(KnowledgeBase).filter(KnowledgeBase.title == "Mock JWT Setup in Python").first()
        self.assertIsNotNone(kb)
        self.assertEqual(kb.content, "FastAPI JWT setup python explanation.")
        self.assertIn("python", kb.tags)
        print("Conversation Knowledge Auto-Extraction: PASS")

if __name__ == "__main__":
    unittest.main()
