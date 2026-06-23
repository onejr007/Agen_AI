import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import logging
import json
import re

logger = logging.getLogger("agent.search")

def clean_text(text: str) -> str:
    """Removes excessive spaces, newlines, and script/style sections."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_url(url: str, timeout: int = 3) -> str:
    """Fetch URL and extract main body content as clean structured Markdown safely."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return ""
        
        # Check if content is HTML
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            return ""

        # Use C-based lxml parser for speed, fallback to html.parser if needed
        try:
            soup = BeautifulSoup(response.content, 'lxml')
        except Exception:
            soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove non-content elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
            
        try:
            from markdownify import markdownify as md
            content_md = md(str(soup), strip=['a', 'img'])
            # Normalize excessive spacing and consecutive newlines
            content_md = re.sub(r'\n\s*\n', '\n\n', content_md).strip()
            # Return up to 1200 characters of clean structured Markdown
            return content_md[:1200]
        except Exception as md_err:
            logger.warning(f"Markdownify failed: {str(md_err)}. Falling back to clean text.")
            text = soup.get_text(separator=' ')
            cleaned = clean_text(text)
            return cleaned[:500]
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {str(e)}")
        return ""

def search_internet(query: str, max_results: int = 2) -> list:
    """
    Search DuckDuckGo for the query.
    First tries direct HTML scraping (highly reliable, no rate-limiting inside containers),
    and falls back to duckduckgo_search library if HTML scraping fails.
    """
    results = []
    logger.info(f"Initiating internet search for: {query}")
    
    # Clean query from conversational filler
    clean_query = query.strip()
    # Remove phrases like: "search on internet for", "search the web for", etc.
    fillers = [
        r"(?i)\bsearch\s+(on|the)\s+(internet|web)\s+for\b",
        r"(?i)\bcari\s+di\s+internet\s+(tentang|untuk)\b",
        r"(?i)\bsearch\s+internet\b",
        r"(?i)\bcari\s+di\s+internet\b",
        r"(?i)\bplease\s+search\s+for\b",
        r"(?i)\bsearch\s+for\b",
        r"(?i)\blook\s+up\s+on\s+internet\b",
        r"(?i)\btolong\s+carikan\s+tentang\b"
    ]
    for pattern in fillers:
        clean_query = re.sub(pattern, "", clean_query)
    clean_query = clean_query.strip("? .! \t\n")
    if not clean_query:
        clean_query = query
        
    logger.info(f"Cleaned query: {clean_query}")

    # Method 1: Direct HTML scraping
    try:
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        r = requests.post(url, data={"q": clean_query}, headers=headers, timeout=5)
        if r.status_code == 200:
            try:
                soup = BeautifulSoup(r.content, 'lxml')
            except Exception:
                soup = BeautifulSoup(r.content, 'html.parser')
            divs = soup.find_all('div', class_='result')
            for div in divs[:max_results]:
                a_title = div.find('a', class_='result__a')
                if not a_title:
                    continue
                title = a_title.text.strip()
                href = a_title.get('href', '')
                
                # Parse snippet
                a_snippet = div.find('a', class_='result__snippet')
                snippet = a_snippet.text.strip() if a_snippet else ""
                
                # Scrape content from the link
                scraped_content = ""
                if href:
                    logger.info(f"Scraping content from: {href}")
                    scraped_content = scrape_url(href)
                    
                results.append({
                    "title": title,
                    "url": href,
                    "snippet": snippet,
                    "content": scraped_content if scraped_content else snippet
                })
            
            if results:
                logger.info(f"Direct HTML search succeeded with {len(results)} results.")
                return results
    except Exception as e:
        logger.warning(f"Direct HTML search failed: {str(e)}. Falling back to DDGS library.")

    # Method 2: Fallback to duckduckgo_search library
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(clean_query, max_results=max_results))
            for r in search_results:
                title = r.get("title", "")
                url = r.get("href", "")
                snippet = r.get("body", "")
                
                scraped_content = ""
                if url:
                    logger.info(f"Scraping content from: {url}")
                    scraped_content = scrape_url(url)
                
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "content": scraped_content if scraped_content else snippet
                })
    except Exception as e:
        logger.error(f"Fallback search library failed: {str(e)}")
        
    return results

def format_search_results(results: list, max_chars: int = None) -> str:
    """Formats search results list into a clean markdown context string, matching character limits."""
    if not results:
        return "No web search results found."
    
    formatted = ["### Internet Search Results\n"]
    current_len = sum(len(x) for x in formatted)
    
    for i, r in enumerate(results, 1):
        content_snippet = r.get('content', '')
        header_lines = [
            f"[{i}] Title: {r.get('title', 'No Title')}",
            f"    URL: {r.get('url', '')}"
        ]
        header_len = sum(len(hl) for hl in header_lines) + 20
        
        if max_chars is not None and current_len + header_len > max_chars:
            break
            
        max_snippet_len = 1200
        if max_chars is not None:
            max_snippet_len = min(1200, max_chars - current_len - header_len)
            if max_snippet_len < 100:
                break
                
        content_snippet = content_snippet[:max_snippet_len]
        if len(r.get('content', '')) > max_snippet_len:
            content_snippet += "..."
            
        formatted.extend(header_lines)
        formatted.append(f"    Content: {content_snippet}")
        formatted.append("-" * 40)
        current_len = sum(len(x) for x in formatted)
    
    return "\n".join(formatted)
