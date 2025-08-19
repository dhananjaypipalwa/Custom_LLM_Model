# internet_tools.py
import asyncio
from typing import List, Dict, Optional, Iterable, Tuple, Set
from dataclasses import dataclass
from urllib.parse import urlparse
import time
import os

import httpx
try:
    from ddgs import DDGS  # preferred
except Exception:
    from duckduckgo_search import DDGS  # fallback
import trafilatura

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    )
}

# Filter noisy/low-signal/blocked domains and API/search endpoints
BLACKLIST_DOMAINS = (
    "bing.com", "msn.com", "news.yahoo.com", "yahoo.com",
    "canva.com", "facebook.com", "pinterest.", "linkedin.com",
    "x.com", "t.co"
)
DROP_PATH_CONTAINS = ("/w/api.php", "/search?", "/login", "/signup")

# ---------- Simple in-memory TTL caches (speed up repeats) ----------
_PAGE_CACHE: Dict[str, Tuple[float, str]] = {}
_SEARCH_CACHE: Dict[str, Tuple[float, List[Dict]]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

def _cache_get(d, k):
    item = d.get(k)
    if not item:
        return None
    ts, val = item
    if (time.time() - ts) > CACHE_TTL_SECONDS:
        d.pop(k, None)
        return None
    return val

def _cache_set(d, k, v):
    d[k] = (time.time(), v)

# -------------------------------------------------------------------

@dataclass
class WebDoc:
    url: str
    title: Optional[str]
    snippet: Optional[str]
    text: str

def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def _is_bad_url(url: str) -> bool:
    try:
        u = urlparse(url)
        if u.scheme not in ("http", "https"):
            return True
        host = u.netloc.lower()
        if any(bad in host for bad in BLACKLIST_DOMAINS):
            return True
        if any(seg in u.path for seg in DROP_PATH_CONTAINS):
            return True
        return False
    except Exception:
        return True

def _dedupe_keep_order(items: Iterable[Dict]) -> List[Dict]:
    seen: Set[str] = set()
    out: List[Dict] = []
    for r in items:
        url = r.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(r)
    return out

def search_ddg_one(query: str, max_results: int = 8, timelimit: Optional[str] = None, prefer_news: bool = False) -> List[Dict]:
    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=max_results, timelimit=timelimit))
    results: List[Dict] = []
    for r in raw:
        url = r.get("href") or r.get("url")
        if not url or _is_bad_url(url):
            continue
        results.append({
            "title": r.get("title"),
            "url": url,
            "snippet": r.get("body") or r.get("snippet")
        })
    return results

def prioritize_results(results: List[Dict], preferred_domains: Optional[List[str]]) -> List[Dict]:
    if not preferred_domains:
        return results
    preferred = tuple(d.lower() for d in preferred_domains)
    def score(r: Dict) -> int:
        h = _host(r.get("url", ""))
        # exact domain match gets higher score; subdomain match gets medium
        if any(h == d for d in preferred):
            return 100
        if any(h.endswith("." + d) for d in preferred):
            return 80
        return 0
    return sorted(results, key=score, reverse=True)

async def _fetch_one(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(
        url,
        headers=HEADERS,
        timeout=httpx.Timeout(connect=2.0, read=4.0, write=4.0, pool=5.0),
    )
    r.raise_for_status()
    return r.text

def _extract_text(html: str, url: str) -> str:
    txt = trafilatura.extract(html, url=url, include_tables=False, include_comments=False)
    return txt or ""

async def fetch_pages(urls: List[str], concurrency: int = 8) -> Dict[str, str]:
    # Use cache first
    out: Dict[str, str] = {}
    to_fetch: List[str] = []
    for u in urls:
        cached = _cache_get(_PAGE_CACHE, u)
        if cached is not None:
            out[u] = cached
        else:
            to_fetch.append(u)

    if not to_fetch:
        return out

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        async def run(u: str):
            async with sem:
                try:
                    html = await _fetch_one(client, u)
                    txt = _extract_text(html, u)
                    _cache_set(_PAGE_CACHE, u, txt)
                    return u, txt
                except Exception:
                    _cache_set(_PAGE_CACHE, u, "")
                    return u, ""
        pairs = await asyncio.gather(*(run(u) for u in to_fetch))

    out.update(dict(pairs))
    return out

async def web_gather(
    query: str,
    max_results: int = 3,
    timelimit: Optional[str] = None,
    prefer_news: bool = False,
    want_docs: int = 2,
    min_chars_each: int = 350,
    trim_chars_each: int = 1200,
    # NEW:
    queries: Optional[List[str]] = None,
    preferred_domains: Optional[List[str]] = None,
    require_any_of: Optional[List[str]] = None
) -> List[WebDoc]:
    """
    If `queries` is provided, we search each and merge/dedupe results.
    `preferred_domains` are prioritized.
    If `require_any_of` is provided, we try to ensure at least one doc from any of those domains.
    """
    all_results: List[Dict] = []
    search_list = queries if queries else [query]
    for q in search_list:
        rs = search_ddg_one(q, max_results=max_results, timelimit=timelimit, prefer_news=prefer_news)
        all_results.extend(rs)
    all_results = _dedupe_keep_order(all_results)
    all_results = prioritize_results(all_results, preferred_domains)

    docs: List[WebDoc] = []
    have_required = False

    # Fetch in small batches; early stop when enough usable docs are gathered
    BATCH = 3
    for i in range(0, len(all_results), BATCH):
        batch = all_results[i:i+BATCH]
        urls = [r["url"] for r in batch]
        url_to_text = await fetch_pages(urls, concurrency=8)

        for r in batch:
            text = (url_to_text.get(r["url"], "") or "").strip()
            if len(text) < min_chars_each:
                continue
            text = text[:trim_chars_each]
            h = _host(r["url"])
            if require_any_of and any(h == d or h.endswith("." + d) for d in require_any_of):
                have_required = True
            docs.append(WebDoc(
                url=r["url"],
                title=r.get("title"),
                snippet=r.get("snippet"),
                text=text
            ))
            if len(docs) >= want_docs:
                break
        if len(docs) >= want_docs:
            break

    # If we required an official domain but didn't get one, return whatever we have (router will warn)
    return docs

def render_citations(docs: List[WebDoc]) -> List[Dict]:
    items = []
    for d in docs:
        items.append({
            "url": d.url,
            "title": d.title or d.url,
            "preview": (d.snippet or d.text[:180]).strip().replace("\n", " ")
        })
    return items

def build_context(docs: List[WebDoc], question: str, max_chars: int = 4000) -> str:
    """Smaller context for faster tokenization/generation."""
    pieces: List[str] = []
    running = 0
    for i, d in enumerate(docs, 1):
        header = f"\n\n### Source {i}: {d.title or d.url}\nURL: {d.url}\n\n"
        body = d.text.strip()
        chunk = (header + body)[: max(0, max_chars - running)]
        if not chunk:
            break
        pieces.append(chunk)
        running += len(chunk)
        if running >= max_chars:
            break
    return (
        "You are a careful researcher. Use the sources below to answer the question.\n"
        "Cite sources inline like [1], [2] that correspond to the numbered sources.\n"
        "If there is no official source among them, say so explicitly and avoid unverified claims.\n"
        f"\nQuestion: {question}\n"
        "\nSources:\n" + "".join(pieces)
    )

async def search_tavily(query: str, max_results: int = 5) -> List[Dict]:
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY not set")

    cache_key = f"{query}::{max_results}"
    cached = _cache_get(_SEARCH_CACHE, cache_key)
    if cached is not None:
        return cached

    url = "https://api.tavily.com/search"
    payload = {"query": query, "max_results": max_results}
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        results = [
            {
                "title": it.get("title"),
                "url": it.get("url"),
                "snippet": it.get("content")
            }
            for it in data.get("results", [])
        ]
        _cache_set(_SEARCH_CACHE, cache_key, results)
        return results
