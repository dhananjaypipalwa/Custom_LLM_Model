# internet_router.py
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

YEAR = time.localtime().tm_year

RECENCY_TERMS = [
    "today", "latest", "breaking", "news", "update", "recent", "this week",
    "this month", "this year", "as of", "currently", "right now", "new release",
    "price today", "stock price", "weather", "schedule", "timetable", "live score"
]
WEB_DOMAINS_HINTS = [".com", ".org", ".net", ".gov", ".io", "reddit", "github", "wikipedia", "kaggle", "arxiv"]

# Expanded to catch more “give me sources/links/docs” phrasing
CITATION_TERMS = ["cite", "citation", "source", "sources", "link", "links", "references", "url", "paper", "pdf", "docs", "documentation"]

DYNAMIC_TOPICS = ["price", "version", "release date", "changelog", "benchmark", "ranking", "download link"]

# New: educational/request-for-guides often implies we should fetch links
EDU_TERMS = ["tutorial", "guide", "how to", "examples", "docs", "documentation"]

YEAR_REGEX = re.compile(rf"\b(20\d{{2}}|{YEAR}|{YEAR-1}|{YEAR-2})\b")
MONTH_REGEX = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b", re.I)

# New: detect “difference”/“compare” questions like “what changed”, “vs v2”
DIFF_REGEX = re.compile(r"\b(what changed|difference|diff|vs\.?|compare|comparison)\b", re.I)

@dataclass
class RouterDecision:
    use_web: bool
    reason: str
    queries: List[str]

def _score_heuristics(question: str, history: Optional[List[Dict[str, str]]] = None) -> int:
    q = (question or "").lower()
    score = 0

    # Recency language
    if any(t in q for t in RECENCY_TERMS):
        score += 3

    # Explicit domains or web-ish hints
    if any(t in q for t in WEB_DOMAINS_HINTS):
        score += 3

    # Asking for citations/links/docs
    if any(t in q for t in CITATION_TERMS):
        score += 2

    # Tutorials/guides/docs imply links
    if any(t in q for t in EDU_TERMS):
        score += 2

    # Dynamic/changing topics
    if any(t in q for t in DYNAMIC_TOPICS):
        score += 2

    # Months/years in query
    if YEAR_REGEX.search(q):
        score += 1
    if MONTH_REGEX.search(q):
        score += 1

    # Difference/compare phrasing (often time-sensitive or source seeking)
    if DIFF_REGEX.search(q):
        score += 2

    # Hard triggers: event-ish
    if re.search(r"\b(what happened|who won|is it out|launched|released)\b", q):
        score += 2

    # Explicit request to use web
    if re.search(r"\b(internet|web|google|search|browse|online)\b", q):
        score += 10  # explicit request

    # Conversation context mentions citations
    if history:
        last = " ".join(h.get("content", "").lower() for h in history[-4:])
        if any(t in last for t in CITATION_TERMS):
            score += 1

    return score

def _make_queries(question: str) -> List[str]:
    q = question.strip()
    variants = [
        q,
        f"{q} site:wikipedia.org",
        f"{q} site:arxiv.org",
        f"{q} overview",
        f"{q} latest",
    ]
    uniq = []
    for v in variants:
        v = v.strip()
        if v and v not in uniq and len(v) <= 140:
            uniq.append(v)
    return uniq[:5]

def decide_web(question: str, history: Optional[List[Dict[str, str]]] = None,
               llm_second_opinion: bool = False,
               llm_callback=None) -> RouterDecision:
    score = _score_heuristics(question, history)
    use_web = score >= 4
    reason = f"Heuristic score={score} ? {'web' if use_web else 'no web'}"
    queries = _make_queries(question)

    # Optional LLM second-opinion
    if llm_second_opinion and llm_callback:
        system = "You are a router. Decide if a web search is needed to answer the user's question accurately right now."
        user = (
            "Return strict JSON with keys: use_web (bool), reason (short), queries (array of 1-5 strings).\n"
            "Use the web if the answer is time-sensitive, factual with potential drift, or if sources/citations are requested.\n"
            f"Question: {question}\n"
            f"Conversation summary: {history[-3:] if history else []}"
        )
        try:
            verdict = llm_callback(system, user) or {}
            if isinstance(verdict.get("use_web"), bool):
                use_web = verdict["use_web"]
            if verdict.get("reason"):
                reason = verdict["reason"]
            if isinstance(verdict.get("queries"), list) and verdict["queries"]:
                queries = [str(x)[:140] for x in verdict["queries"]][:5]
        except Exception as e:
            reason += f" | LLM router skipped: {e}"

    return RouterDecision(use_web=use_web, reason=reason, queries=queries)
