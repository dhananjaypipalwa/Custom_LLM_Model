# emergegpt_router.py
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import os, json, uuid
from typing import Any, Dict, Optional, List
from datetime import datetime
from starlette.background import BackgroundTask
from tool_args import extract_tool_args

from model_loader import tokenizer, model, device
from rag_utils import stream_generate_response
from parser_mistral import parse_resume
from jd_generation import generate_job_description
from match_score import match_resume_dict, load_job_params
from sentence_transformers import SentenceTransformer
from internet_router import decide_web, RouterDecision  # (ok to keep even if unused)
from internet_tools import (
    web_gather,
    build_context,
    render_citations,
    search_tavily,
    fetch_pages,
    WebDoc,
)

import httpx  # NEW: for Ollama forwarding

# for log file
import logging, logging.handlers, queue, time, asyncio  # ADDED

# for log file
# ======================
# Non-blocking logging (QueueHandler + Listener)
# ======================
_log_queue: "queue.Queue[logging.LogRecord]" = queue.Queue(-1)

_stream_handler = logging.StreamHandler()  # prints to terminal
_stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

_file_handler = logging.handlers.RotatingFileHandler(
    "emergegpt.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

_listener = logging.handlers.QueueListener(_log_queue, _stream_handler, _file_handler)
_listener.start()

logger = logging.getLogger("emergegpt")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(logging.handlers.QueueHandler(_log_queue))

# for log file
# ======================
# Helpers to log start/end with timing
# ======================
def _req_meta(request) -> Dict[str, Any]:
    try:
        ip = getattr(request.client, "host", None) or "-"
    except Exception:
        ip = "-"
    ua = request.headers.get("user-agent", "-")
    auth = request.headers.get("authorization", "-")
    return {"ip": ip, "ua": ua, "has_auth": bool(auth and auth.strip() != "-")}

def _log_start(route: str, request, extra: Dict[str, Any]) -> float:
    meta = {**_req_meta(request), **(extra or {})}
    logger.info(f"[HIT] {route} | meta={meta}")
    return time.perf_counter()

def _log_end(route: str, start_ts: float, status: int = 200, extra: Dict[str, Any] = None):
    dur = time.perf_counter() - start_ts
    logger.info(f"[DONE] {route} | status={status} | duration_sec={dur:.3f} | extra={(extra or {})}")

# for log file
def _bg_close_and_log(client: httpx.AsyncClient, route: str, start_ts: float, extra: Dict[str, Any]):
    # best-effort async close without blocking event loop
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(client.aclose())
            else:
                loop.run_until_complete(client.aclose())
        except Exception:
            pass
    finally:
        _log_end(route, start_ts, 200, extra or {})

# for log file (question)
def _trim_text(s: Optional[str], max_len: int = 500) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip()
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."

# for log file (question)
def _extract_question_from_body(body: Dict[str, Any]) -> Optional[str]:
    try:
        if isinstance(body, dict) and "task" in body:
            task = body.get("task")
            data = body.get("input", {}) or {}
            if task in ("chat", "generate"):
                return data.get("prompt")
            if task == "web_chat":
                return data.get("question")
            if task == "openai_chat":
                msgs = data.get("messages", []) or []
                for m in reversed(msgs):
                    if (m or {}).get("role") == "user":
                        return m.get("content")
                for m in msgs:
                    if (m or {}).get("role") == "user":
                        return m.get("content")
                return None
        # OpenAI-style (no "task")
        msgs = (body or {}).get("messages", []) or []
        for m in reversed(msgs):
            if (m or {}).get("role") == "user":
                return m.get("content")
        for m in msgs:
            if (m or {}).get("role") == "user":
                return m.get("content")
    except Exception:
        return None
    return None

router = APIRouter()
model.eval()

# === Auth token ===
API_TOKEN = "emergegpt-secure-token"
async def verify_token(authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing token")

# === Request model (legacy task envelope) ===
class EmergeRequest(BaseModel):
    task: str
    input: dict

# === Heavy singleton: SentenceTransformer (load once) ===
# Avoids multi-second stalls on every /match_resumes call.
_EMBEDDER = SentenceTransformer("BAAI/bge-large-en-v1.5")

# --- Small helper to safely stream JSON fragments with arbitrary tokens ---
def _json_escape_fragment(s: str) -> str:
    # Use json.dumps and strip the surrounding quotes to get a valid JSON string fragment
    return json.dumps(s)[1:-1]

# for log file
def _looks_like_tool_echo(text: str) -> bool:
    """Detects when model just echoed tool JSON/tags instead of answering."""
    t = (text or "").strip()
    if "<tool_result>" in t or "</tool_result>" in t:
        return True
    # crude JSON echo detector (short, pure JSON-like content)
    if t.startswith("{") and t.endswith("}") and len(t) < 1200:
        return True
    return False


def build_mistral_prompt(messages: List[dict]) -> str:
    """
    Mistral chat template:
    <s>[INST] <<SYS>>{sys}<</SYS>> {u1} [/INST] {a1}</s>
    <s>[INST] {u2} [/INST]
    """
    sys_text = ""
    chunks = []
    for m in messages:
        if m["role"] == "system":
            sys_text = m["content"]
            break

    opened = False
    i = 0
    while i < len(messages):
        role = messages[i]["role"]
        content = messages[i]["content"]

        if role == "system":
            i += 1
            continue

        if role == "user":
            if not opened:
                sys_block = f"<<SYS>>\n{sys_text}\n<</SYS>>\n\n" if sys_text else ""
                chunks.append(f"<s>[INST] {sys_block}{content}")
                opened = True
            else:
                # close previous turn and start a new user turn
                chunks.append(f"</s>\n<s>[INST] {content}")
            # if next is assistant text, attach it after closing [/INST]
            if i + 1 < len(messages) and messages[i+1]["role"] == "assistant":
                chunks[-1] = chunks[-1] + f" [/INST] {messages[i+1]['content']}"
                i += 1  # consume assistant (already appended)
            else:
                # keep the INST open; we may append tool results into it
                pass

        elif role == "tool":
            # Append tool output inside the current open [INST] turn as context
            # so the model can read it before answering.
            if chunks:
                chunks[-1] = chunks[-1] + f"\n\n<tool_result>\n{content}\n</tool_result>"
            else:
                # If for some reason the first message is a tool result, start an INST
                chunks.append(f"<s>[INST] <tool_result>\n{content}\n</tool_result>")

        elif role == "assistant":
            # If we have an open [INST] without a closing tag, close it here
            if chunks:
                if not chunks[-1].endswith("[/INST]") and "[INST]" in chunks[-1]:
                    chunks[-1] = chunks[-1] + f" [/INST] {content}"
                else:
                    chunks.append(content)
            else:
                chunks.append(content)

        # else: ignore unknown roles
        i += 1

    # Ensure the final user turn is closed with [/INST] if still open
    if chunks and ("[INST]" in chunks[-1]) and (not chunks[-1].endswith("[/INST]")):
        chunks[-1] = chunks[-1] + " [/INST]"

    return "\n".join(chunks)



# =========================
# NEW: OpenAI-style schema
# =========================
class OAIMsg(BaseModel):
    role: str
    content: str

# --- ADDED: tool schema (minimal) ---
class OAIFunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None  # JSON Schema

class OAITool(BaseModel):
    type: str  # must be "function"
    function: OAIFunctionDef

class OAIReq(BaseModel):
    model: str                      # "emergegpt" | "llama3" | "qwen3" | "deepseek-r1" ...
    messages: List[OAIMsg]
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    # --- ADDED: tool fields ---
    tools: Optional[List[OAITool]] = None
    tool_choice: Optional[str] = None  # "auto" | "none" | "required"

# Ollama base URL (one daemon can host many models)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api").rstrip("/")

def _messages_to_inst_prompt(messages: List[Dict[str, str]]) -> str:
    """Same INST prompt logic you use in `openai_chat`."""
    parts = []
    for msg in messages:
        role, content = msg.get("role"), msg.get("content")
        if role == "system":
            parts.insert(0, f"[INST] {content} [/INST]")
        elif role == "user":
            parts.append(f"[INST] {content} [/INST]")
        elif role == "assistant":
            parts.append(content)
    return "\n".join(parts)

def _format_openai_response_ollama(ollama_response, model_name: str, request_messages: List[OAIMsg]) -> Dict[str, Any]:
    prompt_text = " ".join([m.content for m in request_messages])
    response_text = ollama_response.get("message", {}).get("content", "")
    prompt_tokens = len(prompt_text) // 4
    completion_tokens = len(response_text) // 4
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }

# CHANGED: accept a shared client; do not create/close inside
async def _stream_ollama(client: httpx.AsyncClient, url: str, req_json: Dict[str, Any], model_name: str):
    """Proxy Ollama streaming into OpenAI-style SSE deltas using a caller-provided client."""
    req_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(datetime.now().timestamp())

    # initial role delta
    yield f'data: {json.dumps({"id": req_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]})}\n\n'

    async with client.stream("POST", url, json=req_json, timeout=300.0) as r:
        async for text in r.aiter_text():
            if not text:
                continue
            for line in text.strip().split("\n"):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        delta = data["message"]["content"]
                        yield f'data: {json.dumps({"id": req_id, "object":"chat.completion.chunk","created": created,"model": model_name,"choices":[{"index":0,"delta":{"content": delta},"finish_reason": None}]})}\n\n'
                    if data.get("done"):
                        yield f'data: {json.dumps({"id": req_id,"object":"chat.completion.chunk","created": created,"model": model_name,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]})}\n\n'
                        yield "data: [DONE]\n\n"
                except Exception:
                    continue

# ============================================================
# 1) Legacy route (kept for backward compatibility)
# ============================================================
@router.post("/v1/emergegpt", dependencies=[Depends(verify_token)])
async def emergegpt_router(request: Request):
    # for log file
    _start_ts = _log_start("/v1/emergegpt", request, {"note": "legacy envelope"})

    try:
        body = await request.json()
        task = body.get("task")
        data = body.get("input", {})

        # for log file
        logger.info(f"[INFO] /v1/emergegpt body_keys={list(body.keys())} task={task} input_keys={list(data.keys())}")
        # for log file (question)
        try:
            _q = _extract_question_from_body(body)
            if _q:
                logger.info(f"[Q] /v1/emergegpt | { _trim_text(_q, 800) }")
        except Exception:
            pass

        # -------------------- Chat --------------------
        if task == "chat":
            prompt = data.get("prompt", "")
            full_prompt = f"[INST] {prompt} [/INST]"
            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # for log file
            _log_end("/v1/emergegpt", _start_ts, 200, {"task": "chat", "stream": False})
            return {"response": decoded.split("[/INST]")[-1].strip()}

        # -------------------- Resume Parsing --------------------
        elif task == "resume_parse":
            file_path = data.get("file_path")
            if not file_path or not os.path.exists(file_path):
                # for log file
                _log_end("/v1/emergegpt", _start_ts, 400, {"task": "resume_parse", "error": "file not found"})
                raise HTTPException(status_code=400, detail="Resume file not found.")
            parsed = parse_resume(file_path)
            # for log file
            _log_end("/v1/emergegpt", _start_ts, 200, {"task": "resume_parse"})
            return {"parsed": parsed}

        # -------------------- Job Description Generation --------------------
        elif task == "generate_jd":
            jd = generate_job_description(data)
            # for log file
            _log_end("/v1/emergegpt", _start_ts, 200, {"task": "generate_jd"})
            return {"job_description": jd}

        # -------------------- Resume Matching --------------------
        elif task == "match_resumes":
            job_params = load_job_params()
            model_embed = _EMBEDDER  # singleton

            job_skills_vecs = model_embed.encode(job_params["skills_required"], convert_to_tensor=True, normalize_embeddings=True)
            job_title_vec = model_embed.encode(job_params["job_title"], convert_to_tensor=True, normalize_embeddings=True)
            preferred_edu_vec = model_embed.encode(job_params["preferred_education"], convert_to_tensor=True, normalize_embeddings=True)

            resume = data.get("resume_dict")
            if resume:
                score = match_resume_dict(resume, job_params, model_embed, job_skills_vecs, job_title_vec, preferred_edu_vec)
                # for log file
                _log_end("/v1/emergegpt", _start_ts, 200, {"task": "match_resumes", "mode": "single"})
                return {"match": score}

            folder = "parsed_json"
            os.makedirs("job_matching_result", exist_ok=True)
            all_results = []

            for file in os.listdir(folder):
                if file.endswith(".json"):
                    with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                        resumes = json.load(f)
                        if isinstance(resumes, dict):
                            resumes = [resumes]
                        for r in resumes:
                            result = match_resume_dict(r, job_params, model_embed, job_skills_vecs, job_title_vec, preferred_edu_vec)
                            all_results.append(result)

            all_results.sort(key=lambda x: x["match_score"], reverse=True)
            top_10 = all_results[:10]

            with open("job_matching_result/results.json", "w", encoding="utf-8") as f:
                json.dump({"match_count": len(top_10), "matches": top_10}, f, indent=2)

            # for log file
            _log_end("/v1/emergegpt", _start_ts, 200, {"task": "match_resumes", "mode": "batch", "count": len(top_10)})
            return {"match_count": len(top_10), "matches": top_10}

        # -------------------- Web Search (Tavily ? DuckDuckGo fallback) --------------------
        elif task == "web_search":
            query = data.get("query")
            if not query:
                # for log file
                _log_end("/v1/emergegpt", _start_ts, 400, {"task": "web_search", "error": "missing query"})
                raise HTTPException(status_code=400, detail="Missing 'query' in input")

            max_results = data.get("max_results", 10)  # tighter
            prefer_news = data.get("prefer_news", False)

            # Try Tavily first (reads API key at call time)
            try:
                results = await search_tavily(query, max_results=max_results)
                # for log file
                _log_end("/v1/emergegpt", _start_ts, 200, {"task": "web_search", "provider": "tavily", "count": len(results)})
                return {
                    "results": results,
                    "count": len(results),
                    "provider": "tavily"
                }
            except Exception as e:
                print(f"[web_search] Tavily failed or not configured: {e} -> fallback DuckDuckGo")

            # Fallback to DuckDuckGo
            docs = await web_gather(
                query=query,
                max_results=max_results,
                prefer_news=prefer_news
            )
            # for log file
            _log_end("/v1/emergegpt", _start_ts, 200, {"task": "web_search", "provider": "duckduckgo", "count": len(docs)})
            return {
                "results": render_citations(docs),
                "count": len(docs),
                "provider": "duckduckgo"
            }

        # -------------------- Web-Augmented Chat (search + cite) --------------------
        elif task == "web_chat":
            question = data.get("question")
            max_results = data.get("max_results", 10)
            prefer_news = data.get("prefer_news", False)
            want_docs = data.get("want_docs", 5)
            if not question:
                # for log file
                _log_end("/v1/emergegpt", _start_ts, 400, {"task": "web_chat", "error": "missing question"})
                raise HTTPException(status_code=400, detail="Missing 'question' in input")

            # 1) Search (Tavily ? DuckDuckGo fallback)
            docs: List[WebDoc] = []
            provider_used = "tavily"
            try:
                results = await search_tavily(question, max_results=max_results)
                take_n = max(5, want_docs)
                urls = [r.get("url") for r in results if r.get("url")][:take_n]
                url_to_text = await fetch_pages(urls, concurrency=8)  # faster

                for r in results:
                    url = r.get("url")
                    if not url:
                        continue
                    text = (url_to_text.get(url, "") or "").strip()
                    if len(text) >= 500:  # lower threshold to get started
                        docs.append(WebDoc(
                            url=url,
                            title=r.get("title"),
                            snippet=r.get("snippet"),
                            text=text[:1600]  # tighter trim for quicker prompt
                        ))
                        if len(docs) >= want_docs:
                            break
            except Exception as e:
                print(f"[web_chat] Tavily failed: {e} -> fallback DuckDuckGo")
                provider_used = "duckduckgo"
                docs = await web_gather(
                    query=question,
                    max_results=max_results,
                    prefer_news=prefer_news,
                    want_docs=want_docs,
                    min_chars_each=350,
                    trim_chars_each=1200
                )

            # If Tavily gave nothing usable, try DDG quickly
            if not docs and provider_used != "duckduckgo":
                provider_used = "duckduckgo"
                docs = await web_gather(
                    query=question,
                    max_results=max_results,
                    prefer_news=prefer_news,
                    want_docs=want_docs,
                    min_chars_each=350,
                    trim_chars_each=1200
                )

            if not docs:
                # for log file
                _log_end("/v1/emergegpt", _start_ts, 200, {"task": "web_chat", "provider": provider_used, "docs": 0})
                return {
                    "answer": "I couldnt retrieve enough reliable content to answer. Try rephrasing or broadening the query.",
                    "sources": [],
                    "provider": provider_used
                }

            # 2) Build compact context
            context = build_context(docs, question, max_chars=2800)  # was 4000

            # 3) Prompt model and stream answer with a final event carrying sources
            sys_instructions = (
                "You are a careful researcher. Use ONLY the provided sources to answer.\n"
                "Cite sources inline like [1], [2] matching the order in 'Sources'.\n"
                "If uncertain, say so. Be concise and factual."
            )

            # NOTE: do not change logic — revert to original final_prompt line
            sys_instructions = (
                "You are a careful researcher. Use ONLY the provided sources to answer.\n"
                "Cite sources inline like [1], [2] matching the order in 'Sources'.\n"
                "If uncertain, say so. Be concise and factual."
            )
            final_prompt = f"[INST] {sys_instructions}\n\n{context}\n\nAnswer the question now. [/INST]"

            def stream_answer():
                for token in stream_generate_response(final_prompt):
                    if token:
                        frag = _json_escape_fragment(token)
                        if frag.strip():
                            yield f'data: {{"delta":"{frag}"}}\n\n'
                cites = render_citations(docs)
                yield f'data: {{"done":true,"provider":"{provider_used}","sources":{json.dumps(cites)}}}\n\n'
                yield "data: [DONE]\n\n"

            # for log file
            return StreamingResponse(
                stream_answer(),
                media_type="text/event-stream",
                background=BackgroundTask(_log_end, "/v1/emergegpt", _start_ts, 200, {"task": "web_chat", "provider": provider_used, "stream": True})
            )

        # -------------------- Simple Generation --------------------
        elif task == "generate":
            prompt = data.get("prompt", "")
            if not prompt:
                # for log file
                _log_end("/v1/emergegpt", _start_ts, 400, {"task": "generate", "error": "missing prompt"})
                raise HTTPException(status_code=400, detail="Missing 'prompt' in input")

            def generate_stream():
                for token in stream_generate_response(prompt):
                    yield f"data: {token}\n\n"

            # for log file
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                background=BackgroundTask(_log_end, "/v1/emergegpt", _start_ts, 200, {"task": "generate", "stream": True})
            )

        # -------------------- OpenAI-Style Chat --------------------
        elif task == "openai_chat":
            messages = data.get("messages", [])
            stream = data.get("stream", False)
            temperature = data.get("temperature", 0.5)

            prompt_parts = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role == "system":
                    prompt_parts.insert(0, f"[INST] {content} [/INST]")
                elif role == "user":
                    prompt_parts.append(f"[INST] {content} [/INST]")
                elif role == "assistant":
                    prompt_parts.append(content)
            final_prompt = build_mistral_prompt(messages)
            inputs = tokenizer(final_prompt, return_tensors="pt").to(device)

            if not stream:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.12,
                    no_repeat_ngram_size=3,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                result = decoded.split("[/INST]")[-1].strip()
                # for log file
                _log_end("/v1/emergegpt", _start_ts, 200, {"task": "openai_chat", "stream": False})
                return {
                    "id": "chatcmpl-xxx",
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}
                    ],
                    "usage": {},
                }

            def openai_stream():
                req_id = f"chatcmpl-{uuid.uuid4()}"
                created = int(datetime.now().timestamp())
                model_name = "emergegpt"

                # optional first delta with role
                yield f'data: {json.dumps({"id": req_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]})}\n\n'

                for token in stream_generate_response(final_prompt):
                    if not token:
                        continue
                    frag = _json_escape_fragment(token)
                    if frag.strip():
                        chunk = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {"index": 0, "delta": {"content": frag}, "finish_reason": None}
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                done = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"

            # for log file
            return StreamingResponse(
                openai_stream(),
                media_type="text/event-stream",
                background=BackgroundTask(_log_end, "/v1/emergegpt", _start_ts, 200, {"task": "openai_chat", "stream": True})
            )

        else:
            # for log file
            _log_end("/v1/emergegpt", _start_ts, 400, {"error": "Unsupported task"})
            raise HTTPException(status_code=400, detail="Unsupported task")

    except HTTPException as e:
        # for log file
        _log_end("/v1/emergegpt", _start_ts, e.status_code, {"error": e.detail})
        raise
    except Exception as e:
        # for log file
        _log_end("/v1/emergegpt", _start_ts, 500, {"error": str(e)})
        raise

# ============================================================
# 2) NEW unified endpoint (single public surface)
#    - If body has "task": reuse the same logic above.
#    - Else: OpenAI-style chat routed by "model".
# ============================================================
@router.post("/v1/completions", dependencies=[Depends(verify_token)])
async def unified_completions(request: Request):
    # for log file
    _start_ts = _log_start("/v1/completions", request, {"note": "unified surface"})

    try:
        body = await request.json()
        logger.info(f"[INFO] /v1/completions body_keys={list(body.keys())}")  # for log file
        # for log file (question)
        try:
            _q = _extract_question_from_body(body)
            if _q:
                logger.info(f"[Q] /v1/completions | { _trim_text(_q, 800) }")
        except Exception:
            pass

        # ---- PATH A: Task-style body -> same logic as /v1/emergegpt ----
        if isinstance(body, dict) and "task" in body:
            task = body.get("task")
            data = body.get("input", {})

            if task == "chat":
                prompt = data.get("prompt", "")
                full_prompt = f"[INST] {prompt} [/INST]"
                inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.12,
                    no_repeat_ngram_size=3,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # for log file
                _log_end("/v1/completions", _start_ts, 200, {"task": "chat", "stream": False})
                return {"response": decoded.split("[/INST]")[-1].strip()}

            elif task == "resume_parse":
                file_path = data.get("file_path")
                if not file_path or not os.path.exists(file_path):
                    # for log file
                    _log_end("/v1/completions", _start_ts, 400, {"task": "resume_parse", "error": "file not found"})
                    raise HTTPException(status_code=400, detail="Resume file not found.")
                parsed = parse_resume(file_path)
                # for log file
                _log_end("/v1/completions", _start_ts, 200, {"task": "resume_parse"})
                return {"parsed": parsed}

            elif task == "generate_jd":
                jd = generate_job_description(data)
                # for log file
                _log_end("/v1/completions", _start_ts, 200, {"task": "generate_jd"})
                return {"job_description": jd}

            elif task == "match_resumes":
                job_params = load_job_params()
                model_embed = _EMBEDDER
                job_skills_vecs = model_embed.encode(job_params["skills_required"], convert_to_tensor=True, normalize_embeddings=True)
                job_title_vec = model_embed.encode(job_params["job_title"], convert_to_tensor=True, normalize_embeddings=True)
                preferred_edu_vec = model_embed.encode(job_params["preferred_education"], convert_to_tensor=True, normalize_embeddings=True)

                resume = data.get("resume_dict")
                if resume:
                    score = match_resume_dict(resume, job_params, model_embed, job_skills_vecs, job_title_vec, preferred_edu_vec)
                    # for log file
                    _log_end("/v1/completions", _start_ts, 200, {"task": "match_resumes", "mode": "single"})
                    return {"match": score}

                folder = "parsed_json"
                os.makedirs("job_matching_result", exist_ok=True)
                all_results = []
                for file in os.listdir(folder):
                    if file.endswith(".json"):
                        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                            resumes = json.load(f)
                            if isinstance(resumes, dict):
                                resumes = [resumes]
                            for r in resumes:
                                result = match_resume_dict(r, job_params, model_embed, job_skills_vecs, job_title_vec, preferred_edu_vec)
                                all_results.append(result)
                all_results.sort(key=lambda x: x["match_score"], reverse=True)
                top_10 = all_results[:10]
                with open("job_matching_result/results.json", "w", encoding="utf-8") as f:
                    json.dump({"match_count": len(top_10), "matches": top_10}, f, indent=2)
                # for log file
                _log_end("/v1/completions", _start_ts, 200, {"task": "match_resumes", "mode": "batch", "count": len(top_10)})
                return {"match_count": len(top_10), "matches": top_10}

            elif task == "web_search":
                query = data.get("query")
                if not query:
                    # for log file
                    _log_end("/v1/completions", _start_ts, 400, {"task": "web_search", "error": "missing query"})
                    raise HTTPException(status_code=400, detail="Missing 'query' in input")
                max_results = data.get("max_results", 4)
                prefer_news = data.get("prefer_news", False)
                try:
                    results = await search_tavily(query, max_results=max_results)
                    # for log file
                    _log_end("/v1/completions", _start_ts, 200, {"task": "web_search", "provider": "tavily", "count": len(results)})
                    return {"results": results, "count": len(results), "provider": "tavily"}
                except Exception as e:
                    print(f"[web_search] Tavily failed or not configured: {e} -> fallback DuckDuckGo")
                docs = await web_gather(query=query, max_results=max_results, prefer_news=prefer_news)
                # for log file
                _log_end("/v1/completions", _start_ts, 200, {"task": "web_search", "provider": "duckduckgo", "count": len(docs)})
                return {"results": render_citations(docs), "count": len(docs), "provider": "duckduckgo"}

            elif task == "web_chat":
                question = data.get("question")
                max_results = data.get("max_results", 10)
                prefer_news = data.get("prefer_news", False)
                want_docs = data.get("want_docs", 5)
                if not question:
                    # for log file
                    _log_end("/v1/completions", _start_ts, 400, {"task": "web_chat", "error": "missing question"})
                    raise HTTPException(status_code=400, detail="Missing 'question' in input")

                docs: List[WebDoc] = []
                provider_used = "tavily"
                try:
                    results = await search_tavily(question, max_results=max_results)
                    take_n = max(5, want_docs)
                    urls = [r.get("url") for r in results if r.get("url")][:take_n]
                    url_to_text = await fetch_pages(urls, concurrency=8)
                    for r in results:
                        url = r.get("url")
                        if not url:
                            continue
                        text = (url_to_text.get(url, "") or "").strip()
                        if len(text) >= 500:
                            docs.append(WebDoc(
                                url=url,
                                title=r.get("title"),
                                snippet=r.get("snippet"),
                                text=text[:1600]
                            ))
                            if len(docs) >= want_docs:
                                break
                except Exception as e:
                    print(f"[web_chat] Tavily failed: {e} -> fallback DuckDuckGo")
                    provider_used = "duckduckgo"
                    docs = await web_gather(
                        query=question,
                        max_results=max_results,
                        prefer_news=prefer_news,
                        want_docs=want_docs,
                        min_chars_each=350,
                        trim_chars_each=1200
                    )

                if not docs and provider_used != "duckduckgo":
                    provider_used = "duckduckgo"
                    docs = await web_gather(
                        query=question,
                        max_results=max_results,
                        prefer_news=prefer_news,
                        want_docs=want_docs,
                        min_chars_each=350,
                        trim_chars_each=1200
                    )

                if not docs:
                    # for log file
                    _log_end("/v1/completions", _start_ts, 200, {"task": "web_chat", "provider": provider_used, "docs": 0})
                    return {
                        "answer": "I couldnt retrieve enough reliable content to answer. Try rephrasing or broadening the query.",
                        "sources": [],
                        "provider": provider_used
                    }

                context = build_context(docs, question, max_chars=2800)

                sys_instructions = (
                    "You are a careful researcher. Use ONLY the provided sources to answer.\n"
                    "Cite sources inline like [1], [2] matching the order in 'Sources'.\n"
                    "If uncertain, say so. Be concise and factual."
                )
                final_prompt = f"[INST] {sys_instructions}\n\n{context}\n\nAnswer the question now. [/INST]"

                def stream_answer():
                    for token in stream_generate_response(final_prompt):
                        if token:
                            frag = _json_escape_fragment(token)
                            if frag.strip():
                                yield f'data: {{"delta":"{frag}"}}\n\n'
                    cites = render_citations(docs)
                    yield f'data: {{"done":true,"provider":"{provider_used}","sources":{json.dumps(cites)}}}\n\n'
                    yield "data: [DONE]\n\n"

                # for log file
                return StreamingResponse(
                    stream_answer(),
                    media_type="text/event-stream",
                    background=BackgroundTask(_log_end, "/v1/completions", _start_ts, 200, {"task": "web_chat", "provider": provider_used, "stream": True})
                )

            elif task == "generate":
                prompt = data.get("prompt", "")
                if not prompt:
                    # for log file
                    _log_end("/v1/completions", _start_ts, 400, {"task": "generate", "error": "missing prompt"})
                    raise HTTPException(status_code=400, detail="Missing 'prompt' in input")
                def generate_stream():
                    for token in stream_generate_response(prompt):
                        yield f"data: {token}\n\n"
                # for log file
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream",
                    background=BackgroundTask(_log_end, "/v1/completions", _start_ts, 200, {"task": "generate", "stream": True})
                )

            elif task == "openai_chat":
                messages = data.get("messages", [])
                stream = data.get("stream", False)
                temperature = data.get("temperature", 0.5)
                prompt_parts = []
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content")
                    if role == "system":
                        prompt_parts.insert(0, f"[INST] {content} [/INST]")
                    elif role == "user":
                        prompt_parts.append(f"[INST] {content} [/INST]")
                    elif role == "assistant":
                        prompt_parts.append(content)
                final_prompt = build_mistral_prompt(messages)
                inputs = tokenizer(final_prompt, return_tensors="pt").to(device)

                if not stream:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=4096,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                        repetition_penalty=1.12,
                        no_repeat_ngram_size=3,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    result = decoded.split("[/INST]")[-1].strip()
                    # for log file
                    _log_end("/v1/completions", _start_ts, 200, {"task": "openai_chat", "stream": False})
                    return {
                        "id": "chatcmpl-xxx",
                        "object": "chat.completion",
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}
                        ],
                        "usage": {},
                    }

                def openai_stream():
                    for token in stream_generate_response(final_prompt):
                        if token:
                            frag = _json_escape_fragment(token)
                            if frag.strip():
                                yield f'data: {{"choices":[{{"delta":{{"content":"{frag}"}}}}]}}\n\n'
                    yield "data: [DONE]\n\n"

                # for log file
                return StreamingResponse(
                    openai_stream(),
                    media_type="text/event-stream",
                    background=BackgroundTask(_log_end, "/v1/completions", _start_ts, 200, {"task": "openai_chat", "stream": True})
                )

            else:
                # for log file
                _log_end("/v1/completions", _start_ts, 400, {"error": "Unsupported task"})
                raise HTTPException(status_code=400, detail="Unsupported task")

        # ---- PATH B: OpenAI-style chat (no "task") ----
        try:
            req = OAIReq(**body)
        except Exception as e:
            # for log file
            _log_end("/v1/completions", _start_ts, 400, {"error": f"Invalid body: {e}"})
            raise HTTPException(status_code=400, detail=f"Invalid body. Provide 'task' or OpenAI-style chat. Error: {e}")

        model_name = (req.model or "").lower().strip()

        # A) Local chat through your model
        if model_name in ("emergegpt", "emergeai", "mistral", "mistral-7b"):
            # Build base prompt
            final_prompt = build_mistral_prompt([m.model_dump() for m in req.messages])
            inputs = tokenizer(final_prompt, return_tensors="pt").to(device)

            # If the previous turn included a tool result, tell the model to use it
            if any(m.role == "tool" for m in req.messages):
                final_prompt = (
                    "[INST] <<SYS>>\n"
                    "You have received a tool result inside <tool_result>...</tool_result>.\n"
                    "Now ANSWER THE USER in plain English using that data.\n"
                    "DO NOT repeat the raw JSON. DO NOT show <tool_result>.\n"
                    "Keep it concise and helpful. If numbers exist (like temp), state them clearly.\n"
                    "<</SYS>>\n\n"
                ) + final_prompt

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # CHANGE #1: detect if a tool result already exists in the messages
            has_tool_result = any(m.role == "tool" for m in req.messages)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # --- ADDED: tool-calls (non-streaming) ---
            # CHANGE #2: only force a tool_call when there is NOT yet a tool result
            if req.tools and (req.tool_choice in (None, "auto", "required")) and (not any(m.role == "tool" for m in req.messages)):
                tool = req.tools[0]

                # NEW LOGIC: extract args using schema; clarify if missing when required
                args_obj, missing, err = extract_tool_args(
                    messages=[m.model_dump() for m in req.messages],
                    tool_def=tool.model_dump(),
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )

                if args_obj is not None:
                    args_str = json.dumps(args_obj, ensure_ascii=False)
                    _log_end("/v1/completions", _start_ts, 200, {"model": "emergegpt", "finish_reason": "tool_calls"})
                    return {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion",
                        "created": int(datetime.now().timestamp()),
                        "model": "emergegpt",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "tool_calls": [{
                                    "id": f"call_{uuid.uuid4()}",
                                    "type": "function",
                                    "function": {"name": tool.function.name, "arguments": args_str}
                                }]
                            },
                            "finish_reason": "tool_calls"
                        }],
                        "usage": {}
                    }

                if (req.tool_choice == "required") and missing:
                    question = f"I need {', '.join(missing)} to call {tool.function.name}. Please provide."
                    _log_end("/v1/completions", _start_ts, 200, {"model": "emergegpt", "finish_reason": "stop", "clarify": True})
                    return {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion",
                        "created": int(datetime.now().timestamp()),
                        "model": "emergegpt",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": question},
                            "finish_reason": "stop"
                        }],
                        "usage": {}
                    }
                # else fall through to normal answer path below if auto and nothing extracted

            # --- normal non-streaming generation ---
            if not req.stream:
                # for log file
                temp_to_use = req.temperature
                if has_tool_result:
                    temp_to_use = min(temp_to_use, 0.3)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096 if req.max_tokens is None else req.max_tokens,
                    do_sample=True,
                    temperature=temp_to_use,
                    top_p=req.top_p,
                    pad_token_id=tokenizer.eos_token_id
                )
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                result = decoded.split("[/INST]")[-1].strip()

                # for log file
                if _looks_like_tool_echo(result) and has_tool_result:
                    corrective_prompt = (
                        "[INST] <<SYS>>Rewrite the following tool output for the user as a single, natural sentence. "
                        "Do NOT include JSON, tags or angle brackets. Be concise.<</SYS>>\n\n"
                        f"{result}\n[/INST]"
                    )
                    inputs2 = tokenizer(corrective_prompt, return_tensors="pt").to(device)
                    outputs2 = model.generate(
                        **inputs2,
                        max_new_tokens=256,
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    result2 = tokenizer.decode(outputs2[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
                    if result2:
                        result = result2

                # for log file
                _log_end("/v1/completions", _start_ts, 200, {"model": "emergegpt", "stream": False})
                return {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(datetime.now().timestamp()),
                    "model": "emergegpt",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}],
                    "usage": {}
                }

            # --- streaming path ---
            def local_stream():
                req_id = f"chatcmpl-{uuid.uuid4()}"
                created = int(datetime.now().timestamp())
                mdl = "emergegpt"

                # --- ADDED: tool-calls (streaming one-shot) ---
                # CHANGE #3: stream a tool_call only if there isn't a tool result yet
                if req.tools and (req.tool_choice in ("required", "auto")) and (not has_tool_result):
                    tool = req.tools[0]

                    # NEW LOGIC (streaming): extract args; stream tool_call or clarification
                    args_obj, missing, err = extract_tool_args(
                        messages=[m.model_dump() for m in req.messages],
                        tool_def=tool.model_dump(),
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                    )

                    if args_obj is not None:
                        args_str = json.dumps(args_obj, ensure_ascii=False)
                        # first delta (role)
                        yield f'data: {json.dumps({"id": req_id, "object": "chat.completion.chunk", "created": created, "model": mdl, "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]})}\n\n'
                        # tool_calls delta with real args
                        tool_delta = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": mdl,
                            "choices": [{
                                "index": 0,
                                "delta": {"tool_calls": [{
                                    "index": 0,
                                    "id": f"call_{uuid.uuid4()}",
                                    "type": "function",
                                    "function": {"name": tool.function.name, "arguments": args_str}
                                }]},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(tool_delta)}\n\n"
                        done = {"id": req_id, "object": "chat.completion.chunk", "created": created, "model": mdl,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
                        yield f"data: {json.dumps(done)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    if (req.tool_choice == "required") and missing:
                        # stream a clarifying message instead of a tool call
                        yield f'data: {json.dumps({"id": req_id, "object":"chat.completion.chunk","created": created,"model": mdl,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason": None}]})}\n\n'
                        text_delta = {"id": req_id, "object":"chat.completion.chunk","created": created,"model": mdl,
                                      "choices":[{"index":0,"delta":{"content": f"I need {', '.join(missing)} to call {tool.function.name}. Please provide."},"finish_reason": None}]}
                        yield f"data: {json.dumps(text_delta)}\n\n"
                        stop = {"id": req_id, "object":"chat.completion.chunk","created": created,"model": mdl,
                                "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
                        yield f"data: {json.dumps(stop)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # else: fall through to normal token streaming

                try:
                    # initial role delta
                    yield f'data: {json.dumps({"id": req_id, "object": "chat.completion.chunk", "created": created, "model": mdl, "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]})}\n\n'
                    for token in stream_generate_response(final_prompt):
                        if not token:
                            continue
                        frag = _json_escape_fragment(token)
                        if frag.strip():
                            chunk = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": mdl,
                                "choices": [
                                    {"index": 0, "delta": {"content": frag}, "finish_reason": None}
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                finally:
                    done = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": mdl,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(done)}\n\n"
                    yield "data: [DONE]\n\n"

            # for log file
            return StreamingResponse(
                local_stream(),
                media_type="text/event-stream",
                background=BackgroundTask(_log_end, "/v1/completions", _start_ts, 200, {"model": "emergegpt", "stream": True})
            )

        # B) Ollama (llama3/qwen3/deepseek...)
        ollama_payload = {
            "model": req.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "stream": req.stream,
            "options": {
                "temperature": req.temperature,
                "top_p": req.top_p,
                "top_k": req.top_k,
                **({"num_predict": req.max_tokens} if req.max_tokens is not None else {})
            }
        }
        if req.stop:
            ollama_payload["options"]["stop"] = req.stop

        # --- ADDED: pass-through of tools/tool_choice (safe if backend ignores) ---
        if req.tools:
            ollama_payload["tools"] = [t.model_dump() for t in req.tools]
        if req.tool_choice:
            ollama_payload["tool_choice"] = req.tool_choice

        if req.stream:
            client = httpx.AsyncClient()  # keep it open for the whole SSE
            # for log file
            return StreamingResponse(
                _stream_ollama(client, f"{OLLAMA_API_URL}/chat", ollama_payload, req.model),
                media_type="text/event-stream",
                headers={  # avoid buffering and keep-alive for proxies
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
                background=BackgroundTask(_bg_close_and_log, client, "/v1/completions", _start_ts, {"model": req.model, "stream": True})  # close + log AFTER stream ends
            )

        async with httpx.AsyncClient() as client:
            r = await client.post(f"{OLLAMA_API_URL}/chat", json=ollama_payload, timeout=300.0)
            if r.status_code != 200:
                # for log file
                _log_end("/v1/completions", _start_ts, r.status_code, {"model": req.model, "error": r.text})
                raise HTTPException(status_code=r.status_code, detail=f"Ollama error: {r.text}")
            # for log file
            _log_end("/v1/completions", _start_ts, 200, {"model": req.model, "stream": False})
            return _format_openai_response_ollama(r.json(), req.model, req.messages)

    except HTTPException:
        raise
    except Exception as e:
        # for log file
        _log_end("/v1/completions", _start_ts, 500, {"error": str(e)})
        raise


# --- Alias for OpenAI/Vercel: /v1/chat/completions -> same handler as /v1/completions
from fastapi import Request, Depends  # (already imported above, but safe)

@router.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
async def chat_completions_alias(request: Request):
    # Reuse the same unified OpenAI-style handler
    return await unified_completions(request)
