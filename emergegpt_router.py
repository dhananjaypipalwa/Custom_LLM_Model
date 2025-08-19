# emergegpt_router.py
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import os, json, uuid
from typing import Any, Dict, Optional, List
from datetime import datetime

from model_loader import tokenizer, model, device
from rag_utils import stream_generate_response
from parser_mistral import parse_resume
from jd_generation import generate_job_description
from match_score import match_resume_dict, load_job_params
from sentence_transformers import SentenceTransformer
from internet_router import decide_web, RouterDecision  # (ok to keep even if unused)
# NOTE: do NOT import a module-level TAVILY_API_KEY here; search_tavily reads env at call-time
from internet_tools import (
    web_gather,
    build_context,
    render_citations,
    search_tavily,
    fetch_pages,
    WebDoc,
)

import httpx  # NEW: for Ollama forwarding

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

    # seed with system + first user
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
                chunks.append(f"<s>[INST] {sys_block}{content} [/INST]")
                opened = True
            else:
                chunks.append(f"</s>\n<s>[INST] {content} [/INST]")
            # if next is assistant, append it on same turn
            if i + 1 < len(messages) and messages[i+1]["role"] == "assistant":
                chunks[-1] = chunks[-1] + f" {messages[i+1]['content']}"
                i += 1  # skip assistant (already consumed)
        # ignore other roles here
        i += 1

    # Ensure the last chunk ends with an open INST (no trailing assistant)
    # The template above already leaves the final turn open for generation.
    return "\n".join(chunks)



# =========================
# NEW: OpenAI-style schema
# =========================
class OAIMsg(BaseModel):
    role: str
    content: str

class OAIReq(BaseModel):
    model: str                      # "emergegpt" | "llama3" | "qwen3" | "deepseek-r1" ...
    messages: List[OAIMsg]
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None

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

async def _stream_ollama(client: httpx.AsyncClient, url: str, req_json: Dict[str, Any], model_name: str):
    """Proxy Ollama streaming into OpenAI-style SSE deltas."""
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
    body = await request.json()
    task = body.get("task")
    data = body.get("input", {})

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
        return {"response": decoded.split("[/INST]")[-1].strip()}

    # -------------------- Resume Parsing --------------------
    elif task == "resume_parse":
        file_path = data.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="Resume file not found.")
        parsed = parse_resume(file_path)
        return {"parsed": parsed}

    # -------------------- Job Description Generation --------------------
    elif task == "generate_jd":
        jd = generate_job_description(data)
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

        return {"match_count": len(top_10), "matches": top_10}

    # -------------------- Web Search (Tavily ? DuckDuckGo fallback) --------------------
    elif task == "web_search":
        query = data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' in input")

        max_results = data.get("max_results", 10)  # tighter
        prefer_news = data.get("prefer_news", False)

        # Try Tavily first (reads API key at call time)
        try:
            results = await search_tavily(query, max_results=max_results)
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

        return StreamingResponse(stream_answer(), media_type="text/event-stream")

    # -------------------- Simple Generation --------------------
    elif task == "generate":
        prompt = data.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing 'prompt' in input")

        def generate_stream():
            for token in stream_generate_response(prompt):
                yield f"data: {token}\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

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

        return StreamingResponse(openai_stream(), media_type="text/event-stream")

    else:
        raise HTTPException(status_code=400, detail="Unsupported task")

# ============================================================
# 2) NEW unified endpoint (single public surface)
#    - If body has "task": reuse the same logic above.
#    - Else: OpenAI-style chat routed by "model".
# ============================================================
@router.post("/v1/completions", dependencies=[Depends(verify_token)])
async def unified_completions(request: Request):
    body = await request.json()

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
            return {"response": decoded.split("[/INST]")[-1].strip()}

        elif task == "resume_parse":
            file_path = data.get("file_path")
            if not file_path or not os.path.exists(file_path):
                raise HTTPException(status_code=400, detail="Resume file not found.")
            parsed = parse_resume(file_path)
            return {"parsed": parsed}

        elif task == "generate_jd":
            jd = generate_job_description(data)
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
            return {"match_count": len(top_10), "matches": top_10}

        elif task == "web_search":
            query = data.get("query")
            if not query:
                raise HTTPException(status_code=400, detail="Missing 'query' in input")
            max_results = data.get("max_results", 4)
            prefer_news = data.get("prefer_news", False)
            try:
                results = await search_tavily(query, max_results=max_results)
                return {"results": results, "count": len(results), "provider": "tavily"}
            except Exception as e:
                print(f"[web_search] Tavily failed or not configured: {e} -> fallback DuckDuckGo")
            docs = await web_gather(query=query, max_results=max_results, prefer_news=prefer_news)
            return {"results": render_citations(docs), "count": len(docs), "provider": "duckduckgo"}

        elif task == "web_chat":
            question = data.get("question")
            max_results = data.get("max_results", 10)
            prefer_news = data.get("prefer_news", False)
            want_docs = data.get("want_docs", 5)
            if not question:
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

            return StreamingResponse(stream_answer(), media_type="text/event-stream")

        elif task == "generate":
            prompt = data.get("prompt", "")
            if not prompt:
                raise HTTPException(status_code=400, detail="Missing 'prompt' in input")
            def generate_stream():
                for token in stream_generate_response(prompt):
                    yield f"data: {token}\n\n"
            return StreamingResponse(generate_stream(), media_type="text/event-stream")

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

            return StreamingResponse(openai_stream(), media_type="text/event-stream")

        else:
            raise HTTPException(status_code=400, detail="Unsupported task")

    # ---- PATH B: OpenAI-style chat (no "task") ----
    try:
        req = OAIReq(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid body. Provide 'task' or OpenAI-style chat. Error: {e}")

    model_name = (req.model or "").lower().strip()

    # A) Local chat through your model
    if model_name in ("emergegpt", "emergeai", "mistral", "mistral-7b"):
        final_prompt = build_mistral_prompt([m.model_dump() for m in req.messages])
        inputs = tokenizer(final_prompt, return_tensors="pt").to(device)

        if not req.stream:
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096 if req.max_tokens is None else req.max_tokens,
                do_sample=True,
                temperature=req.temperature,
                top_p=req.top_p,
                pad_token_id=tokenizer.eos_token_id
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = decoded.split("[/INST]")[-1].strip()
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": "emergegpt",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}],
                "usage": {}
            }

        def local_stream():
            for token in stream_generate_response(final_prompt):
                if token:
                    frag = _json_escape_fragment(token)
                    if frag.strip():
                        yield f'data: {{"choices":[{{"delta":{{"content":"{frag}"}}}}]}}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(local_stream(), media_type="text/event-stream")

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

    if req.stream:
        async with httpx.AsyncClient() as client:
            return StreamingResponse(
                _stream_ollama(client, f"{OLLAMA_API_URL}/chat", ollama_payload, req.model),
                media_type="text/event-stream"
            )

    async with httpx.AsyncClient() as client:
        r = await client.post(f"{OLLAMA_API_URL}/chat", json=ollama_payload, timeout=300.0)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=f"Ollama error: {r.text}")
        return _format_openai_response_ollama(r.json(), req.model, req.messages)
