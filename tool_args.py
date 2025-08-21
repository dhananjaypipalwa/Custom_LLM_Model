# tool_args.py
import json, re
from typing import Any, Dict, List, Optional, Tuple

try:
    # pip install jsonschema
    from jsonschema import validate, Draft7Validator
    _HAS_JSONSCHEMA = True
except Exception:
    _HAS_JSONSCHEMA = False


def _last_user_text(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages):
        if (m or {}).get("role") == "user":
            return (m.get("content") or "").strip()
    # fallback: concatenate all
    return " ".join([(m.get("content") or "") for m in messages]).strip()


def _strip_to_json(text: str) -> Optional[str]:
    """
    Try to pull the first JSON object from text. Handles code fences and chatter.
    """
    if text is None:
        return None
    t = text.strip()

    # remove ```json ... ``` wrappers if present
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?", "", t, flags=re.IGNORECASE).strip()
        if t.endswith("```"):
            t = t[:-3].strip()

    # quick path: already a dict string
    if t.startswith("{") and t.endswith("}"):
        return t

    # find the first {...} block
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1].strip()

    return None


def _validate(schema: Dict[str, Any], obj: Dict[str, Any]) -> Tuple[bool, Optional[str], List[str]]:
    """
    Returns (ok, error_message, missing_required_fields)
    Works even if jsonschema isn't installed (basic required check).
    """
    missing: List[str] = []
    if not isinstance(obj, dict):
        return False, "Returned value is not a JSON object", []

    # Basic required check
    req = (schema or {}).get("required") or []
    for k in req:
        if k not in obj or obj[k] in (None, "", []):
            missing.append(k)

    if missing:
        return False, f"Missing required: {missing}", missing

    if _HAS_JSONSCHEMA:
        try:
            validate(instance=obj, schema=schema)
            return True, None, []
        except Exception as e:
            # jsonschema will list the first error; also extract missing if any
            return False, str(e), missing

    # If no jsonschema installed and required is satisfied, accept
    return True, None, []


def _build_extractor_prompt(tool_name: str, schema: Dict[str, Any], conversation: str) -> str:
    """
    A short, strict instruction that makes most instruction-tuned models emit JSON-only.
    """
    schema_str = json.dumps(schema, ensure_ascii=False)
    return (
        "[INST] <<SYS>>You extract arguments for a function call.\n"
        f"Function name: {tool_name}\n"
        "Return ONLY a JSON object that matches EXACTLY the JSON Schema below.\n"
        "No commentary. No markdown fences. No extra keys. JSON object ONLY.\n"
        "If information is not present in the conversation, leave that field out; do NOT invent.\n"
        "<</SYS>>\n\n"
        f"JSON Schema:\n{schema_str}\n\n"
        f"Conversation:\n{conversation}\n\n"
        "Return the JSON now.[/INST]"
    )


def extract_tool_args(
    *,
    messages: List[Dict[str, str]],
    tool_def: Dict[str, Any],
    model,
    tokenizer,
    device,
    max_new_tokens: int = 192
) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]], Optional[str]]:
    """
    Run a deterministic pass with your local model to get JSON args.
    Returns: (args_dict | None, missing_required | None, error_text | None)
    """
    fn = (tool_def or {}).get("function", {}) or {}
    tool_name = fn.get("name") or "tool"
    schema = fn.get("parameters") or {"type": "object"}

    # Build a compact "conversation" text for the extractor
    convo = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        # keep it short-ish
        if len(content) > 1200:
            content = content[:1200] + "..."
        convo.append(f"{role.upper()}: {content}")
    convo_text = "\n".join(convo[-10:])  # last 10 messages max

    prompt = _build_extractor_prompt(tool_name, schema, convo_text)

    # Deterministic decode: low temperature, no sampling fuzz
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id
    )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Take only model's last turn after [/INST]
    text = raw.split("[/INST]")[-1].strip()

    json_str = _strip_to_json(text)
    if not json_str:
        return None, None, "Could not find JSON in model output"

    try:
        obj = json.loads(json_str)
    except Exception as e:
        return None, None, f"Invalid JSON: {e}"

    ok, err, missing = _validate(schema, obj)
    if ok:
        return obj, None, None

    # One repair pass if invalid and we have a concrete error
    repair_prompt = (
        "[INST] <<SYS>>You must return ONLY a JSON object that satisfies this JSON Schema.\n"
        "Fix the JSON according to the validator error. No commentary, JSON ONLY.<</SYS>>\n\n"
        f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Validator error:\n{err}\n\n"
        f"Original JSON:\n{json_str}\n\n"
        "Return the corrected JSON now.[/INST]"
    )
    inputs2 = tokenizer(repair_prompt, return_tensors="pt").to(device)
    outputs2 = model.generate(
        **inputs2,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id
    )
    text2 = tokenizer.decode(outputs2[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    json_str2 = _strip_to_json(text2)
    if not json_str2:
        return None, missing or None, "Repair failed: no JSON produced"

    try:
        obj2 = json.loads(json_str2)
    except Exception as e:
        return None, missing or None, f"Repair produced invalid JSON: {e}"

    ok2, err2, missing2 = _validate(schema, obj2)
    if ok2:
        return obj2, None, None

    # Still invalid -> return missing + error so caller can ask for clarification
    return None, (missing2 or missing or []), (err2 or err or "Validation failed")
