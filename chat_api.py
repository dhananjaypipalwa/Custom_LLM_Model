from fastapi.responses import StreamingResponse
import torch

@router.post("/chat")
async def chat(request: Request):
    prompt_text = await request.body()
    prompt_str = prompt_text.decode().strip()

    full_prompt = f"[INST] {prompt_str} [/INST]"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    def token_stream():
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=2048,
            do_sample=True, 
            streamer=streamer,
        )
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        for token in streamer:
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")
