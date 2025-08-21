# rag_utils.py  (drop-in)
import threading, torch
from transformers import TextIteratorStreamer
from model_loader import tokenizer, model, device

model.eval()

def stream_generate_response(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=4096,            
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.12,       
        no_repeat_ngram_size=3,        
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        streamer=streamer,
    )
    # keep thread (required for true streaming)
    t = threading.Thread(target=lambda: model.generate(**generation_kwargs))
    t.start()

    for token in streamer:
        if not token:
            continue
        yield token