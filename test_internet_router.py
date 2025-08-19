from internet_router import decide_web

def show(q):
    d = decide_web(q)
    print(f"Q: {q}\n -> use_web={d.use_web}, reason={d.reason}, queries={d.queries[:2]}\n")

show("Explain cosine similarity with a tiny example.")
show("What changed in FlashAttention-3 vs v2 this year?")
show("Price of Nvidia H200 in India today, cite sources.")
show("LangGraph tutorials (links please).")
