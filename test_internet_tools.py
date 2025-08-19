import asyncio
from internet_tools import web_gather, render_citations, build_context

async def main():
    docs = await web_gather("flash attention 3 overview", max_results=3, timelimit="y")
    print(f"Docs fetched: {len(docs)}")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d.title} | {d.url} | text_len={len(d.text)}")
    ctx = build_context(docs, "What is FlashAttention-3?", max_chars=1500)
    print("\nContext preview:\n", ctx[:500])
    print("\nCitations:", render_citations(docs))

asyncio.run(main())
