# gen_gate.py
import os, threading
from fastapi import HTTPException
from contextlib import contextmanager

GEN_CONCURRENCY = int(os.getenv("MAX_CONCURRENT_GENS", "1"))
GEN_WAIT_SECS   = float(os.getenv("GEN_WAIT_SECS", "2.0"))

# Public semaphore
GATE = threading.Semaphore(GEN_CONCURRENCY)

def try_acquire_gate(timeout: float = GEN_WAIT_SECS) -> bool:
    return GATE.acquire(timeout=timeout)

def release_gate() -> None:
    try:
        GATE.release()
    except ValueError:
        # release called without a matching acquire; ignore
        pass

@contextmanager
def acquire_generation_slot(wait_seconds: float = GEN_WAIT_SECS):
    """Context manager for non-streaming paths (kept for convenience)."""
    if not GATE.acquire(timeout=wait_seconds):
        raise HTTPException(status_code=503, detail="Model busy. Try again shortly.")
    try:
        yield
    finally:
        release_gate()
