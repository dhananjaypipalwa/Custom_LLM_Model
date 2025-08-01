from fastapi import APIRouter, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import torch
import time

router = APIRouter()

# Load embedding model for vector search
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to MongoDB
client = MongoClient("mongodb+srv://sahilunofficial33:UTDnoN5EAwg8koSs@cluster0.rrkspvm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
chats_collection = client["rag_db"]["chats"]

# Load LoRA model with base model
base_model_path = "mistralai/Mistral-7B-Instruct-v0.1"
lora_model_path = "mistral_resume_parser_model"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, lora_model_path)

# Define your API token
API_TOKEN = "emergegpt-secure-token"

# Input schemas
class RAGRequest(BaseModel):
    context: str
    question: str

class VectorRAGRequest(BaseModel):
    question: str

# Token verification dependency
async def verify_token(authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing token")

@router.post("/rag_chat", dependencies=[Depends(verify_token)])
async def rag_chat(request: RAGRequest):
    prompt = f"You are an expert assistant. Use the following context to answer the question.\n\nContext:\n{request.context}\n\nQuestion:\n{request.question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "").strip()
    return {"answer": answer}

# STREAMING RAG VECTOR CHAT ENDPOINT
@router.post("/rag_vector_chat", dependencies=[Depends(verify_token)])
async def rag_vector_chat(request: VectorRAGRequest):
    question = request.question
    q_embed = embedder.encode(question).tolist()

    # MongoDB vector search
    pipeline = [
        {
            "$search": {
                "index": "chats_search",
                "knnBeta": {
                    "vector": q_embed,
                    "path": "embedding",
                    "k": 1
                }
            }
        },
        {
            "$project": {
                "conversation": 1,
                "score": {"$meta": "searchScore"}
            }
        },
        {
            "$limit": 1
        }
    ]
    results = list(chats_collection.aggregate(pipeline))
    context = "\n".join([doc.get("conversation", "") for doc in results])

    prompt = f"You are an expert assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    def generate_stream():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        streamed_ids = output_ids[0][input_ids.shape[-1]:]

        for token_id in streamed_ids:
            token = tokenizer.decode([token_id], skip_special_tokens=False)
            # Yield token in SSE format
            yield f"data: {token}\n\n"
            time.sleep(0.015)  # Optional: simulated typing delay

    return StreamingResponse(generate_stream(), media_type="text/event-stream")
