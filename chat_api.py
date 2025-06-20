from fastapi import APIRouter, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

router = APIRouter()

# === Load model ===
model_path = "mistral_resume_parser_model"
base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@router.post("/chat")
async def chat(request: Request):
    # Read raw plain text body
    prompt_text = await request.body()
    prompt_str = prompt_text.decode().strip()

    full_prompt = f"[INST] {prompt_str} [/INST]"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("[/INST]")[-1].strip()

    return response  # <== Direct plain string response
