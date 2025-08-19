# model_loader.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# --- Paths ---
base_model_path = "mistralai/Mistral-7B-Instruct-v0.1"
lora_model_path = "mistral_resume_parser_model"

# --- Pick device & dtype safely ---
HAS_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if HAS_CUDA else "cpu")
dtype = torch.bfloat16 if HAS_CUDA else torch.float32

# (Optional speedups only when CUDA is present)
if HAS_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Load base model with safe defaults ---
# Do NOT force flash_attention_2 on CPU. Try it only if CUDA, and fall back to SDPA/default.
load_kwargs = {
    "torch_dtype": dtype,
    "device_map": "cuda:0" if HAS_CUDA else "cpu",
}

base_model = None
if HAS_CUDA:
    # Try Flash-Attn 2, fall back if unavailable
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            attn_implementation="flash_attention_2",
            **load_kwargs,
        )
    except Exception:
        # Safe fallback (works on CUDA too)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            attn_implementation="sdpa",
            **load_kwargs,
        )
else:
    # CPU: never request flash-attn; just load plainly
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **load_kwargs,
    )

# --- Attach LoRA and MERGE into base weights ---
peft_model = PeftModel.from_pretrained(base_model, lora_model_path)
model = peft_model.merge_and_unload()   # merged in memory; adapters freed
model.eval()
