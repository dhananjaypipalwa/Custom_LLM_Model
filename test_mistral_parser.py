from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

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

# === General Chat Prompt ===
chat_prompt = (
    "<|startoftext|>\n"
    "Task: Write a professional job description based on the following parameters.\n"
    "Job Role: Software Developer\n"
    "Skills: Java\n"
    "Experience Required: 2 years\n"
    "Employment Type: Full-time\n"
    "Location: Delhi\n"
    "Output:"
)
chat_inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

chat_outputs = model.generate(
    **chat_inputs,
    max_new_tokens=1024,
    do_sample=True,
    top_p=0.95,
    top_k=50,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

chat_output_text = tokenizer.decode(chat_outputs[0], skip_special_tokens=True)

# === Extract clean description starting from a known heading ===
clean_text = chat_output_text.split("<|endoftext|>")[0].strip()
clean_text = clean_text.split("Output:")[-1].strip()
for keyword in ["Job Title", "Job Summary", "Job Description"]:
    if keyword in clean_text:
        clean_text = clean_text.split(keyword, 1)[-1]
        clean_text = f"{keyword} {clean_text}"
        break

print(clean_text)
