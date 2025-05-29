#without batching

import os
import pdfplumber
from docx import Document
from PIL import Image
import pytesseract
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# === Load tokenizer and LoRA-adapted model ===
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

def call_resume_parser(text):
    prompt = f"<|startoftext|>\nInput: {text}\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2048,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Output:" in result:
        json_part = result.split("Output:", 1)[1]
        json_part = json_part.split("<|endoftext|>")[0].strip()
        return json_part
    return ""

def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
    elif file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(file_path)
        return pytesseract.image_to_string(image).strip()
    return ""

def main():
    folder = "resumes"
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        raw = extract_text(path)
        if raw:
            result = call_resume_parser(raw)
            print(result)  # Print only the JSON (no explanation or heading)

if __name__ == "__main__":
    main()
