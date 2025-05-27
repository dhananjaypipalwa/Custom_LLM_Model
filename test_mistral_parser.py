from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "mistral_resume_parser_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Resume Parsing ===
resume_text = """
Riya Kapoor is a frontend developer with 2 years of experience at Infosys and Wipro. She has a B.Tech in Information Technology from Delhi University and is skilled in React.js, CSS, and JavaScript.
"""
resume_prompt = f"<|startoftext|>\nInput: {resume_text}\nOutput:"
resume_inputs = tokenizer(resume_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
resume_input_ids = resume_inputs["input_ids"].to(device)
resume_attention_mask = resume_inputs["attention_mask"].to(device)

resume_outputs = model.generate(
    input_ids=resume_input_ids,
    attention_mask=resume_attention_mask,
    max_new_tokens=500,
    do_sample=True,
    top_p=0.95,
    top_k=50,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

resume_output_text = tokenizer.decode(resume_outputs[0], skip_special_tokens=True)
print("\n🧾 Parsed Resume Output:\n")
print(resume_output_text.split("Output:")[-1].strip())

# === General Chat Prompt ===
chat_prompt = "Hi, how are you?"
chat_inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

chat_outputs = model.generate(
    **chat_inputs,
    max_new_tokens=50,
    do_sample=True,
    top_p=0.95,
    top_k=50,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

chat_output_text = tokenizer.decode(chat_outputs[0], skip_special_tokens=True)
print("\n💬 Chat Output:\n")
print(chat_output_text)
