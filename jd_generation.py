from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os

# === Load LoRA model once ===
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# === Generation Function ===
def generate_job_description(job_params: dict) -> str:
    # Save job parameters for scoring later
    os.makedirs("job_description", exist_ok=True)
    with open("job_description/job_params.json", "w") as f:
        json.dump(job_params, f, indent=2)

    # Prompt template
    chat_prompt = (
        "<|startoftext|>\n"
        "Task: Write a professional job description based on the following parameters.\n"
        f"Job ID: {job_params['job_id']}\n"
        f"Job Title: {job_params['job_title']}\n"
        f"Company Name: {job_params['company_name']}\n"
        f"Number of Openings: {job_params['number_of_openings']}\n"
        f"Employment Type: {job_params['employment_type']}\n"
        f"Experience Required: {job_params['experience_requirements']} years\n"
        f"Location: {job_params['location']}\n"
        f"Education Requirements: {job_params['education_requirements']}\n"
        f"Skills Required: {', '.join(job_params['skills_required'])}\n"
        f"Spoken Languages: {', '.join(job_params['spoken_languages'])}\n"
        f"Salary: {job_params['salary']}\n"
        f"Joining Date: {job_params['joining_date']}\n"
        f"Shift Timings: {job_params['shift_timings']}\n"
        "Output:\nPlease write a complete and professional job description using the above details.\n"
    )

    # Generate output
    chat_inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    chat_outputs = model.generate(
        **chat_inputs,
        max_new_tokens=2048,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        num_beams=3,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    chat_output_text = tokenizer.decode(chat_outputs[0], skip_special_tokens=True)

    # Extract clean description
    clean_text = chat_output_text.split("<|endoftext|>")[0].strip()
    clean_text = clean_text.split("Output:")[-1].strip()
    for keyword in ["Job Title", "Job Summary", "Job Description"]:
        if keyword in clean_text:
            clean_text = clean_text.split(keyword, 1)[-1]
            clean_text = f"{keyword} {clean_text}"
            break

    return clean_text.replace("\n", "<br>")
