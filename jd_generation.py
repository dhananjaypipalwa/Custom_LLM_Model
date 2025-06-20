from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os

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
        "Now generate a well-written, attractive job description suitable for job portals.\n"
        "Include clearly labeled sections: Job Title, Company, Location, Responsibilities, Required Skills, Qualifications, Salary, Shift Timings, and Joining Date.\n"
        "### Job Description ###\n"
    )

    # Generate output
    chat_inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    chat_outputs = model.generate(
        **chat_inputs,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Extract clean description
    chat_output_text = tokenizer.decode(chat_outputs[0], skip_special_tokens=True)
    job_desc = chat_output_text.split("### Job Description ###")[-1].strip()

    # ---- required change: remove the end-of-text token ----
    job_desc = job_desc.replace("<|endoftext|>", "")

    return job_desc.replace("\n", "<br>").strip()
