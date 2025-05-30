import json
import os
from sentence_transformers import SentenceTransformer, util

# === Configurable Weights ===
WEIGHTS = {
    "skills": 0.35,
    "experience": 0.20,
    "company": 0.20,
    "education": 0.15,
    "location": 0.10
}

# === Tiered Company List ===
top_tier = ["Google", "Microsoft", "OpenAI", "Meta", "Apple", "Amazon", "DeepMind"]
mid_tier = ["Infosys", "TCS", "Wipro", "Capgemini", "Tech Mahindra", "HCL"]

def load_resume_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_job_params():
    with open("job_description/job_params.json", 'r') as f:
        return json.load(f)

def score_skills_semantic(resume_skills, required_skills, model):
    if not resume_skills or not required_skills:
        return 0.0

    resume_vecs = model.encode(resume_skills, convert_to_tensor=True)
    job_vecs = model.encode(required_skills, convert_to_tensor=True)

    sim_matrix = util.cos_sim(resume_vecs, job_vecs)
    max_sim_per_required = [max(sim_row).item() for sim_row in sim_matrix]
    avg_score = sum(max_sim_per_required) / len(max_sim_per_required)
    return avg_score

def score_experience(candidate_years, required_years):
    return min(candidate_years / required_years, 1.0) if required_years else 0.0

def score_company(company):
    if not company:
        return 0.0
    name = company.lower()
    if any(c.lower() in name for c in top_tier):
        return 1.0
    elif any(c.lower() in name for c in mid_tier):
        return 0.7
    return 0.4

def score_education(candidate_edu, preferred_edu):
    if not candidate_edu or not preferred_edu:
        return 0.0
    if preferred_edu.lower() in candidate_edu.lower():
        return 1.0
    return 0.5

def score_location(candidate_loc, job_loc):
    if not candidate_loc or not job_loc:
        return 0.0
    return 1.0 if candidate_loc.lower() == job_loc.lower() else 0.5

def match_resume(resume_path, job_params, model):
    resume = load_resume_json(resume_path)
    full = resume.get("full_output", {})

    skills_score = score_skills_semantic(full.get("skills", []), job_params["skills_required"], model)
    experience_score = score_experience(resume.get("experience_count", 0), job_params["min_experience"])
    company_score = score_company(full.get("experiences", [{}])[0].get("company_name", ""))
    education_score = score_education(resume.get("degree", ""), job_params["preferred_education"])
    location_score = score_location(resume.get("current_location", ""), job_params["location"])

    total = (
        skills_score * WEIGHTS["skills"] +
        experience_score * WEIGHTS["experience"] +
        company_score * WEIGHTS["company"] +
        education_score * WEIGHTS["education"] +
        location_score * WEIGHTS["location"]
    ) * 100

    return {
        "resume": os.path.basename(resume_path),
        "match_score": round(total, 2),
        "details": {
            "skills": round(skills_score * 100, 2),
            "experience": round(experience_score * 100, 2),
            "company": round(company_score * 100, 2),
            "education": round(education_score * 100, 2),
            "location": round(location_score * 100, 2)
        }
    }

def main():
    job_params = load_job_params()
    model = SentenceTransformer("thenlper/gte-large")
    folder = "parsed_json"
    os.makedirs("job_matching_result", exist_ok=True)

    all_results = []

    for file in os.listdir(folder):
        if file.endswith(".json"):
            result = match_resume(os.path.join(folder, file), job_params, model)
            all_results.append(result)
            print(json.dumps(result, indent=2))

    with open("job_matching_result/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
