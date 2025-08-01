from fastapi import APIRouter
from match_score import match_resume_dict as match_resume, load_job_params
from sentence_transformers import SentenceTransformer
import json
import os


router = APIRouter()

@router.post("/match-resumes")
def match_all_resumes():
    job_params = load_job_params()
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    # === Pre-encode job requirement vectors ===
    job_skills_vecs = model.encode(
        job_params["skills_required"],
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    job_title_vec = model.encode(
        job_params["job_title"],
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    preferred_edu_vec = model.encode(
        job_params["preferred_education"],
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    folder = "parsed_json"
    os.makedirs("job_matching_result", exist_ok=True)
    all_results = []

    for file in os.listdir(folder):
        if file.endswith(".json"):
            file_path = os.path.join(folder, file)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for i, resume in enumerate(data):
                    resume_name = resume.get("name") or resume.get("full_output", {}).get("name", "Unknown")
                    print(f"?? Processing resume {i + 1}/{len(data)}: '{resume_name}'")

                    result = match_resume(
                        resume,
                        job_params,
                        model,
                        job_skills_vecs,
                        job_title_vec,
                        preferred_edu_vec
                    )
                    all_results.append(result)

            elif isinstance(data, dict):
                resume_name = data.get("name") or data.get("full_output", {}).get("name", "Unknown")
                print(f"?? Processing resume: '{resume_name}'")

                result = match_resume(
                    data,
                    job_params,
                    model,
                    job_skills_vecs,
                    job_title_vec,
                    preferred_edu_vec
                )
                all_results.append(result)

            else:
                print(f"?? Skipping file (not a valid dict or list): {file_path}")

    # === Sort and select top 10 resumes ===
    all_results.sort(key=lambda x: x["match_score"], reverse=True)
    top_10_results = all_results[:10]

    # === Save top 10 to file ===
    with open("job_matching_result/results.json", "w", encoding="utf-8") as f:
        json.dump({
            "match_count": len(top_10_results),
            "matches": top_10_results
        }, f, indent=2)

    return {
        "match_count": len(top_10_results),
        "matches": top_10_results
    }