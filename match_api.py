# match_api.py

from fastapi import APIRouter
import json
import os
from match_score import match_resume, load_job_params
from sentence_transformers import SentenceTransformer

router = APIRouter()

@router.post("/match-resumes")
def match_all_resumes():
    job_params = load_job_params()
    model = SentenceTransformer("thenlper/gte-large")
    folder = "parsed_json"
    os.makedirs("job_matching_result", exist_ok=True)

    all_results = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            result = match_resume(os.path.join(folder, file), job_params, model)
            all_results.append(result)

    with open("job_matching_result/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return {"matches": all_results}
