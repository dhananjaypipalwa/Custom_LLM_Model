import json
import os
from sentence_transformers import SentenceTransformer, util
from dateutil import parser
from geopy.distance import geodesic
import pandas as pd
from rapidfuzz import process

# === Configurable Weights ===
WEIGHTS = {
    "skills": 0.35,
    "experience": 0.15,
    "role_match": 0.05,
    "education": 0.15,
    "location": 0.10
}

top_tier = ["Google", "Microsoft", "OpenAI", "Meta", "Apple", "Amazon", "DeepMind"]
mid_tier = ["Infosys", "TCS", "Wipro", "Capgemini", "Tech Mahindra", "HCL"]

# === Location logic ===
world_df = pd.read_csv("worldcities.csv")

city_coords_map = {
    str(row["city_ascii"]).strip().lower(): (row["lat"], row["lng"])
    for _, row in world_df.iterrows()
    if pd.notnull(row["city_ascii"]) and pd.notnull(row["lat"]) and pd.notnull(row["lng"])
}

city_country_map = {
    str(row["city_ascii"]).strip().lower(): str(row["country"]).strip().lower()
    for _, row in world_df.iterrows()
    if pd.notnull(row["city_ascii"]) and pd.notnull(row["country"])
}

city_list = list(city_coords_map.keys())

def fuzzy_match_city(city_name, threshold=85):
    match, score, _ = process.extractOne(city_name.lower(), city_list)
    return match if score >= threshold else None

def get_coords(city_name):
    city = str(city_name).strip().lower()
    coords = city_coords_map.get(city)
    if not coords:
        fuzzy = fuzzy_match_city(city)
        coords = city_coords_map.get(fuzzy) if fuzzy else None
    return coords

def get_country(location_name):
    location = str(location_name).strip().lower()
    known_countries = set(world_df["country"].str.strip().str.lower().unique())
    if location in known_countries:
        return location
    country = city_country_map.get(location)
    if not country:
        fuzzy = fuzzy_match_city(location)
        country = city_country_map.get(fuzzy) if fuzzy else None
    return country

def score_location_static(candidate_locs, job_loc):
    if not candidate_locs or not job_loc:
        return 0.0

    if isinstance(candidate_locs, str):
        candidate_locs = [candidate_locs]

    job_coords = get_coords(job_loc)
    job_country = get_country(job_loc)

    if not job_coords:
        return 0.0

    for cand_loc in candidate_locs:
        cand_coords = get_coords(cand_loc)
        cand_country = get_country(cand_loc)

        if cand_coords:
            distance_km = geodesic(job_coords, cand_coords).km
            if distance_km <= 10:
                return 1.0
            elif distance_km <= 50:
                return 0.8
            elif distance_km <= 200:
                return 0.5
            elif distance_km <= 3000:
                return 0.3

        if job_country and cand_country and job_country == cand_country:
            return 0.5

    return 0.0

def load_job_params():
    with open("job_description/job_params.json", "r") as f:
        return json.load(f)

def score_skills_semantic(resume_skills, model, job_skills_vecs, threshold=0.7):
    if not resume_skills or job_skills_vecs is None:
        return 0.0
    resume_vecs = model.encode(resume_skills, convert_to_tensor=True, normalize_embeddings=True)
    sim_matrix = util.cos_sim(job_skills_vecs, resume_vecs)
    matched_required = 0
    extra_relevance_scores = []

    for sim_row in sim_matrix:
        if max(sim_row).item() >= threshold:
            matched_required += 1

    coverage_score = matched_required / len(job_skills_vecs)

    for resume_idx in range(len(resume_skills)):
        sim_to_all_required = util.cos_sim(resume_vecs[resume_idx], job_skills_vecs)
        max_sim = sim_to_all_required.max().item()
        if max_sim < threshold and max_sim >= 0.5:
            extra_relevance_scores.append(max_sim)

    avg_extra_bonus = sum(extra_relevance_scores) / len(job_skills_vecs) if extra_relevance_scores else 0.0
    final_score = (0.8 * coverage_score) + (0.2 * avg_extra_bonus)
    return final_score

def score_company_experience(experiences, min_required_years):
    if not experiences:
        return 0.0, 0.0
    total_weighted_years = 0.0
    total_raw_years = 0.0

    for exp in experiences:
        company = exp.get("company_name", "").lower()
        start = exp.get("startDate", "")
        end = exp.get("endDate", "")
        try:
            if start and end:
                start_date = parser.parse(start)
                end_date = parser.parse(end)
                years = (end_date - start_date).days / 365
            else:
                years = 0.0
        except:
            years = 0.0
        total_raw_years += years
        if any(c in company for c in top_tier):
            total_weighted_years += years * 1.0
        elif any(c in company for c in mid_tier):
            total_weighted_years += years * 0.7
        else:
            total_weighted_years += years * 0.5

    if total_weighted_years >= min_required_years:
        score = 1.0
    elif total_weighted_years >= min_required_years * 0.75:
        score = 0.7
    elif total_weighted_years >= min_required_years * 0.5:
        score = 0.5
    elif total_weighted_years > 0:
        score = 0.3
    else:
        score = 0.0

    return score, round(total_raw_years, 2)

def score_role_match(experiences, model, job_title_vec, threshold=0.75):
    if not experiences or job_title_vec is None:
        return 0.0
    roles = [e.get("role_name", "") or e.get("position", "") or e.get("recent_position", "") for e in experiences]
    if not roles:
        return 0.0
    role_vecs = model.encode(roles, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(job_title_vec, role_vecs)[0]
    max_sim = max(sims).item()
    if max_sim >= threshold:
        return 1.0
    elif max_sim >= 0.5:
        return 0.5
    return 0.0

def score_education(candidate_edu_list, model, preferred_edu_vec):
    if not candidate_edu_list or preferred_edu_vec is None:
        return 0.0
    degrees = [edu.get("degree", "") for edu in candidate_edu_list if edu.get("degree")]
    if not degrees:
        return 0.0
    try:
        degree_embeddings = model.encode(degrees, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(preferred_edu_vec, degree_embeddings)[0]
        max_sim = max(sims).item()
        if max_sim >= 0.85:
            return 1.0
        elif max_sim >= 0.7:
            return 0.5
        else:
            return 0.0
    except:
        return 0.0

def score_languages(candidate_langs, required_langs):
    if not candidate_langs or not required_langs:
        return 0.0
    matched = [lang for lang in required_langs if any(lang.lower() in c.lower() for c in candidate_langs)]
    match_ratio = len(matched) / len(required_langs)
    if match_ratio == 1.0:
        return 5.0
    elif match_ratio >= 0.5:
        return 2.5
    return 0.0

def match_resume_dict(resume, job_params, model, job_skills_vecs, job_title_vec, preferred_edu_vec):
    full = resume.get("full_output", {})
    skills_score = score_skills_semantic(full.get("skills", []), model, job_skills_vecs)
    experiences = full.get("experiences", [])
    company_experience_score, total_years = score_company_experience(experiences, job_params["min_experience"])
    role_match_score = score_role_match(experiences, model, job_title_vec)
    education_score = score_education(full.get("education", []), model, preferred_edu_vec)

    candidate_locs = [resume["current_location"]] if resume.get("current_location") else []
    location_score = score_location_static(candidate_locs, job_params["location"])
    language_bonus = score_languages(full.get("languages", []), job_params.get("spoken_languages", []))

    total = (
        skills_score * WEIGHTS["skills"] +
        company_experience_score * WEIGHTS["experience"] +
        role_match_score * WEIGHTS["role_match"] +
        education_score * WEIGHTS["education"] +
        location_score * WEIGHTS["location"]
    ) * 100

    total += language_bonus

    return {
        "name": resume.get("name", full.get("name", "N/A")),
        "phone": resume.get("phone_numbers", ["N/A"])[0],
        "match_score": round(total, 2),
        "details": {
            "skills": round(skills_score * 100, 2),
            "experience+company": round(company_experience_score * 100, 2),
            "role_match": round(role_match_score * 100, 2),
            "total_experience_years": total_years,
            "education": round(education_score * 100, 2),
            "location": round(location_score * 100, 2),
            "language_bonus": language_bonus
        }
    }

# ==== FASTAPI ROUTER ====
from fastapi import APIRouter

router = APIRouter()

@router.post("/match-resumes")
def match_all_resumes():
    job_params = load_job_params()
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    job_skills_vecs = model.encode(job_params["skills_required"], convert_to_tensor=True, normalize_embeddings=True)
    job_title_vec = model.encode(job_params["job_title"], convert_to_tensor=True, normalize_embeddings=True)
    preferred_edu_vec = model.encode(job_params["preferred_education"], convert_to_tensor=True, normalize_embeddings=True)

    folder = "parsed_json"
    os.makedirs("job_matching_result", exist_ok=True)
    all_results = []

    for file in os.listdir(folder):
        if file.endswith(".json"):
            file_path = os.path.join(folder, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for resume in data:
                    result = match_resume_dict(resume, job_params, model, job_skills_vecs, job_title_vec, preferred_edu_vec)
                    all_results.append(result)
            elif isinstance(data, dict):
                result = match_resume_dict(data, job_params, model, job_skills_vecs, job_title_vec, preferred_edu_vec)
                all_results.append(result)

    all_results.sort(key=lambda x: x["match_score"], reverse=True)
    top_10_results = all_results[:10]

    with open("job_matching_result/results.json", "w", encoding="utf-8") as f:
        json.dump({
            "match_count": len(top_10_results),
            "matches": top_10_results
        }, f, indent=2)

    return {
        "match_count": len(top_10_results),
        "matches": top_10_results
    }
