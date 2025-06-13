from fastapi import APIRouter
from pydantic import BaseModel
from jd_generation import generate_job_description

router = APIRouter()

class JobInput(BaseModel):
    job_id: str
    job_title: str
    company_name: str
    number_of_openings: int
    employment_type: str
    experience_requirements: int
    min_experience: int
    location: str
    education_requirements: str
    preferred_education: str
    preferred_companies: list[str]
    skills_required: list[str]
    spoken_languages: list[str]
    salary: str
    joining_date: str
    shift_timings: str

@router.post("/generate_job_description")
def generate(data: JobInput):
    input_dict = data.dict()
    result = generate_job_description(input_dict)

    # Clean formatting
    cleaned = result.replace("\\n", "\n")  # decode escaped newlines
    cleaned = cleaned.replace("*", "")
    if "Job Description :" in cleaned:
        cleaned = cleaned.split("Job Description :", 1)[-1].strip()

    return {"job_description": cleaned}
