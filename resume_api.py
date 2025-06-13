from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
import json
from parser_mistral import parse_resume  # Make sure parser_mistral.py has this function

router = APIRouter()

@router.post("/parse_resume")
async def upload_resume(file: UploadFile = File(...)):
    # 1. Validate file type
    ext = file.filename.split('.')[-1].lower()
    if ext not in ["pdf", "docx", "png", "jpg", "jpeg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # 2. Save to resumes folder
    os.makedirs("resumes", exist_ok=True)
    resume_id = str(uuid.uuid4())
    save_path = f"resumes/{resume_id}.{ext}"

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

    # 3. Parse the resume
    try:
        parsed_data = parse_resume(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing resume: {e}")

    # 4. Save parsed output
    os.makedirs("parsed_json", exist_ok=True)
    json_path = f"parsed_json/{resume_id}.json"
    try:
        with open(json_path, "w") as f:
            json.dump(parsed_data, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving parsed JSON: {e}")

    # 5. Return parsed data as JSON response
    return JSONResponse(content=parsed_data)
