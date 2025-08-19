from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()
from emergegpt_router import router as emerge_router
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

app.include_router(emerge_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or list your domains
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],  # keep OPTIONS for preflight
    allow_headers=["Authorization", "Content-Type"],
    max_age=3600,
)
