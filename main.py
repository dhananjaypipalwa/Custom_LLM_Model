#main.py

from fastapi import FastAPI
#from jd_api import router as job_router
#from resume_api import router as resume_router
#from match_api import router as match_router
#from chat_api import router as chat_router
from rag_api import router as rag_router

app = FastAPI()

#app.include_router(job_router)
#app.include_router(resume_router)
#app.include_router(match_router)
#app.include_router(chat_router)
app.include_router(rag_router)
