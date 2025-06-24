import ssl
from pymongo import MongoClient

ssl._create_default_https_context = ssl._create_unverified_context

client = MongoClient("mongodb+srv://sahilunofficial33:UTDnoN5EAwg8koSs@cluster0.rrkspvm.mongodb.net/?retryWrites=true&w=majority")

db = client["resume_data"]
source = db["parsed_resume_jsons"]
target = db["candidates"]

# Clear target collection
target.delete_many({})

# Fetch the big doc
big_doc = source.find_one()
resume_list = big_doc.get("data", [])

# Utility to clean keys recursively
def clean_keys(obj):
    if isinstance(obj, dict):
        return {k.lstrip("$"): clean_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_keys(i) for i in obj]
    return obj

# Insert each cleaned resume
count = 0
for resume in resume_list:
    cleaned = clean_keys(resume)
    target.insert_one(cleaned)
    count += 1

print(f"? Inserted {count} cleaned resumes into 'candidates'")
