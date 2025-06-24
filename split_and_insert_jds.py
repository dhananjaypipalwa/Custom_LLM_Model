import ssl
from pymongo import MongoClient

ssl._create_default_https_context = ssl._create_unverified_context

uri = "mongodb+srv://sahilunofficial33:UTDnoN5EAwg8koSs@cluster0.rrkspvm.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)

db = client["jd_data"]
collection = db["jd_job_json"]

raw_doc = collection.find_one()
jd_list = raw_doc.get("data", [])

final_collection = db["raw_job_descriptions"]
final_collection.delete_many({})

# Utility: Recursively replace keys starting with "$"
def sanitize_keys(doc):
    if isinstance(doc, dict):
        return {
            (k.replace('$', '_') if k.startswith('$') else k): sanitize_keys(v)
            for k, v in doc.items()
        }
    elif isinstance(doc, list):
        return [sanitize_keys(item) for item in doc]
    else:
        return doc

count = 0
for jd in jd_list:
    clean_jd = sanitize_keys(jd)
    final_collection.insert_one(clean_jd)
    count += 1

print(f"? Inserted {count} sanitized JDs as separate documents.")
