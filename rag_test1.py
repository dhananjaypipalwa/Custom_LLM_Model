from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# MongoDB connection
mongo_uri = "mongodb+srv://sahilunofficial33:UTDnoN5EAwg8koSs@cluster0.rrkspvm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
db = client["rag_db"]
collection = db["chats"]

# Fetch the document
doc = collection.find_one({"id": "chat1"})
print("Fetched document:", doc)

# Load embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract conversation text
conversation_text = doc["conversation"]

# Generate embedding
embedding = model.encode(conversation_text)

print("Embedding shape:", embedding.shape)
print("Embedding sample:", embedding[:10])  # Print first 10 dimensions as sample

# Convert numpy array to list for MongoDB storage
embedding_list = embedding.tolist()

# Update document with embedding
collection.update_one(
    {"id": "chat1"},
    {"$set": {"embedding": embedding_list}}
)

print("? Embedding stored in MongoDB successfully.")

