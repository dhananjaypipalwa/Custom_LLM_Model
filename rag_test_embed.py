from pymongo import MongoClient

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://sahilunofficial33:UTDnoN5EAwg8koSs@cluster0.rrkspvm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["sample_mflix"]
collection = db["embedded_movies"]

# Your query embedding (example random vector, replace with real embedding)
query_vector = [0.01] * 1536

# Perform vector search
pipeline = [
    {
        "$search": {
            "index": "plot_embedding_index",
            "knnBeta": {
                "vector": query_vector,
                "path": "plot_embedding",
                "k": 3
            }
        }
    },
    {
        "$project": {
            "plot": 1,
            "score": {"$meta": "searchScore"}
        }
    }
]

results = list(collection.aggregate(pipeline))
for doc in results:
    print("Score:", doc["score"])
    print("Plot:", doc.get("plot", "No plot"))
