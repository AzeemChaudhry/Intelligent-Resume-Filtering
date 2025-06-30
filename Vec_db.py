from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer


client = QdrantClient("localhost", port=6333)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  
client.recreate_collection(
collection_name="cv_data",
vectors_config={
    "skills": VectorParams(size=384, distance=Distance.COSINE),
    "education": VectorParams(size=384, distance=Distance.COSINE),
    "work_experience": VectorParams(size=384, distance=Distance.COSINE),
    "projects": VectorParams(size=384, distance=Distance.COSINE),
    }
)




required_fields = ["skills", "education", "work_experience", "projects"]

###-------------------------------------------------------------------------
def zero_vector(dim=384):
    return [0.0] * dim
###-------------------------------------------------------------------------
def join_and_embed(field_list,embedding_model):
    if not field_list:
        return zero_vector()  
    pieces = []
    for item in field_list:
        if isinstance(item, dict):
         
            pieces.append(", ".join(f"{k}: {v}" for k, v in item.items()))
        elif isinstance(item, str):
            pieces.append(item)
        else:
       
            pieces.append(str(item))
    text = " ".join(pieces)
    return embedding_model.encode([text])[0].tolist()

###-----------------------------------------------------------------------
def insert_candidate(candidate, collection_name="cv_data"):
    vector_data = {
        field: join_and_embed(candidate.get(field, []),embedding_model)
        for field in required_fields
    }
    
    payload = {}
    if "name" in candidate:
        payload["name"] = candidate["name"]
    if "filepath" in candidate: 
        payload["filepath"] = candidate["filepath"]

    # Point ID
    point_id = candidate.get("id", hash(candidate.get("name", "unknown")) & 0xFFFFFFFFFFFFFFFF)

    point = PointStruct(
        id=point_id,
        vector=vector_data,
        payload=payload
    )
    
    client.upsert(collection_name=collection_name, points=[point])
    print(f"Inserted: {payload.get('name', 'Unnamed')}")



## creating VEC DB 
def create_vec_db(candidates):
    for i, candidate in enumerate(candidates): 
        print("Inserting candidate")
        insert_candidate(candidate)
    scroll_result = client.scroll(
    collection_name="cv_data",
    with_payload=True,
    with_vectors=True,
    limit=100
)

    # for point in scroll_result[0]:
    #     print(f"\nCandidate ID: {point.id}")
    #     print(f"Name: {point.payload.get('name', 'N/A')}")
    #     print(f"Filepath: {point.payload.get('filepath', 'N/A')}")  # ✅ This line shows the path
    #     print("Vectors:")
    #     for vector_name, vector_values in point.vector.items():
    #         print(f" - {vector_name} ({len(vector_values)} dims)")
    #         print(f"   {vector_values[:10]}...")