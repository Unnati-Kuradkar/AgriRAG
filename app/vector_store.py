from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from sentence_transformers import SentenceTransformer
import uuid

from app.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION


client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

try:
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="source",
        field_schema="keyword"
    )
except Exception:
    pass

def create_collection(vector_size):
    collections = client.get_collections().collections

    collection_names = [collection.name for collection in collections]

    if QDRANT_COLLECTION not in collection_names:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="source",
            field_schema="keyword"
        )


def delete_chunks_by_source(source):
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source)
                )
            ]
        )
    )


def create_embeddings(texts):
    return embedding_model.encode(texts).tolist()


def add_chunks(chunks):
    if not chunks:
        return 0

    source = chunks[0]["source"]

    # Delete old chunks from the same PDF
    delete_chunks_by_source(source)

    texts = [chunk["text"] for chunk in chunks]

    vectors = create_embeddings(texts)

    if vectors:
        create_collection(len(vectors[0]))

    points = []

    for chunk, vector in zip(chunks, vectors):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "source": chunk["source"]
                }
            )
        )

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points
    )

    return len(points)


def search_chunks(question, top_k=10):
    question_vector = create_embeddings([question])[0]

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=question_vector,
        limit=top_k,
        with_payload=True
    )

    retrieved_chunks = []

    for result in results.points:
        retrieved_chunks.append({
            "text": result.payload["text"],
            "page": result.payload["page"],
            "source": result.payload["source"],
            "score": result.score
        })

    return retrieved_chunks