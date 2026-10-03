from pydantic import BaseModel
from typing import List, Optional


class AskRequest(BaseModel):
    question: str


class RetrievedChunk(BaseModel):
    text: str
    page: int
    source: str
    score: float
    rerank_score: Optional[float] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    reranked_chunks: List[RetrievedChunk]


class UploadResponse(BaseModel):
    message: str
    chunks_stored: int