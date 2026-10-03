from fastapi import FastAPI, UploadFile, File

from app.schemas import AskRequest, AskResponse, UploadResponse
from app.rag import process_pdf, ask_question


app = FastAPI(
    title="AgriRAG",
    description="Farmer Crop Advisory RAG Assistant",
    version="1.0"
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    pdf_data = await file.read()

    chunks_stored = process_pdf(
        pdf_data,
        file.filename
    )

    return UploadResponse(
        message="PDF processed successfully",
        chunks_stored=chunks_stored
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = ask_question(request.question)

    return AskResponse(
        question=request.question,
        answer=result["answer"],
        retrieved_chunks=result["retrieved_chunks"],
        reranked_chunks=result["reranked_chunks"]
    )