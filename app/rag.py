import fitz

import requests

from app.config import NUGEN_API_KEY

from app.vector_store import add_chunks, search_chunks
from sentence_transformers import CrossEncoder


reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def process_pdf(pdf_data, filename):
    document = fitz.open(stream=pdf_data, filetype="pdf")

    chunks = []

    chunk_size = 300
    overlap = 50

    for page_number, page in enumerate(document):
        text = page.get_text()

        words = text.split()

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]

            if not chunk_words:
                continue

            chunk_text = " ".join(chunk_words)

            chunks.append({
                "text": chunk_text,
                "page": page_number + 1,
                "source": filename
            })

    document.close()

    chunks_stored = add_chunks(chunks)

    return chunks_stored


def retrieve_chunks(question):
    return search_chunks(question, top_k=10)


def rerank_chunks(question, chunks):
    pairs = []

    for chunk in chunks:
        pairs.append([
            question,
            chunk["text"]
        ])

    scores = reranker_model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    chunks.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return chunks[:3]


def create_prompt(question, top_3_chunks):
    context = ""

    for chunk in top_3_chunks:
        context += (
            f"Source: {chunk['source']}, "
            f"Page: {chunk['page']}\n"
            f"{chunk['text']}\n\n"
        )

    prompt = f"""
You are AgriRAG, a farmer crop advisory assistant.

Answer the farmer's question using only the information
provided in the context below.

If the answer is not available in the context, say:
"I could not find this information in the provided document."

Context:
{context}

Farmer's Question:
{question}

Answer:
"""

    return prompt


def generate_answer(prompt):
    url = "https://api.nugen.in/api/v3/inference/completions"

    headers = {
        "Authorization": f"Bearer {NUGEN_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "model_01m343axrwwe1m0x",
        "prompt": prompt,
        "max_tokens": 400,
        "temperature": 0.2,
        "stream": False
    }

    response = requests.post(
        url,
        headers=headers,
        json=data
    )

    response.raise_for_status()

    result = response.json()

    return result["choices"][0]["text"]


def ask_question(question):
    # Step 1: Retrieve Top 10 chunks
    retrieved_chunks = retrieve_chunks(question)

    # Step 2: Rerank Top 10 and select Top 3
    top_3_chunks = rerank_chunks(
        question,
        retrieved_chunks
    )

    # Step 3: Create prompt using Top 3
    prompt = create_prompt(
        question,
        top_3_chunks
    )

    # Step 4: Generate answer using Nugen
    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "reranked_chunks": top_3_chunks
    }