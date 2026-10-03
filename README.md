# AgriRAG – Agriculture Question Answering System

## Overview

AgriRAG is a Retrieval-Augmented Generation (RAG) based AI system that helps farmers get answers from agricultural advisory documents. The system retrieves relevant information from uploaded PDF documents and generates accurate responses based on the retrieved content.

## Features

* PDF document ingestion
* Automatic text chunking
* Vector embedding generation
* Semantic search using FAISS
* Reranking of retrieved results
* FastAPI backend
* Source-based answer generation
* JSON API responses

## Tech Stack

* Python
* FastAPI
* FAISS
* Sentence Transformers
* Hugging Face
* PyMuPDF
* Pydantic

## Project Structure

AgriRAG/
│
├── app/
│   ├── main.py
│   ├── rag.py
│   ├── vector_store.py
│   ├── schemas.py
│   └── config.py
│
├── data/
│   └── farmer_crop_advisory_vidarbha.pdf
│
├── requirements.txt
├── README.md
└── .gitignore

## API Endpoint

### Ask Question

POST /ask

Example Request:

{
"question": "What is recommended for fertilizer management in soybean?"
}

Example Response:

{
"answer": "Balanced application of nitrogen, phosphorus, potassium and required micronutrients is recommended rather than excessive use of any single fertilizer."
}

## Dataset

The project currently uses agricultural advisory PDF documents for the Vidarbha region. The system can be extended with additional crop advisory documents.

## Future Enhancements

* Multi-language support
* Voice-based farmer queries
* Larger agricultural knowledge base
* Mobile application integration
* LLM-based answer generation

## Author

Unnati Kuradkar
MCA Project – AgriRAG
