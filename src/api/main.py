
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from groq import Groq
from qdrant_client import models

from src.ingestion.ingest_pdf import (
    PDFProcessingError,
    ingest_pdf_bytes,
)
from src.reasoning.answer import (
    COLLECTION_NAME,
    TOP_K,
    embed_question,
    generate_answer,
    retrieve_evidence,
)
from src.vectorstore import create_qdrant_client


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MAX_PDF_BYTES = 25 * 1024 * 1024

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is missing from .env"
        )

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is missing from .env"
        )

    embedding_client = genai.Client()
    groq_client = Groq()
    qdrant = create_qdrant_client()

    try:
        if not qdrant.collection_exists(COLLECTION_NAME):
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                ),
            )

        collection_info = qdrant.get_collection(
            COLLECTION_NAME
        )
        payload_schema = collection_info.payload_schema or {}

        if "doc_id" not in payload_schema:
            qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            print("Created Qdrant payload index for doc_id.")

        app.state.embedding_client = embedding_client
        app.state.groq_client = groq_client
        app.state.qdrant = qdrant

        yield

    finally:
        qdrant.close()
        embedding_client.close()
        groq_client.close()


app = FastAPI(
    title="RAG Document Assistant API",
    description=(
        "Upload text-based PDFs and ask questions about "
        "their contents with evidence and page references."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rag-frontend-six-wheat.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
        description="Your question about the uploaded PDF.",
    )

    document_id: str = Field(
        min_length=36,
        max_length=36,
        description=(
            "The document ID returned by POST /upload."
        ),
    )


@app.get("/")
def root():
    return {
        "name": "RAG Document Assistant API",
        "status": "running",
        "docs": "/docs",
        "upload_endpoint": "POST /upload",
        "question_endpoint": "POST /ask",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "database": "qdrant",
        "collection": COLLECTION_NAME,
    }


@app.post("/upload")
def upload_pdf(
    file: UploadFile = File(
        description="A text-based PDF, up to 25 MB."
    ),
):
    filename = Path(
        file.filename or "uploaded.pdf"
    ).name

    if Path(filename).suffix.lower() != ".pdf":
        file.file.close()
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported.",
        )

    try:
        pdf_bytes = file.file.read(MAX_PDF_BYTES + 1)
    finally:
        file.file.close()

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The PDF exceeds the 25 MB upload limit.",
        )

    document_id = str(uuid.uuid4())

    try:
        result = ingest_pdf_bytes(
            pdf_bytes=pdf_bytes,
            filename=filename,
            document_id=document_id,
            embedding_client=app.state.embedding_client,
            qdrant=app.state.qdrant,
            collection_name=COLLECTION_NAME,
        )

    except PDFProcessingError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "PDF ingestion failed for document %s",
            document_id,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "PDF indexing failed. Check the server logs "
                "for details. Partial indexing was rolled back "
                "when possible."
            ),
        ) from exc

    return {
        "status": "indexed",
        **result,
        "next_step": (
            "Send document_id with POST /ask "
            "to query this PDF."
        ),
    }


@app.post("/ask")
def ask_question(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        document_id = str(uuid.UUID(request.document_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "document_id must be the valid ID "
                "returned by POST /upload."
            ),
        ) from exc

    try:
        vector = embed_question(
            app.state.embedding_client,
            question,
        )

        evidence = retrieve_evidence(
            app.state.qdrant,
            vector,
            limit=TOP_K,
            document_id=document_id,
        )

        answer = generate_answer(
            app.state.groq_client,
            question,
            evidence,
        )

    except Exception as exc:
        logger.exception("Question answering failed")

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not complete the question. "
                "Check the server logs for details."
            ),
        ) from exc

    sources = [
        {
            "source_id": item["source_id"],
            "document_name": (
                item.get("document_name")
                or "Uploaded PDF"
            ),
            "section": item["section"],
            "page_number": item.get("page_number"),
            "chunk_id": item["chunk_id"],
            "source_url": item["source_url"],
            "similarity": item["score"],
            "excerpt": item["text"],
        }
        for item in evidence
    ]

    return {
        "question": question,
        "document_id": document_id,
        "answer": answer,
        "sources": sources,
    }
