
from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

import pymupdf as fitz
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models


EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

EMBED_BATCH_SIZE = 4
BATCH_DELAY_SECONDS = 6
MAX_PDF_PAGES = 500
MAX_RETRIES = 5


class PDFProcessingError(ValueError):
    """Raised when an uploaded PDF cannot be processed."""


def chunk_text(text: str) -> list[str]:
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            boundary = text.rfind(
                " ",
                start + int(CHUNK_SIZE * 0.7),
                end,
            )

            if boundary > start:
                end = boundary

        piece = text[start:end].strip()

        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def extract_pdf_chunks(
    pdf_bytes: bytes,
) -> tuple[list[dict], int, int]:
    try:
        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        )
    except Exception as exc:
        raise PDFProcessingError(
            "The uploaded file is not a valid readable PDF."
        ) from exc

    try:
        if document.is_encrypted:
            raise PDFProcessingError(
                "Password-protected PDFs are not supported."
            )

        page_count = len(document)

        if page_count == 0:
            raise PDFProcessingError(
                "The PDF contains no pages."
            )

        if page_count > MAX_PDF_PAGES:
            raise PDFProcessingError(
                f"The PDF has {page_count} pages. "
                f"The current limit is {MAX_PDF_PAGES} pages."
            )

        chunks = []
        pages_with_text = set()

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")

            if not text or not text.strip():
                continue

            page_chunks = chunk_text(text)

            if page_chunks:
                pages_with_text.add(page_number)

            for index, piece in enumerate(page_chunks):
                chunks.append({
                    "text": piece,
                    "page_number": page_number,
                    "page_chunk_index": index,
                })

        if not chunks:
            raise PDFProcessingError(
                "No selectable text was found in the PDF. "
                "It may be scanned or image-only. OCR is not "
                "supported in this version."
            )

        return chunks, page_count, len(pages_with_text)

    finally:
        document.close()


def normalize_vector(values: list[float]) -> list[float]:
    if len(values) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            "Gemini returned an unexpected embedding dimension."
        )

    norm = sum(value * value for value in values) ** 0.5

    if norm == 0:
        raise RuntimeError(
            "Gemini returned a zero-length embedding."
        )

    return [value / norm for value in values]


def embed_batch_with_retry(
    embedding_client: genai.Client,
    texts: list[str],
) -> list:
    for attempt in range(MAX_RETRIES):
        try:
            response = embedding_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIMENSIONS,
                ),
            )

            embeddings = response.embeddings or []

            if len(embeddings) != len(texts):
                raise RuntimeError(
                    "Gemini returned an unexpected number "
                    "of passage embeddings."
                )

            if any(
                embedding.values is None
                for embedding in embeddings
            ):
                raise RuntimeError(
                    "Gemini returned an empty passage embedding."
                )

            return embeddings

        except Exception as exc:
            message = str(exc).lower()

            is_quota_error = (
                "429" in message
                or "resource_exhausted" in message
            )

            if not is_quota_error or attempt == MAX_RETRIES - 1:
                raise

            match = re.search(
                r"retry in\s+([0-9.]+)\s*s",
                message,
            )

            if match:
                wait_seconds = float(match.group(1)) + 2
            else:
                wait_seconds = 15 * (attempt + 1)

            wait_seconds = min(
                max(wait_seconds, 12),
                60,
            )

            print(
                f"Gemini embedding quota reached. "
                f"Retrying in {wait_seconds:.0f} seconds "
                f"(attempt {attempt + 2}/{MAX_RETRIES})."
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Embedding retries exhausted.")


def delete_document_points(
    qdrant: QdrantClient,
    collection_name: str,
    document_id: str,
) -> None:
    qdrant.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(
                            value=document_id
                        ),
                    )
                ]
            )
        ),
        wait=True,
    )


def ingest_pdf_bytes(
    pdf_bytes: bytes,
    filename: str,
    document_id: str,
    embedding_client: genai.Client,
    qdrant: QdrantClient,
    collection_name: str,
) -> dict:
    chunks, page_count, pages_with_text = extract_pdf_chunks(
        pdf_bytes
    )

    safe_filename = Path(
        filename or "uploaded.pdf"
    ).name[:180]

    indexed_count = 0

    try:
        for batch_start in range(
            0,
            len(chunks),
            EMBED_BATCH_SIZE,
        ):
            batch = chunks[
                batch_start:batch_start + EMBED_BATCH_SIZE
            ]

            texts = [
                chunk["text"]
                for chunk in batch
            ]

            embeddings = embed_batch_with_retry(
                embedding_client,
                texts,
            )

            points = []

            for local_index, (chunk, embedding) in enumerate(
                zip(batch, embeddings)
            ):
                chunk_index = batch_start + local_index

                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"financial-ai-analyst:"
                        f"{document_id}:{chunk_index}",
                    )
                )

                payload = {
                    "doc_id": document_id,
                    "document_name": safe_filename,
                    "chunk_id": (
                        f"{document_id}:{chunk_index:05d}"
                    ),
                    "chunk_index": chunk_index,
                    "page_number": chunk["page_number"],
                    "section": f"Uploaded PDF: {safe_filename}",
                    "filing_date": "Unknown",
                    "source_url": "",
                    "text": chunk["text"],
                }

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=normalize_vector(
                            list(embedding.values)
                        ),
                        payload=payload,
                    )
                )

            qdrant.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )

            indexed_count += len(points)

            print(
                f"Indexed {indexed_count}/{len(chunks)} chunks"
            )

            if batch_start + len(batch) < len(chunks):
                time.sleep(BATCH_DELAY_SECONDS)

        return {
            "document_id": document_id,
            "document_name": safe_filename,
            "pages": page_count,
            "pages_with_text": pages_with_text,
            "chunks_indexed": indexed_count,
        }

    except Exception:
        try:
            delete_document_points(
                qdrant,
                collection_name,
                document_id,
            )
        except Exception:
            print(
                "Warning: could not fully roll back "
                f"document {document_id}. Check Qdrant."
            )

        raise
