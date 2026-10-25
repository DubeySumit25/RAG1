
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv


from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models


# Paths
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
CHUNKS_FILE = ROOT / "data" / "processed" / "apple_chunks.jsonl"
DB_PATH = ROOT / "data" / "qdrant"

# Gemini embedding configuration
COLLECTION_NAME = "financial_reports_gemini"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768

BATCH_SIZE = 32
MAX_ATTEMPTS = 4
PAUSE_BETWEEN_REQUESTS = 2


def load_chunks() -> list[dict]:
    """Load validated text chunks from JSONL."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}\n"
            "Run python src/ingestion/parse_filing.py first."
        )

    records = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            if not isinstance(record.get("text"), str):
                raise ValueError(
                    f"Missing text on line {line_number}"
                )

            if not record["text"].strip():
                continue

            if not record.get("chunk_id"):
                raise ValueError(
                    f"Missing chunk_id on line {line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError("No valid chunks found.")

    return records


def stable_point_id(chunk_id: str) -> str:
    """Generate the same Qdrant ID for the same chunk."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def text_hash(text: str) -> str:
    """Detect changes in chunk contents."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_vector(values: list[float]) -> list[float]:
    """Normalize an embedding vector for cosine search."""
    norm = math.sqrt(sum(value * value for value in values))

    if norm == 0:
        raise ValueError("Gemini returned a zero-length vector.")

    return [value / norm for value in values]


def embed_documents(
    client: genai.Client,
    texts: list[str],
) -> list[list[float]]:
    """Generate document embeddings using Gemini's hosted API."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )

    if not response.embeddings:
        raise RuntimeError("Gemini returned no embeddings.")

    if len(response.embeddings) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embeddings, "
            f"received {len(response.embeddings)}."
        )

    vectors = []

    for embedding in response.embeddings:
        values = embedding.values

        if values is None or len(values) != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                "Unexpected embedding dimensions from Gemini."
            )

        vectors.append(normalize_vector(values))

    return vectors


def embed_with_retry(
    client: genai.Client,
    texts: list[str],
) -> list[list[float]]:
    """Retry temporary API rate limits with exponential backoff."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return embed_documents(client, texts)

        except Exception as exc:
            message = str(exc).lower()

            retryable = any(
                marker in message
                for marker in (
                    "429",
                    "resource_exhausted",
                    "rate limit",
                    "quota",
                    "503",
                    "service unavailable",
                )
            )

            if not retryable or attempt == MAX_ATTEMPTS - 1:
                raise

            matches = re.findall(
                r"(\d+(?:\.\d+)?)\s*s\b",
                message,
            )
            server_delay = float(matches[-1]) if matches else 0

            backoff = min(60 * (2 ** attempt), 180)
            wait_seconds = max(server_delay, backoff)

            print(
                f"\nGemini rate limit or temporary service error."
                f"\nWaiting {wait_seconds:.0f} seconds."
                f"\nRetry {attempt + 1}/{MAX_ATTEMPTS - 1}..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Embedding retries exhausted.")


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in this terminal."
        )

    records = load_chunks()
    total = len(records)

    print(f"Loaded {total} chunks.")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("Connecting to Gemini and Qdrant...")

    DB_PATH.mkdir(parents=True, exist_ok=True)

    embedding_client = genai.Client()
    qdrant = QdrantClient(path=str(DB_PATH))

    try:
        # Create the collection only if it doesn't exist.
        # Existing successfully indexed points are preserved.
        if not qdrant.collection_exists(COLLECTION_NAME):
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSIONS,
                    distance=models.Distance.COSINE,
                ),
            )
            print("Created a new Qdrant collection.")
        else:
            info = qdrant.get_collection(COLLECTION_NAME)
            vector_config = info.config.params.vectors

            if (
                not isinstance(vector_config, models.VectorParams)
                or vector_config.size != EMBEDDING_DIMENSIONS
            ):
                raise RuntimeError(
                    "The existing collection has incompatible vector "
                    "dimensions. Back up data/qdrant before rebuilding "
                    "the financial_reports_gemini collection."
                )

            print("Existing compatible collection found.")

        for start in range(0, total, BATCH_SIZE):
            batch = records[start:start + BATCH_SIZE]

            # Reuse vectors only if the chunk text hasn't changed.
            ids = [
                stable_point_id(record["chunk_id"])
                for record in batch
            ]

            existing_points = qdrant.retrieve(
                collection_name=COLLECTION_NAME,
                ids=ids,
                with_payload=True,
                with_vectors=False,
            )

            existing_by_id = {
                str(point.id): point
                for point in existing_points
            }

            pending = []

            for record in batch:
                point_id = stable_point_id(record["chunk_id"])
                digest = text_hash(record["text"])
                stored = existing_by_id.get(point_id)

                if (
                    stored is not None
                    and stored.payload is not None
                    and stored.payload.get("_text_hash") == digest
                ):
                    continue

                pending.append((record, digest))

            if not pending:
                print(
                    f"Chunks {start + 1}-"
                    f"{min(start + len(batch), total)} "
                    "(already indexed)"
                )
                continue

            texts = [record["text"] for record, _ in pending]

            # Remote embedding call with retry handling.
            vectors = embed_with_retry(embedding_client, texts)

            points = []

            for (record, digest), vector in zip(pending, vectors):
                payload = dict(record)
                payload["_text_hash"] = digest

                points.append(
                    models.PointStruct(
                        id=stable_point_id(record["chunk_id"]),
                        vector=vector,
                        payload=payload,
                    )
                )

            # Upsert replaces stale versions of the same chunk.
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

            count = qdrant.count(
                collection_name=COLLECTION_NAME,
                exact=True,
            ).count

            print(
                f"Progress: batch ending at "
                f"{min(start + len(batch), total)}/{total}; "
                f"stored points in collection: {count}"
            )

            time.sleep(PAUSE_BETWEEN_REQUESTS)

        print("\nIndexing completed.")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Embedding dimensions: {EMBEDDING_DIMENSIONS}")
        print(
            "All current chunks have been indexed or verified "
            "as already indexed."
        )

    finally:
        qdrant.close()
        embedding_client.close()


if __name__ == "__main__":
    main()
    
