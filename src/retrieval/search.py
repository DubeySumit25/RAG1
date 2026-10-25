
import math
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
DB_PATH = ROOT / "data" / "qdrant"

COLLECTION_NAME = "financial_reports_gemini"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768


def embed_question(client: genai.Client, question: str) -> list[float]:
    """Embed a query using Gemini's retrieval-query task."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )

    if not response.embeddings or response.embeddings[0].values is None:
        raise RuntimeError("Gemini returned no query embedding.")

    vector = response.embeddings[0].values

    if len(vector) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Expected {EMBEDDING_DIMENSIONS} dimensions, "
            f"received {len(vector)}."
        )

    # Match the normalization used when indexing documents.
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise RuntimeError("Received a zero-length query vector.")

    return [value / norm for value in vector]


def main() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is missing from this terminal session."
        )

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Qdrant database not found: {DB_PATH}. "
            "Build the index first."
        )

    question = input("\nAsk a question about Apple's report: ").strip()
    if not question:
        print("Please enter a question.")
        return

    embedding_client = genai.Client()
    qdrant = QdrantClient(path=str(DB_PATH))

    try:
        if not qdrant.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found. "
                "Run build_index.py first."
            )

        query_vector = embed_question(embedding_client, question)

        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5,
            with_payload=True,
        ).points

        if not results:
            print("No matching passages found.")
            return

        print(f"\nTop results for: {question}\n")

        for rank, point in enumerate(results, start=1):
            payload = point.payload or {}

            print("=" * 75)
            print(f"Result: {rank} | Similarity score: {point.score:.4f}")
            print(f"Section: {payload.get('section', 'Unknown')}")
            print(f"Filing date: {payload.get('filing_date', 'Unknown')}")
            print(f"Chunk ID: {payload.get('chunk_id', 'Unknown')}")
            print(f"Source: {payload.get('source_url', 'Unknown')}")
            print("\nPassage:")
            print(payload.get("text", "No text stored in payload"))
            print()

    finally:
        qdrant.close()
        embedding_client.close()


if __name__ == "__main__":
    main()
