
import json
import math
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "data" / "qdrant"
OUTPUT_PATH = ROOT / "data" / "processed" / "retrieval_eval_review.json"

COLLECTION_NAME = "financial_reports_gemini"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
TOP_K = 5

QUESTIONS = [
    "What are the main risks Apple identifies in its 2025 annual report?",
    "What risks does Apple describe from relying on single or limited sources for components?",
    "What cybersecurity risks does Apple disclose in its annual report?",
    "What does Apple say about legal proceedings and loss contingencies?",
    "How can public-health disruptions affect Apple's manufacturing and supply chain?",
]


def embed_question(
    client: genai.Client,
    question: str,
) -> list[float]:
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
        raise RuntimeError("Unexpected embedding dimensions.")

    norm = math.sqrt(sum(value * value for value in vector))

    if norm == 0:
        raise RuntimeError("Embedding vector has zero length.")

    return [value / norm for value in vector]


def save_results(cases: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(
        cases,
        ensure_ascii=False,
        indent=2,
    )

    OUTPUT_PATH.write_text(content, encoding="utf-8")

    print(
        f"Saved {len(cases)} completed questions. "
        f"File size: {OUTPUT_PATH.stat().st_size} bytes."
    )


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Check your project .env file."
        )

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Qdrant database not found: {DB_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    embedding_client = genai.Client()
    qdrant = QdrantClient(path=str(DB_PATH))
    evaluation_cases = []

    try:
        if not qdrant.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Collection {COLLECTION_NAME} was not found."
            )

        total_questions = len(QUESTIONS)

        for index, question in enumerate(QUESTIONS, start=1):
            print(f"\nQuestion {index}/{total_questions}:")
            print(question)

            vector = embed_question(embedding_client, question)

            response = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=TOP_K,
                with_payload=True,
            )

            results = []

            for rank, point in enumerate(response.points, start=1):
                payload = point.payload or {}

                result = {
                    "rank": rank,
                    "chunk_id": payload.get("chunk_id", ""),
                    "section": payload.get("section", "Unknown"),
                    "score": point.score,
                    "source_url": payload.get("source_url", ""),
                    "text": payload.get("text", ""),
                    "relevant": None,
                }

                results.append(result)

                print(
                    f"  {rank}. {result['section']} "
                    f"| {result['chunk_id']} "
                    f"| score={result['score']:.4f}"
                )

            evaluation_cases.append({
                "question": question,
                "results": results,
            })

            save_results(evaluation_cases)

            if index < total_questions:
                time.sleep(2)

    finally:
        qdrant.close()
        embedding_client.close()

    print("\nEvaluation review file created successfully.")
    print(f"Location: {OUTPUT_PATH}")
    print(f"Questions saved: {len(evaluation_cases)}")
    print(f"Bytes written: {OUTPUT_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
