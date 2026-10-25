
import json
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CHUNKS_FILE = ROOT / "data" / "processed" / "apple_chunks.jsonl"
DB_PATH = ROOT / "data" / "qdrant"

COLLECTION_NAME = "financial_reports_gemini"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768

CANDIDATE_K = 20
TOP_K = 5
RRF_K = 60

DENSE_WEIGHT = 2.0
BM25_WEIGHT = 1.0
RISK_SECTION_BOOST = 1.5

EXCLUDED_SECTION = "front matter / unclassified"


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[a-z0-9]+(?:['’-][a-z0-9]+)*",
        text.lower(),
    )


def is_searchable_section(record: dict) -> bool:
    section = record.get("section", "").strip().lower()
    return section != EXCLUDED_SECTION


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_FILE}"
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

            text = record.get("text", "")
            chunk_id = record.get("chunk_id")

            if not isinstance(text, str) or not text.strip():
                continue

            if not chunk_id:
                raise ValueError(
                    f"Missing chunk_id on line {line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError("No valid chunks found.")

    return records


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

    if not response.embeddings:
        raise RuntimeError("Gemini returned no embedding.")

    vector = response.embeddings[0].values

    if vector is None or len(vector) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            "Unexpected query embedding dimensions."
        )

    norm = math.sqrt(sum(value * value for value in vector))

    if norm == 0:
        raise RuntimeError("Query embedding has zero length.")

    return [value / norm for value in vector]


def create_candidate(record: dict) -> dict:
    return {
        "chunk_id": record.get("chunk_id", ""),
        "text": record.get("text", ""),
        "section": record.get("section", "Unknown"),
        "filing_date": record.get("filing_date", "Unknown"),
        "source_url": record.get("source_url", ""),
        "dense_rank": None,
        "dense_score": None,
        "bm25_rank": None,
        "bm25_score": None,
        "rrf_score": 0.0,
    }


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is missing from .env"
        )

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Qdrant database not found: {DB_PATH}"
        )

    records = load_chunks()

    bm25_records = [
        record
        for record in records
        if is_searchable_section(record)
    ]

    if not bm25_records:
        raise ValueError(
            "No searchable chunks remain after section filtering."
        )

    corpus = [
        tokenize(record["text"])
        for record in bm25_records
    ]

    if any(not tokens for tokens in corpus):
        raise ValueError(
            "At least one searchable chunk contains no tokens. "
            "Inspect the processed chunks before continuing."
        )

    bm25 = BM25Okapi(corpus)

    question = input(
        "\nAsk a question about Apple's annual report: "
    ).strip()

    if not question:
        print("Please enter a question.")
        return

    query_tokens = tokenize(question)

    if not query_tokens:
        print("The question contains no searchable terms.")
        return

    risk_query = bool(
        re.search(
            r"\brisks?\b|\bthreats?\b|\buncertaint(y|ies)\b",
            question.lower(),
        )
    )

    embedding_client = genai.Client()
    qdrant = QdrantClient(path=str(DB_PATH))

    try:
        if not qdrant.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Collection not found: {COLLECTION_NAME}. "
                "Build the Gemini index first."
            )

        query_vector = embed_question(
            embedding_client,
            question,
        )

        dense_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=CANDIDATE_K,
            with_payload=True,
        )

        candidates = {}

        for rank, point in enumerate(
            dense_response.points,
            start=1,
        ):
            payload = point.payload or {}

            if not is_searchable_section(payload):
                continue

            chunk_id = payload.get("chunk_id")

            if not chunk_id:
                continue

            candidate = candidates.setdefault(
                chunk_id,
                create_candidate(payload),
            )

            candidate["dense_rank"] = rank
            candidate["dense_score"] = point.score
            candidate["rrf_score"] += (
                DENSE_WEIGHT / (RRF_K + rank)
            )

        bm25_scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(bm25_records)),
            key=lambda index: float(bm25_scores[index]),
            reverse=True,
        )

        positive_indices = [
            index
            for index in ranked_indices
            if bm25_scores[index] > 0
        ][:CANDIDATE_K]

        for rank, index in enumerate(
            positive_indices,
            start=1,
        ):
            record = bm25_records[index]
            chunk_id = record["chunk_id"]

            candidate = candidates.setdefault(
                chunk_id,
                create_candidate(record),
            )

            candidate["bm25_rank"] = rank
            candidate["bm25_score"] = float(
                bm25_scores[index]
            )
            candidate["rrf_score"] += (
                BM25_WEIGHT / (RRF_K + rank)
            )

        if risk_query:
            for candidate in candidates.values():
                section = candidate["section"].strip().lower()

                if section.startswith("item 1a. risk factors"):
                    candidate["rrf_score"] *= RISK_SECTION_BOOST

        results = sorted(
            candidates.values(),
            key=lambda candidate: candidate["rrf_score"],
            reverse=True,
        )[:TOP_K]

        print("\nHYBRID SEARCH RESULTS")
        print("=" * 75)
        print(f"Question: {question}")
        print(f"Total loaded chunks: {len(records)}")
        print(f"BM25 searchable chunks: {len(bm25_records)}")
        print(f"Results displayed: {len(results)}")
        print(
            "Risk-specific section boost: "
            f"{'applied' if risk_query else 'not applied'}"
        )

        if not results:
            print("No matching passages found.")
            return

        for rank, result in enumerate(results, start=1):
            print("\n" + "-" * 75)
            print(f"Result: {rank}")
            print(f"Section: {result['section']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"RRF score: {result['rrf_score']:.6f}")
            print(f"Dense rank: {result['dense_rank']}")
            print(f"Dense similarity: {result['dense_score']}")
            print(f"BM25 rank: {result['bm25_rank']}")
            print(f"BM25 score: {result['bm25_score']}")
            print(f"Source: {result['source_url']}")
            print("\nPassage:")
            print(result["text"])

    finally:
        qdrant.close()
        embedding_client.close()


if __name__ == "__main__":
    main()
