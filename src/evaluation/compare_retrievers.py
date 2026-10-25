
import json
import math
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "data" / "qdrant"
CHUNKS_PATH = ROOT / "data" / "processed" / "apple_chunks.jsonl"
OLD_REVIEW_PATH = ROOT / "data" / "processed" / "retrieval_eval_review.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "retriever_comparison_results.json"

COLLECTION = "financial_reports_gemini"
MODEL = "gemini-embedding-001"
DIMENSIONS = 768
TOP_K = 5
CANDIDATE_K = 20
RRF_K = 60


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:['’-][a-z0-9]+)*", text.lower())


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                if record.get("text", "").strip():
                    records.append(record)

    if not records:
        raise ValueError(f"No usable records in {path}")

    return records


def embed_question(client, question):
    response = client.models.embed_content(
        model=MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=DIMENSIONS,
        ),
    )

    if not response.embeddings or response.embeddings[0].values is None:
        raise RuntimeError("Gemini returned no query embedding.")

    vector = response.embeddings[0].values
    if len(vector) != DIMENSIONS:
        raise RuntimeError("Unexpected embedding dimension.")

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise RuntimeError("Query embedding has zero length.")

    return [value / norm for value in vector]


def new_candidate(record):
    return {
        "chunk_id": record.get("chunk_id", ""),
        "text": record.get("text", ""),
        "section": record.get("section", "Unknown"),
        "source_url": record.get("source_url", ""),
        "score": None,
        "dense_rank": None,
        "dense_score": None,
        "bm25_rank": None,
        "bm25_score": None,
        "rrf_score": 0.0,
        "relevant": None,
    }


def metrics(results):
    relevant_ranks = [
        item["rank"]
        for item in results
        if item["relevant"] is True
    ]

    return {
        "hit_at_5": int(bool(relevant_ranks)),
        "precision_at_5": sum(
            item["relevant"] is True for item in results
        ) / TOP_K,
        "mrr_at_5": (
            1 / min(relevant_ranks) if relevant_ranks else 0.0
        ),
    }


def main():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    old_cases = json.loads(
        OLD_REVIEW_PATH.read_text(encoding="utf-8")
    )
    if not old_cases:
        raise ValueError("The existing retrieval evaluation is empty.")

    records = load_jsonl(CHUNKS_PATH)
    corpus = [tokenize(record["text"]) for record in records]
    bm25 = BM25Okapi(corpus)

    previous_labels = {
        case["question"]: {
            result["chunk_id"]: result["relevant"]
            for result in case.get("results", [])
            if result.get("relevant") in (True, False)
        }
        for case in old_cases
    }

    embedding_client = genai.Client()
    qdrant = QdrantClient(path=str(DB_PATH))
    comparison_cases = []

    hit_scores = {"dense": [], "hybrid": []}
    precision_scores = {"dense": [], "hybrid": []}
    mrr_scores = {"dense": [], "hybrid": []}

    try:
        if not qdrant.collection_exists(COLLECTION):
            raise RuntimeError(f"Qdrant collection not found: {COLLECTION}")

        for question_number, old_case in enumerate(old_cases, start=1):
            question = old_case["question"]

            print(f"\n{'=' * 75}")
            print(f"Question {question_number}/{len(old_cases)}")
            print(question)

            vector = embed_question(embedding_client, question)

            dense_response = qdrant.query_points(
                collection_name=COLLECTION,
                query=vector,
                limit=CANDIDATE_K,
                with_payload=True,
            )

            pool = {}
            dense_ranked = []

            for rank, point in enumerate(dense_response.points, start=1):
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id")

                if not chunk_id:
                    continue

                candidate = new_candidate(payload)
                candidate["dense_rank"] = rank
                candidate["dense_score"] = point.score
                candidate["rrf_score"] += 1 / (RRF_K + rank)

                pool[chunk_id] = candidate
                dense_ranked.append(candidate)

            bm25_scores = bm25.get_scores(tokenize(question))
            bm25_indices = sorted(
                range(len(records)),
                key=lambda idx: float(bm25_scores[idx]),
                reverse=True,
            )
            bm25_indices = [
                idx for idx in bm25_indices
                if bm25_scores[idx] > 0
            ][:CANDIDATE_K]

            for rank, idx in enumerate(bm25_indices, start=1):
                record = records[idx]
                chunk_id = record["chunk_id"]

                candidate = pool.get(chunk_id)
                if candidate is None:
                    candidate = new_candidate(record)
                    pool[chunk_id] = candidate

                candidate["bm25_rank"] = rank
                candidate["bm25_score"] = float(bm25_scores[idx])
                candidate["rrf_score"] += 1 / (RRF_K + rank)

            dense_top = [
                dict(item)
                for item in dense_ranked[:TOP_K]
            ]

            hybrid_top = [
                dict(item)
                for item in sorted(
                    pool.values(),
                    key=lambda item: item["rrf_score"],
                    reverse=True,
                )[:TOP_K]
            ]

            for rank, item in enumerate(dense_top, start=1):
                item["rank"] = rank

            for rank, item in enumerate(hybrid_top, start=1):
                item["rank"] = rank

            judgements = dict(previous_labels.get(question, {}))
            candidates_to_label = {}

            for item in dense_top + hybrid_top:
                chunk_id = item["chunk_id"]

                if chunk_id not in judgements:
                    judgements[chunk_id] = None

                candidates_to_label[chunk_id] = item

            for chunk_id, item in candidates_to_label.items():
                if judgements[chunk_id] in (True, False):
                    continue

                print(f"\nNew passage to judge: {chunk_id}")
                print(f"Section: {item['section']}")
                print(item["text"])
                print("\nDoes this passage provide useful evidence for this question?")
                print("Enter y for relevant or n for not relevant.")

                while True:
                    answer = input("[y/n]: ").strip().lower()
                    if answer == "y":
                        judgements[chunk_id] = True
                        break
                    if answer == "n":
                        judgements[chunk_id] = False
                        break
                    print("Please enter y or n.")

            for method_results in (dense_top, hybrid_top):
                for item in method_results:
                    item["relevant"] = judgements[item["chunk_id"]]

            dense_metrics = metrics(dense_top)
            hybrid_metrics = metrics(hybrid_top)

            for method, result_metrics in (
                ("dense", dense_metrics),
                ("hybrid", hybrid_metrics),
            ):
                hit_scores[method].append(result_metrics["hit_at_5"])
                precision_scores[method].append(
                    result_metrics["precision_at_5"]
                )
                mrr_scores[method].append(result_metrics["mrr_at_5"])

            print("\nTop-five comparison:")
            print(f"{'Rank':<6}{'Dense chunk':<36}{'Hybrid chunk':<36}")

            for rank in range(max(len(dense_top), len(hybrid_top))):
                dense_id = (
                    dense_top[rank]["chunk_id"] if rank < len(dense_top)
                    else "-"
                )
                hybrid_id = (
                    hybrid_top[rank]["chunk_id"] if rank < len(hybrid_top)
                    else "-"
                )
                print(f"{rank + 1:<6}{dense_id:<36}{hybrid_id:<36}")

            print(
                f"\nDense: Hit@5={dense_metrics['hit_at_5']}, "
                f"Precision@5={dense_metrics['precision_at_5']:.3f}, "
                f"MRR@5={dense_metrics['mrr_at_5']:.3f}"
            )
            print(
                f"Hybrid: Hit@5={hybrid_metrics['hit_at_5']}, "
                f"Precision@5={hybrid_metrics['precision_at_5']:.3f}, "
                f"MRR@5={hybrid_metrics['mrr_at_5']:.3f}"
            )

            comparison_cases.append({
                "question": question,
                "dense_results": dense_top,
                "hybrid_results": hybrid_top,
                "dense_metrics": dense_metrics,
                "hybrid_metrics": hybrid_metrics,
            })

            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_PATH.write_text(
                json.dumps(
                    comparison_cases,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            if question_number < len(old_cases):
                time.sleep(3)

    finally:
        qdrant.close()
        embedding_client.close()

    count = len(comparison_cases)

    print("\n" + "=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)
    print(f"{'Metric':<20}{'Dense search':<18}{'Hybrid search':<18}")

    for label, values in (
        ("Hit@5", hit_scores),
        ("Precision@5", precision_scores),
        ("MRR@5", mrr_scores),
    ):
        dense_average = sum(values["dense"]) / count
        hybrid_average = sum(values["hybrid"]) / count
        print(f"{label:<20}{dense_average:<18.3f}{hybrid_average:<18.3f}")

    print(f"\nFull report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
