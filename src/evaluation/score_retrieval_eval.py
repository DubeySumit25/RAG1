import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "retrieval_eval_review.json"
TOP_K = 5


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not cases:
        raise ValueError("The evaluation JSON contains no questions.")

    hit_scores = []
    precision_scores = []
    reciprocal_ranks = []

    for case in cases:
        question = case["question"]
        results = case.get("results", [])[:TOP_K]

        if not results:
            raise ValueError(f"No results found for: {question}")

        if any(
            result.get("relevant") not in (True, False)
            for result in results
        ):
            raise ValueError(
                "Some relevance labels are still null. "
                "Open retrieval_eval_review.json and label every "
                "result with true or false."
            )

        relevant_positions = [
            position
            for position, result in enumerate(results, start=1)
            if result["relevant"] is True
        ]

        hit = int(bool(relevant_positions))
        precision = sum(
            result["relevant"] is True for result in results
        ) / TOP_K
        reciprocal_rank = (
            1 / relevant_positions[0]
            if relevant_positions
            else 0.0
        )

        hit_scores.append(hit)
        precision_scores.append(precision)
        reciprocal_ranks.append(reciprocal_rank)

        print(f"\nQuestion: {question}")
        print(f"Relevant passages: {len(relevant_positions)}")
        print(f"Hit@{TOP_K}: {hit}")
        print(f"Precision@{TOP_K}: {precision:.3f}")
        print(f"Reciprocal Rank: {reciprocal_rank:.3f}")

    count = len(cases)

    print("\n" + "=" * 45)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 45)
    print(f"Questions evaluated: {count}")
    print(f"Hit@{TOP_K}: {sum(hit_scores) / count:.3f}")
    print(
        f"Precision@{TOP_K}: "
        f"{sum(precision_scores) / count:.3f}"
    )
    print(
        f"MRR@{TOP_K}: "
        f"{sum(reciprocal_ranks) / count:.3f}"
    )


if __name__ == "__main__":
    main()
