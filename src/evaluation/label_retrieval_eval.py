import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "data" / "processed" / "retrieval_eval_review.json"


def save(cases):
    FILE.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    if not FILE.exists():
        raise FileNotFoundError(f"Review file not found: {FILE}")

    cases = json.loads(FILE.read_text(encoding="utf-8"))
    total = sum(len(case.get("results", [])) for case in cases)
    completed = sum(
        result.get("relevant") in (True, False)
        for case in cases
        for result in case.get("results", [])
    )

    print(f"Questions: {len(cases)}")
    print(f"Passages: {total}")
    print(f"Already labelled: {completed}")
    print("\nFor each passage:")
    print("  y = relevant to the question")
    print("  n = not relevant")
    print("  q = quit and save progress")

    for case in cases:
        question = case["question"]

        for result in case.get("results", []):
            if result.get("relevant") in (True, False):
                continue

            print("\n" + "=" * 70)
            print("QUESTION:", question)
            print(
                f"PASSAGE {result['rank']} | "
                f"Section: {result.get('section', 'Unknown')} | "
                f"Score: {result.get('score', 0):.4f}"
            )
            print("Chunk:", result.get("chunk_id", "Unknown"))
            print("\nPASSAGE TEXT:\n")
            print(result.get("text", ""))

            while True:
                choice = input(
                    "\nIs this passage relevant? [y/n/q]: "
                ).strip().lower()

                if choice == "y":
                    result["relevant"] = True
                    break
                elif choice == "n":
                    result["relevant"] = False
                    break
                elif choice == "q":
                    save(cases)
                    print("\nProgress saved.")
                    return
                else:
                    print("Enter y, n, or q.")

            save(cases)
            print("Label saved.")

    print("\nAll passages have been labelled.")
    print(f"Updated file: {FILE}")


if __name__ == "__main__":
    main()
