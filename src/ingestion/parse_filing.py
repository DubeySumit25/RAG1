
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "apple"
OUTPUT_DIR = ROOT / "data" / "processed"

MAX_CHARS = 1500
OVERLAP_CHARS = 200


def clean_lines(html_path: Path) -> list[str]:
    with html_path.open("rb") as file:
        soup = BeautifulSoup(file.read(), "lxml")

    for tag in soup(["script", "style", "noscript", "head"]):
        tag.decompose()

    metadata_tags = {
        "ix:header",
        "ix:hidden",
        "ix:resources",
        "xbrli:context",
        "xbrli:unit",
    }

    for tag in soup.find_all(
        lambda element: (
            element.name
            and element.name.lower() in metadata_tags
        )
    ):
        tag.decompose()

    lines = []

    for line in soup.get_text(separator="\n").splitlines():
        line = re.sub(
    r"Apple Inc\.\s*\|\s*\d{4}\s+Form 10-K\s*\|\s*\d+",
    " ",
    line,
    flags=re.IGNORECASE,
)
        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        if re.fullmatch(r"\d{1,4}", line):
            continue

        lines.append(line)

    return lines


def is_section_heading(line: str) -> bool:
    if len(line) > 180:
        return False

    return bool(
        re.match(
            r"^(ITEM\s+\d+[A-Z]?\b|PART\s+[IVX]+\b)",
            line,
            flags=re.IGNORECASE,
        )
    )


def group_by_section(
    lines: list[str],
) -> list[tuple[str, str]]:
    groups = []
    current_section = "Front matter / unclassified"
    current_lines = []

    for line in lines:
        if is_section_heading(line):
            if current_lines:
                groups.append(
                    (current_section, "\n".join(current_lines))
                )

            current_section = line
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        groups.append(
            (current_section, "\n".join(current_lines))
        )

    return groups


def split_chunks(
    text: str,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[str]:
    units = []

    for paragraph in text.splitlines():
        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if not paragraph:
            continue

        sentences = re.split(
            r'(?<=[.!?])\s+(?=[A-Z0-9“"(])',
            paragraph,
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) <= max_chars:
                units.append(sentence)
                continue

            words = sentence.split()
            part = []
            part_length = 0

            for word in words:
                additional_length = len(word) + (1 if part else 0)

                if part and part_length + additional_length > max_chars:
                    units.append(" ".join(part))
                    part = [word]
                    part_length = len(word)
                else:
                    part.append(word)
                    part_length += additional_length

            if part:
                units.append(" ".join(part))

    chunks = []
    current = []

    for unit in units:
        candidate = " ".join(current + [unit])

        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))

            overlap = []

            for previous in reversed(current):
                candidate_overlap = [previous] + overlap

                if len(" ".join(candidate_overlap)) > overlap_chars:
                    break

                overlap = candidate_overlap

            while (
                overlap
                and len(" ".join(overlap + [unit])) > max_chars
            ):
                overlap.pop(0)

            current = overlap + [unit]
        else:
            current.append(unit)

    if current:
        chunks.append(" ".join(current))

    return chunks


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data folder not found: {RAW_DIR}"
        )

    html_files = list(RAW_DIR.glob("*.html"))

    if not html_files:
        raise FileNotFoundError(
            f"No HTML filings found in {RAW_DIR}. "
            "Run scripts/fetch_sec.py first."
        )

    html_path = max(
        html_files,
        key=lambda path: path.stat().st_mtime,
    )
    metadata_path = html_path.with_suffix(".json")

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Matching metadata JSON not found: {metadata_path}"
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    lines = clean_lines(html_path)
    sections = group_by_section(lines)

    sections = [
        (section, section_text)
        for section, section_text in sections
        if not section.strip().lower().startswith("item 15")
    ]

    records = []

    for section, section_text in sections:
        chunks = split_chunks(section_text)

        for chunk in chunks:
            records.append({
                "company": metadata.get("company", "Apple"),
                "form": metadata.get("form", "10-K"),
                "filing_date": metadata.get("filing_date"),
                "report_period": metadata.get("report_period"),
                "accession_number": metadata.get("accession_number"),
                "source_url": metadata.get("source_url"),
                "section": section,
                "chunk_id": (
                    f"{metadata.get('accession_number', 'filing')}:"
                    f"{len(records) + 1:05d}"
                ),
                "text": chunk,
            })

    if not records:
        raise RuntimeError("No text chunks were extracted.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "apple_chunks.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Input filing: {html_path.name}")
    print(f"Extracted text lines: {len(lines)}")
    print(f"Chunks created: {len(records)}")
    print(f"Output saved to: {output_path}")
    print("\nSample chunk:")
    print(records[0]["text"][:500])


if __name__ == "__main__":
    main()
