import json
import os
from pathlib import Path

import requests

CIK = "0000320193"  # Apple; 10-digit CIK
COMPANY = "apple"

BASE_DIR = Path("data/raw") / COMPANY
BASE_DIR.mkdir(parents=True, exist_ok=True)

SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT")
if not SEC_USER_AGENT or "@" not in SEC_USER_AGENT:
    raise SystemExit(
        "Set SEC_USER_AGENT to your name and real contact email."
    )

HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}


def get_json(url: str) -> dict:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    # 1. Retrieve filing history.
    submissions_url = (
        f"https://data.sec.gov/submissions/CIK{CIK}.json"
    )
    submissions = get_json(submissions_url)

    recent = submissions["filings"]["recent"]
    forms = recent["form"]

    # 2. Find the newest annual 10-K, ignoring amendments.
    index = next(
        (i for i, form in enumerate(forms) if form == "10-K"),
        None,
    )
    if index is None:
        raise RuntimeError("No recent 10-K found in filing history.")

    accession = recent["accessionNumber"][index]
    document = recent["primaryDocument"][index]
    filing_date = recent["filingDate"][index]
    report_period = recent.get("reportDate", [""] * len(forms))[index]

    # 3. Construct the original filing URL.
    accession_path = accession.replace("-", "")
    cik_path = str(int(CIK))

    filing_url = (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_path}/{accession_path}/{document}"
    )

    # 4. Download and preserve the original HTML.
    response = requests.get(
        filing_url,
        headers=HEADERS,
        timeout=60,
    )
    response.raise_for_status()

    output_html = BASE_DIR / f"{accession}.html"
    output_html.write_bytes(response.content)

    # 5. Preserve enough metadata to cite this filing later.
    metadata = {
        "company": submissions.get("name", COMPANY),
        "cik": CIK,
        "form": "10-K",
        "filing_date": filing_date,
        "report_period": report_period,
        "accession_number": accession,
        "source_url": filing_url,
        "local_html": str(output_html),
    }

    metadata_path = BASE_DIR / f"{accession}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Downloaded filing: {filing_date}")
    print(f"Source: {filing_url}")
    print(f"HTML saved to: {output_html}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()