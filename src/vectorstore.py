
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def create_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()

    if bool(url) != bool(api_key):
        raise RuntimeError(
            "Set both QDRANT_URL and QDRANT_API_KEY "
            "to use Qdrant Cloud."
        )

    if url and api_key:
        return QdrantClient(
            url=url,
            api_key=api_key,
            timeout=60,
        )

    local_path = Path(
        os.getenv("QDRANT_PATH", "data/qdrant")
    ).expanduser()

    if not local_path.is_absolute():
        local_path = ROOT / local_path

    local_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return QdrantClient(
        path=str(local_path),
    )
