"""Load local development settings without overriding process/service secrets."""
from pathlib import Path
from dotenv import load_dotenv

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPOSITORY_ROOT / ".env", override=False)
