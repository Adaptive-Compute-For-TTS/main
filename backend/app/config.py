import os
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
DATABASE_PATH = DATA_DIR / "app.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv(
    "ADAPTIVE_TTS_DATABASE_URL",
    f"sqlite:///{DATABASE_PATH}",
)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ADAPTIVE_TTS_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

PROCESSING_DELAY_SECONDS = float(
    os.getenv("ADAPTIVE_TTS_PROCESSING_DELAY_SECONDS", "0.05")
)
