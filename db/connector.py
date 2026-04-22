"""
Database connection module for auslegalsearchv4.
- Connects to PostgreSQL with pgvector extension enabled.
- Provides SQLAlchemy engine and session makers.
- Checks/creates vector extension if needed.
"""

import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Minimal .env loader (dependency-free). Reads KEY=VALUE lines and sets them in os.environ
# ONLY if the key is not already exported. This makes `source .env` (without export) work.
def _load_dotenv_file():
    try:
        here = os.path.abspath(os.path.dirname(__file__))
        candidates = [
            os.path.abspath(os.path.join(here, "..", ".env")),   # repo root
            os.path.abspath(os.path.join(os.getcwd(), ".env")),  # current working dir
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
                break
    except Exception:
        # Never fail due to dotenv parsing
        pass

# Load .env before reading variables (exported env vars still take precedence)
_load_dotenv_file()

STORAGE_BACKEND = os.environ.get("AUSLEGALSEARCH_STORAGE_BACKEND", "postgres").strip().lower()

DB_HOST = os.environ.get("AUSLEGALSEARCH_DB_HOST")
DB_PORT = os.environ.get("AUSLEGALSEARCH_DB_PORT")
DB_USER = os.environ.get("AUSLEGALSEARCH_DB_USER")
DB_PASSWORD = os.environ.get("AUSLEGALSEARCH_DB_PASSWORD")
DB_NAME = os.environ.get("AUSLEGALSEARCH_DB_NAME")

DB_URL = os.environ.get("AUSLEGALSEARCH_DB_URL")
if STORAGE_BACKEND != "opensearch":
    if not DB_URL:
        # Require explicit env configuration to avoid accidental defaults.
        required = {
            "AUSLEGALSEARCH_DB_HOST": DB_HOST,
            "AUSLEGALSEARCH_DB_PORT": DB_PORT,
            "AUSLEGALSEARCH_DB_USER": DB_USER,
            "AUSLEGALSEARCH_DB_PASSWORD": DB_PASSWORD,
            "AUSLEGALSEARCH_DB_NAME": DB_NAME,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(
                "Missing required DB env vars: "
                + ", ".join(missing)
                + ". Provide AUSLEGALSEARCH_DB_URL or individual AUSLEGALSEARCH_DB_* variables (see BetaDataLoad.md)."
            )
        # Safely quote credentials in case they contain special characters like @ : / # & +
        user_q = quote_plus(DB_USER)
        pwd_q = quote_plus(DB_PASSWORD)
        DB_URL = f"postgresql+psycopg2://{user_q}:{pwd_q}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Production-grade engine config:
# - pool_pre_ping: avoid stale connections
# - pool_size/max_overflow: tuneable via env, sensible defaults
# - pool_recycle: recycle connections periodically to avoid server-side timeouts
# - pool_timeout: bound waiting time for a free connection
# - connect_args: psycopg2 keepalives + connect_timeout + optional statement_timeout
POOL_SIZE = int(os.environ.get("AUSLEGALSEARCH_DB_POOL_SIZE", "10"))
MAX_OVERFLOW = int(os.environ.get("AUSLEGALSEARCH_DB_MAX_OVERFLOW", "20"))
POOL_RECYCLE = int(os.environ.get("AUSLEGALSEARCH_DB_POOL_RECYCLE", "1800"))  # seconds
POOL_TIMEOUT = int(os.environ.get("AUSLEGALSEARCH_DB_POOL_TIMEOUT", "30"))    # seconds
STATEMENT_TIMEOUT_MS = os.environ.get("AUSLEGALSEARCH_DB_STATEMENT_TIMEOUT_MS")  # e.g. "60000"

connect_opts = []
if STATEMENT_TIMEOUT_MS:
    # Sets server-side statement timeout for each session
    connect_opts.append(f"-c statement_timeout={int(STATEMENT_TIMEOUT_MS)}")

if STORAGE_BACKEND == "opensearch":
    engine = None
    SessionLocal = None
else:
    engine = create_engine(
        DB_URL,
        pool_pre_ping=True,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_recycle=POOL_RECYCLE,
        pool_timeout=POOL_TIMEOUT,
        connect_args={
            "connect_timeout": 10,
            # TCP keepalives (Linux)
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
            # Server-side options
            "options": " ".join(connect_opts) if connect_opts else None,
        },
    )
    SessionLocal = sessionmaker(bind=engine)

from sqlalchemy.dialects.postgresql import UUID as UUIDType, JSONB
from pgvector.sqlalchemy import Vector

def ensure_pgvector():
    if engine is None:
        return
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
