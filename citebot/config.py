import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = BASE_DIR / "data" / "sample_docs"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
EVAL_DATASET_PATH = BASE_DIR / "data" / "eval" / "qa_dataset.json"
EVAL_RESULTS_PATH = BASE_DIR / "data" / "eval" / "results.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Final number of chunks passed to the generation model, after re-ranking.
TOP_K = 4
# Candidates fetched by each leg of the hybrid retriever (vector + BM25)
# before re-ranking cuts them down to TOP_K.
FETCH_K = 10

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Last N turns (user+assistant pairs) kept as raw history in prompts.
HISTORY_TURNS = 4
