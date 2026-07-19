# CiteBot

A document chatbot with **RAG (Retrieval-Augmented Generation)** that answers
questions using *only* the content of your documents (PDF, Markdown, TXT) and always
shows which file (and page, if applicable) each answer came from.

Unlike a generic chatbot, CiteBot doesn't "make things up" — if the answer isn't in
the indexed documents, it explicitly says it doesn't know.

## How it works (RAG flow)

```
Upload documents
      │
      ▼
Chunking (RecursiveCharacterTextSplitter)
      │
      ▼
Embeddings (OpenAI text-embedding-3-small)
      │
      ▼
Vector store (Chroma, persisted to disk)
      │
      ▼
User question ──► Retrieval (top-k most similar chunks)
                        │
                        ▼
             Prompt with that context ──► LLM (gpt-4o-mini)
                                                 │
                                                 ▼
                                   Answer + cited sources
```

1. **Ingestion**: documents are parsed based on their extension (`PyPDFLoader` for
   PDF, `TextLoader` for Markdown/TXT), preserving origin metadata (`source`,
   `page`).
2. **Chunking**: documents are split into ~1000-character fragments with 150
   characters of overlap (`citebot/config.py`), to avoid losing context at the
   boundaries.
3. **Embeddings + vector store**: each chunk is converted into a vector with
   `text-embedding-3-small` and stored in Chroma, persisted locally under
   `data/vectorstore/`.
4. **Retrieval**: given a question, the `top_k=4` most similar chunks are retrieved.
5. **Generation**: the LLM receives *only* those chunks as context and answers. The
   prompt (`citebot/rag_chain.py`) explicitly instructs it to say
   **"I don't know based on the documents"** if the context isn't enough.
6. **Citations**: the answer comes with a list of the sources used (file, page, and a
   short excerpt of each chunk).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in OPENAI_API_KEY in .env
```

## Usage

```bash
streamlit run app.py
```

On first run, if no documents have been indexed yet, CiteBot automatically loads the
sample documents in `data/sample_docs/` (a fictional travel policy, onboarding FAQ,
security guide, and a RAG design doc) so you can try it without uploading anything.
You can also upload your own PDF/MD/TXT files from the sidebar and re-index.

## Evaluation

`citebot/evaluate.py` runs a fixed question set (`data/eval/qa_dataset.json`, 15
questions) against a freshly built vectorstore over the sample documents, measuring
two things independently (as the design doc in `data/sample_docs/rag-design-doc.md`
recommends):

- **Retrieval hit rate**: did the retriever bring back the chunk(s) from the expected
  source file(s)?
- **Answer correctness**: does the generated answer convey the same facts as a
  reference answer (graded by an LLM judge), including two "unanswerable" questions
  that should trigger the "I don't know based on the documents" refusal.

```bash
python -m citebot.evaluate
```

Results are printed as a table and saved to `data/eval/results.json`. Current
baseline on the sample docs: 100% retrieval hit rate, 100% answer correctness.

## Design decisions and tradeoffs

- **Chunk size (1000/150)**: larger chunks give more context per fragment but dilute
  retrieval relevance; smaller chunks are more precise but can lose context that
  crosses the cut boundary. 1000/150 is a reasonable starting point for prose-style
  documents like policies/FAQs.
- **top_k=4**: how many chunks get passed to the LLM. Raising it improves recall at
  the cost of more tokens (and more noise) in the prompt; lowering it is cheaper but
  can miss relevant information spread across several fragments.
- **Not enough context**: instead of letting the LLM fill gaps with general knowledge
  (hallucination), the prompt forces an explicit "I don't know" answer, which is the
  key behavior that sets this apart from a generic chatbot.
- **Local Chroma**: requires no external infrastructure or extra accounts, ideal for
  a demo/portfolio project. For production with many concurrent documents, a managed
  vector store would be a better fit.

## Structure

```
citebot/
├── app.py                # Streamlit UI
├── citebot/
│   ├── config.py           # constants (chunk size, top_k, models)
│   ├── ingest.py            # loading, chunking, and vector store
│   ├── rag_chain.py         # retriever + prompt + LLM + citations + history
│   └── evaluate.py          # evaluation harness
├── data/
│   ├── sample_docs/          # sample documents
│   ├── eval/                  # QA dataset + evaluation results
│   └── vectorstore/           # Chroma persistence (gitignored)
└── requirements.txt
```
