# Design Doc: Production RAG Pipeline

## 1. Context and Goal

This document describes the design of a Retrieval-Augmented Generation (RAG)
pipeline for answering questions over an internal knowledge base (product manuals,
internal policies, and technical documentation). The main goal is to reduce
hallucinations from the language model by forcing it to ground every answer in real
document fragments and explicitly cite their origin.

The system must support between 500 and 5,000 documents, with daily updates, and
respond in under 3 seconds per query at the 95th percentile.

## 2. Ingestion and Parsing

Input documents can be PDF, Markdown, HTML, or TXT. Each format uses a different
loader:

- **PDF**: text is extracted page by page, preserving the page number as metadata.
  Scanned PDFs (images) require an additional OCR step before they can be indexed;
  without OCR, those documents end up silently empty, which is a risk worth
  monitoring.
- **HTML**: markup is stripped and headings (`h1`-`h3`) are kept as section metadata,
  useful for citing "according to section X of the manual."
- **Markdown/TXT**: loaded directly, with no additional transformation.

All origin metadata (file name, page, section, last-modified date) is propagated all
the way down to the final chunk, because without it there are no citations possible
further down the pipeline.

## 3. Chunking

Text is split into fragments of a fixed size using a recursive splitter (by
paragraph, then by sentence, then by word if needed) to avoid cutting sentences in
half whenever possible.

The key parameters are:

- **Chunk size**: 800-1200 characters is a good starting point for prose-style
  documents (policies, FAQs, manuals). Larger chunks (2000+) work better for highly
  technical content with tables or code, where cutting in half breaks the meaning.
- **Overlap**: between 10% and 20% of the chunk size. Overlap prevents an idea that
  crosses the boundary between two chunks from being fragmented and invisible to
  retrieval.
- **Semantic chunking** (advanced alternative): instead of cutting by fixed size,
  text can be split by embedding similarity between consecutive sentences, grouping
  those that talk about the same thing. It is more expensive to compute but improves
  the coherence of each chunk, especially in long documents covering multiple
  topics.

## 4. Embeddings and Vector Store

Each chunk is converted into an embedding vector and stored alongside its metadata in
a vector store. For low-to-medium volumes (up to ~100k chunks), a local embedded
vector store (such as Chroma or FAISS) is enough and avoids extra infrastructure. For
larger volumes or high-availability requirements, a managed vector store (Pinecone,
Weaviate, pgvector on Postgres) that supports sharding and replication is preferable.

One important note: embeddings must be regenerated if the embedding model changes,
because vectors from different models are not comparable to each other. This means a
change of embedding model is a full migration, not an incremental tweak.

## 5. Retrieval

Given a user question, it is also converted into a vector using the same embedding
model, and the `k` most similar chunks are retrieved by cosine distance.

Typical values for `k` range from 3 to 8. A low `k` is cheaper and more precise when
the answer is concentrated in a few fragments, but can fail on questions that require
synthesizing information spread across several sections of the document. A high `k`
improves recall but dilutes the relevance of the context and increases token cost on
every call to the generation model.

A common improvement is **hybrid retrieval**: combining embedding-based (semantic)
search with traditional lexical search (BM25), especially useful when questions
include exact terms like product codes, proper names, or item numbers that embeddings
do not always capture well.

## 6. Generation and Prompting

The generation model receives only the retrieved chunks as context, along with an
explicit instruction: answer only based on that context, and openly state that there
is not enough information if the context does not cover the question. This
instruction is the central piece for reducing hallucinations — without it, the model
tends to fill information gaps with general knowledge, which can be incorrect or out
of date with respect to the internal documents.

## 7. Citations

Every answer is accompanied by the list of sources used: source file, page or
section, and a short excerpt of the chunk that supports the answer. This lets the
user verify the answer against the original document, and it is what distinguishes a
well-designed RAG system from a generic chatbot.

## 8. Evaluation

It is recommended to maintain a reference set of questions with known expected
answers, to measure two things separately: whether retrieval brings back the correct
chunks (retriever recall metric) and whether the generated answer is faithful to the
retrieved context (faithfulness metric). A system can fail at either stage
independently, so evaluating them together hides the root cause of errors.

## 9. Known Risks

- Outdated documents that remain indexed and contradict the current version.
- Scanned PDFs without OCR that end up empty with no visible error.
- Ambiguous questions that match chunks from related but incorrect topics, producing
  plausible but wrong answers.
- Full re-indexing costs when the embedding model changes.
