import shutil
import tempfile
import uuid
from pathlib import Path

import streamlit as st

from citebot.config import SAMPLE_DOCS_DIR, VECTORSTORE_DIR
from citebot.feedback import record_feedback
from citebot.ingest import build_vectorstore, chunk_documents, load_documents, load_vectorstore
from citebot.rag_chain import answer_question_stream, estimate_cost_usd, get_retriever

st.set_page_config(page_title="CiteBot", page_icon="📎")
st.title("📎 CiteBot")
st.caption("RAG chatbot that answers only from your documents and cites its sources.")


def index_files(file_paths, persist_dir):
    docs = load_documents(file_paths)
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks, persist_dir)
    n_flagged = sum(1 for c in chunks if c.metadata.get("injection_flagged"))
    return vectorstore, len(docs), len(chunks), n_flagged


def render_sources(sources):
    with st.expander("Sources"):
        for source in sources:
            label = source["file"]
            if "page" in source:
                label += f", p. {source['page']}"
            if source.get("flagged"):
                label += " ⚠️ flagged: possible prompt injection"
            st.markdown(f"**{label}**\n\n> {source['excerpt']}")


def render_usage(usage):
    st.caption(
        f"⏱ {usage.get('latency_seconds', 0):.1f}s · "
        f"🔢 {usage.get('total_tokens', 0)} tokens · "
        f"💵 ${usage.get('cost_usd', 0):.5f}"
    )


def render_feedback(message):
    if message.get("rating"):
        emoji = "👍" if message["rating"] == "up" else "👎"
        st.caption(f"Feedback recorded: {emoji}")
        return

    col1, col2, _ = st.columns([1, 1, 10])
    if col1.button("👍", key=f"up_{message['id']}"):
        message["rating"] = "up"
        record_feedback(
            message["question"],
            message["standalone_question"],
            message["content"],
            message["sources"],
            message["usage"],
            "up",
        )
        st.rerun()
    if col2.button("👎", key=f"down_{message['id']}"):
        message["rating"] = "down"
        record_feedback(
            message["question"],
            message["standalone_question"],
            message["content"],
            message["sources"],
            message["usage"],
            "down",
        )
        st.rerun()


if "vectorstore" not in st.session_state:
    if VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir()):
        st.session_state.vectorstore = load_vectorstore(VECTORSTORE_DIR)
    else:
        sample_files = list(SAMPLE_DOCS_DIR.glob("*"))
        if sample_files:
            vectorstore, n_docs, n_chunks, n_flagged = index_files(sample_files, VECTORSTORE_DIR)
            st.session_state.vectorstore = vectorstore
        else:
            st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, Markdown or TXT",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Index documents"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_paths = []
            for uploaded_file in uploaded_files:
                dest = Path(tmp_dir) / uploaded_file.name
                dest.write_bytes(uploaded_file.getbuffer())
                saved_paths.append(dest)

            with st.spinner("Indexing..."):
                # Each upload gets its own fresh directory: ChromaDB caches
                # SQLite connections per path, so reusing the same path after
                # deleting it causes "readonly database" errors.
                previous_dir = st.session_state.get("upload_vectorstore_dir")
                new_dir = Path(tempfile.mkdtemp(prefix="citebot-upload-", dir=str(VECTORSTORE_DIR.parent)))
                vectorstore, n_docs, n_chunks, n_flagged = index_files(saved_paths, new_dir)
                st.session_state.vectorstore = vectorstore
                st.session_state.upload_vectorstore_dir = new_dir
                st.session_state.messages = []
                if previous_dir is not None:
                    shutil.rmtree(previous_dir, ignore_errors=True)

        success_msg = f"Indexed: {len(uploaded_files)} file(s), {n_chunks} chunks"
        if n_flagged:
            success_msg += f" — ⚠️ {n_flagged} chunk(s) flagged for possible prompt injection"
        st.success(success_msg)

    if st.session_state.get("vectorstore") is not None:
        st.info("Documents are indexed and ready for questions.")
    else:
        st.warning("No documents indexed yet.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])
        if message.get("usage"):
            render_usage(message["usage"])
        if message["role"] == "assistant" and "id" in message:
            render_feedback(message)

question = st.chat_input("Ask something about your documents...")

if question:
    if st.session_state.get("vectorstore") is None:
        st.error("Upload at least one document before asking a question.")
    else:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        retriever = get_retriever(st.session_state.vectorstore)
        with st.chat_message("assistant"):
            with st.spinner("Searching the documents..."):
                token_iterator, standalone_question, sources, usage = answer_question_stream(
                    question, retriever, history=history
                )
            answer = st.write_stream(token_iterator)
            usage["cost_usd"] = estimate_cost_usd(usage)
            if standalone_question != question:
                st.caption(f"Interpreted as: {standalone_question}")
            if sources:
                render_sources(sources)
            render_usage(usage)

            new_message = {
                "id": uuid.uuid4().hex[:8],
                "role": "assistant",
                "content": answer,
                "question": question,
                "standalone_question": standalone_question,
                "sources": sources,
                "usage": usage,
            }
            st.session_state.messages.append(new_message)
            render_feedback(new_message)
