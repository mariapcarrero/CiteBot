import shutil
import tempfile
from pathlib import Path

import streamlit as st

from citebot.config import SAMPLE_DOCS_DIR, VECTORSTORE_DIR
from citebot.ingest import build_vectorstore, chunk_documents, load_documents, load_vectorstore
from citebot.rag_chain import answer_question_stream, get_retriever

st.set_page_config(page_title="CiteBot", page_icon="📎")
st.title("📎 CiteBot")
st.caption("RAG chatbot that answers only from your documents and cites its sources.")


def index_files(file_paths, persist_dir):
    docs = load_documents(file_paths)
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks, persist_dir)
    return vectorstore, len(docs), len(chunks)


if "vectorstore" not in st.session_state:
    if VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir()):
        st.session_state.vectorstore = load_vectorstore(VECTORSTORE_DIR)
    else:
        sample_files = list(SAMPLE_DOCS_DIR.glob("*"))
        if sample_files:
            vectorstore, n_docs, n_chunks = index_files(sample_files, VECTORSTORE_DIR)
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
                vectorstore, n_docs, n_chunks = index_files(saved_paths, new_dir)
                st.session_state.vectorstore = vectorstore
                st.session_state.upload_vectorstore_dir = new_dir
                st.session_state.messages = []
                if previous_dir is not None:
                    shutil.rmtree(previous_dir, ignore_errors=True)

        st.success(f"Indexed: {len(uploaded_files)} file(s), {n_chunks} chunks")

    if st.session_state.get("vectorstore") is not None:
        st.info("Documents are indexed and ready for questions.")
    else:
        st.warning("No documents indexed yet.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    label = source["file"]
                    if "page" in source:
                        label += f", p. {source['page']}"
                    st.markdown(f"**{label}**\n\n> {source['excerpt']}")

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
                token_iterator, standalone_question, sources = answer_question_stream(
                    question, retriever, history=history
                )
            answer = st.write_stream(token_iterator)
            if standalone_question != question:
                st.caption(f"Interpreted as: {standalone_question}")
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        label = source["file"]
                        if "page" in source:
                            label += f", p. {source['page']}"
                        st.markdown(f"**{label}**\n\n> {source['excerpt']}")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
