from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from citebot.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL

LOADERS_BY_SUFFIX = {
    ".pdf": PyPDFLoader,
    ".md": TextLoader,
    ".txt": TextLoader,
}


def load_documents(file_paths):
    documents = []
    for path in file_paths:
        path = Path(path)
        loader_cls = LOADERS_BY_SUFFIX.get(path.suffix.lower())
        if loader_cls is None:
            raise ValueError(f"Tipo de archivo no soportado: {path.suffix}")
        loader = loader_cls(str(path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = path.name
        documents.extend(docs)
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks, persist_dir):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )


def load_vectorstore(persist_dir):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
