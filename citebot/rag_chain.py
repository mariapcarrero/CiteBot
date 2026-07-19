from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from citebot.config import CHAT_MODEL, FETCH_K, HISTORY_TURNS, TOP_K

SYSTEM_PROMPT = """You are an assistant that answers questions using EXCLUSIVELY \
the context provided below, extracted from the user's documents. Use any facts, \
definitions, or acronym expansions stated in the context to answer directly. \
Do not invent information that is not in the context. \
If, after reviewing the context, it truly does not contain information relevant to \
the question, respond exactly: "I don't know based on the documents."

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

CONDENSE_SYSTEM_PROMPT = """Given the conversation history and a follow-up question, \
rewrite the follow-up question as a standalone question that includes all context \
needed to understand it without the history (e.g. resolve pronouns like "it" or \
"that" to what they refer to). If the follow-up question is already standalone, \
return it unchanged. Output ONLY the rewritten question, nothing else.

Conversation history:
{history}"""

condense_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _format_history(history):
    turns = history[-HISTORY_TURNS * 2 :]
    lines = [f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}" for turn in turns]
    return "\n".join(lines)


def condense_question(question, history, llm):
    if not history:
        return question
    chain = condense_prompt | llm | StrOutputParser()
    return chain.invoke({"history": _format_history(history), "question": question}).strip()


RERANK_PROMPT = ChatPromptTemplate.from_template(
    """Rank the numbered document chunks below by relevance to the question. \
Return ONLY a comma-separated list of chunk numbers, ordered from most to least \
relevant, keeping only the top {top_n}. Example output: 3,0,4

Question: {question}

Chunks:
{chunks}"""
)


def _load_all_documents(vectorstore):
    """Reconstruct chunks (with metadata) already persisted in the vectorstore,
    so BM25 can index the same content without needing the original files."""
    stored = vectorstore.get(include=["documents", "metadatas"])
    return [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(stored["documents"], stored["metadatas"])
    ]


def get_retriever(vectorstore):
    """Hybrid retriever: combines semantic (embedding) search with lexical (BM25) \
    search, since embeddings alone can miss exact terms like acronyms, numbers, or \
    proper names."""
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": FETCH_K})

    documents = _load_all_documents(vectorstore)
    if not documents:
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = FETCH_K
    return EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])


def rerank(question, docs, llm, top_n=TOP_K):
    """Listwise LLM re-ranking: ask the model to pick and order the top_n most \
    relevant chunks out of the hybrid retriever's candidates."""
    if len(docs) <= top_n:
        return docs

    numbered = "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs))
    chain = RERANK_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": question, "chunks": numbered, "top_n": top_n})

    seen = set()
    order = []
    for token in raw.split(","):
        token = token.strip()
        if token.isdigit() and 0 <= int(token) < len(docs) and int(token) not in seen:
            order.append(int(token))
            seen.add(int(token))
    for idx in range(len(docs)):
        if len(order) >= top_n:
            break
        if idx not in seen:
            order.append(idx)
            seen.add(idx)

    return [docs[idx] for idx in order[:top_n]]


def _format_context(docs):
    blocks = []
    for doc in docs:
        label = doc.metadata.get("source", "unknown")
        if "page" in doc.metadata:
            label += f", p. {doc.metadata['page'] + 1}"
        blocks.append(f"[{label}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def _to_source(doc):
    excerpt = doc.page_content.strip().replace("\n", " ")
    if len(excerpt) > 200:
        excerpt = excerpt[:200] + "..."
    source = {"file": doc.metadata.get("source", "unknown"), "excerpt": excerpt}
    if "page" in doc.metadata:
        source["page"] = doc.metadata["page"] + 1
    return source


def answer_question_stream(question, retriever, history=None):
    """Resolve the standalone question, retrieve context, and stream the answer.

    Returns (token_iterator, standalone_question, sources). Sources and the
    standalone question are known before the LLM starts generating, since
    retrieval happens up front; only the answer text is streamed.
    """
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    standalone_question = condense_question(question, history or [], llm)

    candidates = retriever.invoke(standalone_question)
    docs = rerank(standalone_question, candidates, llm, top_n=TOP_K)
    context = _format_context(docs)
    sources = [_to_source(doc) for doc in docs]

    chain = prompt | llm | StrOutputParser()
    token_iterator = chain.stream({"context": context, "question": standalone_question})

    return token_iterator, standalone_question, sources
