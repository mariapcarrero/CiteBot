"""RAG evaluation harness.

Runs a fixed question set (data/eval/qa_dataset.json) against a freshly built
vectorstore over the sample documents, and measures two things separately, as
recommended in data/sample_docs/rag-design-doc.md:

  - retrieval hit rate: did the retriever bring back the chunk(s) from the
    expected source file(s)?
  - answer correctness: does the generated answer convey the same facts as the
    reference answer (graded by an LLM judge), and does the model correctly
    refuse ("I don't know based on the documents") on unanswerable questions?

Usage:
    python -m citebot.evaluate
"""

import json
import tempfile
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from citebot.config import (
    CHAT_MODEL,
    EVAL_DATASET_PATH,
    EVAL_MIN_ANSWER_CORRECTNESS_RATE,
    EVAL_MIN_RETRIEVAL_HIT_RATE,
    EVAL_RESULTS_PATH,
    SAMPLE_DOCS_DIR,
    TOP_K,
)
from citebot.ingest import build_vectorstore, chunk_documents, load_documents
from citebot.rag_chain import _format_context, get_retriever, prompt, rerank

REFUSAL_PHRASE = "i don't know based on the documents"

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """You are grading a RAG system's answer against a reference answer. \
Reply with exactly one word: CORRECT or INCORRECT.

Question: {question}
Reference answer: {reference}
Model answer: {answer}

Does the model answer convey the same key facts as the reference answer, \
ignoring differences in wording? Reply CORRECT or INCORRECT only."""
)


def _judge(question, reference, answer, llm):
    chain = JUDGE_PROMPT | llm | StrOutputParser()
    verdict = chain.invoke({"question": question, "reference": reference, "answer": answer})
    return verdict.strip().upper().startswith("CORRECT")


def _is_refusal(answer):
    return REFUSAL_PHRASE in answer.lower()


def run_eval():
    with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    docs = load_documents(list(SAMPLE_DOCS_DIR.glob("*")))
    chunks = chunk_documents(docs)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    answer_chain = prompt | llm | StrOutputParser()

    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        vectorstore = build_vectorstore(chunks, Path(tmp_dir))
        retriever = get_retriever(vectorstore)

        for item in dataset:
            candidates = retriever.invoke(item["question"])
            retrieved = rerank(item["question"], candidates, llm, top_n=TOP_K)
            retrieved_sources = {doc.metadata.get("source") for doc in retrieved}
            context = _format_context(retrieved)
            answer = answer_chain.invoke({"context": context, "question": item["question"]})
            refused = _is_refusal(answer)

            forbidden = item.get("forbidden_substrings", [])
            violated = any(substr.lower() in answer.lower() for substr in forbidden)

            if item["should_refuse"]:
                retrieval_hit = None
                correct = refused and not violated
            else:
                expected = set(item["expected_sources"])
                retrieval_hit = expected.issubset(retrieved_sources)
                correct = (not refused) and (not violated) and _judge(
                    item["question"], item["reference_answer"], answer, llm
                )

            results.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": answer,
                    "retrieved_sources": sorted(s for s in retrieved_sources if s),
                    "retrieval_hit": retrieval_hit,
                    "correct": correct,
                    "should_refuse": item["should_refuse"],
                    "refused": refused,
                    "forbidden_violated": violated,
                }
            )

    retrieval_scores = [r["retrieval_hit"] for r in results if r["retrieval_hit"] is not None]
    correctness_scores = [r["correct"] for r in results]

    summary = {
        "total_questions": len(results),
        "retrieval_hit_rate": sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else None,
        "answer_correctness_rate": sum(correctness_scores) / len(correctness_scores),
        "results": results,
    }

    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _print_report(results, summary)
    return summary


def _print_report(results, summary):
    print(f"{'ID':<4}{'Retrieval':<12}{'Correct':<10}Question")
    print("-" * 70)
    for r in results:
        retrieval_col = "-" if r["retrieval_hit"] is None else ("HIT" if r["retrieval_hit"] else "MISS")
        correct_col = "YES" if r["correct"] else "NO"
        print(f"{r['id']:<4}{retrieval_col:<12}{correct_col:<10}{r['question'][:50]}")
    print("-" * 70)
    if summary["retrieval_hit_rate"] is not None:
        print(f"Retrieval hit rate:      {summary['retrieval_hit_rate']:.0%}")
    print(f"Answer correctness rate: {summary['answer_correctness_rate']:.0%}")
    print(f"Results saved to: {EVAL_RESULTS_PATH}")


def _check_thresholds(summary):
    failures = []
    if (
        summary["retrieval_hit_rate"] is not None
        and summary["retrieval_hit_rate"] < EVAL_MIN_RETRIEVAL_HIT_RATE
    ):
        failures.append(
            f"retrieval hit rate {summary['retrieval_hit_rate']:.0%} "
            f"< required {EVAL_MIN_RETRIEVAL_HIT_RATE:.0%}"
        )
    if summary["answer_correctness_rate"] < EVAL_MIN_ANSWER_CORRECTNESS_RATE:
        failures.append(
            f"answer correctness {summary['answer_correctness_rate']:.0%} "
            f"< required {EVAL_MIN_ANSWER_CORRECTNESS_RATE:.0%}"
        )
    return failures


if __name__ == "__main__":
    import sys

    eval_summary = run_eval()
    threshold_failures = _check_thresholds(eval_summary)
    if threshold_failures:
        print("\nEVAL FAILED: " + "; ".join(threshold_failures))
        sys.exit(1)
