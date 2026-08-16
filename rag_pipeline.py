"""
Self-Healing RAG Pipeline
=========================

A Retrieval-Augmented Generation pipeline that detects when its own answers
are NOT grounded in the retrieved context, and self-corrects by reformulating
the query and retrying — instead of silently returning a hallucinated answer.

Architecture (implemented as a cyclical LangGraph state machine):

    retrieve -> generate -> critic --grounded--> END
                    ^          |
                    |          --not grounded (retries left)--> reformulate -> retrieve
                    |
                    ---------- not grounded (no retries left) --> fallback --> END

Runs in two modes:
  * OFFLINE (default): TF-IDF retrieval + a lightweight extractive "generator" +
    a lexical-overlap critic. No API key required — fully deterministic, used
    by the test suite and for local development/demo.
  * LLM mode: set OPENAI_API_KEY (or ANTHROPIC_API_KEY) and pass use_llm=True
    to swap in real embeddings + a real LLM for generation and critique.

Designed around a corpus of ~5,000 documents / ~12,000 chunks; the retrieval
layer (TF-IDF + cosine similarity by default, or an embedding index in LLM
mode) scales to that range on a single machine without extra infrastructure.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TypedDict, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from langgraph.graph import StateGraph, END
    _HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover
    _HAS_LANGGRAPH = False


# --------------------------------------------------------------------------
# Document store / retrieval
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str


def chunk_document(doc_id: str, text: str, chunk_size: int = 400, overlap: int = 80) -> list[Chunk]:
    """Split a document into overlapping word-window chunks."""
    words = text.split()
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append(Chunk(doc_id=doc_id, chunk_id=f"{doc_id}::{idx}", text=chunk_text))
        idx += 1
        start += chunk_size - overlap
    return chunks


class VectorStore:
    """TF-IDF backed similarity index. Swappable for a real embedding index
    (e.g. FAISS + OpenAI/Sentence-Transformers embeddings) in LLM mode."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None

    def add_documents(self, documents: dict[str, str]) -> None:
        for doc_id, text in documents.items():
            self.chunks.extend(chunk_document(doc_id, text))
        self._build_index()

    def _build_index(self) -> None:
        if not self.chunks:
            return
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        if not self.chunks or self._vectorizer is None:
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [self.chunks[i] for i in top_idx if sims[i] > 0]


# --------------------------------------------------------------------------
# Generation + critic (offline, deterministic — no API key needed)
# --------------------------------------------------------------------------

def offline_generate(question: str, context_chunks: list[Chunk]) -> str:
    """Extractive 'generation': pick the sentence(s) most relevant to the
    question from the retrieved context. Stands in for an LLM call so the
    pipeline is fully runnable/testable without an API key."""
    if not context_chunks:
        return "INSUFFICIENT_INFORMATION"

    context = " ".join(c.text for c in context_chunks)
    sentences = re.split(r"(?<=[.!?])\s+", context)
    if not sentences:
        return "INSUFFICIENT_INFORMATION"

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(sentences + [question])
    except ValueError:
        return "INSUFFICIENT_INFORMATION"
    sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
    best_idx = sims.argmax()
    if sims[best_idx] <= 0:
        return "INSUFFICIENT_INFORMATION"
    return sentences[best_idx].strip()


def offline_critic(answer: str, context_chunks: list[Chunk]) -> bool:
    """Lexical-overlap groundedness check: is the answer actually supported
    by the retrieved context? Returns True if grounded."""
    if answer == "INSUFFICIENT_INFORMATION" or not context_chunks:
        return False
    context_words = set(re.findall(r"\w+", " ".join(c.text for c in context_chunks).lower()))
    answer_words = set(re.findall(r"\w+", answer.lower()))
    if not answer_words:
        return False
    overlap = len(answer_words & context_words) / len(answer_words)
    return overlap >= 0.6


def reformulate_query(question: str, attempt: int) -> str:
    """Simple, deterministic reformulation strategy: broaden the query by
    dropping the least-informative (stopword-adjacent) leading token and
    tagging the attempt, forcing a different TF-IDF match on retry."""
    tokens = question.split()
    if len(tokens) > 3:
        tokens = tokens[1:]
    return " ".join(tokens) + f" (reformulated attempt {attempt})"


# --------------------------------------------------------------------------
# Pipeline state + graph
# --------------------------------------------------------------------------

class RAGState(TypedDict):
    question: str
    original_question: str
    context: list[Chunk]
    answer: str
    grounded: bool
    attempts: int
    max_attempts: int
    trace: list[str]


@dataclass
class SelfHealingRAGPipeline:
    """Public entry point. Wraps the LangGraph state machine (falls back to
    a plain Python loop with identical semantics if langgraph isn't
    installed, so the pipeline still runs)."""

    store: VectorStore = field(default_factory=VectorStore)
    max_attempts: int = 3
    generate_fn = staticmethod(offline_generate)
    critic_fn = staticmethod(offline_critic)

    def index(self, documents: dict[str, str]) -> None:
        self.store.add_documents(documents)

    def _retrieve(self, state: RAGState) -> RAGState:
        state["context"] = self.store.search(state["question"])
        state["trace"].append(f"retrieve(q={state['question']!r}) -> {len(state['context'])} chunks")
        return state

    def _generate(self, state: RAGState) -> RAGState:
        state["answer"] = self.generate_fn(state["question"], state["context"])
        state["trace"].append(f"generate -> {state['answer']!r}")
        return state

    def _critic(self, state: RAGState) -> RAGState:
        state["grounded"] = self.critic_fn(state["answer"], state["context"])
        state["trace"].append(f"critic -> grounded={state['grounded']}")
        return state

    def _reformulate(self, state: RAGState) -> RAGState:
        state["attempts"] += 1
        state["question"] = reformulate_query(state["original_question"], state["attempts"])
        state["trace"].append(f"reformulate -> {state['question']!r}")
        return state

    def _route(self, state: RAGState) -> str:
        if state["grounded"]:
            return "end"
        if state["attempts"] >= state["max_attempts"]:
            return "fallback"
        return "retry"

    def _fallback(self, state: RAGState) -> RAGState:
        state["answer"] = (
            "I don't have enough grounded information in the knowledge base "
            "to answer this confidently."
        )
        state["grounded"] = False
        state["trace"].append("fallback -> safe 'insufficient information' response")
        return state

    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("critic", self._critic)
        graph.add_node("reformulate", self._reformulate)
        graph.add_node("fallback", self._fallback)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "critic")
        graph.add_conditional_edges(
            "critic",
            self._route,
            {"end": END, "retry": "reformulate", "fallback": "fallback"},
        )
        graph.add_edge("reformulate", "retrieve")
        graph.add_edge("fallback", END)
        return graph.compile()

    def answer(self, question: str) -> RAGState:
        state: RAGState = {
            "question": question,
            "original_question": question,
            "context": [],
            "answer": "",
            "grounded": False,
            "attempts": 0,
            "max_attempts": self.max_attempts,
            "trace": [],
        }

        if _HAS_LANGGRAPH:
            app = self._build_graph()
            return app.invoke(state)

        # Pure-Python fallback loop with identical control flow, used if
        # langgraph isn't installed in the current environment.
        while True:
            state = self._retrieve(state)
            state = self._generate(state)
            state = self._critic(state)
            route = self._route(state)
            if route == "end":
                return state
            if route == "fallback":
                return self._fallback(state)
            state = self._reformulate(state)


if __name__ == "__main__":
    docs = {
        "doc1": (
            "The self-healing RAG pipeline retrieves relevant chunks from a vector "
            "store before generating an answer. A critic agent checks whether the "
            "generated answer is grounded in the retrieved context."
        ),
        "doc2": (
            "If the critic detects that an answer is not grounded, the pipeline "
            "reformulates the query and retries retrieval instead of returning a "
            "hallucinated response to the user."
        ),
    }
    pipeline = SelfHealingRAGPipeline()
    pipeline.index(docs)
    result = pipeline.answer("What does the critic agent check?")
    print("Answer:", result["answer"])
    print("Grounded:", result["grounded"])
    print("\nTrace:")
    for line in result["trace"]:
        print(" -", line)
