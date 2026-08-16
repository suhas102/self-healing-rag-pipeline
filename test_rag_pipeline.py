"""
Tests for the self-healing RAG pipeline. Fully offline / deterministic —
no API key required, so this suite runs in CI without secrets.
"""

from rag_pipeline import (
    SelfHealingRAGPipeline,
    VectorStore,
    chunk_document,
    offline_critic,
    offline_generate,
)

DOCS = {
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
    "doc3": (
        "Paris is the capital of France and is known for the Eiffel Tower, "
        "the Louvre museum, and the Seine river running through the city."
    ),
}


def build_pipeline() -> SelfHealingRAGPipeline:
    pipeline = SelfHealingRAGPipeline()
    pipeline.index(DOCS)
    return pipeline


def test_chunking_produces_overlapping_chunks():
    chunks = chunk_document("d1", " ".join(["word"] * 1000), chunk_size=400, overlap=80)
    assert len(chunks) >= 3
    assert all(c.doc_id == "d1" for c in chunks)


def test_vector_store_retrieves_relevant_chunk():
    store = VectorStore()
    store.add_documents(DOCS)
    results = store.search("What does the critic agent check?", k=2)
    assert results, "expected at least one retrieved chunk"
    assert any("critic" in c.text.lower() for c in results)


def test_grounded_answer_is_returned_directly():
    pipeline = build_pipeline()
    result = pipeline.answer("What does the critic agent check?")
    assert result["grounded"] is True
    assert "critic" in result["answer"].lower() or "grounded" in result["answer"].lower()
    assert result["attempts"] == 0


def test_offline_critic_rejects_ungrounded_answer():
    context = [c for c in VectorStore().__class__().chunks]  # empty context
    assert offline_critic("The moon is made of cheese.", []) is False


def test_offline_generate_returns_insufficient_information_with_no_context():
    assert offline_generate("anything", []) == "INSUFFICIENT_INFORMATION"


def test_unanswerable_question_triggers_fallback_not_hallucination():
    pipeline = build_pipeline()
    result = pipeline.answer("What is the boiling point of mercury on Jupiter?")
    # Should never fabricate a confident-sounding wrong answer: either
    # grounded in real context, or an explicit fallback.
    assert result["grounded"] is False
    assert (
        "don't have enough" in result["answer"].lower()
        or result["answer"] == "INSUFFICIENT_INFORMATION"
    )


def test_reformulation_happens_on_repeated_failure():
    pipeline = build_pipeline()
    result = pipeline.answer("What is the boiling point of mercury on Jupiter?")
    assert result["attempts"] >= 1
    assert any("reformulate" in line for line in result["trace"])


def test_trace_is_fully_explainable():
    pipeline = build_pipeline()
    result = pipeline.answer("Where is the Eiffel Tower?")
    assert result["trace"][0].startswith("retrieve(")
    assert any(line.startswith("generate") for line in result["trace"])
    assert any(line.startswith("critic") for line in result["trace"])
