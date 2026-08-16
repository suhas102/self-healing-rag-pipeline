# Self-Healing RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that doesn't just retrieve-and-generate — it **checks its own work**. A critic agent evaluates whether each answer is actually grounded in the retrieved context. If it isn't, the pipeline reformulates the query and retries automatically instead of silently returning a hallucinated answer, falling back to an explicit "insufficient information" response only after retries are exhausted.

## Why this exists

Standard RAG pipelines have no mechanism to notice when they've produced an answer the retrieved context doesn't actually support. This project treats that as a first-class failure mode to detect and recover from, not something to hope doesn't happen.

## Architecture

Modelled as a **stateful, cyclical graph** in [LangGraph](https://github.com/langchain-ai/langgraph) rather than a linear chain, so the pipeline can loop back and self-correct:

```
 retrieve → generate → critic ──grounded──────────────► END
               ▲           │
               │           └─ not grounded, retries left ─► reformulate ─► retrieve
               │
               └─ not grounded, retries exhausted ─► fallback ─► END
```

- **retrieve** — pulls the top-k most relevant chunks for the current query from the vector store.
- **generate** — produces an answer conditioned on the retrieved chunks.
- **critic** — checks whether the answer is actually grounded in the retrieved context.
- **reformulate** — rewrites the query and loops back to retrieval if the critic rejects the answer.
- **fallback** — returns a safe "insufficient information" response rather than a low-confidence guess, once the retry budget is spent.

Designed against a corpus in the **~5,000 document / ~12,000 chunk** range — the default TF-IDF retrieval layer indexes and searches that scale comfortably on a single machine with no external vector database required.

## Two run modes

| Mode | Retrieval | Generation | Critic | Requires API key? |
|---|---|---|---|---|
| **Offline (default)** | TF-IDF + cosine similarity | Extractive sentence selection | Lexical-overlap grounding check | No |
| **LLM mode** | Embedding index (pluggable — OpenAI / Sentence-Transformers) | Real LLM call | LLM-as-judge grounding check | Yes |

The offline mode exists so the *architecture* — the retry loop, the critic, the fallback — is fully testable and demoable without needing to hand out an API key. Swapping in real embeddings and an LLM call for the `generate_fn` / `critic_fn` hooks turns on full LLM mode.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python rag_pipeline.py
```

```python
from rag_pipeline import SelfHealingRAGPipeline

pipeline = SelfHealingRAGPipeline(max_attempts=3)
pipeline.index({"doc1": "...", "doc2": "..."})

result = pipeline.answer("What does the critic agent check?")
print(result["answer"])
print(result["grounded"])
print(result["trace"])   # full step-by-step trace of retrieval/generation/critic decisions
```

To enable LLM mode, set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) and swap `generate_fn` / `critic_fn` on `SelfHealingRAGPipeline` for LLM-backed implementations — the graph, retry logic, and fallback behaviour are unchanged.

## Tests

```bash
pytest -v
```

8 tests cover chunking, retrieval relevance, grounded-answer pass-through, ungrounded-answer rejection, the reformulate-and-retry loop, and the fallback path — all deterministic, no network calls.

## Project status

Personal project, built to explore self-correcting RAG architectures beyond simple retrieve-and-generate. Offline mode is feature-complete; LLM-mode adapters are a drop-in extension point (`generate_fn` / `critic_fn`).
