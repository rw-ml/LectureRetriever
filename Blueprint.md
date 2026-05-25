# Project Overview: Lecture RAG System
This document describes the system design decisions, engineering trade-offs, 
and implementation challenges behind the Lecture RAG system. 
For usage instructions, see the README.
## Goal

The goal of this project was to build a local Retrieval-Augmented Generation (RAG) system for querying lecture slides.

Users upload lecture PDFs, which are transformed into semantically searchable chunks. Questions are answered using a local language model, with responses grounded strictly in retrieved source material and accompanied by explicit source references.

The main focus was to create a system that is:

- **Local** (no external APIs required)
- **Fast enough for interactive use**
- **Traceable and verifiable**
- **Simple to deploy and reproduce**

---

## System Blueprint

The system is split into two independent pipelines sharing a central SQLite database.

### 1. Ingestion Pipeline

Lecture PDFs are processed as follows:

- PDF text extraction using `pdfplumber` accompanied by artifact cleaning
- Dynamic document analysis (auto-detecting slide layouts vs. text-heavy documents)
- Slide-aware chunking with heuristic title propagation (for multi-page continuations)
- Embedding generation using `multilingual-e5-small`
- Storage of chunks and embeddings in a central SQLite database

---

### 2. Query Pipeline

When a user asks a question:

- The question is embedded using the same embedding model
- Relevant chunks are retrieved via cosine similarity search
- Retrieved chunks are reranked using a cross-encoder
- A prompt is constructed from the top-ranked context
- A local LLM (`Qwen3.5-2B`) generates the answer via `vLLM`
- The answer is streamed back together with explicit source references

---

## Key Technical Decisions

### SQLite instead of a Vector Database

Instead of using a dedicated vector database such as pgvector or Pinecone, I chose SQLite for simplicity and local portability.

Retrieval currently uses exact similarity search over stored embeddings.

**Trade-off:**
- Easier local setup
- No external infrastructure
- Limited scalability for large corpora

For the intended use case (small lecture collections), this was an acceptable compromise.

---

### Local LLM Inference with vLLM

A major goal was to avoid external model APIs.

Initial experiments with standard Hugging Face Transformers inference had high latency, which made interaction impractical.

Switching to `vLLM` significantly improved responsiveness:

- slower startup due to model initialization
- near-instant streamed responses after warm-up

This made local inference viable for interactive question answering.


---

## Interesting Challenges

### Challenge 1: Document Structure & Context Loss during Chunking
Lecture slides are inherently fragmented. A naive text-splitter breaks down bullet points into isolated pieces, and crucial slide titles are lost for subsequent sub-points on continuing pages. 

- **Solution:** I implemented a custom `SlidesChunker` 
that uses a sliding window across 3 pages (with a 1-page overlap) 
combined with *heuristic title propagation*. 
The algorithm detects slide titles and automatically injects them 
into continuation chunks (e.g., slides marked as "continued", "step", or "phase"). 

- **The Fallback Trade-off:** For text-heavy documents, 
the system falls back to a `RollingSemanticChunker` based on embedding shifts. 
However, since lecture materials generally maintain high semantic similarity throughout, precise boundary detection here is difficult. 
I deliberately treated this semantic chunker as a best-effort backup rather than over-engineering it.


### Challenge 2: VRAM Resource Crunch in a Single-GPU Setup
The entire pipeline runs locally on a consumer GPU with limited resources (~10GB VRAM). By default, vLLM allocates a large portion of the available VRAM for its KV-Cache (typically 90%). 
This can cause immediate Out-Of-Memory (OOM) exceptions because the embedding model and the cross-encoder reranker also required VRAM on the same device.

- **Solution:** I managed the hardware constraints declaratively inside the `docker-compose.yml`. By restricting `gpu_memory_utilization` to `0.7`, applying model quantization, and explicitly decoupling the prompt context window (`max_model_len = 16384`) from the generation budget (`max_tokens = 8192`), I established a stable memory equilibrium. All three pipeline steps now share the single GPU seamlessly without crashing.

---



### Limitation: Multilingual Inconsistencies

The embedding model supports multilingual retrieval, but the small local generation model still struggles with multilingual consistency.

For example:

- German questions over English lecture material may produce lower-quality answers
- responses may occasionally switch back to English

This is a clear limitation of generator capacity rather than the retrieval pipeline itself.

---

## Possible Future Improvements

- Dedicated frontend for uploads and live streaming
- Formal retrieval evaluation benchmark
- Larger/Improved multilingual generation model (within hardware limitations)
- ANN-based vector retrieval backend for scalability

---

## Main Takeaway

The project demonstrates an end-to-end retrieval-augmented generation system designed for constrained local execution.
It integrates document ingestion, embedding-based retrieval, reranking, and local LLM inference via vLLM within a single-GPU environment.
Key engineering challenges included memory-constrained GPU execution, context window management, and maintaining responsiveness under a shared-model setup.
While generation quality is bounded by the capacity of a 2B parameter model, 
the system provides a reproducible blueprint for local RAG pipelines and can be extended to larger models or dedicated vector databases.