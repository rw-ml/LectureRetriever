#  Lecture RAG System (Streaming, Source-Grounded QA)

A local RAG system for querying lecture slides and retrieving verifiable answers with explicit source references.

Built for study workflows: users upload lecture PDFs and receive fast, source-grounded responses using semantic search and LLM inference with `vLLM`.

> Focus: Fast, local, and verifiable knowledge retrieval for exam preparation and study workflows.

---

##  Motivation

When preparing for exams, students often need to:

- Quickly find relevant information across many slides
- Understand concepts in context
- Verify answers against original material

This project addresses that by combining:
- **semantic search over lecture slides**
- **LLM-based summarization**
- **explicit source grounding**

The goal is not just answering questions, but enabling **traceable learning**.

---

## Key Features

- **PDF Slide Ingestion**
  - Extracts structured text (source, page, content) from lecture PDFs

- **Semantic Retrieval (RAG)**
  - Retrieves relevant slide chunks based on query embeddings

- **Source-Grounded Answers**
  - Answers are derived strictly from retrieved lecture content

- **Streaming Responses (vLLM)**
  - Near-instant responses after initial warm-up

- **Lightweight & Local**
  - Runs fully locally (no external APIs required)

---
## Engineering Highlights

- End-to-end RAG pipeline (ingestion → retrieval → generation)
- Local LLM serving with vLLM
- Semantic retrieval + reranking pipeline
- Streaming API with FastAPI
- Containerized deployment with Docker

---

## Why vLLM: Performance

Comparison to standard Transformers inference with a RTX 3080:

| Metric              | Transformers | vLLM |
|--------------------|--------------|------|
| First response     | \>30s        | ~20s (initial warmup) |
| Subsequent queries | \>30s        | near-instant streaming |



- **Drawback:** vLLM introduces a higher startup cost and time due to model compilation
- After warmup, response latency is significantly reduced
- Streaming enables immediate feedback to the user

This trade-off makes vLLM well-suited for interactive applications.


---



## Architecture Overview

### Lecture Upload Pipeline
pdf-Slides -> Text Extraction (pdfplumber) -> Cleaning & Structuring -> Slide-aware Chunking -> Embeddings (E5-small) -> SQLite Storage
### Information Retrieval
User Query -> Query Embedding -> Top-K Retrieval (K=5) -> Reranking (MiniLM) -> Generation: vLLM (Qwen3.5-2B) -> Streaming Response 

---

##  Data Processing Pipeline

###  Input

- PDF lecture slides (primary use case)
- PDF text documents (supported, not extensively tested)

---

### Text Cleaning

Per-slide preprocessing includes:

- Removal of:
  - multiple spaces / newlines
  - isolated numeric artifacts (from PDF parsing)
- Heuristic title propagation:
  - Handles slide continuations (e.g. *“continued”, “step”, “phase”*)

---

## Model & Design Decisions

### LLM Selection

The system uses `Qwen/Qwen3.5-2B`, chosen for:

- Strong quality-to-size ratio
- Compatibility with limited GPU resources (10GB VRAM)
- Comparable performance to larger models (4B) in initial tests

Smaller models (e.g. 0.8B) showed degraded generation quality.

---

### Embedding Model

- `intfloat/multilingual-e5-small`
- Chosen for:
  - Multilingual capability
  - Low computational overhead
  - Good performance for semantic retrieval

---

### Reranking

- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Improves retrieval precision with minimal latency increase

---

### Chunking Strategy

Slide-aware chunking was implemented to:

- Preserve semantic context across slides
- Avoid fragmentation of related content
- Maintain retrieval relevance

## Getting Started

### Requirements

- Docker
- NVIDIA GPU, `cuda>=12.9.0` (recommended)
- Python 3.10+
- vLLM

---

### Run the backend

```bash
docker build -t lecture-rag-api .
docker run --gpus all -p 8000:8000 lecture-rag-api
```


## Usage Example
- Limitation: Currently no Front-End

### Very simple web Interface 
- for example for Lecture Uploads

Open `http://localhost:8000/docs` in browser, select corresponding function. Downside: responses are not streamed.

### Request
```bash
curl -N -X POST http://localhost:8000/ask_stream \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What are SOLID principles?\",\"lecture_name\":\"SWT2\"}"
```

## Limitations

- Frontend currently not implemented
- PDF parsing quality depends on slide structure
- SQLite may not scale to very large datasets
- Small, not as powerful models for reranking and response generation 