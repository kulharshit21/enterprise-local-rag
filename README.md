# 🚀 Enterprise RAG System (Local & Private)

> **A FAANG-tier, fully local Retrieval-Augmented Generation (RAG) system built for privacy, performance, and scalability.**

## 📖 Overview

This project implements a production-grade RAG pipeline that runs **100% locally** without any external API dependencies (no OpenAI, no Anthropic). It leverages state-of-the-art open-source models and vector search technologies to provide high-accuracy answers from your documents.

**Key Design Principles:**
*   **Privacy First**: No data leaves your machine.
*   **Zero Cost**: No recurring API fees.
*   **High Performance**: GPU-accelerated indexing, retrieval, and generation.
*   **Scalability**: Adaptive vector store (FAISS) that scales from 1k to 10M+ vectors.
*   **Robustness**: Comprehensive error handling, retries, and observability.

---

## ✨ Key Features

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM** | **Meta LLaMA 3 8B Instruct** | Quantized (Q4_K_M) via `llama-cpp-python` for fast inference on consumer text-generation-webui hardware. |
| **Embeddings** | **BGE-large-en-v1.5** | High-performance dense vector embeddings (1024d). |
| **Vector Store** | **FAISS (GPU/CPU)** | Industry-standard similarity search with adaptive indexing (Flat/IVF). |
| **Hybrid Search** | **Dense + Sparse (BM25)** | Combines semantic search with keyword matching for precision. |
| **Reranking** | **BGE-Reranker-Large** | Cross-encoder reranking to boost top-k retrieval accuracy. |
| **API** | **FastAPI** | Async, high-performance REST API with autodocs. |
| **Security** | **JWT + RBAC** | Role-Based Access Control (Admin, Researcher, Viewer) with secure auth. |
| **Observability** | **Prometheus + Structlog** | Structured logging and metrics for monitoring latency and health. |

---

## 🏗️ Architecture

```mermaid
graph TD
    User[User / Client] -->|Query| API[FastAPI Gateway]
    API -->|Auth Check| Auth[JWT Authentication]
    API -->|REST| Pipeline[RAG Pipeline]
    
    subgraph "Ingestion Flow"
        Doc[Document] -->|Parses| Chunker[Semantic Chunker]
        Chunker -->|Embeds| Embedder[BGE Embedder]
        Embedder -->|Indexes| VectorStore[FAISS Vector DB]
        Chunker -->|Keywords| Sparse[BM25 Index]
    end
    
    subgraph "Retrieval Flow"
        Pipeline -->|Query| Embedder
        Embedder -->|Dense Search| VectorStore
        Pipeline -->|Keywords| Sparse
        VectorStore & Sparse -->|Merge| RRF[Reciprocal Rank Fusion]
        RRF -->|Candidates| Reranker[BGE Cross-Encoder]
        Reranker -->|Top-K| Context[Context Window]
    end
    
    subgraph "Generation Flow"
        Context -->|Prompt| LLM[LLaMA 3 8B (Local)]
        LLM -->|Stream| API
        LLM -->|Verify| Hallucination[Faithfulness Check]
    end
```

---

## 🛠️ Prerequisites

*   **OS**: Windows, Linux, or macOS
*   **Python**: 3.10+
*   **RAM**: 16 GB minimum (32 GB recommended)
*   **GPU**: NVIDIA GPU with >= 8 GB VRAM (Recommended for speed, but runs on CPU)
*   **Disk**: ~10 GB free space for models

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd RAG
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies
**Note:** For GPU support, ensure you have CUDA installed.
```bash
# Install PyTorch with CUDA support first (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 4. Setup Models
Create a `models/` directory and download the LLaMA 3 GGUF model:
```bash
mkdir models
# Download LLaMA 3 8B Instruct (Q4_K_M) - ~4.9 GB
huggingface-cli download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --local-dir models
```
*Note: The BGE embedding and reranker models will download automatically on first run.*

### 5. Configure Environment
Copy the example environment file:
```bash
cp .env.example .env
```
Edit `.env` to point to your model path and set a secure secret:
```ini
LLAMA_MODEL_PATH=./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
JWT_SECRET=your-secure-random-secret
```

---

## ⚡ Usage

### Start the API Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
*   **API Docs**: http://localhost:8000/docs
*   **Health Check**: http://localhost:8000/health

### Internal Test Chat
Run the included test script to verify the full pipeline:
```bash
python test_chat.py
```

### Ingest Documents
You can ingest PDF, TXT, or MD files via the API:
```bash
curl -X POST "http://localhost:8000/ingest" \
     -H "Authorization: Bearer <your_token>" \
     -F "file=@/path/to/document.pdf"
```

---

## 📚 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/auth/register` | Register a new user |
| `POST` | `/auth/token` | Login and get JWT token |
| `POST` | `/ingest` | Upload and index a document |
| `POST` | `/query` | Ask a question (RAG) |
| `GET` | `/health` | System health status |
| `GET` | `/metrics` | Prometheus metrics |

---

## 📂 Project Structure

```
RAG/
├── api/                 # FastAPI routes and dependencies
├── config.py            # Configuration management
├── data/                # Data storage (documents, DBs)
├── embeddings/          # Dense (BGE) and Sparse (BM25) embedders
├── evaluation/          # Metrics and evaluation logic
├── generation/          # LLM client (llama-cpp) and hallucination detection
├── ingestion/           # Document loading, chunking, and metadata
├── models/              # Local model files (GGUF)
├── observability/       # Logging and metrics
├── pipeline.py          # Core RAG orchestration
├── retrieval/           # FAISS vector store, reranker, and hybrid search
├── security/            # Auth, rate limiting, and RBAC
└── tests/               # Unit and integration tests
```

---

## 🤝 Contributing
1.  Fork the repo
2.  Create a feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m 'Add amazing feature'`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Open a Pull Request

---

## 📄 License
MIT License
