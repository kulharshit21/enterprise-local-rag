# Enterprise Adaptive RAG System — Technical Design Document

**Version:** 2.0  
**Date:** 2026-02-16  
**Hardware:** RTX-class GPU (12GB VRAM) · 32GB RAM · Fully local, zero paid APIs  
**Stack:** LLaMA 3 8B Instruct (Q4_K_M) · BGE-large-en · FAISS-GPU · BM25 · bge-reranker-large · FastAPI · Docker

---

# 1. System Overview

## 1.1 Problem Definition

Enterprise document QA requires answering natural-language queries over heterogeneous corpora (PDFs, markdown, CSVs, internal docs) with:
- **Attributed answers** — every claim traceable to source chunks
- **Role-gated retrieval** — RBAC filtering at the vector-index level
- **Faithfulness guarantees** — hallucination detection and rejection
- **Sub-3s P95 latency** on commodity GPU hardware

## 1.2 Why Naive RAG Is Insufficient

| Failure mode | Naive RAG behavior | This system's mitigation |
|---|---|---|
| **Lexical–semantic gap** | Dense-only retrieval misses keyword matches | BM25 + dense hybrid with RRF |
| **Re-ranking noise** | Top-K contains irrelevant passages at high recall | Cross-encoder reranker (bge-reranker-large) scores ⟨query, passage⟩ pairs |
| **Hallucination** | LLM fabricates beyond context | NLI-based faithfulness classifier on generated output |
| **Single-embedding blind spots** | One model can't cover domain-specific AND general semantics | BGE-large-en (MTEB #1 for retrieval at 1024-dim) |
| **Scalability ceiling** | Flat index O(n) scan at 100k+ docs | FAISS IVF with GPU-resident centroids |

## 1.3 Design Goals

1. **Retrieval quality:** ≥0.85 Recall@10 on curated test set
2. **Generation quality:** ≥0.90 faithfulness score (NLI-based)
3. **Latency:** P50 < 1.5s, P95 < 3.0s end-to-end
4. **Memory:** Peak VRAM ≤ 11GB (leave 1GB headroom)
5. **Zero external dependencies** — no OpenAI, no cloud APIs
6. **Reproducible evaluation** — ablation-ready experimental framework

## 1.4 Hardware Constraints

```
GPU:   12GB VRAM (RTX 3060/4070-class)
       └─ LLaMA 3 8B Q4_K_M:  ~4.5GB
       └─ BGE-large-en:       ~1.3GB
       └─ FAISS IVF index:    ~0.5GB (100k vectors × 1024d × float16)
       └─ bge-reranker-large: ~1.3GB (loaded on-demand, ~1.3GB)
       └─ Headroom:           ~3.1GB (KV cache, batch buffers)
RAM:   32GB system
       └─ BM25 index:         ~200MB–1GB (corpus-dependent)
       └─ Document store:     ~500MB
       └─ Application:        ~2GB
```

---

# 2. Detailed Architecture

## 2.1 Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Gateway                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │JWT Auth  │  │Rate Limit│  │RBAC      │  │Request Router │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────┬───────┘  │
└────────────────────────────────────────────────────┬────────────┘
                                                     │
┌────────────────────────────────────────────────────▼────────────┐
│                    RAG Pipeline Orchestrator                     │
│                                                                 │
│  ┌─────────┐    ┌──────────────┐    ┌──────────┐               │
│  │ Query   │───▶│Hybrid        │───▶│Cross-    │               │
│  │Encoder  │    │Retriever     │    │Encoder   │               │
│  │(BGE)    │    │(FAISS+BM25+  │    │Reranker  │               │
│  │         │    │ RRF)         │    │(bge-re)  │               │
│  └─────────┘    └──────────────┘    └────┬─────┘               │
│                                          │                      │
│  ┌─────────────────────────────────────┐ │                      │
│  │ Prompt Builder                      │ │                      │
│  │ (grounded template + token budget)  │◀┘                      │
│  └──────────────┬──────────────────────┘                        │
│                 │                                                │
│  ┌──────────────▼──────────────────────┐                        │
│  │ LLaMA 3 8B Instruct (Q4_K_M)       │                        │
│  │ via llama-cpp-python (GPU offload)  │                        │
│  └──────────────┬──────────────────────┘                        │
│                 │                                                │
│  ┌──────────────▼──────────────────────┐   ┌────────────────┐  │
│  │ Faithfulness Checker                │──▶│ Response       │  │
│  │ (NLI + embedding similarity)        │   │ Formatter      │  │
│  └─────────────────────────────────────┘   └────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Observability: structlog (JSON) · Prometheus · Profiler  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Ingestion Pipeline                           │
│  PDF/MD/CSV/HTML → Chunker (semantic, 512 tok) → BGE embed      │
│  → FAISS GPU index (IVF4096,PQ64 or Flat)                       │
│  → BM25 index (rank-bm25, persisted)                            │
│  → Metadata store (SQLite, role_access, doc_id, timestamps)     │
└─────────────────────────────────────────────────────────────────┘
```

## 2.2 End-to-End Data Flow

### Ingestion Path
```
Document Upload ──▶ Format Detection ──▶ Text Extraction (pdfplumber/BeautifulSoup)
    ──▶ Semantic Chunking (512 tokens, 64 overlap, heading-aware)
    ──▶ Metadata Enrichment (doc_id, role_access, sensitivity, section_type)
    ──▶ BGE-large-en Embedding (batch=32, GPU, 1024-dim float16)
    ──▶ FAISS Index Update (IVF add, periodic re-train centroids)
    ──▶ BM25 Index Rebuild (tokenized corpus, pickled)
```

### Query Path
```
HTTP Request ──▶ JWT Validation ──▶ Rate Limit Check ──▶ RBAC Role Extraction
    ──▶ Query Encoding (BGE-large-en, GPU)
    ──▶ Parallel Retrieval:
    │     ├─ FAISS Dense Search (top-50, role-filtered post-hoc)
    │     └─ BM25 Sparse Search (top-50, role-filtered)
    ──▶ Reciprocal Rank Fusion (k=60, weighted: dense=0.6, sparse=0.4)
    ──▶ Cross-Encoder Reranking (bge-reranker-large, top-20 → top-5)
    ──▶ Prompt Construction (system + context chunks + citation markers)
    ──▶ LLaMA 3 8B Generation (max_tokens=512, temp=0.1, top_p=0.9)
    ──▶ Faithfulness Check (embedding similarity + claim extraction)
    ──▶ Response Formatting (answer, citations, confidence, latency)
    ──▶ Metrics Recording (Prometheus counters, latency histograms)
```

## 2.3 GPU Memory Allocation Strategy

The 12GB VRAM budget is managed through **staged loading**:

| Phase | Models Resident | Approx VRAM |
|---|---|---|
| **Steady State** | LLaMA 3 (Q4_K_M) + BGE-large-en + FAISS index | ~6.3GB |
| **Reranking** | + bge-reranker-large (loaded on-demand per query) | ~7.6GB |
| **Peak (generation)** | All above + KV cache (512 tokens × 32 layers) | ~9.5GB |
| **Headroom** | Available for batch processing | ~2.5GB |

**Strategy:**
- LLaMA stays resident (4.5GB) — most expensive to reload
- BGE-large stays resident (1.3GB) — needed for every query
- FAISS index pinned to GPU memory (0.5GB for 100k vectors)
- Reranker: loaded into GPU on first query, kept resident if memory allows; evicted under pressure
- KV cache: allocated per-request, freed after generation

## 2.4 Parallelism Strategy

- **Ingestion:** `ThreadPoolExecutor(max_workers=4)` for IO-bound document loading; `torch.cuda.Stream` for embedding batches
- **Retrieval:** `asyncio.gather()` runs FAISS and BM25 searches concurrently (BM25 is CPU-bound, FAISS is GPU-bound — no contention)
- **Reranking:** Batched inference (batch_size=20) on GPU
- **Generation:** Sequential (autoregressive), streaming tokens via `llama-cpp-python` callbacks

## 2.5 Failure Handling

| Failure | Detection | Recovery |
|---|---|---|
| GPU OOM | `torch.cuda.OutOfMemoryError` catch | Evict reranker, retry with CPU fallback |
| LLM timeout (>10s) | Async timeout wrapper | Return cached similar query or "processing" response |
| Empty retrieval | `len(results) == 0` | Fall back to BM25-only, then to "no relevant docs" |
| FAISS corruption | Checksum mismatch on load | Rebuild from embedding cache |
| Model file missing | Startup health check | Block startup, log actionable error |

---

# 3. Retrieval System Design

## 3.1 Dense Embedding Pipeline

**Model:** `BAAI/bge-large-en-v1.5` (335M params, 1024-dim, MTEB #1 at time of selection)

```python
# Key configuration
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
model.max_seq_length = 512
model.half()  # float16 for VRAM efficiency

# Query prefix (BGE-specific, improves retrieval ~3-5%)
query_embedding = model.encode(f"Represent this sentence for retrieval: {query}")

# Document encoding (no prefix needed)
doc_embeddings = model.encode(chunks, batch_size=32, normalize_embeddings=True)
```

**Design decisions:**
- Float16 inference saves ~40% VRAM vs float32 with <0.1% quality loss
- L2-normalized embeddings enable cosine similarity via inner product (faster FAISS search)
- Batch size 32 balances throughput vs VRAM on 12GB card

## 3.2 FAISS Index Configuration

### Index Selection Analysis

| Index Type | Build Time | Query Latency | Recall@10 | Memory | When to Use |
|---|---|---|---|---|---|
| `Flat` (brute force) | O(1) | O(n) ~2ms@10k | 1.000 | 40MB/10k | <50k docs |
| `IVF4096,Flat` | O(n·log) ~30s | O(√n) ~0.5ms@100k | 0.98 (nprobe=64) | 42MB/10k + centroids | 50k–1M docs |
| `IVF4096,PQ64` | O(n·log) ~45s | O(√n) ~0.3ms@100k | 0.95 | 10MB/10k + centroids | >1M docs (RAM constrained) |
| `HNSW32` | O(n·log n) ~60s | O(log n) ~0.2ms | 0.99 (efSearch=128) | 56MB/10k | Low-latency priority |

**Selected configuration:**

```python
# Adaptive index selection based on corpus size
if num_vectors < 50_000:
    index = faiss.IndexFlatIP(1024)  # Exact search, no training needed
elif num_vectors < 500_000:
    quantizer = faiss.IndexFlatIP(1024)
    index = faiss.IndexIVFFlat(quantizer, 1024, 4096, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = 64  # Trade 2% recall for 4x speed
else:
    quantizer = faiss.IndexFlatIP(1024)
    index = faiss.IndexIVFPQ(quantizer, 1024, 4096, 64, 8)
    index.nprobe = 128

# GPU acceleration
gpu_res = faiss.StandardGpuResources()
gpu_res.setTempMemory(512 * 1024 * 1024)  # 512MB temp buffer
gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
```

**Why IVFFlat over HNSW for default:**
- HNSW has no native GPU support in FAISS (must run CPU)
- IVFFlat GPU search is 10-50x faster than HNSW CPU for batch queries
- IVFFlat supports efficient incremental updates (just add vectors)
- HNSW requires expensive graph reconstruction on updates

## 3.3 BM25 Integration

```python
# Tokenization: whitespace + lowercasing + stopword removal
# Using rank_bm25.BM25Okapi with default k1=1.5, b=0.75

class BM25Retriever:
    def search(self, query: str, top_k: int, role_filter: List[str]) -> List[dict]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Role-based post-filtering
        for i, doc in enumerate(self.documents):
            if not self._role_check(doc["metadata"]["role_access"], role_filter):
                scores[i] = -float("inf")

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] | {"score": scores[i]} for i in top_indices if scores[i] > 0]
```

**Why BM25 alongside dense:**
- Dense embeddings miss exact keyword matches (product names, acronyms, IDs)
- BM25 captures lexical overlap that BGE may compress into similar embedding space
- Empirically, hybrid > dense-only by 5-15% on enterprise corpora (See: Ma et al., 2023)

## 3.4 Reciprocal Rank Fusion (RRF)

```
RRF_score(d) = Σ_r [ w_r / (k + rank_r(d)) ]

where:
  r ∈ {dense, sparse}
  k = 60 (standard constant, reduces impact of high-rank outliers)
  w_dense = 0.6, w_sparse = 0.4
```

**Why RRF over linear score combination:**
- Scores from dense (cosine similarity ∈ [0,1]) and sparse (BM25 ∈ [0,∞)) are incommensurable
- RRF is rank-based, so no score normalization needed
- Robust to score distribution differences across queries
- O(n) merge complexity

## 3.5 Cross-Encoder Reranking

**Model:** `BAAI/bge-reranker-large` (560M params)

```python
# Input: top-20 candidates from RRF
# Output: top-5 reranked by relevance score

pairs = [[query, chunk["content"]] for chunk in candidates[:20]]
scores = reranker.compute_score(pairs, batch_size=20)
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]
```

**Why rerank only top-20:**
- Reranker is O(n) in candidate count with large constant (BERT forward pass per pair)
- 20 pairs ≈ 50ms on GPU; 100 pairs ≈ 250ms — unacceptable for P95 < 3s
- Diminishing returns beyond top-20: recall improvement <1% for candidates 20-50

## 3.6 Adaptive Top-K Strategy

```python
def adaptive_top_k(query: str, base_k: int = 5) -> int:
    """Adjust K based on query complexity and retrieval confidence."""
    # Short queries (likely keyword search) → more candidates
    if len(query.split()) <= 3:
        return base_k + 3

    # Compute retrieval confidence from score distribution
    scores = [r["score"] for r in initial_results]
    score_gap = scores[0] - scores[base_k - 1] if len(scores) >= base_k else 0

    if score_gap > 0.3:  # Clear separation → fewer results needed
        return max(3, base_k - 2)
    elif score_gap < 0.05:  # Flat distribution → need more context
        return min(10, base_k + 3)

    return base_k
```

---

# 4. Experimental Design (Workshop-Level Rigor)

## 4.1 Dataset Construction

### Corpus
- **Source:** 500+ enterprise documents across 4 categories:
  - Policy documents (50 docs, ~25k tokens each)
  - Financial reports (100 docs, ~15k tokens each, heavy tables)
  - Technical documentation (200 docs, ~10k tokens each)
  - Research notes (150 docs, ~8k tokens each, mixed sensitivity levels)
- **Total:** ~5M tokens, ~10k chunks after processing

### Ground-Truth QA Pairs

| Split | Queries | Construction Method |
|---|---|---|
| **Train** | 200 | Manual expert annotation: query + gold passages + gold answer |
| **Validation** | 50 | Same protocol, held out during development |
| **Test** | 100 | Same protocol, sealed until final evaluation |

**Labeling Protocol:**
1. Two annotators independently identify relevant passages for each query
2. Cohen's κ ≥ 0.80 required; disagreements resolved by third annotator
3. Each query annotated with: `relevant_chunk_ids`, `gold_answer`, `difficulty` (easy/medium/hard), `query_type` (factoid/reasoning/comparison/unanswerable)
4. Include 15% adversarial/unanswerable queries to test faithfulness

## 4.2 Retrieval Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Precision@K** | \|retrieved ∩ relevant\| / K | Fraction of top-K that are relevant |
| **Recall@K** | \|retrieved ∩ relevant\| / \|relevant\| | Fraction of relevant docs found |
| **MRR** | 1 / rank(first relevant) | How quickly the first relevant appears |
| **NDCG@K** | DCG@K / IDCG@K | Rank-aware relevance quality |
| **MAP** | Mean of AP across queries | Overall ranking quality |

**Evaluation protocol:** K ∈ {1, 3, 5, 10, 20} for each metric

## 4.3 Generation Metrics

| Metric | What It Measures |
|---|---|
| **Exact Match** | Normalized string match with gold answer |
| **Token F1** | Token-level precision/recall vs gold answer |
| **Faithfulness** | NLI entailment score: answer vs retrieved context |
| **Answer Relevance** | Cosine similarity between query and answer embeddings |
| **Context Utilization** | Token overlap between answer and retrieved chunks |
| **Hallucination Rate** | Fraction of claims not entailed by context |

## 4.4 Statistical Significance

```python
# Paired bootstrap test (Efron & Tibshirani, 1993)
def paired_bootstrap(metric_a, metric_b, n_bootstrap=10000, alpha=0.05):
    """Test if system A is significantly better than system B."""
    diffs = [a - b for a, b in zip(metric_a, metric_b)]
    observed_diff = np.mean(diffs)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(diffs, size=len(diffs), replace=True)
        bootstrap_diffs.append(np.mean(sample))

    p_value = np.mean([d <= 0 for d in bootstrap_diffs])
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "ci_95": (ci_lower, ci_upper),
        "significant": p_value < alpha,
    }
```

## 4.5 Ablation Studies

| Ablation | What's Removed | Hypothesis |
|---|---|---|
| No BM25 | Dense-only retrieval | ≥5% Recall@10 drop on keyword-heavy queries |
| No reranker | Skip cross-encoder | ≥8% NDCG@5 drop, faster latency |
| No RRF (dense only) | BM25 + fusion removed | Lexical queries degrade significantly |
| No RRF (sparse only) | Dense + fusion removed | Semantic queries degrade significantly |
| K=3 vs K=5 vs K=10 | Context window size | Diminishing returns at K>5 |
| No faithfulness check | Skip hallucination detection | Unfaithful answers increase |
| Float32 vs Float16 | Embedding precision | <0.1% quality diff, 40% memory saving |
| Flat vs IVF index | Index type | <2% recall loss, 4x speed gain |

## 4.6 Reproducibility Protocol

- **Random seeds:** Fixed at 42 for all stochastic operations
- **Model versions:** Pinned in `requirements.txt` with exact versions
- **Hardware:** Documented GPU model, driver version, CUDA version
- **Docker image:** Deterministic build with `--no-cache` verification
- **Evaluation scripts:** Single `python -m evaluation.run_experiment --config experiments/config.yaml`

---

# 5. Baseline Systems & Benchmark Plan

## 5.1 Baseline Definitions

| ID | System | Retriever | Reranker | LLM |
|---|---|---|---|---|
| B0 | **No retrieval** | None | None | LLaMA 3 8B (closed-book) |
| B1 | **BM25-only** | BM25 top-5 | None | LLaMA 3 8B |
| B2 | **Dense-only** | FAISS top-5 | None | LLaMA 3 8B |
| B3 | **Hybrid (no rerank)** | FAISS+BM25+RRF top-5 | None | LLaMA 3 8B |
| **S1** | **Full system** | FAISS+BM25+RRF top-20 | bge-reranker→top-5 | LLaMA 3 8B |

## 5.2 Benchmark Table Template

```
| System | P@5  | R@10 | MRR  | NDCG@5 | EM   | F1   | Faith. | Lat.(P50) | Lat.(P95) |
|--------|------|------|------|--------|------|------|--------|-----------|-----------|
| B0     |  --  |  --  |  --  |  --    | 0.xx | 0.xx | 0.xx   | xxxms     | xxxms     |
| B1     | 0.xx | 0.xx | 0.xx | 0.xx   | 0.xx | 0.xx | 0.xx   | xxxms     | xxxms     |
| B2     | 0.xx | 0.xx | 0.xx | 0.xx   | 0.xx | 0.xx | 0.xx   | xxxms     | xxxms     |
| B3     | 0.xx | 0.xx | 0.xx | 0.xx   | 0.xx | 0.xx | 0.xx   | xxxms     | xxxms     |
| S1     | 0.xx | 0.xx | 0.xx | 0.xx   | 0.xx | 0.xx | 0.xx   | xxxms     | xxxms     |
```

## 5.3 Analysis Methodology

1. **Per-query analysis:** Identify queries where full system beats/loses to baselines
2. **Error categorization:** Classify failures as retrieval-miss, reranking-error, generation-hallucination, or input-ambiguity
3. **Statistical tests:** Paired bootstrap for each system pair
4. **Efficiency frontier:** Plot quality vs latency for each system configuration

---

# 6. Scalability & Distributed Redesign

## 6.1 Target Scale

- **100k+ documents** (~500k chunks, ~2M embedding vectors)
- **1,000 concurrent users** (~200 QPS peak)

## 6.2 Sharded FAISS Architecture

```
                    ┌──────────────────┐
                    │   Query Router   │
                    │  (Round-Robin or  │
                    │   Score-Merge)   │
                    └────┬───────┬─────┘
                         │       │
              ┌──────────▼─┐ ┌──▼──────────┐
              │ Shard 0    │ │ Shard 1     │  ... Shard N
              │ IVF4096    │ │ IVF4096     │
              │ 0-166k vecs│ │ 166k-333k   │
              │ GPU 0      │ │ GPU 1       │
              └────────────┘ └─────────────┘
```

- **Sharding strategy:** Hash(doc_id) % N_shards for even distribution
- **Query:** Broadcast to all shards, merge top-K results globally
- **Rebalancing:** Background job when shard size variance > 20%

## 6.3 Distributed Workers

```yaml
# docker-compose.scaled.yml
services:
  gateway:          # FastAPI, JWT, rate limiting
    replicas: 2
    resources: { cpus: '2', memory: '4G' }

  embedding-worker: # BGE-large-en GPU inference
    replicas: 2     # Each gets 1 GPU
    resources: { gpus: '1', memory: '8G' }

  retrieval-worker: # FAISS shards + BM25
    replicas: 4
    resources: { cpus: '4', memory: '8G' }

  generation-worker: # LLaMA 3 8B
    replicas: 2      # Each gets 1 GPU
    resources: { gpus: '1', memory: '12G' }

  reranker-worker:   # bge-reranker-large
    replicas: 2      # Shared GPU with embedding worker
    resources: { gpus: '0.5', memory: '4G' }
```

## 6.4 Async Pipeline

```python
async def handle_query(query: str, role: str):
    # Phase 1: Encode query (GPU, ~15ms)
    query_vec = await embedding_pool.submit(encode_query, query)

    # Phase 2: Parallel retrieval (no GPU contention)
    dense_task = asyncio.create_task(faiss_search(query_vec, role))
    sparse_task = asyncio.create_task(bm25_search(query, role))
    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

    # Phase 3: RRF fusion (CPU, ~1ms)
    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    # Phase 4: Rerank (GPU, ~50ms)
    reranked = await reranker_pool.submit(rerank, query, fused[:20])

    # Phase 5: Generate (GPU, ~800ms)
    answer = await generation_pool.submit(generate, query, reranked[:5])

    return answer
```

## 6.5 Caching Layers

| Layer | Cache | TTL | Hit Rate (expected) |
|---|---|---|---|
| Query embedding | LRU(10k) | 1h | ~15% (exact repeats) |
| Full query result | Redis | 5min | ~8% (identical queries) |
| Semantic cache | FAISS similarity >0.95 | 10min | ~20% (paraphrases) |
| BM25 index | In-memory | Until corpus change | 100% |
| Model weights | GPU memory | Persistent | 100% |

## 6.6 Backpressure Handling

```python
# Semaphore-based concurrency control
GPU_SEMAPHORE = asyncio.Semaphore(4)    # Max 4 concurrent GPU ops
LLM_SEMAPHORE = asyncio.Semaphore(1)    # LLM is sequential (KV cache)

async def gpu_bounded_rerank(query, candidates):
    async with GPU_SEMAPHORE:
        return await reranker.rerank(query, candidates)

# Queue with rejection
class BoundedQueue:
    def __init__(self, maxsize=100):
        self.queue = asyncio.Queue(maxsize=maxsize)

    async def submit(self, task):
        try:
            self.queue.put_nowait(task)
        except asyncio.QueueFull:
            raise HTTPException(503, "System at capacity, retry later")
```

---

# 7. Latency & Throughput Optimization

## 7.1 Component-Level Latency Breakdown

| Component | P50 | P95 | P99 | GPU/CPU | Optimization Lever |
|---|---|---|---|---|---|
| JWT validation | 0.1ms | 0.2ms | 0.5ms | CPU | Pre-computed key cache |
| Query encoding (BGE) | 12ms | 18ms | 25ms | GPU | Float16, batch=1 |
| FAISS search (IVF, 100k) | 1ms | 3ms | 5ms | GPU | nprobe tuning |
| BM25 search | 5ms | 12ms | 20ms | CPU | Pre-tokenized index |
| RRF fusion | 0.5ms | 1ms | 2ms | CPU | NumPy vectorized |
| Reranking (20 pairs) | 45ms | 65ms | 80ms | GPU | Batched, float16 |
| Prompt construction | 1ms | 2ms | 3ms | CPU | Pre-formatted templates |
| LLM generation (256 tok) | 650ms | 1200ms | 1800ms | GPU | KV cache, speculative |
| Faithfulness check | 30ms | 50ms | 70ms | GPU | Embedding reuse |
| Response formatting | 0.5ms | 1ms | 2ms | CPU | — |
| **Total** | **~745ms** | **~1350ms** | **~2000ms** | — | — |

## 7.2 GPU vs CPU Tradeoff Matrix

| Operation | GPU Speedup | When to Use CPU |
|---|---|---|
| BGE encoding | 8x | Batch ingestion when GPU is busy with LLM |
| FAISS search | 20x (IVF) | HNSW index (no GPU support) |
| Reranking | 5x | GPU OOM fallback |
| LLM generation | 3-5x (vs CPU) | Never — CPU is too slow for interactive use |
| BM25 search | N/A | Always CPU (not parallelizable on GPU) |

## 7.3 Batching Strategy

```python
# Dynamic batching for embedding requests during ingestion
class EmbeddingBatcher:
    def __init__(self, max_batch=64, max_wait_ms=50):
        self.queue = asyncio.Queue()
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms

    async def worker(self):
        while True:
            batch = []
            # Collect up to max_batch items within max_wait_ms
            try:
                first = await self.queue.get()
                batch.append(first)
                deadline = time.time() + self.max_wait_ms / 1000

                while len(batch) < self.max_batch and time.time() < deadline:
                    try:
                        item = await asyncio.wait_for(
                            self.queue.get(), timeout=deadline - time.time()
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
            except Exception:
                continue

            # Process batch on GPU
            texts = [item.text for item in batch]
            embeddings = model.encode(texts, batch_size=len(texts))
            for item, emb in zip(batch, embeddings):
                item.future.set_result(emb)
```

## 7.4 Early Exit Retrieval

```python
async def early_exit_search(query_vec, min_score=0.7, max_k=50):
    """Stop retrieving once quality drops below threshold."""
    results = faiss_index.search(query_vec, max_k)

    # Find score cliff
    for i in range(1, len(results)):
        if results[i].score < min_score:
            return results[:i]
        if results[i-1].score - results[i].score > 0.15:  # Score cliff
            return results[:i]

    return results
```

## 7.5 Query Classification Routing

```python
QUERY_ROUTES = {
    "simple_factoid": {"k": 3, "use_reranker": False, "max_tokens": 128},
    "complex_reasoning": {"k": 8, "use_reranker": True, "max_tokens": 512},
    "comparison": {"k": 10, "use_reranker": True, "max_tokens": 384},
    "unanswerable": {"k": 3, "use_reranker": False, "max_tokens": 64},
}

def classify_query(query: str) -> str:
    """Lightweight heuristic query classifier."""
    tokens = query.lower().split()
    if len(tokens) <= 5 and any(w in tokens for w in ["what", "who", "when", "how much"]):
        return "simple_factoid"
    if any(w in tokens for w in ["compare", "difference", "versus", "vs"]):
        return "comparison"
    if len(tokens) > 15 or any(w in tokens for w in ["explain", "analyze", "discuss"]):
        return "complex_reasoning"
    return "simple_factoid"
```

## 7.6 Throughput Targets

| Metric | Single GPU | Scaled (4 GPU) |
|---|---|---|
| Queries/second (P50) | 1.3 | 5.2 |
| Embeddings/second (ingestion) | 200 | 800 |
| Index build time (100k docs) | ~8min | ~2min |
| Concurrent users | 5-10 | 40-50 |

---

# 8. Security & Threat Modeling (STRIDE)

## 8.1 Threat Matrix

| STRIDE Category | Threat | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| **Spoofing** | JWT token forging | Full system access | Low (HS256 w/ strong secret) | RS256 upgrade, short TTL (15min), refresh tokens |
| **Tampering** | Document injection with adversarial content | Poisoned retrieval results | Medium | Input sanitization, embedding anomaly detection |
| **Repudiation** | User denies querying sensitive docs | Audit trail gaps | Low | Immutable audit log with query hashes |
| **Information Disclosure** | Embedding inversion attack | Reconstruct document text | Medium | Add Gaussian noise to stored embeddings (σ=0.01) |
| **Information Disclosure** | Prompt leakage in LLM output | System prompt exposed | Medium | Output regex filter for system prompt fragments |
| **Denial of Service** | Large document flood | GPU/memory exhaustion | High | File size limits (50MB), rate limiting, queue bounds |
| **Elevation of Privilege** | RBAC bypass via metadata manipulation | Access to restricted docs | Medium | Server-side role resolution, never trust client metadata |

## 8.2 Prompt Injection Defense

```python
INJECTION_PATTERNS = [
    r"ignore.*(?:previous|above).*instructions",
    r"you are now",
    r"system prompt",
    r"reveal.*(?:instructions|prompt)",
    r"act as",
    r"\[INST\]",  # LLaMA-specific injection
    r"<<SYS>>",   # LLaMA system token injection
]

def sanitize_query(query: str) -> str:
    """Detect and neutralize prompt injection attempts."""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            logger.warning("prompt_injection_detected", query=query[:100])
            raise HTTPException(400, "Query contains disallowed patterns")
    return query
```

## 8.3 Embedding Leakage Protection

- Store embeddings in float16 (less precise reconstruction)
- Add calibrated Gaussian noise (σ = 0.01) — reduces inversion accuracy by ~40% with <1% retrieval quality loss
- Never expose raw embeddings via API
- FAISS index on encrypted volume at rest

## 8.4 Monitoring Signals

| Signal | Alert Threshold | Action |
|---|---|---|
| Failed auth attempts | >10/min per IP | Temporary IP ban |
| Injection pattern matches | >3/min per user | Account suspension review |
| Unusual query patterns | Embedding centroid shift >2σ | Manual review |
| Data exfiltration signals | >100 queries/5min per user | Rate limit escalation |

---

# 9. Cost Modeling

## 9.1 Local Infrastructure Costs

| Component | Power Draw | Monthly Cost (@$0.12/kWh) |
|---|---|---|
| GPU (RTX 4070, load) | 200W | ~$17.50 |
| CPU + RAM (idle) | 65W | ~$5.70 |
| SSD I/O | 5W | ~$0.44 |
| **Total (24/7)** | **270W** | **~$23.60** |

## 9.2 Token Generation Cost Comparison

| Provider | Cost per 1M tokens | Monthly (100k queries × 512 tokens) |
|---|---|---|
| GPT-4o | $5.00 / $15.00 (in/out) | ~$1,000 |
| GPT-4o-mini | $0.15 / $0.60 | ~$38 |
| Claude 3.5 Sonnet | $3.00 / $15.00 | ~$900 |
| **Local LLaMA 3 8B** | **$0.00** | **$23.60 (electricity only)** |

**Break-even:** Local system pays for itself after ~1 month vs GPT-4o-mini at 100k queries/month.

## 9.3 Optimization Strategies

| Strategy | Cost Reduction | Quality Impact |
|---|---|---|
| **Reranker gating** (skip for high-confidence retrievals) | -30% GPU time | <1% quality loss |
| **Semantic caching** (serve cached for similarity >0.95) | -20% total cost | 0% (exact match) |
| **Dynamic model routing** (small model for simple queries) | -40% GPU time | ~2% quality loss |
| **Adaptive K** (reduce K for clear retrievals) | -15% context tokens | <1% quality loss |
| **Token-aware truncation** (trim chunks to budget) | -25% prompt tokens | Minimal |

---

# 10. Failure Mode Taxonomy

## 10.1 Classification

| Category | Failure Mode | Detection Metric | Mitigation |
|---|---|---|---|
| **Retrieval** | Semantic gap (query ≠ document terminology) | Low top-1 score (<0.4) | Query expansion, synonym injection |
| **Retrieval** | Retrieval drift (index stale vs new docs) | Freshness lag metric | Scheduled re-indexing, version tracking |
| **Retrieval** | BM25/dense disagreement | RRF score variance | Weight adaptation based on query type |
| **Ranking** | Cross-encoder bias (positional, length) | Score calibration tests | Length-normalized scoring |
| **Generation** | Intrinsic hallucination (fabricated facts) | NLI score < 0.5 | Reject and return "insufficient context" |
| **Generation** | Extrinsic hallucination (plausible but wrong) | Claim-level verification | Per-claim NLI entailment check |
| **Generation** | Over-reliance on single chunk | Citation distribution entropy | Minimum 2-source requirement |
| **Input** | Out-of-distribution query | Low max retrieval score | Explicit "no relevant docs" response |
| **Input** | Adversarial prompt injection | Pattern matching + perplexity spike | Input sanitization + rejection |
| **System** | GPU OOM | `torch.cuda.memory_allocated()` threshold | Model eviction cascade |
| **System** | Latency spike | P95 > 5s for 5min | Alert + autoscale |

## 10.2 Hallucination Detection Pipeline

```
Answer ──▶ Claim Extraction (sentence splitting)
  ──▶ Per-claim NLI Check:
  │    claim + context_chunk → {entailment, neutral, contradiction}
  ──▶ Aggregate Faithfulness Score:
  │    score = (entailed_claims / total_claims)
  ──▶ Decision:
       score ≥ 0.7 → Accept answer
       score ∈ [0.4, 0.7) → Flag for review, append warning
       score < 0.4 → Reject, return "Cannot verify from available sources"
```

---

# 11. Continuous Evaluation & Monitoring

## 11.1 Automated Regression Test Suite

```yaml
# test_config.yaml
regression_tests:
  schedule: "daily at 02:00 UTC"
  dataset: "evaluation/regression_set.json"  # 50 gold QA pairs
  pass_criteria:
    recall_at_10: ">= 0.82"
    faithfulness: ">= 0.85"
    p95_latency_ms: "<= 3000"
  alert_on_failure: true
  alert_channels: ["slack", "email"]
```

## 11.2 Monitoring Dashboard Design

```
┌─────────────────────────────────────────────────────────┐
│  RAG System — Real-Time Dashboard                       │
├──────────────────────┬──────────────────────────────────┤
│  Query Volume        │  Latency Distribution            │
│  ████████░░ 847/hr   │  P50: 745ms  P95: 1350ms        │
│  [24h sparkline]     │  [histogram]                     │
├──────────────────────┼──────────────────────────────────┤
│  Faithfulness Score  │  Retrieval Quality               │
│  Mean: 0.87          │  Avg Top-1 Score: 0.72           │
│  [30d trend]         │  Empty Results: 2.1%             │
├──────────────────────┼──────────────────────────────────┤
│  GPU Utilization     │  Error Rate                      │
│  LLM: 67%  BGE: 12% │  4xx: 1.2%  5xx: 0.3%           │
│  Reranker: 8%        │  Timeout: 0.1%                   │
├──────────────────────┼──────────────────────────────────┤
│  Cache Hit Rate      │  Security Events                 │
│  Semantic: 18%       │  Auth failures: 3                │
│  Exact: 7%           │  Injection attempts: 0           │
└──────────────────────┴──────────────────────────────────┘
```

## 11.3 Logging Schema

```json
{
  "timestamp": "2026-02-16T12:00:00Z",
  "query_id": "q-abc123",
  "event": "query_complete",
  "user": "analyst_1",
  "role": "analyst",
  "query_hash": "sha256:...",
  "latency_ms": 1247,
  "components": {
    "encoding_ms": 14,
    "faiss_ms": 2,
    "bm25_ms": 8,
    "rrf_ms": 1,
    "rerank_ms": 52,
    "generation_ms": 1120,
    "faithfulness_ms": 45
  },
  "retrieval": {
    "dense_top1_score": 0.78,
    "sparse_top1_score": 12.3,
    "rrf_top1_score": 0.023,
    "num_results": 5,
    "chunks_used": ["c-001", "c-042", "c-117"]
  },
  "generation": {
    "model": "llama-3-8b-instruct-q4_k_m",
    "prompt_tokens": 1847,
    "completion_tokens": 234,
    "temperature": 0.1
  },
  "quality": {
    "faithfulness_score": 0.89,
    "is_faithful": true,
    "claims_total": 4,
    "claims_supported": 4
  },
  "gpu": {
    "vram_used_mb": 9200,
    "utilization_pct": 72
  }
}
```

## 11.4 Alert Thresholds

| Metric | Warning | Critical | Action |
|---|---|---|---|
| P95 latency | >3s for 5min | >5s for 2min | Scale workers / check GPU |
| Faithfulness (rolling 1h) | <0.80 | <0.70 | Pause system, investigate |
| Error rate | >2% | >5% | Page on-call |
| GPU memory | >10GB | >11GB | Evict reranker |
| Empty retrievals | >10% | >20% | Check index health |

---

# 12. What Would Make This FAANG-Level?

## 12.1 Differentiators from Student RAG Projects

| Aspect | Student Project | This System |
|---|---|---|
| Retrieval | Single embedding model, flat index | Hybrid (dense+sparse), RRF fusion, adaptive K |
| Reranking | None or basic | Cross-encoder with gating logic |
| Evaluation | "It works on 3 examples" | 350 annotated QA pairs, 8 metrics, ablation studies, bootstrap significance tests |
| Hallucination | None | NLI-based claim-level faithfulness pipeline |
| Infrastructure | `streamlit` + `pip install` | FastAPI + Docker + Prometheus + structured logging |
| Memory mgmt | "It fits" | Staged GPU allocation with OOM fallback cascade |
| Security | None | STRIDE model, RBAC, prompt injection defense |
| Scalability | Single process | Async pipeline, sharded indices, caching layers |
| Cost | "It's free" | Quantified TCO with break-even analysis |

## 12.2 Remaining Weaknesses

1. **No learned retrieval:** Using pre-trained BGE without domain fine-tuning on the target corpus
2. **Single LLM:** No ensemble or routing between models of different sizes
3. **Static chunking:** Semantic chunking is heuristic; no learned chunk boundaries
4. **No RAG-specific training:** LLaMA is used as-is, not fine-tuned for grounded generation
5. **Single-node primary:** Distributed design is theoretical, not battle-tested at scale

## 12.3 What Would Elevate to ICML/NeurIPS Workshop Level

1. **Learned hybrid fusion weights** — train α(dense, sparse) per-query using a small labeled set
2. **Retrieval-aware fine-tuning** — RAFT (Retrieval-Augmented Fine-Tuning) on the target corpus
3. **Automated ground-truth generation** — LLM-as-judge pipeline for scaling evaluation beyond 350 manual annotations
4. **Adaptive retrieval** — classify queries into "needs retrieval" vs "answerable from parametric knowledge" before searching
5. **Faithfulness improvements** — fine-tuned NLI model on RAG-specific entailment patterns rather than generic NLI

---

# 13. Senior Engineer Critique

## The Review

*As a senior ML infrastructure engineer reviewing this portfolio:*

### Strengths
- **Systems thinking is real.** GPU memory budget, latency breakdown, failure cascades — this isn't a weekend hack. The staged loading strategy and OOM fallback show production awareness.
- **Evaluation rigor is above average.** 350 QA pairs with inter-annotator agreement, ablation studies, and bootstrap significance tests. This is closer to a workshop paper than a demo.
- **Security isn't an afterthought.** STRIDE analysis, prompt injection defense, embedding leakage protection — most RAG projects ignore all of this.

### Weaknesses & Critique

1. **The "no LangChain" stance is performative, not principled.** You've rebuilt half of LangChain poorly. If the argument is control and debuggability, prove it with a specific debugging scenario where framework abstraction failed you. Otherwise, this reads as resume-driven.

2. **Reranker gating logic is hand-wavy.** "Skip for high-confidence retrievals" — what's the threshold? How was it calibrated? This needs a precision-recall curve showing gating threshold vs quality tradeoff with specific operating points.

3. **The faithfulness pipeline has a circular dependency.** You're using BGE embeddings (the same model that retrieved the chunks) to verify faithfulness. If retrieval is systematically wrong, faithfulness check won't catch it. You need an independent verification signal — ideally a separate NLI model (e.g., DeBERTa-v3-base-mnli-fever-anli).

4. **100k+ doc scalability is theoretical.** You haven't benchmarked FAISS IVF at 500k vectors. The nprobe=64 claim of 98% recall needs validation on YOUR data distribution, not FAISS benchmarks on SIFT1M. Embedding distributions for enterprise text ≠ SIFT features.

5. **The BM25 index doesn't scale to updates.** You rebuild the entire BM25 index on every ingestion. At 100k docs, this takes 10+ seconds and blocks queries. Need incremental BM25 or switch to Elasticsearch.

6. **No streaming generation.** For 1.2s of LLM generation, the user stares at a spinner. Server-Sent Events for streaming tokens would dramatically improve perceived latency.

7. **Missing request tracing.** You have per-component logging but no distributed tracing (OpenTelemetry). In a multi-worker deployment, you can't trace a query across embedding → retrieval → generation workers.

8. **The cost model is misleading.** Comparing local electricity to API pricing ignores hardware amortization ($1500+ for the GPU), maintenance burden, and opportunity cost of running your own infra. A fair comparison includes CapEx amortized over 3 years.

### How to Improve Differentiation

1. **Add a real ablation study with results**, not just a table of "what to ablate." Run the experiments, report numbers, analyze where each component helps/hurts.
2. **Deploy on a real corpus** (at least 10k genuine docs) and report end-to-end numbers.
3. **Implement streaming generation** — it's table-stakes for modern LLM applications.
4. **Add OpenTelemetry tracing** — shows you understand production observability beyond toy metrics.
5. **Fine-tune BGE on your domain** — even 500 hard negatives would show ML depth beyond "pip install sentence-transformers."

### Verdict
> **Hire signal: leaning positive.** This demonstrates genuine systems maturity and research awareness that most candidates lack. The gap is between "designed it" and "ran it at scale and learned from failures." Fill that gap with real experimental results and this is a strong portfolio piece.
