"""
RAG Pipeline Orchestrator.
Coordinates the full retrieval-augmented generation flow.
Fully local inference — no paid APIs.
"""

import time
from typing import Dict, Any, Optional, List

from config import settings, ensure_directories
from embeddings.dense import DenseEmbedder
from embeddings.sparse import SparseIndexer
from embeddings.vector_store import VectorStore
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.prompt_builder import PromptBuilder
from generation.llm_client import LLMClient
from generation.response_formatter import ResponseFormatter
from generation.hallucination import HallucinationDetector
from security.rbac import RBACFilter
from observability.logger import QueryLogger, generate_query_id
from observability import metrics as obs_metrics
from evaluation.system_metrics import QueryMetrics, SystemMetricsTracker, Timer
from ingestion.loader import DocumentLoader
from ingestion.metadata import MetadataExtractor


class RAGPipeline:
    """
    End-to-end RAG pipeline orchestrating:
    1. Query encoding (BGE-large-en, GPU)
    2. Parallel hybrid retrieval (FAISS GPU + BM25 CPU)
    3. Reciprocal Rank Fusion
    4. Cross-encoder reranking (bge-reranker-large)
    5. Grounded prompt construction
    6. Local LLM generation (LLaMA 3 8B Q4_K_M)
    7. Faithfulness checking
    8. Response formatting with citations
    9. Per-component metrics logging
    """

    def __init__(self, use_reranker: bool = False, lazy_llm: bool = True):
        ensure_directories()

        # Core components
        self.embedder = DenseEmbedder()
        self.vector_store = VectorStore()
        self.sparse_indexer = SparseIndexer()

        # Retrieval
        self.dense_retriever = DenseRetriever(self.embedder, self.vector_store)
        self.sparse_retriever = SparseRetriever(self.sparse_indexer)
        self.hybrid_retriever = HybridRetriever(
            self.dense_retriever, self.sparse_retriever
        )

        # Optional reranker
        self.use_reranker = use_reranker
        self._reranker = None

        # Generation
        self.prompt_builder = PromptBuilder()
        self._llm_client = None if lazy_llm else LLMClient()
        self.response_formatter = ResponseFormatter()
        self.hallucination_detector = HallucinationDetector(self.embedder)

        # Ingestion
        self.doc_loader = DocumentLoader()
        self.metadata_extractor = MetadataExtractor()

        # Observability
        self.query_logger = QueryLogger()
        self.system_metrics = SystemMetricsTracker()

        # Try to load existing BM25 index
        self.sparse_indexer.load()

    @property
    def llm_client(self) -> LLMClient:
        """Lazy-load LLM to avoid slow startup when only ingesting."""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    @property
    def reranker(self) -> Optional[CrossEncoderReranker]:
        """Lazy-load reranker."""
        if self.use_reranker and self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker if self.use_reranker else None

    def ingest_documents(
        self,
        directory: str = None,
        file_path: str = None,
        role_access: List[str] = None,
        sensitivity: str = "internal",
    ) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system.
        Supports both single file and directory ingestion.
        """
        role_access = role_access or ["viewer"]

        # Load documents
        if directory:
            documents = self.doc_loader.load_directory(
                directory, role_access, sensitivity
            )
        elif file_path:
            documents = [self.doc_loader.load(file_path, role_access, sensitivity)]
        else:
            return {"error": "Provide either directory or file_path"}

        if not documents:
            return {"message": "No documents found", "count": 0}

        # Process into chunks
        all_chunks = self.metadata_extractor.process_documents(
            documents, role_access, sensitivity
        )

        if not all_chunks:
            return {"message": "No chunks generated", "count": 0}

        # Generate embeddings via BGE-large-en
        chunk_texts = [c.content for c in all_chunks]
        embeddings = self.embedder.embed_texts(chunk_texts)

        # Store in FAISS index
        chunk_ids = [c.chunk_id for c in all_chunks]
        metadatas = [c.metadata for c in all_chunks]

        self.vector_store.add_vectors(
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
        )

        # Build BM25 index
        bm25_docs = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "metadata": c.metadata,
            }
            for c in all_chunks
        ]
        self.sparse_indexer.build_index(bm25_docs)
        self.sparse_indexer.save()

        # Update index size gauge
        obs_metrics.INDEX_SIZE.labels(collection="text_chunks").set(
            self.vector_store.size
        )

        return {
            "message": "Ingestion complete",
            "documents_loaded": len(documents),
            "chunks_created": len(all_chunks),
            "index_size": self.vector_store.size,
        }

    def query(
        self,
        question: str,
        user_role: str = "viewer",
        top_k: int = None,
    ) -> Dict[str, Any]:
        """
        Execute a full RAG query pipeline.

        Args:
            question: User's question
            user_role: User's role for RBAC filtering
            top_k: Number of chunks to retrieve

        Returns:
            Structured response with answer, citations, confidence
        """
        query_id = generate_query_id()
        top_k = top_k or settings.TOP_K
        total_start = time.time()

        self.query_logger.log_query_start(query_id, question, user_role)
        obs_metrics.ACTIVE_QUERIES.inc()

        query_metrics = QueryMetrics(query_id=query_id)

        try:
            # Step 1: Get role-based filter
            role_filter = RBACFilter.get_role_filter(user_role)

            # Step 2: Hybrid retrieval (dense + sparse)
            rerank_k = settings.RERANK_TOP_K if self.use_reranker else top_k
            with Timer() as retrieval_timer:
                results = self.hybrid_retriever.retrieve(
                    query=question,
                    top_k=rerank_k,
                    role_filter=role_filter,
                )

            query_metrics.retrieval_latency_ms = retrieval_timer.elapsed_ms

            # Step 3: RBAC safety filter
            results = RBACFilter.filter_results(results, user_role)

            # Step 4: Optional cross-encoder reranking
            if self.reranker and results:
                with Timer() as rerank_timer:
                    results = self.reranker.rerank(question, results, top_k=top_k)
                query_metrics.reranking_latency_ms = rerank_timer.elapsed_ms
            else:
                results = results[:top_k]

            retrieval_ids = [r.get("chunk_id", "") for r in results]
            self.query_logger.log_retrieval(
                query_id, retrieval_ids, retrieval_timer.elapsed_ms
            )

            # Step 5: Build prompt
            prompt_data = self.prompt_builder.build_prompt(question, results)

            # Step 6: Generate answer (local LLaMA 3)
            with Timer() as gen_timer:
                llm_response = self.llm_client.generate(
                    system_prompt=prompt_data["system"],
                    user_prompt=prompt_data["user"],
                )

            query_metrics.generation_latency_ms = gen_timer.elapsed_ms
            query_metrics.prompt_tokens = llm_response["usage"]["prompt_tokens"]
            query_metrics.completion_tokens = llm_response["usage"]["completion_tokens"]
            query_metrics.total_tokens = llm_response["usage"]["total_tokens"]
            query_metrics.num_chunks_retrieved = len(results)

            self.query_logger.log_generation(
                query_id, gen_timer.elapsed_ms, llm_response["usage"]["total_tokens"]
            )

            # Step 7: Hallucination detection
            with Timer() as hal_timer:
                faithfulness = self.hallucination_detector.evaluate(
                    answer=llm_response["content"],
                    context_chunks=results,
                )

            query_metrics.hallucination_check_ms = hal_timer.elapsed_ms

            self.query_logger.log_hallucination_check(
                query_id,
                faithfulness["faithfulness_score"],
                faithfulness["is_faithful"],
            )

            # Step 8: Determine final answer
            if not faithfulness["is_faithful"]:
                final_answer = self.hallucination_detector.get_safe_response()
                confidence = faithfulness["faithfulness_score"]
            else:
                final_answer = llm_response["content"]
                confidence = faithfulness["faithfulness_score"]

            # Step 9: Format response
            total_latency_ms = (time.time() - total_start) * 1000
            query_metrics.total_latency_ms = total_latency_ms

            response = self.response_formatter.format_response(
                raw_answer=final_answer,
                citations=prompt_data["citations"],
                confidence_score=confidence,
                usage=llm_response["usage"],
                latency_ms=total_latency_ms,
            )

            # Add context texts for evaluation
            response["_context_texts"] = [r.get("content", "") for r in results]

            # Step 10: Record metrics
            self.system_metrics.record_query(query_metrics)
            obs_metrics.record_query_metrics(
                user_role=user_role,
                status="success",
                total_latency_s=total_latency_ms / 1000,
                retrieval_latency_s=query_metrics.retrieval_latency_ms / 1000,
                generation_latency_s=query_metrics.generation_latency_ms / 1000,
                faithfulness=confidence,
                prompt_tokens=query_metrics.prompt_tokens,
                completion_tokens=query_metrics.completion_tokens,
            )

            self.query_logger.log_query_complete(
                query_id, total_latency_ms, len(final_answer)
            )
            obs_metrics.ACTIVE_QUERIES.dec()

            return response

        except Exception as e:
            obs_metrics.ACTIVE_QUERIES.dec()
            obs_metrics.QUERY_COUNTER.labels(user_role=user_role, status="error").inc()
            self.query_logger.log_error(query_id, str(e))
            raise
