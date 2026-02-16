"""
FAISS-based vector store with GPU support.
Replaces ChromaDB with direct FAISS index management.
Adaptive index selection: Flat → IVFFlat → IVFPQ based on corpus size.
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from config import settings


class VectorStore:
    """
    FAISS vector index with metadata store.

    Since FAISS has no native metadata support, we maintain a parallel
    metadata dict keyed by integer FAISS ID.

    Index selection:
      < 50k vectors  → IndexFlatIP (exact brute-force)
      < 500k vectors → IndexIVFFlat (approximate, nprobe=64)
      ≥ 500k vectors → IndexIVFPQ (compressed, nprobe=128)
    """

    def __init__(
        self,
        dimension: int = None,
        index_path: str = None,
        use_gpu: bool = None,
    ):
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.use_gpu = use_gpu if use_gpu is not None else settings.FAISS_USE_GPU

        self._index: Optional[faiss.Index] = None
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._doc_map: Dict[str, int] = {}  # chunk_id → faiss_id
        self._next_id: int = 0
        self._gpu_res = None

        # Try to load existing index
        self._load()

    def _load(self):
        """Load FAISS index and metadata from disk."""
        index_file = os.path.join(self.index_path, "index.faiss")
        meta_file = os.path.join(self.index_path, "metadata.json")

        if os.path.exists(index_file) and os.path.exists(meta_file):
            self._index = faiss.read_index(index_file)
            with open(meta_file, "r") as f:
                stored = json.load(f)
            # JSON keys are strings — convert to int
            self._metadata = {int(k): v for k, v in stored.get("metadata", {}).items()}
            self._doc_map = stored.get("doc_map", {})
            self._next_id = stored.get("next_id", 0)

            if self.use_gpu:
                self._move_to_gpu()

    def _save(self):
        """Persist FAISS index and metadata to disk."""
        os.makedirs(self.index_path, exist_ok=True)
        index_file = os.path.join(self.index_path, "index.faiss")
        meta_file = os.path.join(self.index_path, "metadata.json")

        # If on GPU, copy back to CPU for serialization
        index_to_save = self._index
        if self.use_gpu and self._index is not None:
            index_to_save = faiss.index_gpu_to_cpu(self._index)

        if index_to_save is not None:
            faiss.write_index(index_to_save, index_file)

        with open(meta_file, "w") as f:
            json.dump({
                "metadata": {str(k): v for k, v in self._metadata.items()},
                "doc_map": self._doc_map,
                "next_id": self._next_id,
            }, f, indent=2)

    def _move_to_gpu(self):
        """Move FAISS index to GPU."""
        try:
            if self._gpu_res is None:
                self._gpu_res = faiss.StandardGpuResources()
                self._gpu_res.setTempMemory(256 * 1024 * 1024)  # 256MB temp buffer

            self._index = faiss.index_cpu_to_gpu(self._gpu_res, 0, self._index)
        except Exception:
            # GPU not available — fall back to CPU
            self.use_gpu = False

    def _create_index(self, num_vectors: int = 0) -> faiss.Index:
        """Create an appropriate FAISS index based on expected corpus size."""
        if num_vectors < 50_000:
            # Exact search — perfect recall, O(n) query time
            index = faiss.IndexFlatIP(self.dimension)
        elif num_vectors < 500_000:
            # Approximate search — ~98% recall at nprobe=64
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(
                quantizer, self.dimension, 4096, faiss.METRIC_INNER_PRODUCT
            )
            index.nprobe = 64
        else:
            # Compressed — ~95% recall, 4x memory savings
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFPQ(
                quantizer, self.dimension, 4096, 64, 8
            )
            index.nprobe = 128

        return index

    def add_vectors(
        self,
        chunk_ids: List[str],
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """
        Add document vectors to the FAISS index.

        Args:
            chunk_ids: Unique chunk identifiers
            embeddings: (N, D) float32 array of L2-normalized embeddings
            documents: Raw text of each chunk
            metadatas: Metadata dicts for each chunk

        Returns:
            Total number of vectors in index
        """
        if len(chunk_ids) == 0:
            return self.size

        embeddings = np.array(embeddings, dtype=np.float32)

        # Initialize index if needed
        if self._index is None:
            self._index = self._create_index(len(chunk_ids))

        # Train IVF index if needed and not yet trained
        if hasattr(self._index, 'is_trained') and not self._index.is_trained:
            if len(embeddings) >= 256:  # Minimum for IVF training
                # For GPU index, need CPU copy for training
                cpu_index = self._index
                if self.use_gpu:
                    cpu_index = faiss.index_gpu_to_cpu(self._index)
                cpu_index.train(embeddings)
                if self.use_gpu:
                    self._index = faiss.index_cpu_to_gpu(self._gpu_res, 0, cpu_index)
                else:
                    self._index = cpu_index
            else:
                # Not enough vectors for IVF — fall back to Flat
                self._index = faiss.IndexFlatIP(self.dimension)

        # Assign FAISS IDs and store metadata
        faiss_ids = []
        for i, chunk_id in enumerate(chunk_ids):
            fid = self._next_id
            self._next_id += 1
            faiss_ids.append(fid)
            self._doc_map[chunk_id] = fid
            self._metadata[fid] = {
                "chunk_id": chunk_id,
                "content": documents[i] if i < len(documents) else "",
                **(metadatas[i] if i < len(metadatas) else {}),
            }

        # Add to index
        ids_array = np.array(faiss_ids, dtype=np.int64)

        if isinstance(self._index, faiss.IndexFlatIP):
            # Flat index doesn't support add_with_ids, use IDMap wrapper
            if not isinstance(self._index, faiss.IndexIDMap):
                self._index = faiss.IndexIDMap(self._index)
                if self.use_gpu:
                    self._move_to_gpu()
            self._index.add_with_ids(embeddings, ids_array)
        else:
            self._index.add_with_ids(embeddings, ids_array)

        self._save()
        return self.size

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        role_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for nearest neighbors.

        Args:
            query_vector: (D,) float32 query embedding
            top_k: Number of results to return
            role_filter: If set, only return results with matching role_access

        Returns:
            List of dicts with chunk_id, content, metadata, score
        """
        if self._index is None or self.size == 0:
            return []

        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        # Fetch extra results for post-filtering
        fetch_k = top_k * 3 if role_filter else top_k
        fetch_k = min(fetch_k, self.size)

        scores, ids = self._index.search(query_vector, fetch_k)

        results = []
        for score, fid in zip(scores[0], ids[0]):
            if fid == -1:  # FAISS returns -1 for empty slots
                continue

            meta = self._metadata.get(int(fid), {})

            # Role-based filtering
            if role_filter:
                doc_roles = meta.get("role_access", "viewer")
                if isinstance(doc_roles, str):
                    doc_roles = [r.strip() for r in doc_roles.split(",")]
                if not any(r in role_filter for r in doc_roles):
                    continue

            results.append({
                "chunk_id": meta.get("chunk_id", ""),
                "content": meta.get("content", ""),
                "metadata": meta,
                "score": float(score),
            })

            if len(results) >= top_k:
                break

        return results

    def get_collection_count(self, collection_name: str = "text_chunks") -> int:
        """Get number of vectors in the index (ChromaDB-compatible interface)."""
        return self.size

    @property
    def size(self) -> int:
        """Current number of vectors in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    def reset(self):
        """Clear the index and metadata."""
        self._index = None
        self._metadata = {}
        self._doc_map = {}
        self._next_id = 0
        self._save()
