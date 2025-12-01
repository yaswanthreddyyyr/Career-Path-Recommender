"""
FAISS-based similarity search for efficient vector retrieval.

Provides fast approximate nearest neighbor search using cosine similarity.
"""

import numpy as np
import faiss
from pathlib import Path
from typing import Optional, Union


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


class SimilaritySearch:
    """
    Fast similarity search using FAISS.
    
    Supports both exact and approximate nearest neighbor search.
    """
    
    def __init__(
        self, 
        dimension: int = 384,
        index_type: str = "flat",
        use_gpu: bool = False,
    ):
        """
        Initialize the similarity search engine.
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2).
            index_type: Type of FAISS index ('flat' for exact, 'ivf' for approximate).
            use_gpu: Whether to use GPU acceleration (requires faiss-gpu).
        """
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        
        # Index storage
        self._index: Optional[faiss.Index] = None
        self._ids: list[str] = []
        self._embeddings: Optional[np.ndarray] = None
        
        # Create the index
        self._create_index()
    
    def _create_index(self):
        """Create the FAISS index based on configuration."""
        if self.index_type == "flat":
            # Exact search using inner product (cosine similarity with normalized vectors)
            self._index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            # Approximate search using IVF
            quantizer = faiss.IndexFlatIP(self.dimension)
            # Use 100 clusters by default (adjust based on data size)
            self._index = faiss.IndexIVFFlat(quantizer, self.dimension, 100, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # GPU acceleration if requested and available
        if self.use_gpu:
            try:
                self._index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), 
                    0, 
                    self._index
                )
            except Exception:
                print("GPU not available, falling back to CPU")
    
    def add_embeddings(
        self, 
        embeddings: np.ndarray, 
        ids: Optional[list[str]] = None
    ):
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Embeddings matrix of shape (n, dimension).
            ids: Optional list of IDs for each embedding.
        """
        if len(embeddings) == 0:
            return
        
        # Ensure correct dtype and shape
        embeddings = np.array(embeddings, dtype=np.float32)
        
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Normalize for cosine similarity
        embeddings = self._normalize(embeddings)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self._index.is_trained:
            self._index.train(embeddings)
        
        # Add to index
        self._index.add(embeddings)
        
        # Store IDs
        if ids:
            self._ids.extend(ids)
        else:
            # Generate sequential IDs
            start_id = len(self._ids)
            self._ids.extend([str(i) for i in range(start_id, start_id + len(embeddings))])
        
        # Store embeddings for potential re-indexing
        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """
        Search for similar items.
        
        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            threshold: Minimum similarity threshold.
        
        Returns:
            List of (id, similarity_score) tuples.
        """
        if self._index.ntotal == 0:
            return []
        
        # Ensure correct shape
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query
        query_embedding = self._normalize(query_embedding)
        
        # Search
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query_embedding, k)
        
        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and dist >= threshold:  # FAISS returns -1 for invalid results
                item_id = self._ids[idx] if idx < len(self._ids) else str(idx)
                results.append((item_id, float(dist)))
        
        return results
    
    def search_batch(
        self, 
        query_embeddings: np.ndarray, 
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[list[tuple[str, float]]]:
        """
        Search for similar items for multiple queries.
        
        Args:
            query_embeddings: Query embeddings matrix of shape (n_queries, dimension).
            top_k: Number of results per query.
            threshold: Minimum similarity threshold.
        
        Returns:
            List of result lists, one per query.
        """
        if self._index.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]
        
        # Ensure correct shape and type
        query_embeddings = np.array(query_embeddings, dtype=np.float32)
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Normalize queries
        query_embeddings = self._normalize(query_embeddings)
        
        # Search
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query_embeddings, k)
        
        # Convert to results
        all_results = []
        for query_distances, query_indices in zip(distances, indices):
            results = []
            for dist, idx in zip(query_distances, query_indices):
                if idx >= 0 and dist >= threshold:
                    item_id = self._ids[idx] if idx < len(self._ids) else str(idx)
                    results.append((item_id, float(dist)))
            all_results.append(results)
        
        return all_results
    
    def get_embedding_by_id(self, item_id: str) -> Optional[np.ndarray]:
        """
        Get an embedding by its ID.
        
        Args:
            item_id: ID of the item.
        
        Returns:
            Embedding vector or None if not found.
        """
        if item_id in self._ids and self._embeddings is not None:
            idx = self._ids.index(item_id)
            return self._embeddings[idx]
        return None
    
    def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding.
            embedding2: Second embedding.
        
        Returns:
            Cosine similarity score (0 to 1 for normalized vectors).
        """
        embedding1 = self._normalize(np.array(embedding1).reshape(1, -1))[0]
        embedding2 = self._normalize(np.array(embedding2).reshape(1, -1))[0]
        
        return float(np.dot(embedding1, embedding2))
    
    def save(self, path: Union[str, Path], name: str = "index"):
        """
        Save the index and metadata to disk.
        
        Args:
            path: Directory to save to.
            name: Base name for the files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / f"{name}.faiss"
        faiss.write_index(self._index, str(index_path))
        
        # Save IDs
        ids_path = path / f"{name}_ids.npy"
        np.save(ids_path, np.array(self._ids))
        
        # Save embeddings
        if self._embeddings is not None:
            embeddings_path = path / f"{name}_embeddings.npy"
            np.save(embeddings_path, self._embeddings)
        
        print(f"Saved index to {path}")
    
    def load(self, path: Union[str, Path], name: str = "index"):
        """
        Load the index and metadata from disk.
        
        Args:
            path: Directory to load from.
            name: Base name for the files.
        """
        path = Path(path)
        
        # Load FAISS index
        index_path = path / f"{name}.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        self._index = faiss.read_index(str(index_path))
        
        # Load IDs
        ids_path = path / f"{name}_ids.npy"
        if ids_path.exists():
            self._ids = np.load(ids_path, allow_pickle=True).tolist()
        
        # Load embeddings
        embeddings_path = path / f"{name}_embeddings.npy"
        if embeddings_path.exists():
            self._embeddings = np.load(embeddings_path)
        
        print(f"Loaded index from {path}")
    
    @property
    def size(self) -> int:
        """Get the number of items in the index."""
        return self._index.ntotal
    
    def clear(self):
        """Clear the index and reset."""
        self._ids = []
        self._embeddings = None
        self._create_index()


class MultiIndexSearch:
    """
    Manage multiple similarity search indices (e.g., jobs and courses).
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize multi-index search.
        
        Args:
            dimension: Embedding dimension.
        """
        self.dimension = dimension
        self._indices: dict[str, SimilaritySearch] = {}
    
    def create_index(self, name: str, index_type: str = "flat") -> SimilaritySearch:
        """
        Create a new named index.
        
        Args:
            name: Name for the index (e.g., 'jobs', 'courses').
            index_type: Type of FAISS index.
        
        Returns:
            The created SimilaritySearch instance.
        """
        self._indices[name] = SimilaritySearch(
            dimension=self.dimension,
            index_type=index_type,
        )
        return self._indices[name]
    
    def get_index(self, name: str) -> Optional[SimilaritySearch]:
        """Get an index by name."""
        return self._indices.get(name)
    
    def add_to_index(
        self, 
        name: str, 
        embeddings: np.ndarray, 
        ids: Optional[list[str]] = None
    ):
        """Add embeddings to a named index."""
        if name not in self._indices:
            self.create_index(name)
        self._indices[name].add_embeddings(embeddings, ids)
    
    def search(
        self, 
        name: str, 
        query_embedding: np.ndarray, 
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Search in a named index."""
        if name not in self._indices:
            return []
        return self._indices[name].search(query_embedding, top_k)
    
    def save_all(self, path: Union[str, Path]):
        """Save all indices to disk."""
        path = Path(path)
        for name, index in self._indices.items():
            index.save(path, name)
    
    def load_all(self, path: Union[str, Path], names: list[str]):
        """Load multiple indices from disk."""
        path = Path(path)
        for name in names:
            if name not in self._indices:
                self._indices[name] = SimilaritySearch(self.dimension)
            try:
                self._indices[name].load(path, name)
            except FileNotFoundError:
                print(f"Index {name} not found, skipping")

