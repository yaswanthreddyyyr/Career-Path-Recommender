"""
Embedding engine using SentenceTransformers for semantic text representation.

Uses the all-MiniLM-L6-v2 model for generating 384-dimensional embeddings.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union

from sentence_transformers import SentenceTransformer

from src.models.schemas import UserProfile, JobPosting, Course
from src.data.preprocessor import TextPreprocessor


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


class EmbeddingEngine:
    """Generate semantic embeddings using SentenceTransformers."""
    
    # Default model - fast and efficient for semantic similarity
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384  # Dimension of all-MiniLM-L6-v2 embeddings
    
    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the embedding engine.
        
        Args:
            model_name: Name of the SentenceTransformer model to use.
            cache_dir: Directory to cache embeddings. Defaults to data/processed.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or DATA_PROCESSED
        self.preprocessor = TextPreprocessor()
        
        # Lazy load the model
        self._model: Optional[SentenceTransformer] = None
        
        # Cache for embeddings
        self._embedding_cache: dict[str, np.ndarray] = {}
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the SentenceTransformer model, loading if necessary."""
        if self._model is None:
            print(f"Loading SentenceTransformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model
    
    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed.
            normalize: Whether to L2-normalize the embedding.
        
        Returns:
            Embedding vector as numpy array.
        """
        if not text or not text.strip():
            return np.zeros(self.EMBEDDING_DIM)
        
        # Check cache
        cache_key = f"{text}_{normalize}"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Generate embedding
        embedding = self.model.encode(text, normalize_embeddings=normalize)
        embedding = np.array(embedding, dtype=np.float32)
        
        # Cache result
        self._embedding_cache[cache_key] = embedding
        
        return embedding
    
    def embed_texts(
        self, 
        texts: list[str], 
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            normalize: Whether to L2-normalize embeddings.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.
        
        Returns:
            Embeddings matrix of shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.zeros((0, self.EMBEDDING_DIM))
        
        # Filter empty texts
        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            return np.zeros((len(texts), self.EMBEDDING_DIM))
        
        # Generate embeddings
        embeddings = self.model.encode(
            valid_texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Create full matrix with zeros for empty texts
        full_embeddings = np.zeros((len(texts), self.EMBEDDING_DIM), dtype=np.float32)
        for i, idx in enumerate(valid_indices):
            full_embeddings[idx] = embeddings[i]
        
        return full_embeddings
    
    def embed_user_profile(self, profile: UserProfile) -> np.ndarray:
        """
        Generate embedding for a user profile.
        
        Args:
            profile: UserProfile object.
        
        Returns:
            Embedding vector.
        """
        # Create text representation
        text = self.preprocessor.create_user_profile_text(
            skills=profile.skills,
            current_role=profile.current_role,
            education=profile.education,
            experience_years=profile.experience_years,
            career_goals=profile.career_goals,
        )
        
        return self.embed_text(text)
    
    def embed_job(self, job: JobPosting) -> np.ndarray:
        """
        Generate embedding for a job posting.
        
        Args:
            job: JobPosting object.
        
        Returns:
            Embedding vector.
        """
        text = self.preprocessor.create_job_text(
            title=job.title,
            description=job.description,
            required_skills=job.required_skills,
            preferred_skills=job.preferred_skills,
            experience_level=job.experience_level,
        )
        
        return self.embed_text(text)
    
    def embed_course(self, course: Course) -> np.ndarray:
        """
        Generate embedding for a course.
        
        Args:
            course: Course object.
        
        Returns:
            Embedding vector.
        """
        text = self.preprocessor.create_course_text(
            title=course.title,
            description=course.description,
            skills_taught=course.skills_taught,
            difficulty=course.difficulty,
        )
        
        return self.embed_text(text)
    
    def embed_jobs(
        self, 
        jobs: list[JobPosting],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple job postings.
        
        Args:
            jobs: List of JobPosting objects.
            show_progress: Whether to show progress bar.
        
        Returns:
            Embeddings matrix of shape (n_jobs, embedding_dim).
        """
        texts = []
        for job in jobs:
            text = self.preprocessor.create_job_text(
                title=job.title,
                description=job.description,
                required_skills=job.required_skills,
                preferred_skills=job.preferred_skills,
                experience_level=job.experience_level,
            )
            texts.append(text)
        
        return self.embed_texts(texts, show_progress=show_progress)
    
    def embed_courses(
        self, 
        courses: list[Course],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple courses.
        
        Args:
            courses: List of Course objects.
            show_progress: Whether to show progress bar.
        
        Returns:
            Embeddings matrix of shape (n_courses, embedding_dim).
        """
        texts = []
        for course in courses:
            text = self.preprocessor.create_course_text(
                title=course.title,
                description=course.description,
                skills_taught=course.skills_taught,
                difficulty=course.difficulty,
            )
            texts.append(text)
        
        return self.embed_texts(texts, show_progress=show_progress)
    
    def embed_skills(self, skills: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of skills.
        
        Args:
            skills: List of skill names.
        
        Returns:
            Embeddings matrix of shape (n_skills, embedding_dim).
        """
        return self.embed_texts(skills, show_progress=False)
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray, 
        name: str,
        ids: Optional[list[str]] = None,
    ) -> Path:
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Embeddings matrix to save.
            name: Name for the embeddings file.
            ids: Optional list of IDs corresponding to each embedding.
        
        Returns:
            Path to saved embeddings.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        embeddings_path = self.cache_dir / f"{name}_embeddings.npy"
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to {embeddings_path}")
        
        # Save IDs if provided
        if ids:
            ids_path = self.cache_dir / f"{name}_ids.npy"
            np.save(ids_path, np.array(ids))
            print(f"Saved IDs to {ids_path}")
        
        return embeddings_path
    
    def load_embeddings(self, name: str) -> tuple[np.ndarray, Optional[list[str]]]:
        """
        Load embeddings from disk.
        
        Args:
            name: Name of the embeddings file.
        
        Returns:
            Tuple of (embeddings matrix, optional list of IDs).
        """
        embeddings_path = self.cache_dir / f"{name}_embeddings.npy"
        ids_path = self.cache_dir / f"{name}_ids.npy"
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
        
        embeddings = np.load(embeddings_path)
        
        ids = None
        if ids_path.exists():
            ids = np.load(ids_path, allow_pickle=True).tolist()
        
        return embeddings, ids
    
    def clear_cache(self):
        """Clear the in-memory embedding cache."""
        self._embedding_cache.clear()


def compute_cosine_similarity(
    embedding1: np.ndarray, 
    embedding2: np.ndarray
) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector.
        embedding2: Second embedding vector.
    
    Returns:
        Cosine similarity score (-1 to 1).
    """
    # Handle zero vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


def compute_similarity_matrix(
    embeddings1: np.ndarray, 
    embeddings2: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise cosine similarities between two sets of embeddings.
    
    Args:
        embeddings1: First embeddings matrix (n1, dim).
        embeddings2: Second embeddings matrix (n2, dim).
    
    Returns:
        Similarity matrix of shape (n1, n2).
    """
    # Normalize embeddings
    norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    # Avoid division by zero
    norms1 = np.where(norms1 == 0, 1, norms1)
    norms2 = np.where(norms2 == 0, 1, norms2)
    
    normalized1 = embeddings1 / norms1
    normalized2 = embeddings2 / norms2
    
    return np.dot(normalized1, normalized2.T)

