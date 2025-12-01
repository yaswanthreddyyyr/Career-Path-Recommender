"""
Hybrid Search combining semantic embeddings with BM25 keyword matching.

This approach combines the strengths of:
- Semantic search: Understands meaning and context
- BM25: Precise keyword matching for exact skill names

Formula: final_score = α * semantic_score + (1-α) * bm25_score
"""

import math
import numpy as np
from collections import defaultdict
from typing import Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Result from hybrid search."""
    id: str
    semantic_score: float
    bm25_score: float
    combined_score: float


class BM25:
    """
    BM25 (Best Matching 25) ranking function.
    
    BM25 is a probabilistic retrieval model that considers:
    - Term frequency (TF): How often a term appears in a document
    - Inverse document frequency (IDF): How rare a term is across documents
    - Document length normalization
    
    Formula:
    score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
    
    Where:
    - f(qi, D) = frequency of term qi in document D
    - |D| = length of document D
    - avgdl = average document length
    - k1, b = tuning parameters (typically k1=1.5, b=0.75)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25.
        
        Args:
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Document length normalization (0.75 typical)
        """
        self.k1 = k1
        self.b = b
        
        # Corpus statistics
        self._documents: list[list[str]] = []
        self._doc_ids: list[str] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_freqs: dict[str, int] = defaultdict(int)  # Term -> doc count
        self._term_freqs: list[dict[str, int]] = []  # Doc -> term -> freq
        self._idf_cache: dict[str, float] = {}
        self._n_docs: int = 0
    
    def fit(self, documents: list[list[str]], doc_ids: list[str]):
        """
        Fit BM25 on a corpus of documents.
        
        Args:
            documents: List of tokenized documents (list of word lists)
            doc_ids: List of document IDs
        """
        self._documents = documents
        self._doc_ids = doc_ids
        self._n_docs = len(documents)
        
        # Compute document statistics
        self._doc_lengths = []
        self._term_freqs = []
        self._doc_freqs = defaultdict(int)
        
        for doc in documents:
            # Document length
            self._doc_lengths.append(len(doc))
            
            # Term frequencies in this document
            tf = defaultdict(int)
            seen_terms = set()
            
            for term in doc:
                term_lower = term.lower()
                tf[term_lower] += 1
                
                # Count document frequency (once per document)
                if term_lower not in seen_terms:
                    self._doc_freqs[term_lower] += 1
                    seen_terms.add(term_lower)
            
            self._term_freqs.append(dict(tf))
        
        # Average document length
        self._avg_doc_length = sum(self._doc_lengths) / max(self._n_docs, 1)
        
        # Pre-compute IDF for all terms
        self._idf_cache = {}
        for term, df in self._doc_freqs.items():
            # IDF with smoothing to avoid division by zero
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)
            self._idf_cache[term] = idf
    
    def get_idf(self, term: str) -> float:
        """Get IDF score for a term."""
        term_lower = term.lower()
        return self._idf_cache.get(term_lower, 0.0)
    
    def score(self, query: list[str], doc_idx: int) -> float:
        """
        Compute BM25 score for a query against a document.
        
        Args:
            query: Tokenized query (list of terms)
            doc_idx: Index of the document
        
        Returns:
            BM25 score
        """
        if doc_idx >= self._n_docs:
            return 0.0
        
        score = 0.0
        doc_length = self._doc_lengths[doc_idx]
        term_freqs = self._term_freqs[doc_idx]
        
        for term in query:
            term_lower = term.lower()
            
            # Get term frequency in document
            tf = term_freqs.get(term_lower, 0)
            if tf == 0:
                continue
            
            # Get IDF
            idf = self.get_idf(term_lower)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self._avg_doc_length)
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: list[str], top_k: int = 10) -> list[tuple[str, float]]:
        """
        Search for documents matching the query.
        
        Args:
            query: Tokenized query
            top_k: Number of results to return
        
        Returns:
            List of (doc_id, score) tuples
        """
        scores = []
        
        for idx in range(self._n_docs):
            score = self.score(query, idx)
            if score > 0:
                scores.append((self._doc_ids[idx], score))
        
        # Sort by score descending
        scores.sort(key=lambda x: -x[1])
        
        return scores[:top_k]


class HybridSearch:
    """
    Hybrid search combining semantic embeddings with BM25.
    
    This approach addresses limitations of pure semantic search:
    - Exact skill name matching (BM25 excels)
    - Semantic understanding of context (embeddings excel)
    
    The combination provides better recall and precision.
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """
        Initialize hybrid search.
        
        Args:
            alpha: Weight for semantic scores (1-alpha for BM25)
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
        """
        self.alpha = alpha
        self.bm25 = BM25(k1=bm25_k1, b=bm25_b)
        
        self._semantic_index = None
        self._doc_ids: list[str] = []
        self._embeddings: Optional[np.ndarray] = None
    
    def fit(
        self,
        documents: list[str],
        doc_ids: list[str],
        embeddings: np.ndarray,
    ):
        """
        Fit the hybrid search on documents.
        
        Args:
            documents: List of document texts
            doc_ids: List of document IDs
            embeddings: Pre-computed embeddings for documents
        """
        self._doc_ids = doc_ids
        self._embeddings = embeddings
        
        # Tokenize documents for BM25
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25.fit(tokenized_docs, doc_ids)
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        # Split on whitespace and punctuation, lowercase
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _normalize_scores(self, scores: list[tuple[str, float]]) -> dict[str, float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return {}
        
        max_score = max(s[1] for s in scores)
        min_score = min(s[1] for s in scores)
        
        if max_score == min_score:
            return {s[0]: 1.0 for s in scores}
        
        return {
            s[0]: (s[1] - min_score) / (max_score - min_score)
            for s in scores
        }
    
    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query_text: Query text for BM25
            query_embedding: Query embedding for semantic search
            top_k: Number of results
        
        Returns:
            List of SearchResult objects
        """
        # BM25 search
        query_tokens = self._tokenize(query_text)
        bm25_results = self.bm25.search(query_tokens, top_k=top_k * 2)
        bm25_scores = self._normalize_scores(bm25_results)
        
        # Semantic search (cosine similarity with normalized vectors)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = self._embeddings / np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        similarities = np.dot(embeddings_norm, query_embedding)
        
        # Get top semantic results
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]
        semantic_results = [
            (self._doc_ids[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        semantic_scores = self._normalize_scores(semantic_results)
        
        # Combine scores
        all_ids = set(bm25_scores.keys()) | set(semantic_scores.keys())
        
        results = []
        for doc_id in all_ids:
            sem_score = semantic_scores.get(doc_id, 0.0)
            bm25_score = bm25_scores.get(doc_id, 0.0)
            combined = self.alpha * sem_score + (1 - self.alpha) * bm25_score
            
            results.append(SearchResult(
                id=doc_id,
                semantic_score=sem_score,
                bm25_score=bm25_score,
                combined_score=combined,
            ))
        
        # Sort by combined score
        results.sort(key=lambda x: -x.combined_score)
        
        return results[:top_k]


class SkillNormalizer:
    """
    Normalize skills to handle variations and synonyms.
    
    This improves matching by:
    - Handling case variations (python vs Python)
    - Expanding abbreviations (ML -> Machine Learning)
    - Grouping related skills
    """
    
    # Skill synonyms and abbreviations
    SKILL_SYNONYMS = {
        # Programming languages
        "python": ["python", "py", "python3"],
        "javascript": ["javascript", "js", "ecmascript"],
        "typescript": ["typescript", "ts"],
        
        # AI/ML
        "machine learning": ["machine learning", "ml", "machine-learning"],
        "deep learning": ["deep learning", "dl", "deep-learning"],
        "artificial intelligence": ["artificial intelligence", "ai"],
        "natural language processing": ["natural language processing", "nlp"],
        "computer vision": ["computer vision", "cv", "image processing"],
        
        # Frameworks
        "tensorflow": ["tensorflow", "tf"],
        "pytorch": ["pytorch", "torch"],
        "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
        
        # Cloud
        "amazon web services": ["amazon web services", "aws", "amazon aws"],
        "google cloud platform": ["google cloud platform", "gcp", "google cloud"],
        "microsoft azure": ["microsoft azure", "azure"],
        
        # DevOps
        "kubernetes": ["kubernetes", "k8s"],
        "continuous integration": ["continuous integration", "ci"],
        "continuous deployment": ["continuous deployment", "cd"],
        "ci/cd": ["ci/cd", "cicd", "ci-cd"],
        
        # Data
        "structured query language": ["structured query language", "sql"],
        "nosql": ["nosql", "no-sql", "non-relational"],
    }
    
    def __init__(self):
        """Initialize the skill normalizer."""
        # Build reverse lookup: variation -> canonical form
        self._variation_to_canonical: dict[str, str] = {}
        
        for canonical, variations in self.SKILL_SYNONYMS.items():
            for variation in variations:
                self._variation_to_canonical[variation.lower()] = canonical
    
    def normalize(self, skill: str) -> str:
        """
        Normalize a skill name to its canonical form.
        
        Args:
            skill: Skill name to normalize
        
        Returns:
            Canonical skill name
        """
        skill_lower = skill.lower().strip()
        return self._variation_to_canonical.get(skill_lower, skill_lower)
    
    def expand(self, skill: str) -> list[str]:
        """
        Expand a skill to include all known variations.
        
        Args:
            skill: Skill name
        
        Returns:
            List of skill variations
        """
        canonical = self.normalize(skill)
        
        if canonical in self.SKILL_SYNONYMS:
            return self.SKILL_SYNONYMS[canonical]
        
        return [skill.lower()]
    
    def match_skills(
        self,
        user_skills: list[str],
        required_skills: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Match user skills against required skills with normalization.
        
        Args:
            user_skills: List of user's skills
            required_skills: List of required skills
        
        Returns:
            Tuple of (matching_skills, missing_skills)
        """
        # Normalize user skills
        user_normalized = {self.normalize(s) for s in user_skills}
        
        matching = []
        missing = []
        
        for skill in required_skills:
            skill_normalized = self.normalize(skill)
            
            if skill_normalized in user_normalized:
                matching.append(skill)
            else:
                missing.append(skill)
        
        return matching, missing


class QueryExpander:
    """
    Expand queries with related terms to improve recall.
    
    Uses skill relationships and synonyms to find more relevant results.
    """
    
    # Related skill groups
    SKILL_GROUPS = {
        "data_science": [
            "Python", "R", "SQL", "Statistics", "Machine Learning",
            "Data Visualization", "Pandas", "NumPy", "Scikit-learn"
        ],
        "machine_learning": [
            "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
            "Neural Networks", "NLP", "Computer Vision", "Scikit-learn"
        ],
        "web_development": [
            "JavaScript", "TypeScript", "React", "Node.js", "HTML/CSS",
            "REST APIs", "GraphQL", "MongoDB", "PostgreSQL"
        ],
        "devops": [
            "Docker", "Kubernetes", "AWS", "CI/CD", "Terraform",
            "Linux", "Ansible", "Jenkins", "Git"
        ],
        "cloud": [
            "AWS", "GCP", "Azure", "Cloud Computing", "Serverless",
            "Docker", "Kubernetes", "Terraform"
        ],
    }
    
    def __init__(self, expansion_weight: float = 0.5):
        """
        Initialize query expander.
        
        Args:
            expansion_weight: Weight for expanded terms (0-1)
        """
        self.expansion_weight = expansion_weight
        
        # Build skill -> group mapping
        self._skill_to_groups: dict[str, list[str]] = defaultdict(list)
        for group, skills in self.SKILL_GROUPS.items():
            for skill in skills:
                self._skill_to_groups[skill.lower()].append(group)
    
    def expand_skills(self, skills: list[str], max_expansion: int = 5) -> list[str]:
        """
        Expand a list of skills with related skills.
        
        Args:
            skills: Original skills
            max_expansion: Maximum number of skills to add
        
        Returns:
            Expanded skills list
        """
        expanded = set(s.lower() for s in skills)
        
        # Find related skills from the same groups
        related_skills: dict[str, int] = defaultdict(int)
        
        for skill in skills:
            skill_lower = skill.lower()
            groups = self._skill_to_groups.get(skill_lower, [])
            
            for group in groups:
                for related in self.SKILL_GROUPS[group]:
                    if related.lower() not in expanded:
                        related_skills[related] += 1
        
        # Sort by frequency and take top N
        sorted_related = sorted(
            related_skills.items(),
            key=lambda x: -x[1]
        )[:max_expansion]
        
        return list(expanded) + [s[0] for s in sorted_related]
    
    def expand_query(self, query: str, user_skills: list[str]) -> str:
        """
        Expand a query with related terms.
        
        Args:
            query: Original query
            user_skills: User's skills for context
        
        Returns:
            Expanded query
        """
        expanded_skills = self.expand_skills(user_skills, max_expansion=3)
        
        # Add expanded skills to query
        expansion = " ".join(expanded_skills)
        
        return f"{query} {expansion}"

