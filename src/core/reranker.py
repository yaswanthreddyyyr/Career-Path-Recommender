"""
Cross-Encoder Re-ranking for improved recommendation accuracy.

Two-stage retrieval:
1. First stage: Fast retrieval with bi-encoder (FAISS)
2. Second stage: Accurate re-ranking with cross-encoder

Cross-encoders are more accurate because they see both texts together,
allowing for better understanding of their relationship.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class RerankResult:
    """Result after re-ranking."""
    id: str
    original_score: float
    rerank_score: float
    final_score: float


class CrossEncoderReranker:
    """
    Re-rank candidates using a cross-encoder model.
    
    Cross-encoders process query and document together:
    [CLS] query [SEP] document [SEP] -> relevance score
    
    This allows for more nuanced understanding but is slower,
    so we only apply it to top candidates from initial retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_model: bool = True,
    ):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
            use_model: Whether to load the actual model (False for lightweight mode)
        """
        self.model_name = model_name
        self.use_model = use_model
        self._model = None
        
    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None and self.use_model:
            try:
                from sentence_transformers import CrossEncoder
                print(f"Loading cross-encoder: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                print("CrossEncoder not available, using fallback scoring")
                self.use_model = False
        return self._model
    
    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str, float]],  # (id, text, original_score)
        top_k: int = 10,
        alpha: float = 0.6,  # Weight for rerank score vs original
    ) -> list[RerankResult]:
        """
        Re-rank candidates using cross-encoder.
        
        Args:
            query: Query text
            candidates: List of (id, document_text, original_score) tuples
            top_k: Number of results to return
            alpha: Weight for cross-encoder score (1-alpha for original)
        
        Returns:
            Re-ranked results
        """
        if not candidates:
            return []
        
        if self.use_model and self.model is not None:
            # Use cross-encoder for scoring
            pairs = [(query, doc_text) for _, doc_text, _ in candidates]
            rerank_scores = self.model.predict(pairs)
            
            # Normalize scores to [0, 1]
            min_score = min(rerank_scores)
            max_score = max(rerank_scores)
            if max_score > min_score:
                rerank_scores = (rerank_scores - min_score) / (max_score - min_score)
            else:
                rerank_scores = np.ones(len(rerank_scores))
        else:
            # Fallback: use lexical overlap scoring
            rerank_scores = self._fallback_scoring(query, candidates)
        
        # Combine scores
        results = []
        for i, (doc_id, _, orig_score) in enumerate(candidates):
            rerank = float(rerank_scores[i])
            final = alpha * rerank + (1 - alpha) * orig_score
            
            results.append(RerankResult(
                id=doc_id,
                original_score=orig_score,
                rerank_score=rerank,
                final_score=final,
            ))
        
        # Sort by final score
        results.sort(key=lambda x: -x.final_score)
        
        return results[:top_k]
    
    def _fallback_scoring(
        self,
        query: str,
        candidates: list[tuple[str, str, float]],
    ) -> np.ndarray:
        """
        Fallback scoring using lexical features.
        
        Uses:
        - Jaccard similarity of words
        - Term overlap ratio
        - Exact phrase matching bonus
        """
        query_words = set(query.lower().split())
        scores = []
        
        for _, doc_text, _ in candidates:
            doc_words = set(doc_text.lower().split())
            
            # Jaccard similarity
            intersection = len(query_words & doc_words)
            union = len(query_words | doc_words)
            jaccard = intersection / max(union, 1)
            
            # Query coverage
            coverage = intersection / max(len(query_words), 1)
            
            # Phrase match bonus
            phrase_bonus = 0.0
            query_lower = query.lower()
            doc_lower = doc_text.lower()
            
            # Check for consecutive word matches
            query_bigrams = [
                f"{w1} {w2}" 
                for w1, w2 in zip(query.lower().split()[:-1], query.lower().split()[1:])
            ]
            for bigram in query_bigrams:
                if bigram in doc_lower:
                    phrase_bonus += 0.1
            
            # Combined score
            score = 0.4 * jaccard + 0.4 * coverage + min(0.2, phrase_bonus)
            scores.append(score)
        
        return np.array(scores)


class TFIDFSkillWeighter:
    """
    Weight skills using TF-IDF to identify important distinguishing skills.
    
    Common skills (Python, SQL) get lower weight.
    Rare, specialized skills get higher weight.
    """
    
    def __init__(self):
        """Initialize the skill weighter."""
        self._skill_idf: dict[str, float] = {}
        self._skill_counts: dict[str, int] = {}
        self._n_docs: int = 0
    
    def fit(self, skill_lists: list[list[str]]):
        """
        Fit TF-IDF on a corpus of skill lists.
        
        Args:
            skill_lists: List of skill lists (one per job/user)
        """
        self._n_docs = len(skill_lists)
        self._skill_counts = {}
        
        # Count document frequency for each skill
        for skills in skill_lists:
            seen = set()
            for skill in skills:
                skill_lower = skill.lower()
                if skill_lower not in seen:
                    self._skill_counts[skill_lower] = self._skill_counts.get(skill_lower, 0) + 1
                    seen.add(skill_lower)
        
        # Compute IDF
        import math
        self._skill_idf = {}
        for skill, count in self._skill_counts.items():
            # IDF with smoothing
            self._skill_idf[skill] = math.log((self._n_docs + 1) / (count + 1)) + 1
    
    def get_weight(self, skill: str) -> float:
        """
        Get TF-IDF weight for a skill.
        
        Args:
            skill: Skill name
        
        Returns:
            Weight (higher = more important/rare)
        """
        return self._skill_idf.get(skill.lower(), 1.0)
    
    def weight_skills(self, skills: list[str]) -> dict[str, float]:
        """
        Get weights for a list of skills.
        
        Args:
            skills: List of skill names
        
        Returns:
            Dictionary mapping skill to weight
        """
        return {skill: self.get_weight(skill) for skill in skills}
    
    def weighted_match_score(
        self,
        user_skills: list[str],
        required_skills: list[str],
    ) -> float:
        """
        Compute weighted skill match score.
        
        Args:
            user_skills: User's skills
            required_skills: Required skills for a job
        
        Returns:
            Weighted match score (0-1)
        """
        if not required_skills:
            return 1.0
        
        user_skills_lower = {s.lower() for s in user_skills}
        
        total_weight = 0.0
        matched_weight = 0.0
        
        for skill in required_skills:
            weight = self.get_weight(skill)
            total_weight += weight
            
            if skill.lower() in user_skills_lower:
                matched_weight += weight
        
        return matched_weight / max(total_weight, 0.001)


class MultiStageRetriever:
    """
    Multi-stage retrieval pipeline combining fast retrieval with accurate re-ranking.
    
    Pipeline:
    1. Fast candidate retrieval (bi-encoder + FAISS) - O(log n)
    2. Query expansion for better recall
    3. Cross-encoder re-ranking for precision - O(k)
    4. Skill-based scoring adjustment
    """
    
    def __init__(
        self,
        use_reranker: bool = True,
        use_skill_weighting: bool = True,
        retrieval_k: int = 50,  # Candidates for re-ranking
        final_k: int = 10,  # Final results
    ):
        """
        Initialize multi-stage retriever.
        
        Args:
            use_reranker: Whether to use cross-encoder re-ranking
            use_skill_weighting: Whether to use TF-IDF skill weighting
            retrieval_k: Number of candidates for re-ranking
            final_k: Number of final results
        """
        self.use_reranker = use_reranker
        self.use_skill_weighting = use_skill_weighting
        self.retrieval_k = retrieval_k
        self.final_k = final_k
        
        self.reranker = CrossEncoderReranker(use_model=use_reranker)
        self.skill_weighter = TFIDFSkillWeighter()
    
    def fit_skill_weighter(self, skill_lists: list[list[str]]):
        """Fit the skill weighter on job skill data."""
        if self.use_skill_weighting:
            self.skill_weighter.fit(skill_lists)
    
    def retrieve_and_rerank(
        self,
        query: str,
        query_embedding: np.ndarray,
        initial_results: list[tuple[str, str, float]],  # (id, text, score)
        user_skills: Optional[list[str]] = None,
        job_skills: Optional[dict[str, list[str]]] = None,  # job_id -> skills
    ) -> list[dict]:
        """
        Full retrieval pipeline with optional re-ranking.
        
        Args:
            query: Query text
            query_embedding: Query embedding
            initial_results: Initial retrieval results (id, text, score)
            user_skills: User's skills for matching
            job_skills: Job ID to skills mapping
        
        Returns:
            List of result dictionaries with scores
        """
        # Step 1: Re-rank if enabled
        if self.use_reranker and len(initial_results) > 0:
            reranked = self.reranker.rerank(
                query=query,
                candidates=initial_results,
                top_k=self.final_k,
            )
            
            results = [
                {
                    "id": r.id,
                    "semantic_score": r.original_score,
                    "rerank_score": r.rerank_score,
                    "final_score": r.final_score,
                }
                for r in reranked
            ]
        else:
            results = [
                {
                    "id": id_,
                    "semantic_score": score,
                    "rerank_score": score,
                    "final_score": score,
                }
                for id_, _, score in initial_results[:self.final_k]
            ]
        
        # Step 2: Apply skill weighting if available
        if self.use_skill_weighting and user_skills and job_skills:
            for result in results:
                job_id = result["id"]
                if job_id in job_skills:
                    skill_score = self.skill_weighter.weighted_match_score(
                        user_skills, job_skills[job_id]
                    )
                    # Combine with final score
                    result["skill_score"] = skill_score
                    result["final_score"] = (
                        0.7 * result["final_score"] + 0.3 * skill_score
                    )
        
        # Sort by final score
        results.sort(key=lambda x: -x["final_score"])
        
        return results


class DiversityReranker:
    """
    Re-rank results to ensure diversity in recommendations.
    
    Prevents recommending many similar jobs by penalizing
    results too similar to already selected ones.
    
    Uses Maximal Marginal Relevance (MMR):
    MMR = λ * sim(q, d) - (1-λ) * max(sim(d, d_i)) for d_i in selected
    """
    
    def __init__(self, lambda_param: float = 0.7):
        """
        Initialize diversity reranker.
        
        Args:
            lambda_param: Balance between relevance (high) and diversity (low)
        """
        self.lambda_param = lambda_param
    
    def rerank_for_diversity(
        self,
        results: list[tuple[str, float, np.ndarray]],  # (id, score, embedding)
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Re-rank results using MMR for diversity.
        
        Args:
            results: List of (id, relevance_score, embedding) tuples
            top_k: Number of diverse results to return
        
        Returns:
            Re-ranked results (id, mmr_score)
        """
        if not results:
            return []
        
        if len(results) <= top_k:
            return [(r[0], r[1]) for r in results]
        
        selected: list[tuple[str, float, np.ndarray]] = []
        remaining = list(results)
        
        # Select first result (highest relevance)
        remaining.sort(key=lambda x: -x[1])
        selected.append(remaining.pop(0))
        
        # Iteratively select diverse results
        while len(selected) < top_k and remaining:
            best_mmr = float('-inf')
            best_idx = 0
            
            for i, (doc_id, rel_score, embedding) in enumerate(remaining):
                # Compute max similarity to already selected
                max_sim = 0.0
                for _, _, sel_emb in selected:
                    sim = float(np.dot(embedding, sel_emb) / (
                        np.linalg.norm(embedding) * np.linalg.norm(sel_emb) + 1e-8
                    ))
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr = self.lambda_param * rel_score - (1 - self.lambda_param) * max_sim
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return [(doc_id, score) for doc_id, score, _ in selected]

