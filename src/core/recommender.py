"""
Career Recommender Engine - Core recommendation logic.

Provides job matching, skill gap analysis, course recommendations,
and career path visualization.

Optimized with:
- Hybrid search (semantic + BM25)
- Cross-encoder re-ranking
- Skill normalization and query expansion
- TF-IDF skill weighting
- Diversity re-ranking (MMR)
"""

import numpy as np
from pathlib import Path
from typing import Optional
from collections import defaultdict

from src.core.embeddings import EmbeddingEngine
from src.core.similarity import SimilaritySearch, MultiIndexSearch
from src.core.hybrid_search import HybridSearch, SkillNormalizer, QueryExpander, BM25
from src.core.reranker import (
    CrossEncoderReranker,
    TFIDFSkillWeighter,
    MultiStageRetriever,
    DiversityReranker,
)
from src.data.loader import DataLoader
from src.data.preprocessor import TextPreprocessor
from src.models.schemas import (
    UserProfile,
    JobPosting,
    Course,
    SkillGap,
    JobRecommendation,
    CourseRecommendation,
    CareerPath,
    CareerPathNode,
)


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


class CareerRecommender:
    """
    Main recommendation engine for career path planning.
    
    Combines multiple retrieval and ranking techniques:
    1. Semantic similarity (bi-encoder embeddings)
    2. BM25 keyword matching (hybrid search)
    3. Cross-encoder re-ranking (optional)
    4. Skill normalization and expansion
    5. TF-IDF skill weighting
    6. MMR diversity re-ranking
    """
    
    def __init__(
        self,
        data_loader: Optional[DataLoader] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        use_hybrid_search: bool = True,
        use_reranking: bool = False,  # Disabled by default (requires model download)
        use_diversity: bool = True,
        hybrid_alpha: float = 0.7,  # Weight for semantic vs BM25
    ):
        """
        Initialize the career recommender.
        
        Args:
            data_loader: DataLoader instance for loading data.
            embedding_engine: EmbeddingEngine instance for generating embeddings.
            use_hybrid_search: Enable BM25 + semantic hybrid search.
            use_reranking: Enable cross-encoder re-ranking (slower but more accurate).
            use_diversity: Enable MMR diversity re-ranking.
            hybrid_alpha: Weight for semantic scores in hybrid search (0-1).
        """
        self.data_loader = data_loader or DataLoader()
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.preprocessor = TextPreprocessor()
        
        # Configuration
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        self.use_diversity = use_diversity
        self.hybrid_alpha = hybrid_alpha
        
        # Search indices
        self.search = MultiIndexSearch(dimension=self.embedding_engine.EMBEDDING_DIM)
        
        # Hybrid search components
        self.hybrid_jobs = HybridSearch(alpha=hybrid_alpha) if use_hybrid_search else None
        self.hybrid_courses = HybridSearch(alpha=hybrid_alpha) if use_hybrid_search else None
        
        # Optimization components
        self.skill_normalizer = SkillNormalizer()
        self.query_expander = QueryExpander()
        self.skill_weighter = TFIDFSkillWeighter()
        self.diversity_reranker = DiversityReranker(lambda_param=0.7)
        self.reranker = CrossEncoderReranker(use_model=use_reranking) if use_reranking else None
        
        # Data storage
        self._jobs: dict[str, JobPosting] = {}
        self._courses: dict[str, Course] = {}
        self._skills: set[str] = set()
        self._job_texts: dict[str, str] = {}  # For BM25
        self._course_texts: dict[str, str] = {}
        self._job_embeddings: Optional[np.ndarray] = None
        self._course_embeddings: Optional[np.ndarray] = None
        
        # Initialization flag
        self._initialized = False
    
    def initialize(self, force_rebuild: bool = False):
        """
        Initialize the recommender by loading data and building indices.
        
        Args:
            force_rebuild: Whether to rebuild indices even if they exist.
        """
        if self._initialized and not force_rebuild:
            return
        
        print("Initializing Career Recommender (Optimized)...")
        
        # Try to load existing indices
        indices_exist = self._try_load_indices()
        
        if not indices_exist or force_rebuild:
            self._build_indices()
        
        # Initialize optimization components
        self._init_optimizations()
        
        self._initialized = True
        print("Career Recommender initialized successfully!")
    
    def _try_load_indices(self) -> bool:
        """Try to load existing indices from disk."""
        try:
            # Load jobs index
            jobs_index = self.search.create_index("jobs")
            jobs_index.load(DATA_PROCESSED, "jobs")
            
            # Load courses index
            courses_index = self.search.create_index("courses")
            courses_index.load(DATA_PROCESSED, "courses")
            
            # Load data
            self._load_data()
            
            # Load embeddings for hybrid search
            jobs_index = self.search.get_index("jobs")
            self._job_embeddings = jobs_index._embeddings
            
            courses_index = self.search.get_index("courses")
            self._course_embeddings = courses_index._embeddings
            
            print(f"Loaded {jobs_index.size} jobs and {courses_index.size} courses from cache")
            return True
        except FileNotFoundError:
            return False
    
    def _load_data(self):
        """Load data from disk into memory."""
        # Load jobs
        jobs = self.data_loader.get_jobs_as_models()
        for job in jobs:
            self._jobs[job.id] = job
            # Create text representation for BM25
            self._job_texts[job.id] = self.preprocessor.create_job_text(
                title=job.title,
                description=job.description,
                required_skills=job.required_skills,
                preferred_skills=job.preferred_skills,
                experience_level=job.experience_level,
            )
            # Collect skills
            self._skills.update(skill.lower() for skill in job.required_skills)
            self._skills.update(skill.lower() for skill in job.preferred_skills)
        
        # Load courses
        courses = self.data_loader.get_courses_as_models()
        for course in courses:
            self._courses[course.id] = course
            self._course_texts[course.id] = self.preprocessor.create_course_text(
                title=course.title,
                description=course.description,
                skills_taught=course.skills_taught,
                difficulty=course.difficulty,
            )
            self._skills.update(skill.lower() for skill in course.skills_taught)
    
    def _build_indices(self):
        """Build search indices from scratch."""
        print("Building search indices...")
        
        # Load data
        self._load_data()
        
        jobs = list(self._jobs.values())
        courses = list(self._courses.values())
        
        # Generate embeddings
        print(f"Generating embeddings for {len(jobs)} jobs...")
        self._job_embeddings = self.embedding_engine.embed_jobs(jobs)
        
        print(f"Generating embeddings for {len(courses)} courses...")
        self._course_embeddings = self.embedding_engine.embed_courses(courses)
        
        # Build FAISS indices
        jobs_index = self.search.create_index("jobs")
        jobs_index.add_embeddings(self._job_embeddings, [job.id for job in jobs])
        
        courses_index = self.search.create_index("courses")
        courses_index.add_embeddings(self._course_embeddings, [course.id for course in courses])
        
        # Save indices
        jobs_index.save(DATA_PROCESSED, "jobs")
        courses_index.save(DATA_PROCESSED, "courses")
        
        print("Search indices built and saved!")
    
    def _init_optimizations(self):
        """Initialize optimization components after data is loaded."""
        jobs = list(self._jobs.values())
        job_ids = [job.id for job in jobs]
        job_texts = [self._job_texts[job.id] for job in jobs]
        
        courses = list(self._courses.values())
        course_ids = [course.id for course in courses]
        course_texts = [self._course_texts[course.id] for course in courses]
        
        # Initialize hybrid search
        if self.use_hybrid_search and self._job_embeddings is not None:
            print("Initializing hybrid search (BM25 + Semantic)...")
            self.hybrid_jobs = HybridSearch(alpha=self.hybrid_alpha)
            self.hybrid_jobs.fit(job_texts, job_ids, self._job_embeddings)
            
            self.hybrid_courses = HybridSearch(alpha=self.hybrid_alpha)
            self.hybrid_courses.fit(course_texts, course_ids, self._course_embeddings)
        
        # Initialize TF-IDF skill weighter
        print("Initializing TF-IDF skill weighting...")
        all_skill_lists = [job.required_skills + job.preferred_skills for job in jobs]
        self.skill_weighter.fit(all_skill_lists)
    
    def recommend_jobs(
        self,
        profile: UserProfile,
        top_k: int = 10,
        min_similarity: float = 0.3,
    ) -> list[JobRecommendation]:
        """
        Recommend jobs based on user profile using optimized pipeline.
        
        Pipeline:
        1. Query expansion with related skills
        2. Hybrid search (semantic + BM25)
        3. Optional cross-encoder re-ranking
        4. Skill matching with normalization
        5. TF-IDF weighted skill scoring
        6. MMR diversity re-ranking
        
        Args:
            profile: User profile with skills and experience.
            top_k: Number of recommendations to return.
            min_similarity: Minimum similarity threshold.
        
        Returns:
            List of job recommendations with scores.
        """
        self.initialize()
        
        # Step 1: Expand query with related skills
        expanded_skills = self.query_expander.expand_skills(profile.skills, max_expansion=3)
        
        # Create query text
        query_text = self.preprocessor.create_user_profile_text(
            skills=expanded_skills,  # Use expanded skills
            current_role=profile.current_role,
            education=profile.education,
            experience_years=profile.experience_years,
            career_goals=profile.career_goals,
        )
        
        # Generate query embedding
        query_embedding = self.embedding_engine.embed_text(query_text)
        
        # Step 2: Retrieve candidates (hybrid or semantic-only)
        if self.use_hybrid_search and self.hybrid_jobs is not None:
            # Hybrid search combines semantic + BM25
            search_results = self.hybrid_jobs.search(
                query_text=query_text,
                query_embedding=query_embedding,
                top_k=top_k * 3,  # Get more candidates for re-ranking
            )
            candidates = [
                (r.id, self._job_texts.get(r.id, ""), r.combined_score)
                for r in search_results
                if r.combined_score >= min_similarity
            ]
        else:
            # Semantic-only search
            jobs_index = self.search.get_index("jobs")
            results = jobs_index.search(query_embedding, top_k=top_k * 3, threshold=min_similarity)
            candidates = [
                (job_id, self._job_texts.get(job_id, ""), score)
                for job_id, score in results
            ]
        
        # Step 3: Optional cross-encoder re-ranking
        if self.use_reranking and self.reranker is not None and candidates:
            reranked = self.reranker.rerank(
                query=query_text,
                candidates=candidates,
                top_k=top_k * 2,
            )
            candidates = [(r.id, "", r.final_score) for r in reranked]
        
        # Step 4: Build recommendations with skill analysis
        recommendations = []
        user_skills_normalized = {self.skill_normalizer.normalize(s) for s in profile.skills}
        
        # Get embeddings for diversity re-ranking
        candidate_data = []
        
        for job_id, _, score in candidates:
            job = self._jobs.get(job_id)
            if not job:
                continue
            
            # Skill matching with normalization
            matching_skills, missing_skills = self.skill_normalizer.match_skills(
                profile.skills, job.required_skills
            )
            matching_preferred, _ = self.skill_normalizer.match_skills(
                profile.skills, job.preferred_skills
            )
            
            all_matching = list(set(matching_skills + matching_preferred))
            
            # Step 5: TF-IDF weighted skill score
            skill_score = self.skill_weighter.weighted_match_score(
                profile.skills, job.required_skills
            )
            
            # Combine semantic score with skill score
            combined_score = 0.7 * score + 0.3 * skill_score
            
            # Store for diversity re-ranking
            job_idx = list(self._jobs.keys()).index(job_id)
            if self._job_embeddings is not None and job_idx < len(self._job_embeddings):
                candidate_data.append((
                    job_id, combined_score, self._job_embeddings[job_idx],
                    job, all_matching, missing_skills
                ))
        
        # Step 6: MMR diversity re-ranking
        if self.use_diversity and len(candidate_data) > top_k:
            diversity_input = [
                (job_id, score, emb) 
                for job_id, score, emb, _, _, _ in candidate_data
            ]
            diverse_results = self.diversity_reranker.rerank_for_diversity(
                diversity_input, top_k=top_k
            )
            
            # Map back to full data
            id_to_data = {d[0]: d for d in candidate_data}
            for job_id, final_score in diverse_results:
                data = id_to_data.get(job_id)
                if data:
                    _, _, _, job, matching, missing = data
                    recommendations.append(JobRecommendation(
                        job=job,
                        similarity_score=final_score,
                        matching_skills=matching,
                        missing_skills=missing,
                    ))
        else:
            # No diversity re-ranking
            for job_id, score, _, job, matching, missing in candidate_data[:top_k]:
                recommendations.append(JobRecommendation(
                    job=job,
                    similarity_score=score,
                    matching_skills=matching,
                    missing_skills=missing,
                ))
        
        return recommendations
    
    def analyze_skill_gaps(
        self,
        profile: UserProfile,
        target_role: str,
    ) -> list[SkillGap]:
        """
        Analyze skill gaps for a target role with TF-IDF weighting.
        
        Args:
            profile: User profile with current skills.
            target_role: Target job role/title.
        
        Returns:
            List of skill gaps with importance scores.
        """
        self.initialize()
        
        # Find jobs matching the target role
        role_embedding = self.embedding_engine.embed_text(target_role)
        jobs_index = self.search.get_index("jobs")
        results = jobs_index.search(role_embedding, top_k=5, threshold=0.3)
        
        # Collect required skills with TF-IDF weighting
        skill_importance = defaultdict(float)
        
        for job_id, similarity in results:
            job = self._jobs.get(job_id)
            if not job:
                continue
            
            # Weight by job similarity and TF-IDF
            for skill in job.required_skills:
                tfidf_weight = self.skill_weighter.get_weight(skill)
                skill_importance[skill.lower()] += 2.0 * similarity * tfidf_weight
            
            for skill in job.preferred_skills:
                tfidf_weight = self.skill_weighter.get_weight(skill)
                skill_importance[skill.lower()] += 1.0 * similarity * tfidf_weight
        
        # Normalize user skills
        user_skills_normalized = {
            self.skill_normalizer.normalize(s) for s in profile.skills
        }
        
        # Find missing skills
        skill_gaps = []
        max_importance = max(skill_importance.values()) if skill_importance else 1
        
        for skill, importance in sorted(skill_importance.items(), key=lambda x: -x[1]):
            skill_normalized = self.skill_normalizer.normalize(skill)
            
            if skill_normalized not in user_skills_normalized:
                # Find original case skill name
                original_name = skill
                for job in self._jobs.values():
                    for s in job.required_skills + job.preferred_skills:
                        if s.lower() == skill:
                            original_name = s
                            break
                
                skill_gaps.append(SkillGap(
                    skill=original_name,
                    importance=importance / max_importance,
                    category=self._get_skill_category(skill),
                ))
        
        return skill_gaps
    
    def _get_skill_category(self, skill: str) -> str:
        """Get category for a skill."""
        categories = self.data_loader.get_skill_categories()
        skill_lower = skill.lower()
        
        for category, skills in categories.items():
            if any(s.lower() == skill_lower for s in skills):
                return category
        
        return "Other"
    
    def recommend_courses(
        self,
        skill_gaps: list[SkillGap],
        top_k: int = 10,
        min_relevance: float = 0.3,
    ) -> list[CourseRecommendation]:
        """
        Recommend courses to fill skill gaps using hybrid search.
        
        Args:
            skill_gaps: List of skill gaps to address.
            top_k: Number of recommendations to return.
            min_relevance: Minimum relevance threshold.
        
        Returns:
            List of course recommendations.
        """
        self.initialize()
        
        if not skill_gaps:
            return []
        
        # Create query from skill gaps (weighted by importance)
        weighted_skills = []
        for gap in skill_gaps[:10]:
            # Repeat important skills for emphasis
            repeat = max(1, int(gap.importance * 3))
            weighted_skills.extend([gap.skill] * repeat)
        
        skills_text = ", ".join(weighted_skills)
        query_text = f"Learn {skills_text}"
        query_embedding = self.embedding_engine.embed_text(query_text)
        
        # Use hybrid search if available
        if self.use_hybrid_search and self.hybrid_courses is not None:
            search_results = self.hybrid_courses.search(
                query_text=query_text,
                query_embedding=query_embedding,
                top_k=top_k * 2,
            )
            candidates = [
                (r.id, r.combined_score)
                for r in search_results
                if r.combined_score >= min_relevance
            ]
        else:
            courses_index = self.search.get_index("courses")
            candidates = courses_index.search(query_embedding, top_k=top_k * 2, threshold=min_relevance)
        
        # Build recommendations
        gap_skills_normalized = {
            self.skill_normalizer.normalize(gap.skill) for gap in skill_gaps
        }
        gap_skill_importance = {
            self.skill_normalizer.normalize(gap.skill): gap.importance
            for gap in skill_gaps
        }
        
        recommendations = []
        
        for course_id, similarity in candidates[:top_k]:
            course = self._courses.get(course_id)
            if not course:
                continue
            
            # Find which skill gaps this course addresses
            addressed_skills = []
            total_importance = 0.0
            
            for skill in course.skills_taught:
                skill_normalized = self.skill_normalizer.normalize(skill)
                if skill_normalized in gap_skills_normalized:
                    # Find original gap skill name
                    for gap in skill_gaps:
                        if self.skill_normalizer.normalize(gap.skill) == skill_normalized:
                            addressed_skills.append(gap.skill)
                            total_importance += gap.importance
                            break
            
            # Calculate relevance score
            # Boost by number and importance of skills addressed
            relevance = similarity
            if addressed_skills:
                skill_bonus = 0.1 * len(addressed_skills) + 0.2 * total_importance
                relevance = min(1.0, similarity + skill_bonus)
            
            recommendations.append(CourseRecommendation(
                course=course,
                relevance_score=relevance,
                skills_addressed=addressed_skills,
            ))
        
        # Sort by relevance
        recommendations.sort(key=lambda x: (-len(x.skills_addressed), -x.relevance_score))
        
        return recommendations[:top_k]
    
    def recommend_courses_for_profile(
        self,
        profile: UserProfile,
        target_role: Optional[str] = None,
        top_k: int = 10,
    ) -> list[CourseRecommendation]:
        """
        Recommend courses for a user profile.
        
        Args:
            profile: User profile.
            target_role: Optional target role for skill gap analysis.
            top_k: Number of recommendations.
        
        Returns:
            List of course recommendations.
        """
        if target_role:
            # Get skill gaps and recommend courses to fill them
            skill_gaps = self.analyze_skill_gaps(profile, target_role)
            return self.recommend_courses(skill_gaps, top_k=top_k)
        else:
            # Recommend courses based on career goals or general profile
            query_text = profile.career_goals or profile.to_text()
            query_embedding = self.embedding_engine.embed_text(query_text)
            
            courses_index = self.search.get_index("courses")
            results = courses_index.search(query_embedding, top_k=top_k)
            
            recommendations = []
            for course_id, similarity in results:
                course = self._courses.get(course_id)
                if course:
                    recommendations.append(CourseRecommendation(
                        course=course,
                        relevance_score=similarity,
                        skills_addressed=[],
                    ))
            
            return recommendations
    
    def get_career_path(
        self,
        current_role: str,
        target_role: Optional[str] = None,
    ) -> CareerPath:
        """
        Generate a career path from current role to target role.
        
        Args:
            current_role: User's current job role.
            target_role: Target role to reach.
        
        Returns:
            CareerPath with progression nodes.
        """
        self.initialize()
        
        # Find jobs similar to current role
        current_embedding = self.embedding_engine.embed_text(current_role)
        jobs_index = self.search.get_index("jobs")
        
        # Define career levels based on experience
        level_mapping = {
            "Entry-Level": 0,
            "Junior": 1,
            "Mid-Level": 2,
            "Senior": 3,
            "Lead": 4,
            "Principal": 5,
            "Director": 6,
        }
        
        # Get related jobs
        results = jobs_index.search(current_embedding, top_k=20, threshold=0.2)
        
        # Build path nodes
        path_nodes = []
        seen_titles = set()
        
        for job_id, similarity in results:
            job = self._jobs.get(job_id)
            if not job or job.title in seen_titles:
                continue
            
            seen_titles.add(job.title)
            
            # Determine level
            level = 2  # Default to mid-level
            if job.experience_level:
                for level_name, level_num in level_mapping.items():
                    if level_name.lower() in job.experience_level.lower():
                        level = level_num
                        break
            
            path_nodes.append(CareerPathNode(
                role=job.title,
                level=level,
                required_skills=job.required_skills,
                avg_years_experience=level * 2.0,  # Rough estimate
            ))
        
        # Sort by level
        path_nodes.sort(key=lambda x: x.level)
        
        # Determine current level
        current_level = 2
        for node in path_nodes:
            if current_role.lower() in node.role.lower():
                current_level = node.level
                break
        
        # Generate recommended transitions
        transitions = []
        for i, node in enumerate(path_nodes):
            if node.level > current_level and len(transitions) < 3:
                transitions.append(node.role)
        
        return CareerPath(
            target_role=target_role or (path_nodes[-1].role if path_nodes else current_role),
            current_level=current_level,
            path_nodes=path_nodes[:10],  # Limit nodes
            recommended_transitions=transitions,
        )
    
    def get_all_jobs(self) -> list[JobPosting]:
        """Get all available jobs."""
        self.initialize()
        return list(self._jobs.values())
    
    def get_all_courses(self) -> list[Course]:
        """Get all available courses."""
        self.initialize()
        return list(self._courses.values())
    
    def get_all_skills(self) -> list[str]:
        """Get all known skills."""
        self.initialize()
        return sorted(self._skills)
    
    def get_job_by_id(self, job_id: str) -> Optional[JobPosting]:
        """Get a job by its ID."""
        self.initialize()
        return self._jobs.get(job_id)
    
    def get_course_by_id(self, course_id: str) -> Optional[Course]:
        """Get a course by its ID."""
        self.initialize()
        return self._courses.get(course_id)


# Singleton instance for convenience
_recommender: Optional[CareerRecommender] = None


def get_recommender() -> CareerRecommender:
    """Get or create the global recommender instance."""
    global _recommender
    if _recommender is None:
        _recommender = CareerRecommender()
    return _recommender
