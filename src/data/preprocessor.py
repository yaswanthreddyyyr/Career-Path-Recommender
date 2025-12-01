"""
Text preprocessing utilities for the Smart Career Path Recommender.

Handles cleaning and normalizing text data for embedding generation.
"""

import re
from typing import Optional


class TextPreprocessor:
    """Preprocess text data for embedding generation."""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_special_chars: bool = True,
        remove_extra_whitespace: bool = True,
        min_length: int = 3,
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            lowercase: Whether to convert text to lowercase.
            remove_special_chars: Whether to remove special characters.
            remove_extra_whitespace: Whether to normalize whitespace.
            min_length: Minimum word length to keep.
        """
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.remove_extra_whitespace = remove_extra_whitespace
        self.min_length = min_length
        
        # Common stop words to optionally remove
        self.stop_words = {
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "also", "now", "about", "this", "that", "these", "those",
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean.
        
        Returns:
            Cleaned text.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)
        
        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)
        
        # Remove special characters but keep alphanumeric, spaces, and basic punctuation
        if self.remove_special_chars:
            text = re.sub(r"[^\w\s.,!?;:\-+#/()]", "", text)
        
        # Normalize whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stop words from text.
        
        Args:
            text: Text to process.
        
        Returns:
            Text with stop words removed.
        """
        words = text.split()
        filtered_words = [
            word for word in words 
            if word.lower() not in self.stop_words and len(word) >= self.min_length
        ]
        return " ".join(filtered_words)
    
    def preprocess(
        self, 
        text: str, 
        remove_stopwords: bool = False
    ) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Text to preprocess.
            remove_stopwords: Whether to remove stop words.
        
        Returns:
            Preprocessed text.
        """
        text = self.clean_text(text)
        
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        return text
    
    def preprocess_skills(self, skills: list[str]) -> str:
        """
        Preprocess a list of skills into a single text string.
        
        Args:
            skills: List of skill names.
        
        Returns:
            Preprocessed skills as a comma-separated string.
        """
        if not skills:
            return ""
        
        cleaned_skills = [self.clean_text(skill) for skill in skills]
        return ", ".join(s for s in cleaned_skills if s)
    
    def create_job_text(
        self,
        title: str,
        description: str,
        required_skills: Optional[list[str]] = None,
        preferred_skills: Optional[list[str]] = None,
        experience_level: Optional[str] = None,
    ) -> str:
        """
        Create a unified text representation of a job posting.
        
        Args:
            title: Job title.
            description: Job description.
            required_skills: Required skills list.
            preferred_skills: Preferred skills list.
            experience_level: Experience level requirement.
        
        Returns:
            Unified text for embedding.
        """
        parts = []
        
        # Title is important, so include it prominently
        if title:
            parts.append(f"Job Title: {self.clean_text(title)}")
        
        # Description
        if description:
            parts.append(f"Description: {self.clean_text(description)}")
        
        # Skills
        if required_skills:
            skills_text = self.preprocess_skills(required_skills)
            parts.append(f"Required Skills: {skills_text}")
        
        if preferred_skills:
            skills_text = self.preprocess_skills(preferred_skills)
            parts.append(f"Preferred Skills: {skills_text}")
        
        # Experience level
        if experience_level:
            parts.append(f"Experience Level: {self.clean_text(experience_level)}")
        
        return " ".join(parts)
    
    def create_course_text(
        self,
        title: str,
        description: str,
        skills_taught: Optional[list[str]] = None,
        difficulty: Optional[str] = None,
    ) -> str:
        """
        Create a unified text representation of a course.
        
        Args:
            title: Course title.
            description: Course description.
            skills_taught: Skills taught in the course.
            difficulty: Course difficulty level.
        
        Returns:
            Unified text for embedding.
        """
        parts = []
        
        if title:
            parts.append(f"Course: {self.clean_text(title)}")
        
        if description:
            parts.append(f"Description: {self.clean_text(description)}")
        
        if skills_taught:
            skills_text = self.preprocess_skills(skills_taught)
            parts.append(f"Skills Taught: {skills_text}")
        
        if difficulty:
            parts.append(f"Difficulty: {self.clean_text(difficulty)}")
        
        return " ".join(parts)
    
    def create_user_profile_text(
        self,
        skills: list[str],
        current_role: Optional[str] = None,
        education: Optional[str] = None,
        experience_years: int = 0,
        career_goals: Optional[str] = None,
    ) -> str:
        """
        Create a unified text representation of a user profile.
        
        Args:
            skills: User's skills.
            current_role: Current job title.
            education: Educational background.
            experience_years: Years of experience.
            career_goals: Career aspirations.
        
        Returns:
            Unified text for embedding.
        """
        parts = []
        
        if current_role:
            parts.append(f"Current Role: {self.clean_text(current_role)}")
        
        if skills:
            skills_text = self.preprocess_skills(skills)
            parts.append(f"Skills: {skills_text}")
        
        if education:
            parts.append(f"Education: {self.clean_text(education)}")
        
        if experience_years > 0:
            parts.append(f"Experience: {experience_years} years")
        
        if career_goals:
            parts.append(f"Career Goals: {self.clean_text(career_goals)}")
        
        return " ".join(parts)


def extract_skills_from_text(text: str, known_skills: list[str]) -> list[str]:
    """
    Extract known skills from a text string.
    
    Args:
        text: Text to search for skills.
        known_skills: List of known skill names to look for.
    
    Returns:
        List of found skills.
    """
    text_lower = text.lower()
    found_skills = []
    
    for skill in known_skills:
        # Check for exact match (case-insensitive)
        skill_lower = skill.lower()
        
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(skill_lower) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    
    return found_skills


def normalize_skill_name(skill: str) -> str:
    """
    Normalize a skill name for consistent matching.
    
    Args:
        skill: Skill name to normalize.
    
    Returns:
        Normalized skill name.
    """
    # Common skill name variations
    variations = {
        "ml": "machine learning",
        "dl": "deep learning",
        "ai": "artificial intelligence",
        "nlp": "natural language processing",
        "cv": "computer vision",
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "sql": "sql",
        "nosql": "nosql",
        "k8s": "kubernetes",
        "aws": "aws",
        "gcp": "google cloud platform",
        "azure": "microsoft azure",
    }
    
    skill_lower = skill.lower().strip()
    return variations.get(skill_lower, skill)

