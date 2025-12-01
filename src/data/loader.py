"""
Data loader utilities for the Smart Career Path Recommender.

Handles loading job postings, courses, and skills data from various formats.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional

from src.models.schemas import JobPosting, Course


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


class DataLoader:
    """Load and manage datasets for the recommender system."""
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the data directory. Defaults to project's data/raw.
        """
        self.data_path = data_path or DATA_RAW
        self._jobs_df: Optional[pd.DataFrame] = None
        self._courses_df: Optional[pd.DataFrame] = None
        self._skills_df: Optional[pd.DataFrame] = None
    
    def load_jobs(self, filename: str = "jobs.json") -> pd.DataFrame:
        """
        Load job postings data.
        
        Args:
            filename: Name of the jobs data file.
        
        Returns:
            DataFrame containing job postings.
        """
        filepath = self.data_path / filename
        
        if filepath.suffix == ".json":
            with open(filepath, "r") as f:
                data = json.load(f)
            self._jobs_df = pd.DataFrame(data)
        elif filepath.suffix == ".csv":
            self._jobs_df = pd.read_csv(filepath)
            # Parse list columns if they're stored as strings
            for col in ["required_skills", "preferred_skills"]:
                if col in self._jobs_df.columns:
                    self._jobs_df[col] = self._jobs_df[col].apply(
                        lambda x: eval(x) if isinstance(x, str) else x
                    )
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return self._jobs_df
    
    def load_courses(self, filename: str = "courses.json") -> pd.DataFrame:
        """
        Load courses data.
        
        Args:
            filename: Name of the courses data file.
        
        Returns:
            DataFrame containing courses.
        """
        filepath = self.data_path / filename
        
        if filepath.suffix == ".json":
            with open(filepath, "r") as f:
                data = json.load(f)
            self._courses_df = pd.DataFrame(data)
        elif filepath.suffix == ".csv":
            self._courses_df = pd.read_csv(filepath)
            # Parse list columns if they're stored as strings
            if "skills_taught" in self._courses_df.columns:
                self._courses_df["skills_taught"] = self._courses_df["skills_taught"].apply(
                    lambda x: eval(x) if isinstance(x, str) else x
                )
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return self._courses_df
    
    def load_skills(self, filename: str = "skills.json") -> pd.DataFrame:
        """
        Load skills taxonomy data.
        
        Args:
            filename: Name of the skills data file.
        
        Returns:
            DataFrame containing skills taxonomy.
        """
        filepath = self.data_path / filename
        
        if filepath.suffix == ".json":
            with open(filepath, "r") as f:
                data = json.load(f)
            self._skills_df = pd.DataFrame(data)
        elif filepath.suffix == ".csv":
            self._skills_df = pd.read_csv(filepath)
            if "related_skills" in self._skills_df.columns:
                self._skills_df["related_skills"] = self._skills_df["related_skills"].apply(
                    lambda x: eval(x) if isinstance(x, str) else x
                )
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return self._skills_df
    
    def get_jobs_as_models(self) -> list[JobPosting]:
        """
        Get job postings as Pydantic models.
        
        Returns:
            List of JobPosting models.
        """
        if self._jobs_df is None:
            self.load_jobs()
        
        jobs = []
        for _, row in self._jobs_df.iterrows():
            job = JobPosting(
                id=str(row.get("id", "")),
                title=str(row.get("title", "")),
                company=row.get("company"),
                description=str(row.get("description", "")),
                required_skills=row.get("required_skills", []),
                preferred_skills=row.get("preferred_skills", []),
                experience_level=row.get("experience_level"),
                location=row.get("location"),
                salary_range=row.get("salary_range"),
            )
            jobs.append(job)
        
        return jobs
    
    def get_courses_as_models(self) -> list[Course]:
        """
        Get courses as Pydantic models.
        
        Returns:
            List of Course models.
        """
        if self._courses_df is None:
            self.load_courses()
        
        courses = []
        for _, row in self._courses_df.iterrows():
            course = Course(
                id=str(row.get("id", "")),
                title=str(row.get("title", "")),
                provider=row.get("provider"),
                description=str(row.get("description", "")),
                skills_taught=row.get("skills_taught", []),
                difficulty=row.get("difficulty"),
                duration=row.get("duration"),
                url=row.get("url"),
                rating=row.get("rating"),
            )
            courses.append(course)
        
        return courses
    
    def get_all_skills(self) -> list[str]:
        """
        Get a list of all unique skills from the skills taxonomy.
        
        Returns:
            List of unique skill names.
        """
        if self._skills_df is None:
            self.load_skills()
        
        return self._skills_df["skill"].tolist()
    
    def get_skill_categories(self) -> dict[str, list[str]]:
        """
        Get skills organized by category.
        
        Returns:
            Dictionary mapping category names to skill lists.
        """
        if self._skills_df is None:
            self.load_skills()
        
        categories = {}
        for _, row in self._skills_df.iterrows():
            category = row.get("category", "Other")
            skill = row.get("skill", "")
            
            if category not in categories:
                categories[category] = []
            categories[category].append(skill)
        
        return categories
    
    def get_related_skills(self, skill: str) -> list[str]:
        """
        Get skills related to a given skill.
        
        Args:
            skill: The skill to find related skills for.
        
        Returns:
            List of related skill names.
        """
        if self._skills_df is None:
            self.load_skills()
        
        skill_lower = skill.lower()
        for _, row in self._skills_df.iterrows():
            if row["skill"].lower() == skill_lower:
                return row.get("related_skills", [])
        
        return []
    
    @property
    def jobs(self) -> pd.DataFrame:
        """Get jobs DataFrame, loading if necessary."""
        if self._jobs_df is None:
            self.load_jobs()
        return self._jobs_df
    
    @property
    def courses(self) -> pd.DataFrame:
        """Get courses DataFrame, loading if necessary."""
        if self._courses_df is None:
            self.load_courses()
        return self._courses_df
    
    @property
    def skills(self) -> pd.DataFrame:
        """Get skills DataFrame, loading if necessary."""
        if self._skills_df is None:
            self.load_skills()
        return self._skills_df

