"""
Pydantic models for the Smart Career Path Recommender.

Defines data structures for user profiles, job postings, courses,
recommendations, and career paths.
"""

from typing import Optional
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """User profile with skills, experience, and career goals."""
    
    skills: list[str] = Field(default_factory=list, description="List of user skills")
    current_role: Optional[str] = Field(None, description="Current job title/role")
    education: Optional[str] = Field(None, description="Education level")
    experience_years: int = Field(0, ge=0, description="Years of professional experience")
    career_goals: Optional[str] = Field(None, description="Career aspirations and goals")
    
    def to_text(self) -> str:
        """Convert profile to text representation for embedding."""
        parts = []
        
        if self.current_role:
            parts.append(f"Current Role: {self.current_role}")
        
        if self.skills:
            parts.append(f"Skills: {', '.join(self.skills)}")
        
        if self.experience_years > 0:
            parts.append(f"Experience: {self.experience_years} years")
        
        if self.education:
            parts.append(f"Education: {self.education}")
        
        if self.career_goals:
            parts.append(f"Career Goals: {self.career_goals}")
        
        return "\n".join(parts) if parts else ""


class JobPosting(BaseModel):
    """Job posting with requirements and details."""
    
    id: str = Field(..., description="Unique job identifier")
    title: str = Field(..., description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    description: str = Field(..., description="Job description")
    required_skills: list[str] = Field(default_factory=list, description="Required skills")
    preferred_skills: list[str] = Field(default_factory=list, description="Preferred skills")
    experience_level: Optional[str] = Field(None, description="Experience level (e.g., Entry-Level, Mid-Level)")
    location: Optional[str] = Field(None, description="Job location")
    salary_range: Optional[str] = Field(None, description="Salary range")


class Course(BaseModel):
    """Course with learning content and metadata."""
    
    id: str = Field(..., description="Unique course identifier")
    title: str = Field(..., description="Course title")
    provider: Optional[str] = Field(None, description="Course provider/platform")
    description: str = Field(..., description="Course description")
    skills_taught: list[str] = Field(default_factory=list, description="Skills taught in the course")
    difficulty: Optional[str] = Field(None, description="Difficulty level (e.g., Beginner, Intermediate)")
    duration: Optional[str] = Field(None, description="Course duration")
    url: Optional[str] = Field(None, description="Course URL")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Course rating (0-5)")


class SkillGap(BaseModel):
    """Skill gap identified for a target role."""
    
    skill: str = Field(..., description="Missing skill name")
    importance: float = Field(..., ge=0, le=1, description="Importance score (0-1)")
    category: Optional[str] = Field(None, description="Skill category")


class JobRecommendation(BaseModel):
    """Job recommendation with matching analysis."""
    
    job: JobPosting = Field(..., description="Recommended job posting")
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity score (0-1)")
    matching_skills: list[str] = Field(default_factory=list, description="Skills that match")
    missing_skills: list[str] = Field(default_factory=list, description="Required skills user lacks")


class CourseRecommendation(BaseModel):
    """Course recommendation with relevance analysis."""
    
    course: Course = Field(..., description="Recommended course")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score (0-1)")
    skills_addressed: list[str] = Field(default_factory=list, description="Skill gaps this course addresses")


class CareerPathNode(BaseModel):
    """A node in a career progression path."""
    
    role: str = Field(..., description="Job role/title")
    level: int = Field(..., ge=0, description="Career level (0=Entry, higher=Senior)")
    required_skills: list[str] = Field(default_factory=list, description="Skills required for this role")
    avg_years_experience: Optional[float] = Field(None, ge=0, description="Average years of experience needed")


class CareerPath(BaseModel):
    """Career progression path from current to target role."""
    
    target_role: Optional[str] = Field(None, description="Target job role")
    current_level: int = Field(..., ge=0, description="Current career level")
    path_nodes: list[CareerPathNode] = Field(default_factory=list, description="Progression path nodes")
    recommended_transitions: list[str] = Field(default_factory=list, description="Recommended next roles")


class RecommendationRequest(BaseModel):
    """Request model for job/course recommendations."""
    
    profile: UserProfile = Field(..., description="User profile")
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations to return")
    target_role: Optional[str] = Field(None, description="Optional target role for recommendations")


class SkillGapRequest(BaseModel):
    """Request model for skill gap analysis."""
    
    profile: UserProfile = Field(..., description="User profile with current skills")
    target_role: str = Field(..., description="Target role to analyze gaps for")


