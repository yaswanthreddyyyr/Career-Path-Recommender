"""
Models package for the Smart Career Path Recommender.

Contains Pydantic schemas for data validation and serialization.
"""

from src.models.schemas import (
    UserProfile,
    JobPosting,
    Course,
    SkillGap,
    JobRecommendation,
    CourseRecommendation,
    CareerPath,
    CareerPathNode,
    RecommendationRequest,
    SkillGapRequest,
)

__all__ = [
    "UserProfile",
    "JobPosting",
    "Course",
    "SkillGap",
    "JobRecommendation",
    "CourseRecommendation",
    "CareerPath",
    "CareerPathNode",
    "RecommendationRequest",
    "SkillGapRequest",
]


