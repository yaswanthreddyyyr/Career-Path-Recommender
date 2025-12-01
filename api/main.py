"""
FastAPI backend for the Smart Career Path Recommender.

Provides REST API endpoints for job recommendations, skill gap analysis,
course recommendations, and career path visualization.
"""

import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.recommender import CareerRecommender, get_recommender
from src.models.schemas import (
    UserProfile,
    JobPosting,
    Course,
    SkillGap,
    JobRecommendation,
    CourseRecommendation,
    CareerPath,
    RecommendationRequest,
    SkillGapRequest,
)


# Global recommender instance
recommender: Optional[CareerRecommender] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the recommender on startup."""
    global recommender
    print("Initializing Career Recommender...")
    recommender = get_recommender()
    recommender.initialize()
    print("Career Recommender ready!")
    yield
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Smart Career Path Recommender API",
    description="""
    An intelligent career recommendation system that bridges individual skill profiles,
    learning resources, and real-world job opportunities.
    
    ## Features
    
    - **Job Recommendations**: Find matching jobs based on your skills and experience
    - **Skill Gap Analysis**: Identify missing skills for target roles
    - **Course Recommendations**: Get targeted learning resources to fill skill gaps
    - **Career Path Planning**: Visualize career progression paths
    
    ## Usage
    
    1. Submit your profile with skills, experience, and career goals
    2. Get personalized job recommendations
    3. Analyze skill gaps for desired roles
    4. Find courses to build missing skills
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class JobRecommendationsResponse(BaseModel):
    """Response containing job recommendations."""
    recommendations: list[JobRecommendation]
    total_jobs_searched: int


class CourseRecommendationsResponse(BaseModel):
    """Response containing course recommendations."""
    recommendations: list[CourseRecommendation]
    total_courses_searched: int


class SkillGapsResponse(BaseModel):
    """Response containing skill gap analysis."""
    target_role: str
    skill_gaps: list[SkillGap]
    courses: list[CourseRecommendation]


class CareerPathResponse(BaseModel):
    """Response containing career path information."""
    career_path: CareerPath


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    jobs_loaded: int
    courses_loaded: int


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Smart Career Path Recommender API",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and data status."""
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    return HealthResponse(
        status="healthy",
        jobs_loaded=len(recommender.get_all_jobs()),
        courses_loaded=len(recommender.get_all_courses()),
    )


@app.post(
    "/api/recommend/jobs",
    response_model=JobRecommendationsResponse,
    tags=["Recommendations"],
    summary="Get job recommendations",
)
async def recommend_jobs(request: RecommendationRequest):
    """
    Get personalized job recommendations based on user profile.
    
    - **profile**: User profile with skills, experience, and career goals
    - **top_k**: Number of recommendations to return (default: 10)
    """
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        recommendations = recommender.recommend_jobs(
            profile=request.profile,
            top_k=request.top_k,
        )
        
        return JobRecommendationsResponse(
            recommendations=recommendations,
            total_jobs_searched=len(recommender.get_all_jobs()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/recommend/courses",
    response_model=CourseRecommendationsResponse,
    tags=["Recommendations"],
    summary="Get course recommendations",
)
async def recommend_courses(request: RecommendationRequest):
    """
    Get course recommendations based on user profile and optional target role.
    
    - **profile**: User profile with skills and career goals
    - **target_role**: Optional target role for skill gap-based recommendations
    - **top_k**: Number of recommendations to return (default: 10)
    """
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        recommendations = recommender.recommend_courses_for_profile(
            profile=request.profile,
            target_role=request.target_role,
            top_k=request.top_k,
        )
        
        return CourseRecommendationsResponse(
            recommendations=recommendations,
            total_courses_searched=len(recommender.get_all_courses()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/analyze/skills",
    response_model=SkillGapsResponse,
    tags=["Analysis"],
    summary="Analyze skill gaps",
)
async def analyze_skills(request: SkillGapRequest):
    """
    Analyze skill gaps between user profile and target role.
    
    Returns missing skills and recommended courses to fill them.
    
    - **profile**: User profile with current skills
    - **target_role**: Target job role to analyze gaps for
    """
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        # Get skill gaps
        skill_gaps = recommender.analyze_skill_gaps(
            profile=request.profile,
            target_role=request.target_role,
        )
        
        # Get course recommendations for skill gaps
        courses = recommender.recommend_courses(
            skill_gaps=skill_gaps,
            top_k=10,
        )
        
        return SkillGapsResponse(
            target_role=request.target_role,
            skill_gaps=skill_gaps,
            courses=courses,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/career-path/{role}",
    response_model=CareerPathResponse,
    tags=["Career Path"],
    summary="Get career path",
)
async def get_career_path(
    role: str,
    target_role: Optional[str] = Query(None, description="Target role to plan path towards"),
):
    """
    Get a career progression path starting from a given role.
    
    - **role**: Current or starting role
    - **target_role**: Optional target role to plan path towards
    """
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        career_path = recommender.get_career_path(
            current_role=role,
            target_role=target_role,
        )
        
        return CareerPathResponse(career_path=career_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Data endpoints

@app.get(
    "/api/jobs",
    response_model=list[JobPosting],
    tags=["Data"],
    summary="Get all jobs",
)
async def get_all_jobs(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
):
    """Get all available job postings."""
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    jobs = recommender.get_all_jobs()
    return jobs[offset:offset + limit]


@app.get(
    "/api/jobs/{job_id}",
    response_model=JobPosting,
    tags=["Data"],
    summary="Get job by ID",
)
async def get_job(job_id: str):
    """Get a specific job posting by ID."""
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    job = recommender.get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    return job


@app.get(
    "/api/courses",
    response_model=list[Course],
    tags=["Data"],
    summary="Get all courses",
)
async def get_all_courses(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of courses to return"),
    offset: int = Query(0, ge=0, description="Number of courses to skip"),
):
    """Get all available courses."""
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    courses = recommender.get_all_courses()
    return courses[offset:offset + limit]


@app.get(
    "/api/courses/{course_id}",
    response_model=Course,
    tags=["Data"],
    summary="Get course by ID",
)
async def get_course(course_id: str):
    """Get a specific course by ID."""
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    course = recommender.get_course_by_id(course_id)
    if not course:
        raise HTTPException(status_code=404, detail=f"Course not found: {course_id}")
    
    return course


@app.get(
    "/api/skills",
    response_model=list[str],
    tags=["Data"],
    summary="Get all skills",
)
async def get_all_skills():
    """Get all known skills in the system."""
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    return recommender.get_all_skills()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

