# Smart Career Path Recommender

An intelligent, data-driven system that bridges the gap between individual skill profiles, learning resources, and real-world job opportunities. Built using modern Information Retrieval (IR) and Natural Language Processing (NLP) techniques.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Core Algorithm](#core-algorithm)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Team](#team)

## Overview

In today's digital economy, individuals face an overwhelming number of career choices and online learning options. Traditional platforms operate in silos—learning platforms suggest popular courses while job portals match resumes to postings. This fragmented approach fails to provide holistic career guidance.

**Smart Career Path Recommender** addresses this gap by creating a unified recommendation framework that:
- Integrates job market data with learning resources
- Uses semantic understanding instead of keyword matching
- Provides personalized, actionable career guidance

## Features

| Feature | Description |
|---------|-------------|
| **Job Recommendations** | Find matching job roles based on skills, experience, and career goals using semantic similarity |
| **Skill Gap Analysis** | Identify specific skills needed to transition to target roles |
| **Course Recommendations** | Get targeted learning resources to fill identified skill gaps |
| **Career Path Visualization** | Interactive career progression maps showing potential transitions |

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                                 │
│                    (Streamlit Dashboard / API)                          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          FastAPI BACKEND                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   /jobs     │  │  /courses   │  │   /skills   │  │/career-path │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      RECOMMENDATION ENGINE                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │  Job Matching    │  │  Skill Gap       │  │  Course          │      │
│  │  Algorithm       │  │  Analysis        │  │  Recommender     │      │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘      │
└───────────┼─────────────────────┼─────────────────────┼─────────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SIMILARITY SEARCH                                │
│                    (FAISS Vector Index)                                 │
│         ┌─────────────┐              ┌─────────────┐                    │
│         │ Jobs Index  │              │Courses Index│                    │
│         └─────────────┘              └─────────────┘                    │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        EMBEDDING ENGINE                                  │
│              (SentenceTransformers: all-MiniLM-L6-v2)                   │
│                                                                         │
│    Text Input ──► Tokenization ──► Transformer ──► 384-dim Vector      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Algorithm

### 1. Text Embedding with Transformer Models

We use the **all-MiniLM-L6-v2** model from SentenceTransformers, which converts text into dense 384-dimensional vectors that capture semantic meaning.

**Why this model?**
- Optimized for semantic similarity tasks
- Fast inference (~14,000 sentences/second on GPU)
- Strong performance on paraphrase and similarity benchmarks
- Compact 384-dimensional embeddings

**Embedding Process:**
```
Input Text ──► Tokenization ──► BERT Encoder ──► Mean Pooling ──► L2 Normalize ──► 384-dim Vector
```

### 2. Similarity Computation

We use **cosine similarity** to measure the semantic closeness between vectors:

```
                    A · B
cos(θ) = ─────────────────────
          ||A|| × ||B||
```

Where:
- `A` and `B` are embedding vectors
- `A · B` is the dot product
- `||A||` and `||B||` are L2 norms

For normalized vectors (L2 norm = 1), cosine similarity simplifies to dot product, which FAISS computes efficiently using **IndexFlatIP** (Inner Product).

### 3. Vector Search with FAISS

**FAISS (Facebook AI Similarity Search)** enables efficient similarity search at scale:

```python
# Index Creation
index = faiss.IndexFlatIP(384)  # 384-dim, Inner Product

# Add embeddings
index.add(job_embeddings)  # Shape: (n_jobs, 384)

# Search
distances, indices = index.search(query_embedding, k=10)
```

**Complexity:**
- Exact search: O(n × d) where n = number of vectors, d = dimension
- For larger datasets, IVF (Inverted File) index reduces to O(n/c × d) where c = number of clusters

### 4. Job Matching Algorithm

```python
def recommend_jobs(user_profile, top_k=10):
    # Step 1: Create text representation
    user_text = f"""
        Current Role: {user_profile.current_role}
        Skills: {', '.join(user_profile.skills)}
        Experience: {user_profile.experience_years} years
        Career Goals: {user_profile.career_goals}
    """
    
    # Step 2: Generate embedding
    user_embedding = model.encode(user_text, normalize=True)
    
    # Step 3: Search FAISS index
    similarities, job_indices = jobs_index.search(user_embedding, top_k)
    
    # Step 4: Rank and return results
    recommendations = []
    for idx, sim in zip(job_indices, similarities):
        job = jobs[idx]
        matching_skills = set(user_profile.skills) & set(job.required_skills)
        missing_skills = set(job.required_skills) - set(user_profile.skills)
        
        recommendations.append({
            'job': job,
            'similarity_score': sim,
            'matching_skills': matching_skills,
            'missing_skills': missing_skills
        })
    
    return recommendations
```

### 5. Skill Gap Analysis

The skill gap analysis identifies skills needed for a target role:

```python
def analyze_skill_gaps(user_skills, target_role):
    # Step 1: Find jobs matching target role
    role_embedding = model.encode(target_role)
    _, job_indices = jobs_index.search(role_embedding, k=5)
    
    # Step 2: Aggregate required skills with weights
    skill_importance = defaultdict(float)
    for idx in job_indices:
        job = jobs[idx]
        for skill in job.required_skills:
            skill_importance[skill] += 2.0  # Higher weight
        for skill in job.preferred_skills:
            skill_importance[skill] += 1.0  # Lower weight
    
    # Step 3: Identify gaps
    user_skills_lower = {s.lower() for s in user_skills}
    gaps = [
        SkillGap(skill=skill, importance=score/max_score)
        for skill, score in skill_importance.items()
        if skill.lower() not in user_skills_lower
    ]
    
    return sorted(gaps, key=lambda x: -x.importance)
```

### 6. Course Recommendation Pipeline

```
Skill Gaps ──► Create Query ──► Embed Query ──► Search Courses ──► Rank by Relevance
                   │
                   ▼
          "Learn Python, Machine Learning, TensorFlow"
```

Relevance scoring combines:
- **Semantic similarity** (from FAISS search)
- **Skill coverage bonus** (+0.2 per skill gap addressed)

### 7. Career Path Generation

Career paths are generated by:
1. Finding related roles via semantic similarity
2. Clustering roles by experience level
3. Ordering by career progression (Entry → Mid → Senior → Lead)

```
Entry Level ──► Mid Level ──► Senior ──► Lead/Principal
     │              │            │            │
Data Analyst ─► Data Scientist ─► Senior DS ─► Principal DS
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) | Semantic text representation |
| **Vector Search** | FAISS | Efficient similarity search |
| **Backend** | FastAPI | REST API with async support |
| **Frontend** | Streamlit | Interactive dashboard |
| **Visualization** | Plotly | Charts and career graphs |
| **Data Models** | Pydantic | Request/response validation |
| **Data Processing** | Pandas, NumPy | Data manipulation |

## Project Structure

```
ISR Project/
├── data/
│   ├── raw/                    # Source datasets
│   │   ├── jobs.json           # Job postings (15 samples)
│   │   ├── courses.json        # Course catalog (20 samples)
│   │   └── skills.json         # Skills taxonomy (20 skills)
│   └── processed/              # Cached embeddings & FAISS indices
│       ├── jobs.faiss          # Job embeddings index
│       ├── jobs_ids.npy        # Job ID mapping
│       ├── courses.faiss       # Course embeddings index
│       └── courses_ids.npy     # Course ID mapping
├── src/
│   ├── core/
│   │   ├── embeddings.py       # EmbeddingEngine class
│   │   ├── similarity.py       # SimilaritySearch with FAISS
│   │   └── recommender.py      # CareerRecommender main engine
│   ├── data/
│   │   ├── loader.py           # DataLoader for JSON/CSV
│   │   └── preprocessor.py     # TextPreprocessor for cleaning
│   └── models/
│       └── schemas.py          # Pydantic models (UserProfile, JobPosting, etc.)
├── api/
│   └── main.py                 # FastAPI application
├── ui/
│   └── app.py                  # Streamlit dashboard
├── scripts/
│   └── download_data.py        # Data generation script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.10+
- pip package manager

### Setup Steps

1. **Clone and navigate to the project:**
   ```bash
   cd "ISR Project"
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate sample data:**
   ```bash
   python scripts/download_data.py
   ```

## Usage

### Option 1: Streamlit Dashboard (Recommended)

```bash
streamlit run ui/app.py
```

Open http://localhost:8501 in your browser.

**Dashboard Features:**
- Sidebar: Enter your profile (skills, experience, goals)
- Tab 1: View job recommendations with match scores
- Tab 2: Analyze skill gaps for target roles
- Tab 3: Get course recommendations
- Tab 4: Visualize career progression paths

### Option 2: FastAPI Backend

```bash
uvicorn api.main:app --reload --port 8000
```

- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Option 3: Python API

```python
from src.core.recommender import get_recommender
from src.models.schemas import UserProfile

# Initialize
recommender = get_recommender()
recommender.initialize()

# Create profile
profile = UserProfile(
    skills=["Python", "SQL", "Data Analysis"],
    current_role="Data Analyst",
    experience_years=3,
    career_goals="Become a Data Scientist"
)

# Get recommendations
jobs = recommender.recommend_jobs(profile, top_k=5)
for rec in jobs:
    print(f"{rec.job.title}: {rec.similarity_score:.0%} match")

# Analyze skill gaps
gaps = recommender.analyze_skill_gaps(profile, "Data Scientist")
for gap in gaps[:5]:
    print(f"Learn: {gap.skill} (importance: {gap.importance:.0%})")

# Get course recommendations
courses = recommender.recommend_courses(gaps, top_k=5)
for rec in courses:
    print(f"{rec.course.title} - covers: {rec.skills_addressed}")
```

## API Reference

### POST /api/recommend/jobs

Get job recommendations based on user profile.

**Request Body:**
```json
{
  "profile": {
    "skills": ["Python", "Machine Learning", "SQL"],
    "current_role": "Data Analyst",
    "experience_years": 3,
    "career_goals": "Transition to Data Science"
  },
  "top_k": 10
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "job": {
        "id": "job_001",
        "title": "Data Scientist",
        "company": "Tech Corp",
        "description": "...",
        "required_skills": ["Python", "ML", "SQL"],
        "salary_range": "$120,000 - $160,000"
      },
      "similarity_score": 0.85,
      "matching_skills": ["Python", "SQL"],
      "missing_skills": ["TensorFlow"]
    }
  ],
  "total_jobs_searched": 15
}
```

### POST /api/analyze/skills

Analyze skill gaps for a target role.

**Request Body:**
```json
{
  "profile": {
    "skills": ["Python", "SQL"]
  },
  "target_role": "Machine Learning Engineer"
}
```

**Response:**
```json
{
  "target_role": "Machine Learning Engineer",
  "skill_gaps": [
    {"skill": "TensorFlow", "importance": 0.95, "category": "AI/ML Frameworks"},
    {"skill": "PyTorch", "importance": 0.90, "category": "AI/ML Frameworks"},
    {"skill": "Docker", "importance": 0.75, "category": "DevOps"}
  ],
  "courses": [...]
}
```

### GET /api/career-path/{role}

Get career progression path from a starting role.

**Example:** `GET /api/career-path/Data%20Analyst?target_role=Data%20Scientist`

**Response:**
```json
{
  "career_path": {
    "target_role": "Data Scientist",
    "current_level": 2,
    "path_nodes": [
      {"role": "Data Analyst", "level": 1, "required_skills": ["SQL", "Excel"]},
      {"role": "Data Scientist", "level": 2, "required_skills": ["Python", "ML"]},
      {"role": "Senior Data Scientist", "level": 3, "required_skills": ["Deep Learning"]}
    ],
    "recommended_transitions": ["Data Scientist", "ML Engineer"]
  }
}
```

## Evaluation Metrics

The system can be evaluated using:

### Retrieval Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Precision@K** | `relevant_in_top_k / k` | Fraction of top-k results that are relevant |
| **Recall@K** | `relevant_in_top_k / total_relevant` | Fraction of relevant items in top-k |
| **MRR** | `1 / rank_of_first_relevant` | Mean Reciprocal Rank |
| **NDCG@K** | Normalized DCG | Considers position and relevance grades |

### Example Evaluation

```python
# For a user seeking "Data Scientist" roles
recommended_jobs = recommender.recommend_jobs(profile, top_k=10)

# If 7 out of 10 recommendations are relevant:
precision_at_10 = 7 / 10  # = 0.70

# If there are 15 relevant jobs total:
recall_at_10 = 7 / 15     # = 0.47
```

## Data Sources

The system supports multiple data sources:

1. **Sample Data** (included): 15 jobs, 20 courses, 20 skills
2. **Kaggle Datasets**: LinkedIn job postings, Coursera courses
3. **Custom Data**: Add your own JSON/CSV files to `data/raw/`

### Data Schema

**Jobs (`jobs.json`):**
```json
{
  "id": "job_001",
  "title": "Data Scientist",
  "company": "Tech Corp",
  "description": "Job description...",
  "required_skills": ["Python", "ML"],
  "preferred_skills": ["TensorFlow"],
  "experience_level": "Mid-Level",
  "location": "San Francisco",
  "salary_range": "$120K-$160K"
}
```

**Courses (`courses.json`):**
```json
{
  "id": "course_001",
  "title": "Machine Learning Specialization",
  "provider": "Coursera",
  "description": "Course description...",
  "skills_taught": ["Machine Learning", "Python"],
  "difficulty": "Intermediate",
  "duration": "3 months",
  "rating": 4.9
}
```

## Future Enhancements

- [ ] Real-time job data from LinkedIn/Indeed APIs
- [ ] User authentication and profile persistence
- [ ] Collaborative filtering for improved recommendations
- [ ] Resume parsing for automatic skill extraction
- [ ] Salary prediction based on skills and experience
- [ ] Interview preparation recommendations

## Team

| Name | Role | NetID |
|------|------|-------|
| Ubaid Khan Mohammed | Developer | mohammed.u |
| Manoj Siva Sai Deepank | Developer | deeps45 |
| Yaswanth Yaradoddi | Developer | yaswanthreddy |

## License

This project is developed for academic purposes as part of the **Information Storage and Retrieval** course (October 2025).

---

*Built with modern IR and NLP techniques to transform career guidance into a personalized, intelligent, and scalable system.*
