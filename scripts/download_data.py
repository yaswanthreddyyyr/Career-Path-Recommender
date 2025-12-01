"""
Data download script for the Smart Career Path Recommender.

Downloads job postings, courses, and skills data from Kaggle and other sources.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def ensure_directories():
    """Create necessary data directories."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"Created directories: {DATA_RAW}, {DATA_PROCESSED}")


def download_kaggle_dataset(dataset_name: str, output_path: Path) -> bool:
    """
    Download a dataset from Kaggle using kagglehub.
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'arshkon/linkedin-job-postings')
        output_path: Path to save the downloaded files
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import kagglehub
        
        print(f"Downloading {dataset_name} from Kaggle...")
        path = kagglehub.dataset_download(dataset_name)
        print(f"Downloaded to: {path}")
        
        # Copy files to our data directory
        import shutil
        for file in Path(path).glob("*"):
            dest = output_path / file.name
            if file.is_file():
                shutil.copy2(file, dest)
                print(f"  Copied: {file.name}")
        
        return True
    except Exception as e:
        print(f"Failed to download {dataset_name}: {e}")
        return False


def create_sample_jobs_data() -> pd.DataFrame:
    """Create sample job postings data for development."""
    jobs_data = [
        {
            "id": "job_001",
            "title": "Data Scientist",
            "company": "Tech Corp",
            "description": "We are looking for a Data Scientist to analyze large datasets and build predictive models. You will work with machine learning algorithms, statistical analysis, and data visualization.",
            "required_skills": ["Python", "Machine Learning", "SQL", "Statistics", "Data Visualization"],
            "preferred_skills": ["TensorFlow", "PyTorch", "Spark", "AWS"],
            "experience_level": "Mid-Level",
            "location": "San Francisco, CA",
            "salary_range": "$120,000 - $160,000"
        },
        {
            "id": "job_002",
            "title": "Machine Learning Engineer",
            "company": "AI Innovations",
            "description": "Join our team to develop and deploy machine learning models at scale. Work on deep learning, NLP, and computer vision projects.",
            "required_skills": ["Python", "TensorFlow", "PyTorch", "Docker", "Kubernetes"],
            "preferred_skills": ["MLOps", "AWS SageMaker", "Model Optimization"],
            "experience_level": "Senior",
            "location": "New York, NY",
            "salary_range": "$150,000 - $200,000"
        },
        {
            "id": "job_003",
            "title": "Software Engineer",
            "company": "StartupXYZ",
            "description": "Build scalable backend systems and APIs. Work with microservices architecture and cloud technologies.",
            "required_skills": ["Python", "Java", "REST APIs", "PostgreSQL", "Git"],
            "preferred_skills": ["Kubernetes", "Redis", "GraphQL", "CI/CD"],
            "experience_level": "Mid-Level",
            "location": "Austin, TX",
            "salary_range": "$100,000 - $140,000"
        },
        {
            "id": "job_004",
            "title": "Data Analyst",
            "company": "Analytics Plus",
            "description": "Analyze business data to provide actionable insights. Create reports and dashboards for stakeholders.",
            "required_skills": ["SQL", "Excel", "Tableau", "Statistics", "Data Visualization"],
            "preferred_skills": ["Python", "R", "Power BI"],
            "experience_level": "Entry-Level",
            "location": "Chicago, IL",
            "salary_range": "$60,000 - $85,000"
        },
        {
            "id": "job_005",
            "title": "Full Stack Developer",
            "company": "WebDev Inc",
            "description": "Develop web applications using modern frameworks. Work on both frontend and backend development.",
            "required_skills": ["JavaScript", "React", "Node.js", "HTML/CSS", "MongoDB"],
            "preferred_skills": ["TypeScript", "GraphQL", "AWS", "Docker"],
            "experience_level": "Mid-Level",
            "location": "Seattle, WA",
            "salary_range": "$110,000 - $150,000"
        },
        {
            "id": "job_006",
            "title": "DevOps Engineer",
            "company": "CloudOps",
            "description": "Manage CI/CD pipelines and cloud infrastructure. Implement automation and monitoring solutions.",
            "required_skills": ["AWS", "Docker", "Kubernetes", "Terraform", "Linux"],
            "preferred_skills": ["Python", "Ansible", "Prometheus", "Jenkins"],
            "experience_level": "Senior",
            "location": "Remote",
            "salary_range": "$140,000 - $180,000"
        },
        {
            "id": "job_007",
            "title": "Product Manager",
            "company": "ProductLab",
            "description": "Lead product development from ideation to launch. Work with engineering and design teams.",
            "required_skills": ["Product Strategy", "Agile", "User Research", "Data Analysis", "Communication"],
            "preferred_skills": ["SQL", "A/B Testing", "Roadmapping", "Stakeholder Management"],
            "experience_level": "Senior",
            "location": "Boston, MA",
            "salary_range": "$130,000 - $170,000"
        },
        {
            "id": "job_008",
            "title": "Backend Engineer",
            "company": "ServerSide Co",
            "description": "Design and implement backend services and APIs. Focus on performance and scalability.",
            "required_skills": ["Python", "Django", "PostgreSQL", "REST APIs", "Redis"],
            "preferred_skills": ["FastAPI", "Celery", "Elasticsearch", "AWS"],
            "experience_level": "Mid-Level",
            "location": "Denver, CO",
            "salary_range": "$105,000 - $145,000"
        },
        {
            "id": "job_009",
            "title": "NLP Engineer",
            "company": "LanguageAI",
            "description": "Develop natural language processing solutions. Work on text classification, named entity recognition, and language models.",
            "required_skills": ["Python", "NLP", "Transformers", "PyTorch", "BERT"],
            "preferred_skills": ["Hugging Face", "spaCy", "LLMs", "Vector Databases"],
            "experience_level": "Mid-Level",
            "location": "San Jose, CA",
            "salary_range": "$130,000 - $170,000"
        },
        {
            "id": "job_010",
            "title": "Data Engineer",
            "company": "DataFlow Systems",
            "description": "Build and maintain data pipelines. Work with big data technologies and ETL processes.",
            "required_skills": ["Python", "SQL", "Spark", "Airflow", "AWS"],
            "preferred_skills": ["Kafka", "Snowflake", "dbt", "Data Modeling"],
            "experience_level": "Mid-Level",
            "location": "Atlanta, GA",
            "salary_range": "$115,000 - $155,000"
        },
        {
            "id": "job_011",
            "title": "Frontend Developer",
            "company": "UIUXStudio",
            "description": "Create beautiful and responsive web interfaces. Focus on user experience and performance.",
            "required_skills": ["JavaScript", "React", "HTML/CSS", "TypeScript", "Git"],
            "preferred_skills": ["Next.js", "Tailwind CSS", "Testing", "Figma"],
            "experience_level": "Mid-Level",
            "location": "Los Angeles, CA",
            "salary_range": "$95,000 - $135,000"
        },
        {
            "id": "job_012",
            "title": "AI Research Scientist",
            "company": "Research Labs",
            "description": "Conduct cutting-edge AI research. Publish papers and develop novel algorithms.",
            "required_skills": ["Python", "Deep Learning", "Mathematics", "Research", "PyTorch"],
            "preferred_skills": ["PhD", "Publications", "Reinforcement Learning", "Computer Vision"],
            "experience_level": "Senior",
            "location": "Palo Alto, CA",
            "salary_range": "$180,000 - $250,000"
        },
        {
            "id": "job_013",
            "title": "Security Engineer",
            "company": "SecureNet",
            "description": "Implement security measures and conduct vulnerability assessments. Protect systems from cyber threats.",
            "required_skills": ["Security", "Python", "Linux", "Networking", "Penetration Testing"],
            "preferred_skills": ["CISSP", "Cloud Security", "SIEM", "Incident Response"],
            "experience_level": "Senior",
            "location": "Washington, DC",
            "salary_range": "$135,000 - $175,000"
        },
        {
            "id": "job_014",
            "title": "Mobile Developer",
            "company": "AppMakers",
            "description": "Develop iOS and Android applications. Focus on performance and user experience.",
            "required_skills": ["Swift", "Kotlin", "Mobile Development", "REST APIs", "Git"],
            "preferred_skills": ["React Native", "Flutter", "CI/CD", "App Store Optimization"],
            "experience_level": "Mid-Level",
            "location": "Miami, FL",
            "salary_range": "$100,000 - $140,000"
        },
        {
            "id": "job_015",
            "title": "Business Intelligence Analyst",
            "company": "InsightCorp",
            "description": "Create BI dashboards and reports. Transform data into actionable business insights.",
            "required_skills": ["SQL", "Tableau", "Power BI", "Data Analysis", "Excel"],
            "preferred_skills": ["Python", "ETL", "Data Warehousing", "Looker"],
            "experience_level": "Mid-Level",
            "location": "Minneapolis, MN",
            "salary_range": "$80,000 - $110,000"
        }
    ]
    
    return pd.DataFrame(jobs_data)


def create_sample_courses_data() -> pd.DataFrame:
    """Create sample courses data for development."""
    courses_data = [
        {
            "id": "course_001",
            "title": "Machine Learning Specialization",
            "provider": "Coursera",
            "description": "Learn machine learning from Andrew Ng. Covers supervised learning, unsupervised learning, and best practices.",
            "skills_taught": ["Machine Learning", "Python", "TensorFlow", "Neural Networks"],
            "difficulty": "Intermediate",
            "duration": "3 months",
            "url": "https://coursera.org/ml-specialization",
            "rating": 4.9
        },
        {
            "id": "course_002",
            "title": "Deep Learning Specialization",
            "provider": "Coursera",
            "description": "Master deep learning fundamentals. Build neural networks, CNNs, RNNs, and more.",
            "skills_taught": ["Deep Learning", "TensorFlow", "Neural Networks", "Computer Vision", "NLP"],
            "difficulty": "Advanced",
            "duration": "4 months",
            "url": "https://coursera.org/deep-learning",
            "rating": 4.8
        },
        {
            "id": "course_003",
            "title": "Python for Data Science",
            "provider": "edX",
            "description": "Learn Python programming for data analysis. Covers pandas, numpy, and visualization.",
            "skills_taught": ["Python", "Pandas", "NumPy", "Data Visualization", "Matplotlib"],
            "difficulty": "Beginner",
            "duration": "6 weeks",
            "url": "https://edx.org/python-data-science",
            "rating": 4.6
        },
        {
            "id": "course_004",
            "title": "SQL for Data Analysis",
            "provider": "Udacity",
            "description": "Master SQL for querying databases. Learn joins, aggregations, and window functions.",
            "skills_taught": ["SQL", "Database Management", "Data Analysis", "PostgreSQL"],
            "difficulty": "Beginner",
            "duration": "4 weeks",
            "url": "https://udacity.com/sql-data-analysis",
            "rating": 4.5
        },
        {
            "id": "course_005",
            "title": "AWS Cloud Practitioner",
            "provider": "AWS",
            "description": "Learn AWS cloud fundamentals. Prepare for the AWS Cloud Practitioner certification.",
            "skills_taught": ["AWS", "Cloud Computing", "EC2", "S3", "IAM"],
            "difficulty": "Beginner",
            "duration": "6 weeks",
            "url": "https://aws.amazon.com/training",
            "rating": 4.7
        },
        {
            "id": "course_006",
            "title": "Docker and Kubernetes",
            "provider": "Udemy",
            "description": "Learn containerization with Docker and orchestration with Kubernetes.",
            "skills_taught": ["Docker", "Kubernetes", "DevOps", "Container Orchestration"],
            "difficulty": "Intermediate",
            "duration": "8 weeks",
            "url": "https://udemy.com/docker-kubernetes",
            "rating": 4.6
        },
        {
            "id": "course_007",
            "title": "React - The Complete Guide",
            "provider": "Udemy",
            "description": "Build modern web applications with React. Covers hooks, Redux, and Next.js.",
            "skills_taught": ["React", "JavaScript", "Redux", "Next.js", "Frontend Development"],
            "difficulty": "Intermediate",
            "duration": "10 weeks",
            "url": "https://udemy.com/react-complete-guide",
            "rating": 4.8
        },
        {
            "id": "course_008",
            "title": "Natural Language Processing",
            "provider": "Coursera",
            "description": "Learn NLP techniques including text classification, NER, and transformers.",
            "skills_taught": ["NLP", "Python", "Transformers", "BERT", "Text Processing"],
            "difficulty": "Advanced",
            "duration": "3 months",
            "url": "https://coursera.org/nlp-specialization",
            "rating": 4.7
        },
        {
            "id": "course_009",
            "title": "Statistics for Data Science",
            "provider": "edX",
            "description": "Master statistics fundamentals for data science. Covers probability, hypothesis testing, and regression.",
            "skills_taught": ["Statistics", "Probability", "Hypothesis Testing", "Regression", "Data Analysis"],
            "difficulty": "Intermediate",
            "duration": "8 weeks",
            "url": "https://edx.org/statistics-data-science",
            "rating": 4.5
        },
        {
            "id": "course_010",
            "title": "FastAPI Modern Python Web Development",
            "provider": "Udemy",
            "description": "Build fast APIs with Python and FastAPI. Learn async programming and API best practices.",
            "skills_taught": ["FastAPI", "Python", "REST APIs", "Async Programming", "Pydantic"],
            "difficulty": "Intermediate",
            "duration": "6 weeks",
            "url": "https://udemy.com/fastapi",
            "rating": 4.7
        },
        {
            "id": "course_011",
            "title": "Data Engineering with Python",
            "provider": "DataCamp",
            "description": "Learn to build data pipelines. Covers ETL, Airflow, and data warehousing.",
            "skills_taught": ["Data Engineering", "Python", "Airflow", "ETL", "Data Pipelines"],
            "difficulty": "Intermediate",
            "duration": "8 weeks",
            "url": "https://datacamp.com/data-engineering",
            "rating": 4.4
        },
        {
            "id": "course_012",
            "title": "Tableau for Data Visualization",
            "provider": "Coursera",
            "description": "Create impactful visualizations and dashboards with Tableau.",
            "skills_taught": ["Tableau", "Data Visualization", "Dashboard Design", "Data Analysis"],
            "difficulty": "Beginner",
            "duration": "4 weeks",
            "url": "https://coursera.org/tableau",
            "rating": 4.6
        },
        {
            "id": "course_013",
            "title": "Git and GitHub for Beginners",
            "provider": "Udemy",
            "description": "Learn version control with Git and collaboration with GitHub.",
            "skills_taught": ["Git", "GitHub", "Version Control", "Collaboration"],
            "difficulty": "Beginner",
            "duration": "2 weeks",
            "url": "https://udemy.com/git-github",
            "rating": 4.5
        },
        {
            "id": "course_014",
            "title": "PyTorch for Deep Learning",
            "provider": "Udacity",
            "description": "Master PyTorch for building deep learning models. Covers CNNs, RNNs, and GANs.",
            "skills_taught": ["PyTorch", "Deep Learning", "Neural Networks", "Computer Vision"],
            "difficulty": "Advanced",
            "duration": "3 months",
            "url": "https://udacity.com/pytorch",
            "rating": 4.8
        },
        {
            "id": "course_015",
            "title": "Product Management Fundamentals",
            "provider": "Coursera",
            "description": "Learn product management skills. Covers roadmapping, user research, and agile.",
            "skills_taught": ["Product Management", "Agile", "User Research", "Roadmapping", "Product Strategy"],
            "difficulty": "Beginner",
            "duration": "6 weeks",
            "url": "https://coursera.org/product-management",
            "rating": 4.5
        },
        {
            "id": "course_016",
            "title": "Apache Spark for Big Data",
            "provider": "edX",
            "description": "Process big data with Apache Spark. Learn PySpark and distributed computing.",
            "skills_taught": ["Spark", "PySpark", "Big Data", "Distributed Computing", "Data Processing"],
            "difficulty": "Advanced",
            "duration": "8 weeks",
            "url": "https://edx.org/apache-spark",
            "rating": 4.6
        },
        {
            "id": "course_017",
            "title": "Cybersecurity Fundamentals",
            "provider": "Coursera",
            "description": "Learn security fundamentals. Covers network security, cryptography, and threat analysis.",
            "skills_taught": ["Security", "Networking", "Cryptography", "Threat Analysis", "Penetration Testing"],
            "difficulty": "Intermediate",
            "duration": "3 months",
            "url": "https://coursera.org/cybersecurity",
            "rating": 4.7
        },
        {
            "id": "course_018",
            "title": "Node.js Backend Development",
            "provider": "Udemy",
            "description": "Build scalable backend applications with Node.js. Covers Express, MongoDB, and REST APIs.",
            "skills_taught": ["Node.js", "JavaScript", "Express", "MongoDB", "REST APIs"],
            "difficulty": "Intermediate",
            "duration": "8 weeks",
            "url": "https://udemy.com/nodejs",
            "rating": 4.6
        },
        {
            "id": "course_019",
            "title": "MLOps: Machine Learning Operations",
            "provider": "Coursera",
            "description": "Learn to deploy and manage ML models in production. Covers CI/CD for ML.",
            "skills_taught": ["MLOps", "Machine Learning", "Docker", "Kubernetes", "Model Deployment"],
            "difficulty": "Advanced",
            "duration": "3 months",
            "url": "https://coursera.org/mlops",
            "rating": 4.7
        },
        {
            "id": "course_020",
            "title": "Power BI for Business Intelligence",
            "provider": "Microsoft Learn",
            "description": "Create interactive reports and dashboards with Power BI.",
            "skills_taught": ["Power BI", "Data Visualization", "Business Intelligence", "DAX", "Data Modeling"],
            "difficulty": "Intermediate",
            "duration": "6 weeks",
            "url": "https://learn.microsoft.com/power-bi",
            "rating": 4.6
        }
    ]
    
    return pd.DataFrame(courses_data)


def create_sample_skills_data() -> pd.DataFrame:
    """Create sample skills taxonomy data."""
    skills_data = [
        {"skill": "Python", "category": "Programming Languages", "related_skills": ["Java", "R", "JavaScript"]},
        {"skill": "Machine Learning", "category": "AI/ML", "related_skills": ["Deep Learning", "Statistics", "Data Science"]},
        {"skill": "SQL", "category": "Databases", "related_skills": ["PostgreSQL", "MySQL", "NoSQL"]},
        {"skill": "TensorFlow", "category": "AI/ML Frameworks", "related_skills": ["PyTorch", "Keras", "Scikit-learn"]},
        {"skill": "PyTorch", "category": "AI/ML Frameworks", "related_skills": ["TensorFlow", "Deep Learning", "Neural Networks"]},
        {"skill": "AWS", "category": "Cloud", "related_skills": ["GCP", "Azure", "Cloud Computing"]},
        {"skill": "Docker", "category": "DevOps", "related_skills": ["Kubernetes", "Containerization", "CI/CD"]},
        {"skill": "Kubernetes", "category": "DevOps", "related_skills": ["Docker", "Cloud", "Orchestration"]},
        {"skill": "React", "category": "Frontend", "related_skills": ["JavaScript", "TypeScript", "Vue.js"]},
        {"skill": "JavaScript", "category": "Programming Languages", "related_skills": ["TypeScript", "Node.js", "React"]},
        {"skill": "Data Visualization", "category": "Data Analysis", "related_skills": ["Tableau", "Power BI", "Matplotlib"]},
        {"skill": "Statistics", "category": "Mathematics", "related_skills": ["Probability", "Machine Learning", "Data Analysis"]},
        {"skill": "NLP", "category": "AI/ML", "related_skills": ["Transformers", "BERT", "Text Processing"]},
        {"skill": "Deep Learning", "category": "AI/ML", "related_skills": ["Neural Networks", "TensorFlow", "PyTorch"]},
        {"skill": "Git", "category": "Tools", "related_skills": ["GitHub", "Version Control", "CI/CD"]},
        {"skill": "REST APIs", "category": "Backend", "related_skills": ["FastAPI", "Flask", "GraphQL"]},
        {"skill": "Spark", "category": "Big Data", "related_skills": ["Hadoop", "Data Engineering", "PySpark"]},
        {"skill": "Agile", "category": "Methodologies", "related_skills": ["Scrum", "Product Management", "Project Management"]},
        {"skill": "Linux", "category": "Operating Systems", "related_skills": ["Bash", "System Administration", "DevOps"]},
        {"skill": "Transformers", "category": "AI/ML", "related_skills": ["BERT", "NLP", "Hugging Face"]},
    ]
    
    return pd.DataFrame(skills_data)


def try_download_kaggle_datasets():
    """Attempt to download real datasets from Kaggle."""
    datasets = [
        ("arshkon/linkedin-job-postings", "LinkedIn job postings"),
        ("shivamb/real-or-fake-fake-jobposting-prediction", "Job postings with skills"),
    ]
    
    for dataset_id, description in datasets:
        print(f"\nAttempting to download: {description}")
        if download_kaggle_dataset(dataset_id, DATA_RAW):
            return True
    
    return False


def save_sample_data():
    """Save sample data for development."""
    print("\nGenerating sample data for development...")
    
    # Create sample datasets
    jobs_df = create_sample_jobs_data()
    courses_df = create_sample_courses_data()
    skills_df = create_sample_skills_data()
    
    # Save to CSV
    jobs_df.to_csv(DATA_RAW / "jobs.csv", index=False)
    courses_df.to_csv(DATA_RAW / "courses.csv", index=False)
    skills_df.to_csv(DATA_RAW / "skills.csv", index=False)
    
    # Also save as JSON for easier loading with nested fields
    jobs_df.to_json(DATA_RAW / "jobs.json", orient="records", indent=2)
    courses_df.to_json(DATA_RAW / "courses.json", orient="records", indent=2)
    skills_df.to_json(DATA_RAW / "skills.json", orient="records", indent=2)
    
    print(f"Saved {len(jobs_df)} job postings to {DATA_RAW / 'jobs.csv'}")
    print(f"Saved {len(courses_df)} courses to {DATA_RAW / 'courses.csv'}")
    print(f"Saved {len(skills_df)} skills to {DATA_RAW / 'skills.csv'}")


def main():
    """Main function to download and prepare data."""
    print("=" * 60)
    print("Smart Career Path Recommender - Data Download")
    print("=" * 60)
    
    # Ensure directories exist
    ensure_directories()
    
    # Try to download from Kaggle (requires kaggle credentials)
    kaggle_success = False
    try:
        kaggle_success = try_download_kaggle_datasets()
    except Exception as e:
        print(f"\nKaggle download failed: {e}")
        print("This is normal if you don't have Kaggle credentials configured.")
    
    # Always create sample data for development
    save_sample_data()
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nData saved to: {DATA_RAW}")
    print("\nYou can now run the embedding generation step.")


if __name__ == "__main__":
    main()

