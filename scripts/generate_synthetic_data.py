"""
Synthetic Data Generator for Smart Career Path Recommender.

Generates realistic job postings and courses by mixing and matching
skills, roles, companies, and templates.
"""

import json
import random
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# --- Constants & Templates ---

ROLES = {
    "Data Scientist": ["Python", "SQL", "Machine Learning", "Statistics", "Data Visualization", "Pandas", "Scikit-learn", "TensorFlow", "PyTorch"],
    "Data Analyst": ["SQL", "Excel", "Tableau", "Power BI", "Data Analysis", "Python", "Statistics", "Communication"],
    "Machine Learning Engineer": ["Python", "TensorFlow", "PyTorch", "Docker", "Kubernetes", "AWS", "MLOps", "Git", "CI/CD"],
    "Software Engineer": ["Python", "Java", "JavaScript", "Git", "SQL", "REST APIs", "Agile", "Problem Solving"],
    "Frontend Developer": ["JavaScript", "React", "HTML", "CSS", "TypeScript", "Redux", "Figma", "Responsive Design"],
    "Backend Developer": ["Python", "Java", "Node.js", "SQL", "NoSQL", "API Design", "Docker", "AWS", "Microservices"],
    "DevOps Engineer": ["AWS", "Docker", "Kubernetes", "Linux", "Jenkins", "Terraform", "Ansible", "CI/CD", "Scripting"],
    "Product Manager": ["Product Strategy", "Agile", "User Research", "Roadmapping", "Communication", "Data Analysis", "JIRA", "Leadership"],
    "Full Stack Developer": ["JavaScript", "Python", "React", "Node.js", "SQL", "Git", "AWS", "HTML/CSS"],
    "Cloud Architect": ["AWS", "Azure", "GCP", "Cloud Computing", "System Design", "Security", "Networking", "Docker"],
    "Cybersecurity Analyst": ["Network Security", "Python", "Linux", "Penetration Testing", "SIEM", "Firewalls", "Risk Assessment"],
    "UI/UX Designer": ["Figma", "Sketch", "User Research", "Prototyping", "Wireframing", "Adobe XD", "HTML/CSS"],
    "QA Engineer": ["Selenium", "Python", "Java", "Testing", "JIRA", "Automation", "SQL", "Git"],
}

COMPANIES = [
    "TechCorp", "DataFlow Systems", "InnovateAI", "CloudScale", "WebSolutions", 
    "FutureTech", "SmartSystems", "GlobalConnect", "NextGen Software", "AlphaBit",
    "CyberGuard", "EcoTech", "HealthPlus", "FinServe", "EduLearn", "RetailGiant",
    "MediaStream", "AutoDrive", "GreenEnergy", "SpaceXplore"
]

LOCATIONS = [
    "Remote", "San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA",
    "Boston, MA", "Chicago, IL", "Los Angeles, CA", "Denver, CO", "Atlanta, GA",
    "London, UK", "Toronto, Canada", "Berlin, Germany", "Bangalore, India"
]

EXPERIENCE_LEVELS = ["Entry-Level", "Junior", "Mid-Level", "Senior", "Lead", "Principal"]

COURSE_PROVIDERS = ["Coursera", "Udemy", "edX", "Udacity", "Pluralsight", "DataCamp", "Codecademy"]

# Description Templates
JOB_DESC_TEMPLATES = [
    "We are seeking a talented {title} to join our team at {company}. You will be working on cutting-edge projects involving {skill1} and {skill2}.",
    "Join {company} as a {title}! We are looking for someone with strong experience in {skill1}, {skill2}, and {skill3}. The ideal candidate is passionate about technology.",
    "{company} is hiring a {title}. In this role, you will leverage {skill1} and {skill2} to build scalable solutions. Experience with {skill3} is a plus.",
    "Are you an experienced {title}? {company} needs you to lead our initiatives in {skill1}. Proficiency in {skill2} and {skill3} is required.",
    "Exciting opportunity for a {title} at {company}. You will focus on {skill1} and collaborate with cross-functional teams using {skill2}."
]

COURSE_DESC_TEMPLATES = [
    "Master {skill1} and {skill2} with this comprehensive course. Learn best practices and build real-world projects.",
    "This course covers the fundamentals of {skill1} and advances to complex topics in {skill2}. Perfect for beginners and intermediates.",
    "Become an expert in {skill1} with this intensive training. Includes hands-on labs using {skill2}.",
    "Learn {skill1} from scratch. This course guides you through {skill2} and {skill3} with practical examples.",
    "Advanced specialization in {skill1}. Deep dive into {skill2} and master industry-standard tools."
]

def generate_jobs(n=500):
    """Generate synthetic job postings."""
    jobs = []
    for i in range(n):
        role = random.choice(list(ROLES.keys()))
        company = random.choice(COMPANIES)
        location = random.choice(LOCATIONS)
        level = random.choice(EXPERIENCE_LEVELS)
        
        # Select skills
        role_skills = ROLES[role]
        num_req = random.randint(3, 5)
        num_pref = random.randint(2, 4)
        
        # Shuffle and split skills
        selected_skills = random.sample(role_skills, min(len(role_skills), num_req + num_pref))
        required = selected_skills[:num_req]
        preferred = selected_skills[num_req:]
        
        # Generate description
        template = random.choice(JOB_DESC_TEMPLATES)
        description = template.format(
            title=role,
            company=company,
            skill1=required[0] if required else "technology",
            skill2=required[1] if len(required) > 1 else "innovation",
            skill3=preferred[0] if preferred else "collaboration"
        )
        
        # Salary estimation (rough logic)
        base_salary = 60000
        if "Senior" in level or "Lead" in level: base_salary += 60000
        elif "Mid" in level: base_salary += 30000
        if "San Francisco" in location or "New York" in location: base_salary += 20000
        
        salary_range = f"${base_salary:,} - ${base_salary + 40000:,}"

        jobs.append({
            "id": f"job_{i+1:04d}",
            "title": role if random.random() > 0.2 else f"{level} {role}",
            "company": company,
            "description": description,
            "required_skills": required,
            "preferred_skills": preferred,
            "experience_level": level,
            "location": location,
            "salary_range": salary_range,
            "posted_date": (datetime.now() - timedelta(days=random.randint(0, 60))).strftime("%Y-%m-%d")
        })
    
    return jobs

def generate_courses(n=200):
    """Generate synthetic courses."""
    courses = []
    all_skills = sorted(list({s for skills in ROLES.values() for s in skills}))
    
    for i in range(n):
        provider = random.choice(COURSE_PROVIDERS)
        
        # Focus on 1-3 skills
        num_skills = random.randint(1, 3)
        course_skills = random.sample(all_skills, num_skills)
        main_skill = course_skills[0]
        
        # Generate title
        prefixes = ["Mastering", "Introduction to", "Advanced", "The Complete Guide to", "Learn", "Professional"]
        title = f"{random.choice(prefixes)} {main_skill}"
        if len(course_skills) > 1:
            title += f" and {course_skills[1]}"
            
        # Generate description
        template = random.choice(COURSE_DESC_TEMPLATES)
        description = template.format(
            skill1=main_skill,
            skill2=course_skills[1] if len(course_skills) > 1 else "industry tools",
            skill3=course_skills[2] if len(course_skills) > 2 else "more"
        )
        
        difficulty = random.choice(["Beginner", "Intermediate", "Advanced"])
        rating = round(random.uniform(3.5, 5.0), 1)
        duration = f"{random.randint(4, 12)} weeks"

        courses.append({
            "id": f"course_{i+1:04d}",
            "title": title,
            "provider": provider,
            "description": description,
            "skills_taught": course_skills,
            "difficulty": difficulty,
            "duration": duration,
            "url": f"https://example.com/courses/{i+1}",
            "rating": rating
        })
    
    return courses

def main():
    print("Generating synthetic data...")
    
    # Ensure directory exists
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    # 5,000 jobs is a "sweet spot" - enough density for the graph, 
    # but avoids excessive duplication/wait times.
    jobs = generate_jobs(5000)
    courses = generate_courses(1000)
    
    # Save jobs
    jobs_df = pd.DataFrame(jobs)
    jobs_df.to_csv(DATA_RAW / "jobs.csv", index=False)
    jobs_df.to_json(DATA_RAW / "jobs.json", orient="records", indent=2)
    print(f"Generated {len(jobs)} jobs -> {DATA_RAW / 'jobs.csv'}")
    
    # Save courses
    courses_df = pd.DataFrame(courses)
    courses_df.to_csv(DATA_RAW / "courses.csv", index=False)
    courses_df.to_json(DATA_RAW / "courses.json", orient="records", indent=2)
    print(f"Generated {len(courses)} courses -> {DATA_RAW / 'courses.csv'}")
    
    print("\nDone! You can now re-run the application to index this new data.")

if __name__ == "__main__":
    main()

