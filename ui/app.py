"""
Streamlit Dashboard for the Smart Career Path Recommender.

Provides an interactive UI for job recommendations, skill gap analysis,
course recommendations, and career path visualization.
"""

import sys
from pathlib import Path
from collections import Counter

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.recommender import CareerRecommender, get_recommender
from src.core.cv_parser import CareerResumeParser
from src.models.schemas import UserProfile


# Page configuration
st.set_page_config(
    page_title="Smart Career Path Recommender",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
    }
    .skill-tag {
        display: inline-block;
        background-color: #E8F0FE;
        color: #1967D2;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
    .missing-skill-tag {
        background-color: #FCE8E6;
        color: #D93025;
    }
    .job-card {
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .course-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_recommender():
    """Load and cache the recommender system."""
    recommender = get_recommender()
    recommender.initialize()
    return recommender


def render_header():
    """Render the page header."""
    st.markdown('<h1 class="main-header">Smart Career Path Recommender</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Discover your ideal career path with AI-powered recommendations</p>',
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render the sidebar with user profile input."""
    st.sidebar.header("Your Profile")
    
    # Initialize defaults
    default_skills = "Python, Machine Learning, SQL, Data Analysis"
    default_role = "Data Analyst"
    default_edu_index = 1  # Bachelor's Degree
    default_experience = 3
    
    # Resume Upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Resume (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"],
        help="Upload your resume to auto-fill your profile"
    )
    
    if uploaded_file:
        try:
            parser = CareerResumeParser()
            with st.spinner("Parsing resume..."):
                # Save parsed profile in session state to persist
                if "parsed_profile" not in st.session_state or st.session_state.get("uploaded_file_name") != uploaded_file.name:
                    st.session_state.parsed_profile = parser.parse(uploaded_file, uploaded_file.name)
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.sidebar.success("Resume parsed successfully!")
            
            parsed = st.session_state.parsed_profile
            
            # Update defaults if parsed data exists
            if parsed.skills:
                default_skills = ", ".join(parsed.skills)
            
            if parsed.experience_years > 0:
                default_experience = parsed.experience_years
                
            if parsed.education:
                if "PhD" in parsed.education:
                    default_edu_index = 3
                elif "Master" in parsed.education:
                    default_edu_index = 2
                elif "Bachelor" in parsed.education:
                    default_edu_index = 1
                else:
                    default_edu_index = 4 # Other
                    
        except Exception as e:
            st.sidebar.error(f"Error parsing resume: {str(e)}")
    
    # Skills input
    st.sidebar.subheader("Skills")
    skills_input = st.sidebar.text_area(
        "Enter your skills (comma-separated)",
        value=default_skills,
        height=100,
        help="List your technical and soft skills"
    )
    skills = [s.strip() for s in skills_input.split(",") if s.strip()]
    
    # Current role
    current_role = st.sidebar.text_input(
        "Current Role",
        value=default_role,
        help="Your current job title"
    )
    
    # Education
    education_options = ["High School", "Bachelor's Degree", "Master's Degree", "PhD", "Other"]
    education = st.sidebar.selectbox(
        "Education Level",
        education_options,
        index=default_edu_index
    )
    
    # Experience
    experience = st.sidebar.slider(
        "Years of Experience",
        min_value=0,
        max_value=20,
        value=default_experience,
        help="Total years of professional experience"
    )
    
    # Career goals
    st.sidebar.subheader("Career Goals")
    career_goals = st.sidebar.text_area(
        "Describe your career aspirations",
        value="Transition to a Data Scientist role with focus on machine learning",
        height=100
    )
    
    # Target role for skill gap analysis
    target_role = st.sidebar.text_input(
        "Target Role",
        value="Data Scientist",
        help="The role you want to transition to"
    )
    
    return UserProfile(
        skills=skills,
        current_role=current_role,
        education=education,
        experience_years=experience,
        career_goals=career_goals,
    ), target_role


def render_job_recommendations(recommender, profile, top_k=10):
    """Render job recommendations section."""
    st.header("Job Recommendations")
    
    with st.spinner("Finding matching jobs..."):
        recommendations = recommender.recommend_jobs(profile, top_k=top_k)
    
    if not recommendations:
        st.warning("No matching jobs found. Try adjusting your profile.")
        return
    
    # Check for cold start indication (neutral score implies cold start)
    is_cold_start = all(r.similarity_score == 0.5 for r in recommendations)
    if is_cold_start:
        st.info("👋 We noticed your profile is a bit empty. Here are some popular **Entry-Level** roles to get you started! Add skills to see personalized matches.")
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jobs Found", len(recommendations))
    with col2:
        avg_score = sum(r.similarity_score for r in recommendations) / len(recommendations)
        st.metric("Avg Match Score", f"{avg_score:.0%}")
    with col3:
        top_match = recommendations[0].similarity_score if recommendations else 0
        st.metric("Top Match", f"{top_match:.0%}")
    
    st.divider()
    
    # Display job cards
    for i, rec in enumerate(recommendations):
        with st.expander(
            f"**{rec.job.title}** at {rec.job.company or 'Company'} - Match: {rec.similarity_score:.0%}",
            expanded=(i == 0)
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Description:** {rec.job.description}")
                
                if rec.job.location:
                    st.write(f"**Location:** {rec.job.location}")
                if rec.job.salary_range:
                    st.write(f"**Salary:** {rec.job.salary_range}")
                if rec.job.experience_level:
                    st.write(f"**Level:** {rec.job.experience_level}")
            
            with col2:
                # Similarity gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=rec.similarity_score * 100,
                    title={'text': "Match Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 50], 'color': "#FCE8E6"},
                            {'range': [50, 75], 'color': "#FEF7E0"},
                            {'range': [75, 100], 'color': "#E6F4EA"},
                        ],
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
            
            # Skills section
            st.write("**Matching Skills:**")
            if rec.matching_skills:
                for skill in rec.matching_skills:
                    st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
            else:
                st.write("No direct skill matches")
            
            st.write("")
            st.write("**Skills to Develop:**")
            if rec.missing_skills:
                for skill in rec.missing_skills:
                    st.markdown(f'<span class="skill-tag missing-skill-tag">{skill}</span>', unsafe_allow_html=True)
            else:
                st.write("You have all required skills!")


def render_skill_gap_analysis(recommender, profile, target_role):
    """Render skill gap analysis section."""
    st.header("🎯 Skill Gap Analysis")
    
    with st.spinner(f"Analyzing skill gaps for {target_role}..."):
        skill_gaps = recommender.analyze_skill_gaps(profile, target_role)
    
    if not skill_gaps:
        st.success(f"Great news! You already have the skills needed for {target_role}!")
        return
    
    # Summary
    st.write(f"To transition to **{target_role}**, consider developing these skills:")
    
    # Create skill gap chart
    gap_df = pd.DataFrame([
        {"Skill": gap.skill, "Importance": gap.importance, "Category": gap.category or "Other"}
        for gap in skill_gaps[:10]
    ])
    
    fig = px.bar(
        gap_df,
        x="Importance",
        y="Skill",
        color="Category",
        orientation="h",
        title="Skills to Develop (by importance)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Importance Score",
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed skill gaps
    col1, col2 = st.columns(2)
    
    high_priority = [g for g in skill_gaps if g.importance >= 0.7]
    medium_priority = [g for g in skill_gaps if 0.3 <= g.importance < 0.7]
    
    with col1:
        st.subheader("High Priority Skills")
        for gap in high_priority[:5]:
            st.write(f"• **{gap.skill}** ({gap.category or 'General'})")
    
    with col2:
        st.subheader("Medium Priority Skills")
        for gap in medium_priority[:5]:
            st.write(f"• **{gap.skill}** ({gap.category or 'General'})")
    
    return skill_gaps


def render_course_recommendations(recommender, skill_gaps, top_k=10):
    """Render course recommendations section."""
    st.header("Recommended Courses")
    
    if not skill_gaps:
        st.info("Complete the skill gap analysis to get personalized course recommendations.")
        return
    
    with st.spinner("Finding relevant courses..."):
        recommendations = recommender.recommend_courses(skill_gaps, top_k=top_k)
    
    if not recommendations:
        st.warning("No matching courses found.")
        return
    
    # Course cards
    for rec in recommendations:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(rec.course.title)
                st.write(f"**Provider:** {rec.course.provider or 'Online'}")
                st.write(f"**Description:** {rec.course.description}")
                
                if rec.course.difficulty:
                    st.write(f"📊 **Difficulty:** {rec.course.difficulty}")
                if rec.course.duration:
                    st.write(f"⏱️ **Duration:** {rec.course.duration}")
                
                if rec.skills_addressed:
                    st.write("**Skills Covered:**")
                    for skill in rec.skills_addressed:
                        st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
            
            with col2:
                st.metric("Relevance", f"{rec.relevance_score:.0%}")
                if rec.course.rating:
                    st.write(f"⭐ Rating: {rec.course.rating}/5.0")
                if rec.course.url:
                    st.link_button("View Course", rec.course.url)
            
            st.divider()


def render_career_path(recommender, current_role, target_role):
    """Render career path visualization."""
    st.header("Career Path")
    
    with st.spinner("Generating career path..."):
        career_path = recommender.get_career_path(current_role, target_role)
    
    if not career_path.path_nodes:
        st.warning("Could not generate career path. Try different roles.")
        return
    
    # Create career path visualization
    nodes = career_path.path_nodes
    
    # Prepare data for visualization
    path_data = []
    for i, node in enumerate(nodes):
        path_data.append({
            "Role": node.role,
            "Level": node.level,
            "Skills": ", ".join(node.required_skills[:3]) + "..." if len(node.required_skills) > 3 else ", ".join(node.required_skills),
            "Years": node.avg_years_experience or i * 2,
        })
    
    path_df = pd.DataFrame(path_data)
    
    # Career progression chart
    fig = px.scatter(
        path_df,
        x="Years",
        y="Level",
        text="Role",
        size=[30] * len(path_df),
        color="Level",
        color_continuous_scale="Viridis",
        title="Career Progression Path",
    )
    fig.update_traces(textposition="top center", marker=dict(sizemode='diameter'))
    fig.update_layout(
        height=400,
        xaxis_title="Years of Experience",
        yaxis_title="Career Level",
        showlegend=False,
    )
    
    # Add connecting lines
    fig.add_trace(go.Scatter(
        x=path_df["Years"],
        y=path_df["Level"],
        mode="lines",
        line=dict(color="gray", width=1, dash="dash"),
        showlegend=False,
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Current position indicator
    st.write(f"**Your Current Level:** {career_path.current_level}")
    
    # Recommended transitions
    if career_path.recommended_transitions:
        st.subheader("Recommended Next Steps")
        for i, transition in enumerate(career_path.recommended_transitions, 1):
            st.write(f"{i}. **{transition}**")
    
    # Detailed path nodes
    st.subheader("Career Path Details")
    for node in nodes:
        with st.expander(f"Level {node.level}: {node.role}"):
            st.write(f"**Average Experience:** {node.avg_years_experience:.0f} years")
            st.write("**Required Skills:**")
            for skill in node.required_skills:
                st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)


def render_market_insights(recommender):
    """Render market insights dashboard."""
    st.header("Market Insights")
    
    jobs = recommender.get_all_jobs()
    if not jobs:
        st.warning("No job data available.")
        return
    
    # 1. Top Skills in Demand
    all_skills = []
    for job in jobs:
        all_skills.extend(job.required_skills)
        all_skills.extend(job.preferred_skills)
    
    skill_counts = Counter(all_skills).most_common(15)
    skills_df = pd.DataFrame(skill_counts, columns=["Skill", "Count"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Skills in Demand")
        fig = px.bar(
            skills_df, 
            x="Count", 
            y="Skill", 
            orientation="h",
            color="Count",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. Top Job Titles
    titles = [job.title for job in jobs]
    title_counts = Counter(titles).most_common(10)
    titles_df = pd.DataFrame(title_counts, columns=["Role", "Count"])
    
    with col2:
        st.subheader("🏆 Most Popular Roles")
        fig = px.pie(
            titles_df, 
            values="Count", 
            names="Role",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 3. Salary vs Experience (if salary data is parseable)
    # Simple parsing logic for the synthetic data format "$XX,XXX - $YY,YYY"
    salary_data = []
    for job in jobs:
        if job.salary_range and job.experience_level:
            try:
                # Extract lower bound
                lower = int(job.salary_range.split("-")[0].replace("$", "").replace(",", "").strip())
                salary_data.append({
                    "Role": job.title,
                    "Salary": lower,
                    "Experience": job.experience_level
                })
            except:
                continue
                
    if salary_data:
        st.subheader("💰 Salary Distribution by Experience")
        salary_df = pd.DataFrame(salary_data)
        
        # Order experience levels
        level_order = ["Entry-Level", "Junior", "Mid-Level", "Senior", "Lead", "Principal"]
        
        fig = px.box(
            salary_df,
            x="Experience",
            y="Salary",
            color="Experience",
            category_orders={"Experience": level_order},
            points="all"
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application entry point."""
    render_header()
    
    # Load recommender
    with st.spinner("Loading AI models..."):
        recommender = load_recommender()
    
    # Get user profile from sidebar
    profile, target_role = render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Job Recommendations",
        "Skill Gap Analysis",
        "Course Recommendations",
        "Career Path",
        "Market Insights"
    ])
    
    skill_gaps = None
    
    with tab1:
        render_job_recommendations(recommender, profile)
    
    with tab2:
        skill_gaps = render_skill_gap_analysis(recommender, profile, target_role)
    
    with tab3:
        if skill_gaps is None:
            with st.spinner(f"Analyzing skill gaps for {target_role}..."):
                skill_gaps = recommender.analyze_skill_gaps(profile, target_role)
            # Check if skill_gaps is still None or empty (could happen if analysis fails or no gaps)
            if skill_gaps is None:
                skill_gaps = []
                
        render_course_recommendations(recommender, skill_gaps)
    
    with tab4:
        render_career_path(recommender, profile.current_role or "Analyst", target_role)
        
    with tab5:
        render_market_insights(recommender)




if __name__ == "__main__":
    main()
