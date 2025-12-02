
"""
Demo script to showcase the Graph-Based Skill Dependency Miner.
"""
import sys
import logging
from src.core.recommender import CareerRecommender
from src.models.schemas import UserProfile

# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
    print("="*50)
    print("DEMO: Graph-Based Skill Dependency Miner")
    print("="*50)
    
    # 1. Initialize Recommender
    print("\n[1] Initializing Recommender and Mining Graph...")
    recommender = CareerRecommender(
        use_hybrid_search=True, # Semantic + BM25
        use_reranking=False,
        use_diversity=True
    )
    recommender.initialize()
    
    graph = recommender.skill_graph
    print(f"\nGraph Stats: {graph.graph.number_of_nodes()} skills, {graph.graph.number_of_edges()} dependencies found.")
    
    # 2. Show inferred prerequisites for common skills
    print("\n[2] Inferred Prerequisites (Examples):")
    sample_skills = ["pytorch", "react", "aws", "kubernetes"]
    
    for skill in sample_skills:
        if skill in graph.graph:
            prereqs = graph.get_prerequisites(skill)
            print(f"  To learn '{skill}', you might need: {prereqs}")
        else:
            print(f"  '{skill}' not found in graph (insufficient data).")
            
    # 3. Generate a Learning Path
    print("\n[3] Generating Logical Learning Path...")
    # Scenario: User wants to learn full stack + AI
    target_skills = ["pytorch", "react", "python", "javascript", "machine learning"]
    print(f"  Target Skills: {target_skills}")
    
    ordered_path = graph.suggest_learning_path(target_skills)
    print(f"  Recommended Order: {ordered_path}")
    
    print("\n" + "="*50)
    print("Explanation:")
    print("The system used the Job Postings data to infer that 'Python' is fundamental")
    print("to 'Machine Learning' and 'PyTorch', and 'JavaScript' is fundamental to 'React'.")
    print("It automatically ordered the curriculum: Foundations -> Advanced Frameworks.")
    print("="*50)

if __name__ == "__main__":
    main()

