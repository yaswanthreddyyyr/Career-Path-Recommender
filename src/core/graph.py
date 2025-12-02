"""
Skill Dependency Graph Miner.

This module builds a directed graph of skills by analyzing co-occurrence patterns
in job postings. It infers potential prerequisites and skill clusters automatically,
allowing for logically ordered learning paths.

Novelty:
- Uses conditional probability P(A|B) to infer directionality (Dependency).
- Constructs a graph without manual taxonomy.
"""

import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from src.models.schemas import JobPosting

class SkillGraphBuilder:
    def __init__(self, min_support: int = 5, min_confidence: float = 0.6):
        """
        Initialize the Skill Graph Builder.
        
        Args:
            min_support: Minimum number of times a pair must appear together.
            min_confidence: Threshold for directional inference P(B|A).
        """
        self.graph = nx.DiGraph()
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.skill_counts = defaultdict(int)
        self.co_occurrence = defaultdict(lambda: defaultdict(int))
        
    def fit(self, jobs: List[JobPosting]):
        """
        Build the graph from a list of job postings.
        """
        print(f"Mining skill relationships from {len(jobs)} jobs...")
        
        # 1. Count frequencies and co-occurrences
        for job in jobs:
            # Combine required and preferred skills
            skills = list(set(job.required_skills + job.preferred_skills))
            skills = [s.lower() for s in skills] # Normalize
            
            for skill in skills:
                self.skill_counts[skill] += 1
            
            # Count pairs
            for i in range(len(skills)):
                for j in range(len(skills)):
                    if i != j:
                        self.co_occurrence[skills[i]][skills[j]] += 1
                        
        # 2. Build Graph Edges based on rules
        self._build_edges()
        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def _build_edges(self):
        """
        Construct edges based on conditional probability.
        If P(A | B) is high, it means A is likely present whenever B is present.
        This often implies A is a 'foundation' or 'component' of B.
        We draw edge A -> B (Learn A first).
        """
        for source, targets in self.co_occurrence.items():
            for target, count in targets.items():
                if count < self.min_support:
                    continue
                
                # Calculate association metrics
                # P(Target | Source) = count(Source, Target) / count(Source)
                # If Source implies Target strongly, maybe Source is a sub-skill? 
                # Or maybe Source is a prerequisite?
                
                # Let's use a simpler heuristic for 'Prerequisite':
                # General skills appear more often than specialized skills.
                # If they co-occur often, the more frequent one is likely the parent/prereq.
                
                freq_source = self.skill_counts[source]
                freq_target = self.skill_counts[target]
                
                # Confidence: How often do they appear together relative to the rarer skill?
                # This measures connection strength.
                overlap_ratio = count / min(freq_source, freq_target)
                
                if overlap_ratio >= 0.3: # Strong connection
                    # Directionality: flow from Common -> Rare
                    if freq_source > freq_target * 1.2: 
                        weight = overlap_ratio
                        self.graph.add_edge(source, target, weight=weight)

    def get_prerequisites(self, skill: str, depth: int = 1) -> List[str]:
        """Get inferred prerequisites for a skill."""
        skill = skill.lower()
        if skill not in self.graph:
            return []
            
        preds = list(self.graph.predecessors(skill))
        # Sort by weight
        preds.sort(key=lambda x: self.graph[x][skill]['weight'], reverse=True)
        return preds

    def suggest_learning_path(self, target_skills: List[str]) -> List[str]:
        """
        Topological sort of the subgraph induced by target skills and their ancestors.
        Returns a logical learning order.
        """
        relevant_nodes = set([s.lower() for s in target_skills])
        
        # Expand with ancestors (prerequisites)
        to_process = list(relevant_nodes)
        seen = set(to_process)
        
        while to_process:
            current = to_process.pop()
            if current in self.graph:
                for pred in self.graph.predecessors(current):
                    if pred not in seen:
                        seen.add(pred)
                        to_process.append(pred)
                        relevant_nodes.add(pred)
        
        # Extract subgraph
        subgraph = self.graph.subgraph(relevant_nodes)
        
        # Topological sort (handles cycles by falling back to simple sort)
        try:
            ordered = list(nx.topological_sort(subgraph))
        except nx.NetworkXUnfeasible:
            # Cycle detected, fallback to frequency sort
            ordered = sorted(list(relevant_nodes), key=lambda x: -self.skill_counts[x])
            
        return ordered

