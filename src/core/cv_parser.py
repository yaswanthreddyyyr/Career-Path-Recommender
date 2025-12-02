import re
from pathlib import Path
from typing import Optional, List, Set
import pandas as pd
import pdfplumber
import docx
from src.models.schemas import UserProfile

class CareerResumeParser:
    """Parses resumes to extract user profile information."""

    def __init__(self, skills_path: Optional[str] = None):
        """
        Initialize the parser.
        
        Args:
            skills_path: Path to the skills CSV file.
        """
        self.known_skills = self._load_skills(skills_path)
        
        # Education keywords mapping to standardized levels
        self.education_map = {
            "phd": "PhD",
            "doctorate": "PhD",
            "master": "Master's Degree",
            "m.sc": "Master's Degree",
            "m.tech": "Master's Degree",
            "mba": "Master's Degree",
            "bachelor": "Bachelor's Degree",
            "b.sc": "Bachelor's Degree",
            "b.tech": "Bachelor's Degree",
            "b.e": "Bachelor's Degree",
            "engineer": "Bachelor's Degree",
            "degree": "Bachelor's Degree",
            "b.a": "Bachelor's Degree",
            "m.a": "Master's Degree",
            "undergraduate": "Bachelor's Degree",
            "postgraduate": "Master's Degree"
        }

    def _load_skills(self, path: Optional[str]) -> Set[str]:
        """Load known skills from CSV."""
        if not path:
            # Default to a relative path if not provided
            path = Path(__file__).parent.parent.parent / "data/raw/skills.csv"
            
        skills = set()
        if Path(path).exists():
            try:
                df = pd.read_csv(path)
                if "skill" in df.columns:
                    skills = set(df["skill"].str.lower().str.strip())
            except Exception as e:
                print(f"Error loading skills: {e}")
                
        return skills

    def extract_text(self, file, filename: str) -> str:
        """Extract text from a resume file."""
        text = ""
        filename = filename.lower()
        
        try:
            if filename.endswith(".pdf"):
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                            
            elif filename.endswith(".docx"):
                doc = docx.Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
                    
            elif filename.endswith(".txt"):
                text = file.read().decode("utf-8")
                
        except Exception as e:
            print(f"Error extracting text: {e}")
            
        return text

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text based on known skills list."""
        if not self.known_skills:
            return []
            
        text_lower = text.lower()
        found_skills = set()
        
        # Simple string matching for now
        for skill in self.known_skills:
            # Create a regex to match the skill as a whole word or phrase
            # Escape special characters in skill name
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
                
        return [s.title() for s in found_skills]

    def extract_education(self, text: str) -> Optional[str]:
        """Extract education level."""
        text_lower = text.lower()
        
        # Priority order: PhD > Master > Bachelor
        for keyword in ["phd", "doctorate"]:
            if keyword in text_lower:
                return "PhD"
                
        for keyword in ["master", "m.sc", "m.tech", "mba", "postgraduate", "m.a"]:
            if keyword in text_lower:
                return "Master's Degree"
                
        for keyword in ["bachelor", "b.sc", "b.tech", "b.e", "engineer", "undergraduate", "b.a"]:
            if keyword in text_lower:
                return "Bachelor's Degree"
            
        return "Other"

    def extract_experience_years(self, text: str) -> int:
        """Estimate years of experience."""
        years = 0
        current_year = 2024
        
        # Regex for year ranges
        # More robust regex to handle spaces and different formats
        # Matches: 2010-2012, 2010 - 2012, 2010 to 2012
        date_pattern = r'(\b20\d{2}\b)\s*(?:-|–|to)\s*(\b20\d{2}\b|present|current|now)'
        matches = re.findall(date_pattern, text, re.IGNORECASE)
        
        total_months = 0
        
        for start, end in matches:
            try:
                start_year = int(start)
                if end.lower() in ["present", "current", "now"]:
                    end_year = current_year
                else:
                    end_year = int(end)
                    
                diff = end_year - start_year
                if 0 <= diff < 30: # Sanity check
                    # We add to years directly. 
                    # Note: this is a simple sum of durations, so overlapping jobs might double count
                    years += diff
            except ValueError:
                continue

        # Check for explicit "X years of experience" statements
        # Matches: "5+ years", "5 years", "five years"
        explicit_pattern = r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\+?\s*years?\s+(?:of\s+)?experience'
        exp_matches = re.findall(explicit_pattern, text, re.IGNORECASE)
        
        stated_years = 0
        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        for match in exp_matches:
            try:
                if match.lower() in word_to_num:
                    val = word_to_num[match.lower()]
                else:
                    val = int(match)
                stated_years = max(stated_years, val)
            except ValueError:
                continue

        # Use the larger of calculated vs stated, but prefer stated if found as it's usually more accurate
        # for summary sections.
        if stated_years > 0:
            return stated_years
            
        return years

    def extract_current_role(self, text: str) -> Optional[str]:
        """Attempt to extract current role."""
        # This is difficult without Named Entity Recognition (NER).
        # We will try a few heuristics:
        # 1. Look for lines that contain "Senior", "Lead", "Manager", "Developer", "Analyst", "Engineer", "Scientist"
        # 2. Prefer lines near the top or near "Experience" sections.
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        common_roles = [
            "Software Engineer", "Data Scientist", "Data Analyst", "Product Manager",
            "Project Manager", "Developer", "Consultant", "Designer", "Architect",
            "Administrator", "Technician", "Specialist"
        ]
        
        # Heuristic 1: Check first few lines for exact matches with common roles
        for line in lines[:10]:
            for role in common_roles:
                if role.lower() in line.lower():
                    # Return the line or the role? 
                    # Often the line is just the role, e.g. "Senior Data Scientist"
                    if len(line.split()) < 5: # Assuming title lines are short
                        return line
        
        # Heuristic 2: Look for lines with typical role keywords
        role_keywords = ["Engineer", "Developer", "Analyst", "Scientist", "Manager", "Consultant"]
        for line in lines[:20]: # Check top 20 lines
            for keyword in role_keywords:
                if keyword in line and len(line.split()) < 5:
                     return line
                     
        return None

    def parse(self, file, filename: str) -> UserProfile:
        """Parse file and return UserProfile."""
        text = self.extract_text(file, filename)
        
        skills = self.extract_skills(text)
        education = self.extract_education(text)
        experience = self.extract_experience_years(text)
        current_role = self.extract_current_role(text)
        
        return UserProfile(
            skills=skills,
            education=education,
            experience_years=experience,
            current_role=current_role,
            career_goals=None
        )
