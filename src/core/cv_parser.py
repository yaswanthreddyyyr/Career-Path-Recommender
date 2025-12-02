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
            "b.s": "Bachelor's Degree",
            "bs": "Bachelor's Degree",
            "m.a": "Master's Degree",
            "m.s": "Master's Degree",
            "ms": "Master's Degree",
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

    def _remove_education_section(self, text: str) -> str:
        """Remove the Education section from the text to avoid counting study years as experience."""
        lines = text.split('\n')
        new_lines = []
        in_education = False
        
        # Headers that signal the start of a new section (ending the previous one)
        headers = ["experience", "work history", "employment", "skills", "projects", "summary", "profile", "certifications", "languages", "interests"]
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line is the "Education" header
            if "education" in line_lower and len(line_lower) < 20:
                in_education = True
                continue
            
            # Check if we've hit another header
            if in_education:
                is_header = any(line_lower == h or (line_lower.startswith(h) and len(line_lower) < 30) for h in headers)
                if is_header:
                    in_education = False
            
            if not in_education:
                new_lines.append(line)
                
        return "\n".join(new_lines)

    def extract_experience_years(self, text: str) -> int:
        """Estimate years of experience."""
        # Remove education section first
        text_no_edu = self._remove_education_section(text)
        
        years = 0.0
        current_year = 2024
        
        # Regex for year ranges with optional months
        # Matches: 2010-2012, 01/2010 - 12/2012, 2010 to Present
        date_pattern = r'((?:\d{1,2}[/.-])?\b(?:19|20)\d{2}\b)\s*(?:-|–|to)\s*((?:\d{1,2}[/.-])?\b(?:19|20)\d{2}\b|present|current|now)'
        
        lines = text_no_edu.split('\n')
        matches = []
        
        # Education keywords to exclude from experience calculation (fallback if section removal fails)
        edu_keywords = ["university", "college", "school", "degree", "bachelor", "master", "phd", "diploma", "certificate", "b.sc", "m.sc", "b.tech", "m.tech"]
        
        for line in lines:
            # Skip lines that look like education
            if any(k in line.lower() for k in edu_keywords):
                continue
                
            line_matches = re.findall(date_pattern, line, re.IGNORECASE)
            matches.extend(line_matches)
        
        for start, end in matches:
            try:
                # Parse start year
                start_parts = re.split(r'[/.-]', start)
                if len(start_parts) > 1:
                    start_year = int(start_parts[-1])
                    start_month = int(start_parts[0])
                else:
                    start_year = int(start)
                    start_month = 1

                # Parse end year
                if end.lower() in ["present", "current", "now"]:
                    end_year = current_year
                    end_month = 12 # Assume full current year roughly
                else:
                    end_parts = re.split(r'[/.-]', end)
                    if len(end_parts) > 1:
                        end_year = int(end_parts[-1])
                        end_month = int(end_parts[0])
                    else:
                        end_year = int(end)
                        end_month = 12

                # Calculate duration in years
                start_date = start_year + (start_month - 1) / 12.0
                end_date = end_year + (end_month) / 12.0 # Include the end month
                
                diff = end_date - start_date
                
                if 0 <= diff < 40: # Sanity check (increased from 30)
                    years += diff
            except (ValueError, IndexError):
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

        # Round calculated years
        calc_years = int(round(years))
        
        # Use the larger of calculated vs stated
        if stated_years > 0:
            return max(stated_years, calc_years)
            
        return calc_years

    def extract_current_role(self, text: str) -> Optional[str]:
        """Attempt to extract current role."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        common_roles = [
            "Software Engineer", "Data Scientist", "Data Analyst", "Product Manager",
            "Project Manager", "Developer", "Consultant", "Designer", "Architect",
            "Administrator", "Technician", "Specialist"
        ]
        
        role_keywords = ["Engineer", "Developer", "Analyst", "Scientist", "Manager", "Consultant", "Lead", "Director", "Head", "Chief"]
        
        # Strategy 1: Look for Experience Section
        exp_headers = ["experience", "work history", "employment", "professional experience", "work experience"]
        header_idx = -1
        
        for i, line in enumerate(lines):
            if any(header == line.lower().strip() for header in exp_headers) or \
               any(line.lower().strip().startswith(h) and len(line) < 30 for h in exp_headers):
                header_idx = i
                break
        
        if header_idx != -1:
            # Look at lines immediately following the header
            # We look at the next 5 non-empty lines
            search_window = lines[header_idx+1:header_idx+10]
            for line in search_window:
                # Check if line contains a role keyword
                if any(keyword in line for keyword in role_keywords) or \
                   any(role.lower() in line.lower() for role in common_roles):
                    # It's likely a role if it's short
                    if len(line.split()) < 8:
                        return line
        
        # Strategy 2: Fallback to top of file (Summary/Header area)
        # Check first few lines for exact matches with common roles
        for line in lines[:10]:
            for role in common_roles:
                if role.lower() in line.lower():
                    if len(line.split()) < 6: 
                        return line
        
        # Strategy 3: Look for lines with typical role keywords in top section
        for line in lines[:20]:
            for keyword in role_keywords:
                if keyword in line and len(line.split()) < 6:
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
