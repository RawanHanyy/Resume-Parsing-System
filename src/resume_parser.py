"""
Resume Parser - Core NLP Module
================================
Extracts structured information from resumes using spaCy, regex, and keyword matching.
"""

import re
import json
import spacy
from pathlib import Path
from typing import Optional
import pdfplumber
import docx2txt


# Load spaCy model 
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")


# Skill taxonomy 
SKILL_TAXONOMY = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "r",
        "golang", "rust", "kotlin", "swift", "scala", "php", "ruby",
    ],
    "frameworks": [
        "django", "flask", "fastapi", "react", "vue", "angular", "node.js",
        "spring", "tensorflow", "pytorch", "keras", "scikit-learn", "pandas",
        "numpy", "spark", "hadoop",
    ],
    "databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "cassandra", "sqlite", "oracle", "dynamodb",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
        "ci/cd", "git", "linux", "bash",
    ],
    "data_science": [
        "machine learning", "deep learning", "nlp", "computer vision",
        "data analysis", "statistics", "tableau", "power bi", "excel",
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem solving",
        "project management", "agile", "scrum",
    ],
}

# Flatten for fast lookup
ALL_SKILLS: set[str] = {
    skill for category in SKILL_TAXONOMY.values() for skill in category
}


# Section headers 
SECTION_HEADERS = {
    "experience": re.compile(
        r"(work experience|experience|employment|professional experience|career)",
        re.IGNORECASE,
    ),
    "education": re.compile(
        r"(education|academic|qualifications|degrees?)", re.IGNORECASE
    ),
    "skills": re.compile(
        r"(skills|technical skills|competencies|expertise|technologies)",
        re.IGNORECASE,
    ),
    "projects": re.compile(r"(projects?|portfolio|works?)", re.IGNORECASE),
    "certifications": re.compile(
        r"(certifications?|licenses?|credentials?)", re.IGNORECASE
    ),
    "summary": re.compile(
        r"(summary|objective|profile|about me|overview)", re.IGNORECASE
    ),
}

# Degree patterns
DEGREE_PATTERNS = re.compile(
    r"\b(B\.?S\.?|B\.?E\.?|B\.?Tech\.?|B\.?Sc\.?|Bachelor(?:\'s)?|"
    r"M\.?S\.?|M\.?E\.?|M\.?Tech\.?|M\.?Sc\.?|Master(?:\'s)?|MBA|"
    r"Ph\.?D\.?|Doctorate|Associate)\b",
    re.IGNORECASE,
)

# Email / phone / LinkedIn
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?)(\d{3,4}[\s.-]?\d{3,5})"
)
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
GITHUB_RE = re.compile(r"github\.com/[\w\-]+", re.IGNORECASE)

# Year ranges for experience
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DATE_RANGE_RE = re.compile(
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)?[\s,]*(20|19)\d{2}"
    r"\s*[-–—to]+\s*"
    r"(?:(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)?[\s,]*(20|19)\d{2}|[Pp]resent|[Cc]urrent)",
    re.IGNORECASE,
)

class ResumeParser:
    """
    Parses a resume file (PDF / DOCX / TXT) and returns structured JSON.
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.raw_text: str = ""
        self.doc = None  # spaCy doc
        self.parsed: dict = {}

    # Public entry-point 
    def parse(self) -> dict:
        self.raw_text = self._extract_text()
        self.doc = nlp(self.raw_text[:100_000])   # spaCy token limit guard
        self.parsed = {
            "contact":        self._extract_contact(),
            "name":           self._extract_name(),
            "summary":        self._extract_summary(),
            "skills":         self._extract_skills(),
            "education":      self._extract_education(),
            "experience":     self._extract_experience(),
            "projects":       self._extract_projects(),
            "certifications": self._extract_certifications(),
            "total_experience_years": self._estimate_experience_years(),
        }
        return self.parsed

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.parsed, indent=indent, ensure_ascii=False)

    # Text extraction 
    def _extract_text(self) -> str:
        suffix = self.filepath.suffix.lower()
        if suffix == ".pdf":
            return self._read_pdf()
        elif suffix in (".docx", ".doc"):
            return docx2txt.process(str(self.filepath))
        else:
            return self.filepath.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf(self) -> str:
        text_parts = []
        with pdfplumber.open(self.filepath) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)

    # Contact info 
    def _extract_contact(self) -> dict:
        email = EMAIL_RE.search(self.raw_text)
        phone = PHONE_RE.search(self.raw_text)
        linkedin = LINKEDIN_RE.search(self.raw_text)
        github = GITHUB_RE.search(self.raw_text)
        return {
            "email":    email.group() if email else None,
            "phone":    phone.group().strip() if phone else None,
            "linkedin": linkedin.group() if linkedin else None,
            "github":   github.group() if github else None,
        }

    # Name (first PERSON entity in first 10 lines) 
    def _extract_name(self) -> Optional[str]:
        first_chunk = "\n".join(self.raw_text.splitlines()[:10])
        chunk_doc = nlp(first_chunk)
        for ent in chunk_doc.ents:
            if ent.label_ == "PERSON":
                return ent.text.strip()
        # Fallback: first non-empty line that looks like a name
        for line in self.raw_text.splitlines():
            line = line.strip()
            if line and len(line.split()) <= 5 and not EMAIL_RE.search(line):
                return line
        return None

    # Summary section 
    def _extract_summary(self) -> Optional[str]:
        section = self._get_section("summary")
        return section[:500].strip() if section else None

    # Skills 
    def _extract_skills(self) -> dict:
        text_lower = self.raw_text.lower()
        found: dict[str, list[str]] = {}
        for category, skills in SKILL_TAXONOMY.items():
            matched = [s for s in skills if re.search(r"\b" + re.escape(s) + r"\b", text_lower)]
            if matched:
                found[category] = matched
        return found

    # Education 
    def _extract_education(self) -> list[dict]:
        section = self._get_section("education")
        if not section:
            return []
        entries = []
        for line in section.splitlines():
            line = line.strip()
            if not line:
                continue
            degree_match = DEGREE_PATTERNS.search(line)
            years = YEAR_RE.findall(line)
            entries.append({
                "raw":    line,
                "degree": degree_match.group() if degree_match else None,
                "years":  years if years else None,
            })
        # Keep only lines that look like degree entries
        return [e for e in entries if e["degree"] or e["years"]]

    # Work experience 
    def _extract_experience(self) -> list[dict]:
        section = self._get_section("experience")
        if not section:
            return []
        entries = []
        current: dict = {}
        for line in section.splitlines():
            line = line.strip()
            if not line:
                continue
            date_match = DATE_RANGE_RE.search(line)
            if date_match:
                if current:
                    entries.append(current)
                current = {"raw_line": line, "date_range": date_match.group(), "bullets": []}
            elif current and line.startswith(("•", "-", "–", "*", "·")):
                current["bullets"].append(line.lstrip("•-–*· "))
            elif current:
                if "title" not in current:
                    current["title"] = line
        if current:
            entries.append(current)
        return entries

    # Projects 
    def _extract_projects(self) -> list[str]:
        section = self._get_section("projects")
        if not section:
            return []
        return [ln.strip() for ln in section.splitlines() if ln.strip()][:20]

    # Certifications 
    def _extract_certifications(self) -> list[str]:
        section = self._get_section("certifications")
        if not section:
            # Fallback: scan whole text for cert patterns
            cert_re = re.compile(
                r"(AWS|Azure|GCP|Google|Cisco|Oracle|PMP|CISSP|CPA|CFA|"
                r"Certified|Certificate|Certification)[\w\s,.-]{0,80}",
                re.IGNORECASE,
            )
            return list({m.group().strip() for m in cert_re.finditer(self.raw_text)})[:10]
        return [ln.strip() for ln in section.splitlines() if ln.strip()][:15]

    # Estimate years of experience 
    def _estimate_experience_years(self) -> Optional[float]:
        ranges = DATE_RANGE_RE.findall(self.raw_text)
        years = YEAR_RE.findall(self.raw_text)
        if not years:
            return None
        int_years = sorted(set(map(int, years)))
        if len(int_years) >= 2:
            return float(int_years[-1] - int_years[0])
        return None

    # Helper: extract named section text 
    def _get_section(self, section_name: str) -> str:
        pattern = SECTION_HEADERS[section_name]
        lines = self.raw_text.splitlines()
        start = None
        for i, line in enumerate(lines):
            if pattern.search(line):
                start = i + 1
                break
        if start is None:
            return ""
        # Collect until the next section header
        end = len(lines)
        other_patterns = [
            p for k, p in SECTION_HEADERS.items() if k != section_name
        ]
        for i in range(start, len(lines)):
            if any(p.search(lines[i]) for p in other_patterns):
                end = i
                break
        return "\n".join(lines[start:end])
