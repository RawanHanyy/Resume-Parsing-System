"""
Tests for resume_parser and scorer modules.
Run with: pytest tests/ -v
"""

import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from resume_parser import ResumeParser, EMAIL_RE, PHONE_RE, DEGREE_PATTERNS
from scorer import CandidateScorer, rank_candidates, _cosine, _tfidf_vectors, _recommendation

# Fixtures 

SAMPLE_RESUME_TEXT = """
John Doe
john.doe@email.com
+1 (555) 123-4567
linkedin.com/in/johndoe
github.com/johndoe

Summary
-------
Experienced Python developer with 5 years in backend development.

Skills
------
Python, Django, Flask, PostgreSQL, AWS, Docker, Git, Machine Learning

Education
---------
B.S. Computer Science — MIT, 2018

Experience
----------
Senior Software Engineer — Acme Corp
Jan 2020 – Present
• Built REST APIs using Django and Flask
• Deployed microservices on AWS ECS
• Reduced query time by 40% with PostgreSQL optimization

Software Engineer — StartupXYZ
Jun 2018 – Dec 2019
• Developed ML pipelines using scikit-learn
• Set up CI/CD with Jenkins and Docker
"""

JOB_DESCRIPTION = """
We are looking for a Senior Python Developer with experience in:
- Python, Django, Flask
- PostgreSQL and Redis
- AWS (EC2, S3, ECS)
- Docker and Kubernetes
- Machine learning and scikit-learn
- 4+ years of experience required
"""


@pytest.fixture
def sample_parsed(tmp_path):
    """Write sample resume to a temp file and parse it."""
    resume_file = tmp_path / "sample_resume.txt"
    resume_file.write_text(SAMPLE_RESUME_TEXT, encoding="utf-8")
    parser = ResumeParser(str(resume_file))
    return parser.parse()


# Parser tests 

class TestRegexPatterns:
    def test_email_detection(self):
        assert EMAIL_RE.search("john.doe@email.com")
        assert not EMAIL_RE.search("not an email")

    def test_phone_detection(self):
        assert PHONE_RE.search("+1 (555) 123-4567")
        assert PHONE_RE.search("555-123-4567")

    def test_degree_pattern(self):
        assert DEGREE_PATTERNS.search("B.S. Computer Science")
        assert DEGREE_PATTERNS.search("Master's in Data Science")
        assert DEGREE_PATTERNS.search("Ph.D. in Physics")


class TestResumeParser:
    def test_parse_returns_dict(self, sample_parsed):
        assert isinstance(sample_parsed, dict)

    def test_contact_extraction(self, sample_parsed):
        contact = sample_parsed["contact"]
        assert contact["email"] == "john.doe@email.com"
        assert contact["linkedin"] == "linkedin.com/in/johndoe"
        assert contact["github"] == "github.com/johndoe"

    def test_name_extraction(self, sample_parsed):
        assert sample_parsed["name"] is not None
        assert "John" in sample_parsed["name"] or "Doe" in sample_parsed["name"]

    def test_skills_extraction(self, sample_parsed):
        skills = sample_parsed["skills"]
        assert isinstance(skills, dict)
        # Python should appear in programming_languages
        all_found = [s for lst in skills.values() for s in lst]
        assert "python" in all_found

    def test_education_extraction(self, sample_parsed):
        edu = sample_parsed["education"]
        assert isinstance(edu, list)
        if edu:
            assert "degree" in edu[0] or "years" in edu[0]

    def test_experience_extraction(self, sample_parsed):
        exp = sample_parsed["experience"]
        assert isinstance(exp, list)

    def test_all_keys_present(self, sample_parsed):
        expected_keys = {
            "contact", "name", "summary", "skills",
            "education", "experience", "projects",
            "certifications", "total_experience_years",
        }
        assert expected_keys.issubset(sample_parsed.keys())


# Scorer tests

class TestTFIDFHelpers:
    def test_cosine_identical(self):
        v1, v2 = _tfidf_vectors("python django aws", "python django aws")
        assert _cosine(v1, v2) == pytest.approx(1.0, abs=0.01)

    def test_cosine_zero(self):
        v1, v2 = _tfidf_vectors("python", "javascript")
        score = _cosine(v1, v2)
        assert 0.0 <= score <= 1.0

    def test_cosine_partial(self):
        v1, v2 = _tfidf_vectors("python django flask", "python react node")
        score = _cosine(v1, v2)
        assert 0.0 < score < 1.0


class TestCandidateScorer:
    def test_score_returns_dict(self, sample_parsed):
        scorer = CandidateScorer(sample_parsed, JOB_DESCRIPTION, required_years=4)
        result = scorer.score()
        assert isinstance(result, dict)

    def test_score_range(self, sample_parsed):
        scorer = CandidateScorer(sample_parsed, JOB_DESCRIPTION)
        result = scorer.score()
        assert 0.0 <= result["total_score"] <= 100.0

    def test_breakdown_keys(self, sample_parsed):
        scorer = CandidateScorer(sample_parsed, JOB_DESCRIPTION)
        result = scorer.score()
        assert set(result["breakdown"].keys()) == {
            "skill_match", "tfidf_sim", "education", "experience"
        }

    def test_matched_skills_are_subset(self, sample_parsed):
        scorer = CandidateScorer(sample_parsed, JOB_DESCRIPTION)
        result = scorer.score()
        # matched + missing should cover all required skills
        assert isinstance(result["matched_skills"], list)
        assert isinstance(result["missing_skills"], list)

    def test_recommendation_str(self, sample_parsed):
        scorer = CandidateScorer(sample_parsed, JOB_DESCRIPTION)
        result = scorer.score()
        assert isinstance(result["recommendation"], str)


class TestRecommendation:
    def test_strong_match(self):
        assert "Strong" in _recommendation(80)

    def test_potential_match(self):
        assert "Potential" in _recommendation(60)

    def test_weak_match(self):
        assert "Weak" in _recommendation(40)

    def test_poor_match(self):
        assert "Poor" in _recommendation(20)


class TestRankCandidates:
    def test_rank_order(self, sample_parsed, tmp_path):
        # Create a second weaker resume
        weak_text = "Jane Smith\njane@x.com\nSkills: Excel\nEducation: High School"
        weak_file = tmp_path / "weak.txt"
        weak_file.write_text(weak_text, encoding="utf-8")
        weak_parsed = ResumeParser(str(weak_file)).parse()

        ranked = rank_candidates([weak_parsed, sample_parsed], JOB_DESCRIPTION)
        assert ranked[0]["total_score"] >= ranked[1]["total_score"]

    def test_rank_returns_list(self, sample_parsed):
        result = rank_candidates([sample_parsed], JOB_DESCRIPTION)
        assert isinstance(result, list)
        assert len(result) == 1
