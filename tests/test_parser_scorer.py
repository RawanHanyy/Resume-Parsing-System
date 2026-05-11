"""
Tests for resume_parser and scorer modules.
Run with: pytest tests/ -v
"""

import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from classifier import ResumeClassifier
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

    def test_experience_year_estimation_uses_full_years(self, tmp_path):
        resume_file = tmp_path / "dated_resume.txt"
        resume_file.write_text(
            "Alex Smith\nExperience\nDeveloper\nJan 2019 - Dec 2023\n- Built APIs",
            encoding="utf-8",
        )

        parsed = ResumeParser(str(resume_file)).parse()

        assert parsed["total_experience_years"] == 4.0

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
        assert isinstance(result["required_skills"], list)

    def test_no_required_skills_does_not_score_as_full_match(self, sample_parsed):
        unrelated_jd = "We need a restaurant manager for menu planning and kitchen operations."
        result = CandidateScorer(sample_parsed, unrelated_jd).score()

        assert result["breakdown"]["skill_match"] == 0.0
        assert result["matched_skills"] == []
        assert result["missing_skills"] == []
        assert result["required_skills"] == []

    def test_missing_required_skills_are_reported(self, sample_parsed):
        frontend_jd = "We need React, Angular, and Vue experience."
        result = CandidateScorer(sample_parsed, frontend_jd).score()

        assert result["breakdown"]["skill_match"] == 0.0
        assert result["matched_skills"] == []
        assert result["missing_skills"] == ["angular", "react", "vue"]
        assert result["required_skills"] == ["angular", "react", "vue"]

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


class TestResumeClassifier:
    def test_bert_training_initializes_tokenizer_before_encoding(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "dataset.csv"
        rows = ["Resume_str,Category"]
        for idx in range(6):
            rows.append(f"Python developer resume {idx},Python Developer")
            rows.append(f"Data scientist resume {idx},Data Scientist")
        csv_path.write_text("\n".join(rows), encoding="utf-8")

        class FakeHistory:
            history = {
                "accuracy": [1.0],
                "val_accuracy": [1.0],
                "loss": [0.1],
                "val_loss": [0.1],
            }

        class FakeModel:
            def fit(self, *args, **kwargs):
                return FakeHistory()

        def fake_build_bert(self):
            self.bert_tokenizer = object()
            return FakeModel()

        def fake_encode_bert(self, texts):
            assert hasattr(self, "bert_tokenizer")
            return {"input_ids": list(texts)}

        monkeypatch.setattr(ResumeClassifier, "_build_bert", fake_build_bert)
        monkeypatch.setattr(ResumeClassifier, "_encode_bert", fake_encode_bert)

        clf = ResumeClassifier(model_type="bert")
        clf.train(str(csv_path))

        assert clf.model is not None
        assert clf.X_test is not None
