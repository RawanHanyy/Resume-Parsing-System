"""
Candidate Scorer
================
Scores a parsed resume against a job description using
TF-IDF cosine similarity + weighted skill matching.
"""

from __future__ import annotations

import re
import math
from typing import Optional
from collections import Counter

from resume_parser import ALL_SKILLS, skill_pattern


# Weights for each scoring component 
WEIGHTS = {
    "skill_match":    0.45,
    "tfidf_sim":      0.30,
    "education":      0.15,
    "experience":     0.10,
}

DEGREE_RANK = {
    "phd": 4, "doctorate": 4,
    "master": 3, "ms": 3, "msc": 3, "mba": 3, "me": 3, "mtech": 3,
    "bachelor": 2, "bs": 2, "be": 2, "btech": 2, "bsc": 2,
    "associate": 1,
}

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z][a-z0-9+#.]*\b", text.lower())


def _tfidf_vectors(doc1: str, doc2: str) -> tuple[dict, dict]:
    """Return TF-IDF-like vectors for two documents (log-TF, binary IDF)."""
    t1 = _tokenize(doc1)
    t2 = _tokenize(doc2)
    vocab = set(t1) | set(t2)

    def tf(tokens: list[str]) -> dict[str, float]:
        c = Counter(tokens)
        n = len(tokens) or 1
        return {w: (1 + math.log(c[w])) if c[w] > 0 else 0.0 for w in vocab}

    return tf(t1), tf(t2)


def _cosine(v1: dict, v2: dict) -> float:
    dot = sum(v1[k] * v2.get(k, 0) for k in v1)
    mag1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    mag2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def _extract_required_skills(jd_text: str) -> set[str]:
    jd_lower = jd_text.lower()
    return {s for s in ALL_SKILLS if skill_pattern(s).search(jd_lower)}


def _degree_score(education: list[dict]) -> float:
    best = 0
    for entry in education:
        raw = (entry.get("raw") or "").lower()
        for keyword, rank in DEGREE_RANK.items():
            if keyword in raw:
                best = max(best, rank)
    return min(best / 4.0, 1.0)   # normalise to [0, 1]


def _experience_score(parsed: dict, required_years: Optional[float]) -> float:
    candidate_years = parsed.get("total_experience_years") or 0
    if required_years is None or required_years == 0:
        return 1.0
    ratio = candidate_years / required_years
    return min(ratio, 1.0)

class CandidateScorer:
    """
    Score a parsed resume dict against a plain-text job description.

    Usage
    -----
    scorer = CandidateScorer(parsed_resume, job_description_text)
    result = scorer.score()
    # result["total_score"]  ← float 0-100
    """

    def __init__(
        self,
        parsed_resume: dict,
        job_description: str,
        required_years: Optional[float] = None,
    ):
        self.resume = parsed_resume
        self.jd = job_description
        self.required_years = required_years

    # Resume text reconstruction 
    def _resume_text(self) -> str:
        parts = []
        skills_dict = self.resume.get("skills", {})
        for skill_list in skills_dict.values():
            parts.extend(skill_list)
        for edu in self.resume.get("education", []):
            parts.append(edu.get("raw", ""))
        for exp in self.resume.get("experience", []):
            parts.append(exp.get("title", ""))
            parts.extend(exp.get("bullets", []))
        parts.extend(self.resume.get("projects", []))
        parts.extend(self.resume.get("certifications", []))
        if self.resume.get("summary"):
            parts.append(self.resume["summary"])
        return " ".join(parts)

    # Skill match score 
    def _skill_score(self) -> tuple[float, list[str], list[str]]:
        required = _extract_required_skills(self.jd)
        candidate_skills: set[str] = set()
        for skill_list in self.resume.get("skills", {}).values():
            candidate_skills.update(skill_list)

        if not required:
            return 0.0, [], []

        matched = required & candidate_skills
        missing = required - candidate_skills
        score = len(matched) / len(required)
        return score, sorted(matched), sorted(missing)

    # Main scoring 
    def score(self) -> dict:
        resume_text = self._resume_text()

        skill_raw, matched_skills, missing_skills = self._skill_score()
        tfidf_raw = _cosine(*_tfidf_vectors(resume_text, self.jd))
        edu_raw = _degree_score(self.resume.get("education", []))
        exp_raw = _experience_score(self.resume, self.required_years)

        total = (
            WEIGHTS["skill_match"] * skill_raw
            + WEIGHTS["tfidf_sim"]  * tfidf_raw
            + WEIGHTS["education"]  * edu_raw
            + WEIGHTS["experience"] * exp_raw
        ) * 100

        return {
            "name":           self.resume.get("name", "Unknown"),
            "email":          (self.resume.get("contact") or {}).get("email"),
            "total_score":    round(total, 2),
            "breakdown": {
                "skill_match":  round(skill_raw  * 100, 2),
                "tfidf_sim":    round(tfidf_raw   * 100, 2),
                "education":    round(edu_raw     * 100, 2),
                "experience":   round(exp_raw     * 100, 2),
            },
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "required_skills": sorted(_extract_required_skills(self.jd)),
            "recommendation": _recommendation(total),
        }

def _recommendation(score: float) -> str:
    if score >= 75:
        return "Strong Match — Recommend for Interview"
    elif score >= 55:
        return "Potential Match — Consider for Screening"
    elif score >= 35:
        return "Weak Match — Review Manually"
    else:
        return "Poor Match — Likely to Reject"

def rank_candidates(resumes: list[dict], job_description: str, required_years: Optional[float] = None) -> list[dict]:
    """
    Rank a list of parsed resume dicts against a job description.
    Returns sorted list (highest score first).
    """
    results = []
    for r in resumes:
        scorer = CandidateScorer(r, job_description, required_years)
        results.append(scorer.score())
    return sorted(results, key=lambda x: x["total_score"], reverse=True)
