"""
Generate a deterministic synthetic resume classification dataset.

The notebook and classifier expect a CSV with resume text plus a category label.
This script creates that dataset locally so the training workflow is reproducible
without relying on a manual Kaggle download step.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path


SEED = 42
ROWS_PER_CATEGORY = 24


CATEGORIES = {
    "Data Scientist": {
        "skills": [
            "Python", "pandas", "NumPy", "scikit-learn", "TensorFlow",
            "PyTorch", "SQL", "Tableau", "feature engineering", "NLP",
            "machine learning", "deep learning", "statistics", "A/B testing",
        ],
        "projects": [
            "built churn prediction pipelines",
            "developed recommendation systems",
            "trained text classification models",
            "delivered forecasting dashboards",
            "optimized anomaly detection workflows",
        ],
        "tools": ["Jupyter", "BigQuery", "Airflow", "Git", "Docker"],
    },
    "Python Developer": {
        "skills": [
            "Python", "Django", "FastAPI", "Flask", "PostgreSQL", "Redis",
            "REST APIs", "pytest", "Celery", "Docker", "AWS", "Git",
        ],
        "projects": [
            "implemented backend services",
            "designed REST APIs",
            "optimized database queries",
            "built internal automation tools",
            "integrated payment gateways",
        ],
        "tools": ["Linux", "NGINX", "Kubernetes", "Terraform", "CI/CD"],
    },
    "Java Developer": {
        "skills": [
            "Java", "Spring Boot", "Hibernate", "Maven", "JUnit",
            "Microservices", "MySQL", "Kafka", "Docker", "AWS",
        ],
        "projects": [
            "implemented microservice APIs",
            "improved transaction processing",
            "built event-driven services",
            "maintained enterprise backends",
            "refactored legacy monolith modules",
        ],
        "tools": ["Jenkins", "Git", "Prometheus", "Grafana", "OpenAPI"],
    },
    "DevOps Engineer": {
        "skills": [
            "AWS", "Docker", "Kubernetes", "Terraform", "Ansible",
            "CI/CD", "Linux", "Bash", "Prometheus", "Grafana", "Helm",
        ],
        "projects": [
            "automated infrastructure provisioning",
            "built deployment pipelines",
            "improved cluster observability",
            "reduced release downtime",
            "managed container orchestration platforms",
        ],
        "tools": ["GitHub Actions", "ArgoCD", "EKS", "ELK", "Vault"],
    },
    "Business Analyst": {
        "skills": [
            "requirements gathering", "SQL", "Excel", "Power BI",
            "stakeholder management", "process modeling", "dashboards",
            "KPI analysis", "documentation", "user stories",
        ],
        "projects": [
            "translated business requirements into user stories",
            "built executive dashboards",
            "mapped current and future state processes",
            "supported product discovery workshops",
            "analyzed operational KPIs",
        ],
        "tools": ["Jira", "Confluence", "Visio", "Tableau", "Miro"],
    },
    "HR Specialist": {
        "skills": [
            "recruitment", "talent acquisition", "employee onboarding",
            "ATS", "HRIS", "interview scheduling", "policy compliance",
            "candidate screening", "employee relations", "training",
        ],
        "projects": [
            "managed end-to-end recruitment cycles",
            "streamlined onboarding workflows",
            "coordinated interview processes",
            "improved applicant screening procedures",
            "supported employee engagement initiatives",
        ],
        "tools": ["Workday", "LinkedIn Recruiter", "Google Workspace", "Excel", "BambooHR"],
    },
    "UI UX Designer": {
        "skills": [
            "Figma", "wireframing", "prototyping", "user research",
            "usability testing", "design systems", "interaction design",
            "accessibility", "responsive design", "visual design",
        ],
        "projects": [
            "designed mobile app prototypes",
            "ran usability testing sessions",
            "built scalable design systems",
            "improved landing page conversion flows",
            "created accessible interface patterns",
        ],
        "tools": ["Adobe XD", "Miro", "Notion", "Maze", "Illustrator"],
    },
    "Network Engineer": {
        "skills": [
            "routing", "switching", "TCP/IP", "firewalls", "VPN",
            "network monitoring", "Cisco", "LAN/WAN", "troubleshooting",
            "load balancing", "DNS", "DHCP",
        ],
        "projects": [
            "maintained enterprise network infrastructure",
            "resolved critical connectivity incidents",
            "configured branch office VPNs",
            "improved network monitoring coverage",
            "upgraded switching environments",
        ],
        "tools": ["Wireshark", "SolarWinds", "Cisco IOS", "Fortinet", "Nagios"],
    },
}


DEGREES = [
    "Bachelor of Science in Computer Science",
    "Bachelor of Engineering in Information Technology",
    "Master of Science in Computer Science",
    "Master of Science in Data Analytics",
]

UNIVERSITIES = [
    "Cairo University",
    "Ain Shams University",
    "Alexandria University",
    "Helwan University",
    "Mansoura University",
]

SOFT_SKILLS = [
    "strong communication skills",
    "cross-functional collaboration",
    "problem solving mindset",
    "analytical thinking",
    "attention to detail",
    "stakeholder coordination",
]

NAMES = [
    "Ahmed Hassan", "Sara Ali", "Mona Ibrahim", "Omar Khaled", "Youssef Adel",
    "Nour Mostafa", "Salma Magdy", "Hana Tarek", "Karim Fathy", "Laila Samir",
    "Mahmoud Nabil", "Rana Hossam", "Mariam Essam", "Ali Reda", "Nada Emad",
]


def build_resume(rng: random.Random, name: str, category: str, spec: dict[str, list[str]]) -> str:
    years = rng.randint(1, 9)
    degree = rng.choice(DEGREES)
    university = rng.choice(UNIVERSITIES)
    core_skills = rng.sample(spec["skills"], k=min(6, len(spec["skills"])))
    secondary_skills = rng.sample(spec["tools"], k=min(3, len(spec["tools"])))
    project_a, project_b = rng.sample(spec["projects"], k=2)
    soft_skill = rng.choice(SOFT_SKILLS)
    cert = rng.choice(
        [
            "Google Professional Certificate",
            "AWS Certification coursework",
            "Scrum fundamentals training",
            "advanced analytics specialization",
            "professional development workshops",
        ]
    )

    return (
        f"{name}\n"
        f"Target Role: {category}\n"
        f"Email: {name.lower().replace(' ', '.')}@example.com | Phone: +20 10 {rng.randint(1000,9999)} {rng.randint(1000,9999)}\n"
        f"Summary: {category} with {years} years of experience, {soft_skill}, and hands-on delivery across production projects.\n"
        f"Skills: {', '.join(core_skills + secondary_skills)}.\n"
        f"Experience: {project_a}; {project_b}; collaborated with product, engineering, and operations teams.\n"
        f"Education: {degree} from {university}.\n"
        f"Achievements: improved team outcomes, documented workflows, and completed {cert}.\n"
    )


def generate_rows() -> list[dict[str, str]]:
    rng = random.Random(SEED)
    rows: list[dict[str, str]] = []

    for category, spec in CATEGORIES.items():
        for idx in range(ROWS_PER_CATEGORY):
            name = f"{rng.choice(NAMES)} {idx + 1}"
            resume = build_resume(rng, name, category, spec)
            rows.append(
                {
                    "Resume_str": resume,
                    "Category": category,
                    "job_position_name": category,
                }
            )

    rng.shuffle(rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Resume_str", "Category", "job_position_name"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rows = generate_rows()

    primary = repo_root / "data" / "resume_dataset.csv"
    fixed = repo_root / "data" / "resume_dataset_fixed.csv"

    write_csv(primary, rows)
    write_csv(fixed, rows)

    print(f"[✓] Wrote {len(rows)} rows to {primary}")
    print(f"[✓] Wrote {len(rows)} rows to {fixed}")


if __name__ == "__main__":
    main()
