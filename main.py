"""
resume-parser CLI
=================

Usage examples
--------------
# Parse a single resume
python main.py parse --file resume.pdf

# Rank multiple resumes against a job description
python main.py rank --folder ./resumes --jd job_description.txt --years 3

# Parse and output JSON
python main.py parse --file resume.pdf --output result.json
"""

import sys
import json
import argparse
from pathlib import Path

# allow running from repo root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from resume_parser import ResumeParser
from scorer import rank_candidates, CandidateScorer


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


def cmd_parse(args: argparse.Namespace) -> None:
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    parser = ResumeParser(str(filepath))
    result = parser.parse()

    output = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[✓] Saved to {args.output}")
    else:
        print(output)


def cmd_rank(args: argparse.Namespace) -> None:
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"[ERROR] Folder not found: {folder}")
        sys.exit(1)

    jd_path = Path(args.jd)
    if not jd_path.exists():
        print(f"[ERROR] Job description file not found: {jd_path}")
        sys.exit(1)

    jd_text = jd_path.read_text(encoding="utf-8", errors="ignore")

    # Collect resume files
    resume_files = [
        f for f in folder.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not resume_files:
        print("[ERROR] No supported resume files found in folder.")
        sys.exit(1)

    print(f"[→] Parsing {len(resume_files)} resume(s)…")

    parsed_resumes = []
    for f in resume_files:
        try:
            rp = ResumeParser(str(f))
            parsed = rp.parse()
            parsed["_source_file"] = f.name
            parsed_resumes.append(parsed)
            print(f"    [✓] {f.name}")
        except Exception as e:
            print(f"    [✗] {f.name}: {e}")

    required_years = float(args.years) if args.years else None
    ranked = rank_candidates(parsed_resumes, jd_text, required_years)

    print("\n" + "=" * 60)
    print(f"{'RANK':<6} {'NAME':<25} {'SCORE':>7}  RECOMMENDATION")
    print("=" * 60)
    for i, r in enumerate(ranked, 1):
        print(f"  {i:<4} {(r['name'] or 'Unknown'):<25} {r['total_score']:>6.1f}%  {r['recommendation']}")

    print("=" * 60)

    if args.output:
        Path(args.output).write_text(
            json.dumps(ranked, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[✓] Full results saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume Parsing System — NLP-based ATS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # parse sub-command
    p_parse = sub.add_parser("parse", help="Parse a single resume file")
    p_parse.add_argument("--file", required=True, help="Path to resume (PDF/DOCX/TXT)")
    p_parse.add_argument("--output", help="Save JSON output to file")

    # rank sub-command
    p_rank = sub.add_parser("rank", help="Rank multiple resumes against a job description")
    p_rank.add_argument("--folder", required=True, help="Folder containing resume files")
    p_rank.add_argument("--jd", required=True, help="Path to job description .txt file")
    p_rank.add_argument("--years", type=float, help="Required years of experience")
    p_rank.add_argument("--output", help="Save ranked results JSON to file")

    args = parser.parse_args()

    if args.command == "parse":
        cmd_parse(args)
    elif args.command == "rank":
        cmd_rank(args)


if __name__ == "__main__":
    main()
