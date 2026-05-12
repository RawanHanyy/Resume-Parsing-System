# Resume Parsing System

## Project overview

This project is an NLP-based resume screening system. It reads resumes, extracts useful information, compares candidates against a job description, and returns ranked results.

The project has two connected parts:

1. An ATS-style resume parsing and ranking system
2. A deep learning resume classification component for the course requirement

The system is designed to reduce manual HR screening work by automatically analyzing resumes and producing explainable scores.

## What the project does

The project can:

- read resumes in `PDF`, `DOCX`, and `TXT` format
- extract candidate information such as name, email, phone, GitHub, LinkedIn, skills, education, and experience
- compare candidate skills and resume content with a job description
- rank candidates from best match to worst match
- show results in a Streamlit website
- run the same logic from the command line
- train and evaluate deep learning models on categorized resume data

## High-level architecture

```text
                 +----------------------+
                 |   Resume Files       |
                 | PDF / DOCX / TXT     |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |  ResumeParser        |
                 | src/resume_parser.py |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Structured Resume    |
                 | name, contact,       |
                 | skills, education,   |
                 | experience, projects |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | CandidateScorer      |
                 | src/scorer.py        |
                 +----------+-----------+
                            |
          +-----------------+------------------+
          |                 |                  |
          v                 v                  v
   Skill matching     TF-IDF similarity   Education/Experience
          \                 |                  /
           \                |                 /
            +---------------+----------------+
                            |
                            v
                 +----------------------+
                 | Final candidate rank |
                 +----------------------+
```

## Website and CLI flow

```text
Streamlit website (app.py)
    User uploads resumes + pastes job description
        -> resumes are parsed
        -> candidates are scored
        -> rankings and breakdowns are displayed

CLI (main.py)
    parse command
        -> parse one resume
        -> print or save JSON

    rank command
        -> parse all resumes in a folder
        -> score them against job description
        -> print ranked output
```

## Project structure

```text
Resume-Parsing-System-clone/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── CheckFinal.md
├── documentation_report.tex
├── sample_resume.txt
├── sample_job_description.txt
├── sample_inputs/
├── notebooks/
├── src/
│   ├── classifier.py
│   ├── preprocess.py
│   ├── resume_parser.py
│   └── scorer.py
├── tests/
└── docs/
```

## Code explanation

### `app.py`

This file builds the website using Streamlit.

It is responsible for:

- setting the page layout
- taking a job description as text input
- uploading one or more resumes
- sending each uploaded resume to `ResumeParser`
- sending parsed results to `rank_candidates`
- displaying a ranking table
- displaying per-candidate score details
- allowing results to be downloaded as JSON

In simple terms, `app.py` is the presentation layer of the project.

### `main.py`

This file is the command-line entry point.

It supports two commands:

- `parse`
  Parses one resume and outputs structured JSON.

- `rank`
  Parses all resumes in a folder, scores them against a job description, and prints a ranked list.

This file is useful for testing the core logic without using the website.

### `src/resume_parser.py`

This is the core resume parsing module.

It does the following:

- reads text from `PDF`, `DOCX`, or `TXT`
- uses regex to extract email, phone, LinkedIn, and GitHub
- uses `spaCy` to help identify the candidate name
- detects sections such as `summary`, `skills`, `education`, `experience`, `projects`, and `certifications`
- matches skills using a built-in skill taxonomy
- estimates total years of experience using date patterns

The output of this file is a structured Python dictionary like:

```json
{
  "name": "John Doe",
  "contact": {
    "email": "john@example.com"
  },
  "skills": {
    "programming_languages": ["python"],
    "frameworks": ["django"]
  }
}
```

### `src/scorer.py`

This file contains the candidate ranking logic.

It compares the parsed resume against the job description and produces:

- total score
- score breakdown
- matched skills
- missing skills
- recommendation label

The scoring combines:

- skill match score
- TF-IDF cosine similarity
- education score
- experience score

At the end, candidates are sorted by total score from highest to lowest.

### `src/preprocess.py`

This file is used by the classification pipeline.

It cleans resume text before training or prediction by:

- converting text to lowercase
- removing URLs
- removing emails
- removing phone numbers
- removing special characters
- removing standalone digits
- removing stopwords

This creates cleaner text for model training.

### `src/classifier.py`

This file contains the deep learning resume classification logic.

It can train two types of models:

- `BiLSTM`
- `DistilBERT`

It also handles:

- dataset loading
- text preprocessing
- class filtering
- train/test split
- label encoding
- training
- evaluation
- prediction
- saving plots and model artifacts

## Resume parsing pipeline

```text
Resume file
   |
   v
Read raw text
   |
   v
Extract contact info with regex
   |
   v
Extract name with spaCy + fallback rules
   |
   v
Detect sections using section headers
   |
   v
Extract skills using taxonomy matching
   |
   v
Extract education and experience
   |
   v
Estimate total experience years
   |
   v
Return structured resume data
```

## Candidate scoring logic

The final score is built from four weighted components:

| Component | Weight |
|---|---:|
| Skill Match | 45% |
| TF-IDF Similarity | 30% |
| Education | 15% |
| Experience | 10% |

### 1. Skill match

The system extracts skills from the job description and compares them to the skills found in the resume.

Example:

- job description requires: `python`, `django`, `aws`
- candidate has: `python`, `django`
- skill match = `2 / 3`

### 2. TF-IDF similarity

The system reconstructs resume text from extracted fields and compares it with the job description using cosine similarity.

This helps measure how similar the overall resume content is to the target role, even beyond exact skill matching.

### 3. Education score

The code maps education levels to ranks:

- PhD / Doctorate = highest
- Master's = high
- Bachelor's = medium
- Associate = lower

The best detected degree is normalized into a score between `0` and `1`.

### 4. Experience score

The parser estimates candidate years of experience from date ranges inside the resume.

The score is based on:

```text
candidate_years / required_years
```

The value is capped at `1.0`.

## Recommendation labels

The final score is translated into a human-readable recommendation:

- `75-100` -> `Strong Match - Recommend for Interview`
- `55-74` -> `Potential Match - Consider for Screening`
- `35-54` -> `Weak Match - Review Manually`
- `0-34` -> `Poor Match - Likely to Reject`

## Skill taxonomy

The parser uses a built-in taxonomy of skill groups, including:

- programming languages
- frameworks
- mobile technologies
- databases
- cloud and DevOps tools
- data science tools
- soft skills

This taxonomy is stored in `src/resume_parser.py` and is used for both resume skill extraction and job description skill detection.

## Deep learning component

The second part of the project focuses on resume category classification.

This is different from candidate ranking:

- ranking compares resumes to a job description
- classification predicts which category a resume belongs to

Example categories can include:

- Python Developer
- Java Developer
- Data Scientist
- HR Specialist
- Business Analyst

## Dataset preparation for the models

Before training, the code performs several filtering steps:

```text
Raw dataset
   |
   v
Clean text
   |
   v
Remove duplicate resumes
   |
   v
Drop classes with too few samples
   |
   v
Keep top classes by frequency
   |
   v
Encode labels
   |
   v
Train/test split
```

This is important because resume datasets often contain:

- duplicate data
- too many tiny classes
- class imbalance

Filtering makes the training problem more realistic and stable.

## Model 1: BiLSTM

The LSTM model in `src/classifier.py` uses this architecture:

```text
Input text
   |
   v
Tokenizer
   |
   v
Padded sequences
   |
   v
Embedding layer
   |
   v
Bidirectional LSTM
   |
   v
Dropout
   |
   v
Bidirectional LSTM
   |
   v
Dropout
   |
   v
Dense layer (ReLU)
   |
   v
Dropout
   |
   v
Softmax output
```

### Why BiLSTM was used

- it can learn sequence patterns in resume text
- bidirectional reading helps the model understand context from both left and right
- it is lighter than a large transformer model
- it often works well on medium-sized text classification datasets

### Important settings in the code

- `MAX_WORDS = 10000`
- `MAX_LEN = 300`
- `EMBED_DIM = 128`
- `LSTM_UNITS = 128`
- `BATCH_SIZE = 32`
- `EPOCHS = 20`

## Model 2: DistilBERT

The second model uses `DistilBERT`, which is a smaller and faster transformer than full BERT.

### Why DistilBERT was used

- it is transformer-based, so it captures context strongly
- it is smaller than BERT, which makes it more practical for student projects
- it is widely used for text classification tasks

### Transformer pipeline

```text
Resume text
   |
   v
DistilBERT tokenizer
   |
   v
Token IDs + attention masks
   |
   v
DistilBERT encoder
   |
   v
Classification head
   |
   v
Predicted resume category
```

## Why the project uses both classic NLP and deep learning

The project solves two related but different problems:

### ATS resume ranking

This part needs:

- explainable logic
- skill matching
- direct comparison with a job description

That is why rule-based extraction and weighted scoring are used.

### Resume category classification

This part needs:

- supervised learning
- category prediction from resume text

That is why LSTM and DistilBERT are used.

## How to run the website

Create the environment and install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## How to run from the command line

Parse one resume:

```bash
python main.py parse --file resume.pdf
```

Save parsed output:

```bash
python main.py parse --file resume.pdf --output result.json
```

Rank resumes inside a folder:

```bash
python main.py rank --folder ./sample_inputs --jd sample_job_description.txt --years 3
```

## How to run tests

```bash
pytest -q
```

## Example system usage

```text
Job description:
  Python developer with Django, PostgreSQL, AWS, and Docker

Candidate resume:
  Contains Python, Django, AWS, Git, and 4 years experience

System result:
  High skill match
  Good similarity
  Good experience score
  Final recommendation: Potential Match or Strong Match
```

## Strengths of the project

- supports multiple resume file formats
- combines rule-based NLP and scoring
- gives explainable ranking results
- has both website and CLI interfaces
- includes a deep learning classification extension
- includes tests and generated result artifacts

## Current limitations

- the parser depends on resume formatting quality
- some resumes may not have clear section headers
- skill extraction depends on the predefined taxonomy
- education and experience extraction are heuristic-based
- dataset size can limit transformer performance

## Summary

This project is a complete resume analysis system that combines:

- information extraction
- job-description matching
- candidate ranking
- explainable scoring
- a website interface
- a CLI interface
- deep learning classification models

The codebase is structured so that the parser, scorer, website, and learning models are separated clearly, which makes the project easier to explain, test, and extend.
