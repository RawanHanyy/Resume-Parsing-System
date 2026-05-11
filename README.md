# 📄 Resume Parsing System

> **NLP-powered Applicant Tracking System (ATS)** that automatically parses resumes and ranks candidates against job descriptions — eliminating manual screening effort for HR teams.

---

## 🗂️ Project Structure

```
resume-parser/
│
├── src/                         ← all Python logic lives here
│   ├── preprocess.py            ← cleans text before the model sees it
│   ├── resume_parser.py         ← extracts info from resume files
│   ├── scorer.py                ← scores candidates vs job description
│   └── classifier.py           ← trains LSTM + BERT, evaluates, predicts
│
├── notebooks/
│   └── train_model.ipynb        ← run this to train and evaluate
│
├── models/                      ← created automatically when you train
│   ├── lstm_best.keras
│   ├── bert_best.keras
│   └── tokenizer.pkl
│
├── data/
│   └── resume_dataset.csv       ← download from Kaggle
│
├── docs/                        ← created automatically by the notebook
│   ├── confusion_matrix_lstm.png
│   ├── training_curves_lstm.png
│   └── model_comparison.png
│
├── tests/
│   └── test_parser_scorer.py    ← run with: pytest tests/
│
├── app.py                       ← streamlit web UI
├── main.py                      ← CLI entry point
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/resume-parser.git
cd resume-parser
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 🚀 Usage

### CLI — Parse a single resume
```bash
python main.py parse --file resume.pdf
python main.py parse --file resume.docx --output result.json
```

### CLI — Rank multiple resumes
```bash
python main.py rank \
  --folder ./data/sample_resumes \
  --jd job_description.txt \
  --years 3 \
  --output ranked.json
```

### Web UI (Streamlit)
```bash
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501)

---

## 🧠 How It Works

### 1. Text Extraction
Supports **PDF** (via `pdfplumber`), **DOCX** (via `docx2txt`), and **plain text**.

### 2. NLP Information Extraction
Uses **spaCy `en_core_web_sm`** for named entity recognition (NER) and custom regex pipelines to extract:

| Field | Method |
|---|---|
| Name | NER (PERSON entity) + first-line heuristic |
| Email / Phone / LinkedIn | Regex |
| Skills | Keyword matching against 60+ skill taxonomy |
| Education | Degree regex + section parsing |
| Experience | Date-range regex + section parsing |
| Certifications | Keyword + section parsing |

### 3. Scoring Algorithm

Each candidate is scored on a **0–100 scale** using four weighted components:

| Component | Weight | Method |
|---|---|---|
| Skill Match | 45% | Jaccard overlap vs. JD skills |
| Semantic Similarity | 30% | TF-IDF cosine similarity |
| Education | 15% | Degree rank (PhD > Master > Bachelor > Associate) |
| Experience | 10% | Years ratio vs. required years |

### 4. Recommendations

| Score | Label |
|---|---|
| ≥ 75% | ✅ Strong Match — Recommend for Interview |
| 55–74% | 🟡 Potential Match — Consider for Screening |
| 35–54% | 🟠 Weak Match — Review Manually |
| < 35% | ❌ Poor Match — Likely to Reject |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📊 Sample Output

```json
{
  "name": "John Doe",
  "email": "john.doe@email.com",
  "total_score": 82.4,
  "breakdown": {
    "skill_match": 88.0,
    "tfidf_sim": 76.3,
    "education": 50.0,
    "experience": 100.0
  },
  "matched_skills": ["python", "django", "postgresql", "aws", "docker"],
  "missing_skills": ["kubernetes", "redis"],
  "recommendation": "Strong Match — Recommend for Interview"
}
```

---

## 🛠️ Technologies

- **Python 3.10+**
- **spaCy** — NER and linguistic analysis
- **pdfplumber** — PDF text extraction
- **docx2txt** — Word document parsing
- **Streamlit** — Web UI
- **pandas** — Data display
- **pytest** — Testing

---

## 📌 Future Improvements

- [ ] Fine-tuned spaCy NER model trained on resume corpora
- [ ] BERT/Sentence-Transformers for semantic similarity
- [ ] Database integration (PostgreSQL) for multi-session storage
- [ ] REST API with FastAPI
- [ ] Resume anonymization for bias-free screening
- [ ] Multi-language support
