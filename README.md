# Resume Parsing System

This project is an NLP-based resume screening system for hiring and recruitment. It reads resumes, extracts structured candidate information, compares candidates against a job description, and returns ranked results. The repository also includes a deep learning notebook that trains and evaluates resume classification models to satisfy the course requirement for LSTM / transformer-based NLP.

## Project Goal

The main goal is to reduce manual HR screening effort by:

- parsing resumes automatically
- extracting useful candidate data such as name, email, skills, education, and experience
- scoring candidates against a target job description
- ranking candidates with explainable score breakdowns
- training and evaluating deep learning NLP models on labeled resume data

## What The System Does

The project has two parts.

### 1. ATS Resume Parsing and Ranking

This is the practical recruitment system.

- Input: PDF, DOCX, or TXT resumes
- Processing: text extraction, information extraction, skill matching, TF-IDF similarity, education scoring, experience scoring
- Output: ranked candidates with recommendation labels

Recommendation labels:

- `75-100`: Strong Match
- `55-74`: Potential Match
- `35-54`: Weak Match
- `0-34`: Poor Match

### 2. Deep Learning Classification Notebook

This is the academic NLP part of the project.

- uses a labeled resume dataset
- trains a `BiLSTM` model
- trains a `DistilBERT` transformer model
- evaluates both models using a train/test split
- produces accuracy values, confusion matrices, and training curves

## Project Structure

```text
Resume-Parsing-System/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── CheckFinal.md
├── documentation_report.tex
├── notebooks/
│   ├── train_model.ipynb
│   └── train_model.executed.ipynb
├── scripts/
│   └── generate_resume_dataset.py
├── src/
│   ├── classifier.py
│   ├── preprocess.py
│   ├── resume_parser.py
│   └── scorer.py
├── tests/
│   └── test_parser_scorer.py
├── docs/
│   ├── label_distribution.png
│   ├── confusion_matrix_lstm.png
│   ├── confusion_matrix_bert.png
│   ├── training_curves_lstm.png
│   ├── training_curves_bert.png
│   └── model_comparison.png
└── data/
    ├── resume_dataset.csv
    └── resume_dataset_fixed.csv
```

## Technologies Used

- Python
- spaCy
- pdfplumber
- docx2txt
- scikit-learn
- TensorFlow / Keras
- Hugging Face Transformers
- Streamlit
- Jupyter
- pytest

## How It Works

### Resume Parsing Flow

1. Read resume file
2. Extract raw text
3. Detect contact information using regex
4. Extract name and sections using spaCy and heuristics
5. Match skills against the built-in skill taxonomy
6. Score the resume against the job description
7. Return ranked output

### Candidate Scoring

The total score is calculated from four weighted parts:

| Component | Weight |
|---|---:|
| Skill Match | 45% |
| TF-IDF Similarity | 30% |
| Education | 15% |
| Experience | 10% |

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Generate the dataset for the notebook

```bash
python scripts/generate_resume_dataset.py
```

This creates:

- `data/resume_dataset.csv`
- `data/resume_dataset_fixed.csv`

## How To Run

### Run the Streamlit app

```bash
streamlit run app.py
```

### Run the CLI parser

```bash
python main.py parse --file resume.pdf
```

### Run the CLI ranking flow

```bash
python main.py rank --folder ./data/sample_resumes --jd job_description.txt --years 3
```

### Run the notebook

```bash
jupyter notebook notebooks/train_model.ipynb
```

### Run tests

```bash
pytest -q
```

## What We Needed To Do

Based on the project brief in `CheckFinal.md`, the required work was:

- solve the recruitment NLP problem using models such as LSTM or transformers
- prepare a dataset and split it into training and testing sets
- evaluate the model using accuracy, confusion matrix, and similar metrics
- build a working NLP application around the chosen problem

## What We Completed

The repository now includes:

- a working resume parsing and candidate ranking pipeline
- a Streamlit interface
- a command-line interface
- a reproducible dataset generation script
- an executable training notebook
- an LSTM classification model
- a transformer classification model
- evaluation plots and metrics
- automated tests
- an updated report file for Overleaf

## Dataset and Experiment Setup

The classification workflow uses a generated labeled dataset for reproducibility.

- total samples: `192`
- number of categories: `8`
- split: `80% train / 20% test`
- notebook: `notebooks/train_model.ipynb`

Resume categories used in training:

- Business Analyst
- Data Scientist
- DevOps Engineer
- HR Specialist
- Java Developer
- Network Engineer
- Python Developer
- UI UX Designer

## Results

The notebook was executed successfully and produced the following results.

| Model | Test Accuracy |
|---|---:|
| BiLSTM | `87.18%` |
| DistilBERT | `79.49%` |

Current conclusion:

- the `BiLSTM` model performed better than `DistilBERT` on the current dataset
- the difference in this run was `7.69` percentage points
- both models trained successfully and generated evaluation artifacts

Generated evaluation files:

- `docs/label_distribution.png`
- `docs/confusion_matrix_lstm.png`
- `docs/confusion_matrix_bert.png`
- `docs/training_curves_lstm.png`
- `docs/training_curves_bert.png`
- `docs/model_comparison.png`

## Testing Status

Automated tests currently pass:

```text
25 passed
```

## Current Limitations

- the ATS skill extraction is still keyword-based
- experience estimation is approximate
- the classifier is not yet integrated into the ranking app
- the classification dataset is synthetic, not a real collected resume corpus
- PDF structure can still affect extraction quality

## Suggested Next Steps

- integrate the classifier into the main ATS flow
- improve semantic matching with stronger embedding models
- expand the skill taxonomy and synonyms
- improve experience calculation from exact date spans
- add API or database support

## Final Status

The project is now in a submission-ready state for the stated requirements:

- [ ] Fine-tuned spaCy NER model trained on resume corpora
- [ ] BERT/Sentence-Transformers for semantic similarity
- [ ] Database integration (PostgreSQL) for multi-session storage
- [ ] REST API with FastAPI
- [ ] Resume anonymization for bias-free screening
- [ ] Multi-language support

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author

Built as an NLP capstone project demonstrating practical applications of natural language processing in HR automation.
