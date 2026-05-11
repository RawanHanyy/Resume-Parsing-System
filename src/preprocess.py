"""
preprocess.py
=============
Cleans raw resume text before feeding into the ML model.

Used by: classifier.py (during training and prediction)
NOT used by: resume_parser.py (handles its own text reading)
"""

import re

STOPWORDS = {
    "i","me","my","we","our","you","your","he","she","it","they","them",
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","was","are","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","this","that","these","those","as","if","then","than","so","up",
    "out","about","into","through","during","before","after","above","below",
    "between","each","both","few","more","most","other","some","such","no",
    "not","only","same","also","just","because","while","although","however",
    "therefore","thus",
}

def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline.

    Input : raw resume text (messy, with URLs, emails, symbols)
    Output: clean lowercase string ready for tokenization

    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove emails
      4. Remove phone numbers
      5. Remove special characters (keep + and # for c++, c#)
      6. Remove standalone digits (years like 2019 add noise)
      7. Collapse whitespace
      8. Remove stopwords
      9. Remove single-character tokens
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", " ", text)
    text = re.sub(r"(\+?\d[\d\s.\-()]{7,}\d)", " ", text)
    text = re.sub(r"[^\w\s+#]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)
