"""
Resume Category Classifier
===========================
Trains and evaluates two models:
  1. Bidirectional LSTM
  2. BERT (DistilBERT) Transformer

Dataset: Kaggle Resume Dataset (~2400 resumes, 25 job categories)

Usage
-----
from classifier import ResumeClassifier
clf = ResumeClassifier(model_type="lstm")  # or "bert"
clf.train("data/resume_dataset.csv")
clf.evaluate()
prediction = clf.predict("Experienced Python developer with Django and AWS...")
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (
    accuracy_score, classification_report, confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.models     import Sequential, load_model
from tensorflow.keras.layers     import (
    Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D
)
from tensorflow.keras.preprocessing.text     import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks              import EarlyStopping, ModelCheckpoint

from preprocess import clean_text


# Constants 
MAX_WORDS    = 10_000   # vocabulary size
MAX_LEN      = 300      # max tokens per resume
EMBED_DIM    = 128      # embedding dimensions
LSTM_UNITS   = 128      # LSTM hidden units
BATCH_SIZE   = 32
EPOCHS       = 20       # early stopping will kick in before this
TEST_SIZE    = 0.20     # 80% train / 20% test
RANDOM_STATE = 42


class ResumeClassifier:
    """
    Train, evaluate, and predict resume job categories.

    Parameters
    ----------
    model_type : "lstm" | "bert"
    """

    def __init__(self, model_type: str = "lstm"):
        assert model_type in ("lstm", "bert"), "model_type must be 'lstm' or 'bert'"
        self.model_type    = model_type
        self.model         = None
        self.tokenizer     = None          # Keras tokenizer (LSTM only)
        self.label_encoder = LabelEncoder()
        self.history       = None
        self.X_test        = None
        self.y_test        = None
        self.num_classes   = None

    # 1. Load & preprocess data 
    def _load_data(self, csv_path: str):
        """
        Load Kaggle resume CSV.
        Expected columns: 'Resume_str' (text) and 'Category' (label).
        """
        df = pd.read_csv(csv_path)

        # Flexible column detection
        text_col = next(
            c for c in df.columns
            if "resume" in c.lower() or "text" in c.lower()
        )
        label_col = next(
            c for c in df.columns
            if "category" in c.lower() or "label" in c.lower()
        )

        print(f"[✓] Loaded {len(df)} resumes from '{csv_path}'")
        print(f"    Text column  : {text_col}")
        print(f"    Label column : {label_col}")

        df["clean_text"] = df[text_col].astype(str).apply(clean_text)
        df["label_enc"]  = self.label_encoder.fit_transform(df[label_col])
        self.num_classes = len(self.label_encoder.classes_)

        print(f"    Categories   : {self.num_classes}")
        print(f"    Classes      : {list(self.label_encoder.classes_)}\n")

        return df["clean_text"].values, df["label_enc"].values

    # 2. Tokenize (LSTM) 
    def _tokenize_lstm(self, texts_train, texts_test):
        self.tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts_train)

        X_train = pad_sequences(
            self.tokenizer.texts_to_sequences(texts_train),
            maxlen=MAX_LEN, padding="post", truncating="post"
        )
        X_test = pad_sequences(
            self.tokenizer.texts_to_sequences(texts_test),
            maxlen=MAX_LEN, padding="post", truncating="post"
        )
        return X_train, X_test

    # 3. Build LSTM model 
    def _build_lstm(self) -> tf.keras.Model:
        """
        Bidirectional LSTM architecture:
          Embedding → BiLSTM → Dropout → Dense → Output
        """
        model = Sequential([
            # Embedding layer: turns word indices into dense vectors
            Embedding(
                input_dim    = MAX_WORDS,
                output_dim   = EMBED_DIM,
                input_length = MAX_LEN,
                name         = "embedding"
            ),

            # Bidirectional LSTM: reads sequence forwards AND backwards
            Bidirectional(
                LSTM(LSTM_UNITS, return_sequences=True),
                name="bilstm_1"
            ),
            Dropout(0.3, name="dropout_1"),

            # Second LSTM layer
            Bidirectional(
                LSTM(64),
                name="bilstm_2"
            ),
            Dropout(0.3, name="dropout_2"),

            # Dense hidden layer
            Dense(128, activation="relu", name="dense_1"),
            Dropout(0.3, name="dropout_3"),

            # Output layer: one neuron per category
            Dense(self.num_classes, activation="softmax", name="output")
        ])

        model.compile(
            optimizer = "adam",
            loss      = "sparse_categorical_crossentropy",
            metrics   = ["accuracy"]
        )

        model.summary()
        return model

    # 4. Build BERT model
    def _build_bert(self):
        """
        DistilBERT fine-tuned for sequence classification.
        Requires: pip install transformers
        """
        from transformers import (
            TFDistilBertForSequenceClassification,
            DistilBertTokenizerFast
        )

        self.bert_tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )
        model = TFDistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels = self.num_classes
        )
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss      = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics   = ["accuracy"]
        )
        return model

    def _encode_bert(self, texts):
        encodings = self.bert_tokenizer(
            list(texts),
            truncation  = True,
            padding     = True,
            max_length  = MAX_LEN,
            return_tensors = "tf"
        )
        return dict(encodings)

    # 5. Train 
    def train(self, csv_path: str):
        texts, labels = self._load_data(csv_path)

        # Train / test split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            texts, labels,
            test_size    = TEST_SIZE,
            random_state = RANDOM_STATE,
            stratify     = labels      # ensure equal class distribution
        )
        self.y_test = y_test

        os.makedirs("models", exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor   = "val_accuracy",
                patience  = 3,
                restore_best_weights = True,
                verbose   = 1
            ),
            ModelCheckpoint(
                filepath       = f"models/{self.model_type}_best.keras",
                save_best_only = True,
                monitor        = "val_accuracy",
                verbose        = 1
            )
        ]

        # LSTM path 
        if self.model_type == "lstm":
            X_train, X_test = self._tokenize_lstm(X_train_raw, X_test_raw)
            self.X_test = X_test
            self.model  = self._build_lstm()

            self.history = self.model.fit(
                X_train, y_train,
                validation_data = (X_test, y_test),
                epochs          = EPOCHS,
                batch_size      = BATCH_SIZE,
                callbacks       = callbacks,
                verbose         = 1
            )

            # Save tokenizer and encoder
            with open("models/tokenizer.pkl", "wb") as f:
                pickle.dump(self.tokenizer, f)

        # BERT path 
        elif self.model_type == "bert":
            X_train = self._encode_bert(X_train_raw)
            X_test  = self._encode_bert(X_test_raw)
            self.X_test = X_test
            self.model  = self._build_bert()

            self.history = self.model.fit(
                X_train, y_train,
                validation_data = (X_test, y_test),
                epochs          = 3,    # BERT needs fewer epochs
                batch_size      = 16,
                callbacks       = callbacks,
                verbose         = 1
            )

        # Save label encoder
        with open("models/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

        print(f"\n[✓] Training complete. Model saved to models/")

    # 6. Evaluate 
    def evaluate(self, save_plots: bool = True):
        """
        Compute and display:
          - Test accuracy
          - Classification report (precision, recall, F1 per class)
          - Confusion matrix heatmap
          - Training curves (accuracy & loss)
        """
        if self.model is None or self.X_test is None:
            raise RuntimeError("Call train() first.")

        # Predictions 
        if self.model_type == "lstm":
            y_prob = self.model.predict(self.X_test)
        else:
            outputs = self.model.predict(self.X_test)
            y_prob  = tf.nn.softmax(outputs.logits).numpy()

        y_pred = np.argmax(y_prob, axis=1)

        # Metrics 
        acc = accuracy_score(self.y_test, y_pred)
        print(f"\n{'='*60}")
        print(f"  TEST ACCURACY : {acc * 100:.2f}%")
        print(f"{'='*60}\n")

        class_names = self.label_encoder.classes_
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=class_names))

        os.makedirs("docs", exist_ok=True)

        # Confusion matrix 
        cm = confusion_matrix(self.y_test, y_pred)
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm,
            annot      = True,
            fmt        = "d",
            cmap       = "Blues",
            xticklabels = class_names,
            yticklabels = class_names,
            ax          = ax
        )
        ax.set_title(f"Confusion Matrix — {self.model_type.upper()} Model", fontsize=14)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"docs/confusion_matrix_{self.model_type}.png", dpi=150)
            print("[✓] Saved: docs/confusion_matrix.png")
        plt.show()

        # Training curves 
        if self.history:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Accuracy
            ax1.plot(self.history.history["accuracy"],     label="Train Accuracy")
            ax1.plot(self.history.history["val_accuracy"], label="Val Accuracy")
            ax1.set_title("Model Accuracy over Epochs")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Accuracy")
            ax1.legend()
            ax1.grid(True)

            # Loss
            ax2.plot(self.history.history["loss"],     label="Train Loss")
            ax2.plot(self.history.history["val_loss"], label="Val Loss")
            ax2.set_title("Model Loss over Epochs")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            if save_plots:
                plt.savefig(f"docs/training_curves_{self.model_type}.png", dpi=150)
                print("[✓] Saved: docs/training_curves.png")
            plt.show()

        return acc

    # 7. Predict a single resume 
    def predict(self, resume_text: str) -> dict:
        """
        Predict the job category of a single resume text.
        Returns predicted category + confidence scores for all classes.
        """
        if self.model is None:
            raise RuntimeError("Call train() or load() first.")

        cleaned = clean_text(resume_text)

        if self.model_type == "lstm":
            seq    = self.tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
            probs  = self.model.predict(padded, verbose=0)[0]

        else:
            encoded = self._encode_bert([cleaned])
            outputs = self.model.predict(encoded, verbose=0)
            probs   = tf.nn.softmax(outputs.logits[0]).numpy()

        top_idx   = int(np.argmax(probs))
        top_label = self.label_encoder.classes_[top_idx]
        confidence = float(probs[top_idx]) * 100

        # Top 3 predictions
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {
                "category":   self.label_encoder.classes_[i],
                "confidence": round(float(probs[i]) * 100, 2)
            }
            for i in top3_idx
        ]

        return {
            "predicted_category": top_label,
            "confidence":         round(confidence, 2),
            "top_3":              top3
        }

    # 8. Save / Load 
    def save(self, path: str = "models/"):
        self.model.save(os.path.join(path, f"{self.model_type}_model.keras"))
        print(f"[✓] Model saved to {path}")

    def load(self, path: str = "models/"):
        self.model = load_model(os.path.join(path, f"{self.model_type}_model.keras"))
        with open(os.path.join(path, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)
        if self.model_type == "lstm":
            with open(os.path.join(path, "tokenizer.pkl"), "rb") as f:
                self.tokenizer = pickle.load(f)
        print(f"[✓] Model loaded from {path}")
