"""
Text preprocessing pipeline using NLTK.
Handles tokenization, lemmatization, encoding, padding, and Word2Vec training.
"""

import os
from string import punctuation
from itertools import chain

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec


def ensure_nltk_data():
    """Download required NLTK data if not present."""
    for resource in ["punkt", "stopwords", "wordnet"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def tokenize(texts, lowercase=True):
    """Tokenize a list of texts into word sequences."""
    if lowercase:
        texts = [t.lower() for t in texts]
    return [word_tokenize(t) for t in texts]


def remove_punctuation(token_lists):
    """Remove punctuation tokens."""
    return [[w for w in tokens if w not in punctuation] for tokens in token_lists]


def lemmatize(token_lists):
    """Lemmatize token sequences."""
    lm = WordNetLemmatizer()
    return [[lm.lemmatize(w) for w in tokens] for tokens in token_lists]


def remove_stopwords(token_lists):
    """Remove English stopwords."""
    stop = set(stopwords.words("english"))
    return [[w for w in tokens if w not in stop] for tokens in token_lists]


def filter_vocab(token_lists, max_vocab=185):
    """Keep only the most common N words."""
    flat = list(chain.from_iterable(token_lists))
    freq = nltk.FreqDist(flat)
    common = {w for w, _ in freq.most_common(max_vocab - 1)}
    return [[w for w in tokens if w in common] for tokens in token_lists]


def encode_tokens(train_tokens, target_tokens):
    """Fit LabelEncoder on train tokens, transform target tokens. Returns (encoder, encoded)."""
    le = LabelEncoder()
    flat_train = list(chain.from_iterable(train_tokens))
    le.fit(flat_train)
    le.classes_ = np.append(le.classes_, "<oov>")

    encoded = []
    for sentence in target_tokens:
        safe = ["<oov>" if w not in le.classes_ else w for w in sentence]
        encoded.append(list(le.transform(safe)))
    return le, encoded


def pad_encoded_sequences(encoded, max_len=8, oov_index=0):
    """Pad encoded sequences to uniform length."""
    return pad_sequences(encoded, padding="post", maxlen=max_len, truncating="post", value=oov_index)


def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """Split dataframe into train/test/val and return as dict of DataFrames."""
    val_proportion = val_size / (test_size + val_size)

    x_train, x_temp, y_train, y_temp = train_test_split(
        df["Questions"], df["Intent"],
        test_size=(test_size + val_size), random_state=random_state
    )
    x_test, x_val, y_test, y_val = train_test_split(
        x_temp, y_temp, test_size=val_proportion, random_state=random_state
    )

    return {
        "train": pd.DataFrame({"Questions": x_train, "Intent": y_train}),
        "test": pd.DataFrame({"Questions": x_test, "Intent": y_test}),
        "val": pd.DataFrame({"Questions": x_val, "Intent": y_val}),
    }


def preprocess_tokens(texts, remove_punct=True, do_lemmatize=True, do_remove_stopwords=False, max_vocab=185):
    """Full preprocessing pipeline: tokenize → clean → filter."""
    tokens = tokenize(texts)
    if remove_punct:
        tokens = remove_punctuation(tokens)
    if do_lemmatize:
        tokens = lemmatize(tokens)
    if do_remove_stopwords:
        tokens = remove_stopwords(tokens)
    tokens = filter_vocab(tokens, max_vocab)
    return tokens


def train_word2vec(tokens, output_path, window=3, min_count=2, workers=4):
    """Train Word2Vec model on token sequences and save to disk."""
    w2v = Word2Vec(window=window, min_count=min_count, workers=workers)
    w2v.build_vocab(corpus_iterable=tokens)
    w2v.train(tokens, total_examples=w2v.corpus_count, epochs=w2v.epochs)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w2v.save(output_path)
    return output_path


def run(data_dir: str, output_dir: str) -> dict:
    """
    Full preprocessing step.
    Loads source CSV, splits data, trains Word2Vec, saves split CSVs.
    Returns paths and metadata.
    """
    from pipeline.data_generation import import_dataframe

    ensure_nltk_data()

    source_csv = os.path.join(data_dir, "final_data.csv")
    df = import_dataframe(source_csv)

    # Split
    splits = split_data(df)
    split_dir = os.path.join(output_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    for name, split_df in splits.items():
        split_df.to_csv(os.path.join(split_dir, f"{name}_set.csv"), index=False)

    # Preprocess train tokens and train Word2Vec
    train_texts = list(splits["train"]["Questions"])
    tokens = preprocess_tokens(train_texts)

    w2v_path = train_word2vec(tokens, os.path.join(output_dir, "models", "w2v.model"))

    # Encode labels
    le = LabelEncoder()
    le.fit(df["Intent"])

    return {
        "split_dir": split_dir,
        "w2v_path": w2v_path,
        "num_train": len(splits["train"]),
        "num_test": len(splits["test"]),
        "num_val": len(splits["val"]),
        "label_classes": list(le.classes_),
    }
