"""
Named Entity Recognition model training using spaCy.
Trains a custom NER on annotated data and optionally merges with en_core_web_lg.
"""

import os
import json
import random

import spacy
import pandas as pd
from spacy.training import Example
from sklearn.model_selection import train_test_split


def load_annotations(annotations_path: str):
    """Load annotated training data from JSON file."""
    with open(annotations_path) as f:
        data = json.loads(f.read())
    return data


def split_annotations(annotations, test_size=0.25, random_state=42):
    """Split annotations into train/test."""
    train, test = train_test_split(annotations, test_size=test_size, random_state=random_state)
    return train, test


def train_ner(train_data, epochs=10, batch_size=5, drop=0.25):
    """
    Train a blank spaCy NER model on annotated data.
    Returns the trained nlp object.
    """
    nlp = spacy.blank("en")

    if "ner" not in nlp.pipe_names:
        nlp.add_pipe("ner", last=True)

    ner = nlp.get_pipe("ner")
    for _, annot in train_data:
        for ent in annot["entities"]:
            ner.add_label(ent[2])

    not_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*not_pipes):
        optimizer = nlp.begin_training()
        for epoch in range(epochs):
            random.shuffle(train_data)
            losses = {}
            for batch in spacy.util.minibatch(train_data, batch_size):
                for question, label in batch:
                    doc = nlp.make_doc(question)
                    example = Example.from_dict(doc, label)
                    nlp.update([example], drop=drop, sgd=optimizer, losses=losses)
            print(f"Epoch {epoch + 1}/{epochs} — Loss: {losses.get('ner', 0):.4f}")

    return nlp


def evaluate_ner(nlp, test_data):
    """Evaluate NER model on test data, return scores dict."""
    examples = []
    for question, annot in test_data:
        doc = nlp.make_doc(question)
        example = Example.from_dict(doc, annot)
        examples.append(example)
    scores = nlp.evaluate(examples)
    return {
        "ents_p": scores.get("ents_p", 0),
        "ents_r": scores.get("ents_r", 0),
        "ents_f": scores.get("ents_f", 0),
    }


def save_model(nlp, output_path: str):
    """Save spaCy model to disk."""
    os.makedirs(output_path, exist_ok=True)
    nlp.to_disk(output_path)
    return output_path


def run(data_dir: str, output_dir: str, epochs=10, batch_size=5) -> dict:
    """
    Full NER training step.
    Loads annotations, trains spaCy NER, evaluates, and saves.
    """
    annotations_path = os.path.join(data_dir, "annotated.json")
    model_dir = os.path.join(output_dir, "models")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Load and split
    annotations = load_annotations(annotations_path)
    train_data, test_data = split_annotations(annotations)

    # Train
    nlp = train_ner(train_data, epochs=epochs, batch_size=batch_size)

    # Save model
    ner_model_path = os.path.join(model_dir, "ner_model")
    save_model(nlp, ner_model_path)

    # Evaluate
    metrics = evaluate_ner(nlp, test_data)

    # Save metrics
    metrics_path = os.path.join(metrics_dir, "ner_report.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_path": ner_model_path,
        "metrics_path": metrics_path,
        "ents_f1": metrics["ents_f"],
        "num_train": len(train_data),
        "num_test": len(test_data),
    }
