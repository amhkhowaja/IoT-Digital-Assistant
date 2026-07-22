"""
ETL Predict: Run custom models (intent + NER) on extracted texts.
"""

import os
import numpy as np
import spacy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


def load_intent_model(model_path, train_csv_path):
    """Load trained intent classifier and prepare tokenizer + label encoder."""
    model = load_model(model_path)

    # Rebuild tokenizer from training data
    train_df = pd.read_csv(train_csv_path)
    tokenizer = Tokenizer(num_words=120, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["Questions"])

    # Get label classes
    label_classes = sorted(train_df["Intent"].unique())

    return model, tokenizer, label_classes


def predict_intents(texts, model, tokenizer, label_classes, max_length=8):
    """Predict intents for a list of texts using the trained model."""
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

    predictions = model.predict(padded, verbose=0)
    results = []
    for pred in predictions:
        idx = np.argmax(pred)
        results.append({
            "intent": label_classes[idx],
            "confidence": float(pred[idx]),
        })
    return results


def load_ner_model(model_path):
    """Load trained spaCy NER model."""
    return spacy.load(model_path)


def predict_entities(texts, nlp):
    """Extract entities from texts using spaCy NER model."""
    results = []
    for text in texts:
        doc = nlp(text)
        entities = [
            {"entity": ent.label_, "value": ent.text, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]
        results.append(entities)
    return results


def run(records, output_dir, **kwargs):
    """
    Predict step entry point.
    Runs both intent classification and NER on extracted records.
    """
    models_dir = os.path.join(output_dir, "models")
    splits_dir = os.path.join(output_dir, "splits")

    # Load models
    intent_model, tokenizer, label_classes = load_intent_model(
        model_path=os.path.join(models_dir, "bilstm.h5"),
        train_csv_path=os.path.join(splits_dir, "train_set.csv"),
    )
    ner_model = load_ner_model(os.path.join(models_dir, "ner_model"))

    texts = [r["text"] for r in records]

    # Predict
    intent_preds = predict_intents(texts, intent_model, tokenizer, label_classes)
    entity_preds = predict_entities(texts, ner_model)

    # Combine
    custom_predictions = []
    for i, record in enumerate(records):
        custom_predictions.append({
            "text": record["text"],
            "custom_intent": intent_preds[i]["intent"],
            "custom_intent_confidence": intent_preds[i]["confidence"],
            "custom_entities": entity_preds[i],
        })

    return custom_predictions
