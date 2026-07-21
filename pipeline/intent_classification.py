"""
Intent classification model training.
Trains BiLSTM and CNN models, evaluates, and saves artifacts.
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def build_bilstm(vocab_size=120, embedding_dim=8, max_length=8, num_classes=5):
    """Build a Bidirectional LSTM model."""
    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.Bidirectional(layers.LSTM(embedding_dim)),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
    return model


def build_cnn(vocab_size=120, embedding_dim=8, max_length=8, num_classes=5):
    """Build a 1D CNN model."""
    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.Conv1D(16, 8, activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
    return model


def prepare_data(train_csv: str, test_csv: str, val_csv: str, vocab_size=120, max_length=8):
    """Load split CSVs, tokenize, pad, encode labels. Returns ready-to-train data."""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    val_df = pd.read_csv(val_csv)

    # Encode labels
    le = LabelEncoder()
    le.fit(pd.concat([train_df["Intent"], test_df["Intent"], val_df["Intent"]]))
    y_train = le.transform(train_df["Intent"])
    y_test = le.transform(test_df["Intent"])
    y_val = le.transform(val_df["Intent"])

    # Tokenize and pad
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["Questions"])

    x_train = pad_sequences(tokenizer.texts_to_sequences(train_df["Questions"]),
                            maxlen=max_length, padding="post", truncating="post")
    x_test = pad_sequences(tokenizer.texts_to_sequences(test_df["Questions"]),
                           maxlen=max_length, padding="post", truncating="post")
    x_val = pad_sequences(tokenizer.texts_to_sequences(val_df["Questions"]),
                          maxlen=max_length, padding="post", truncating="post")

    return {
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": y_test,
        "x_val": x_val, "y_val": y_val,
        "label_encoder": le,
        "tokenizer": tokenizer,
        "num_classes": len(le.classes_),
    }


def train_model(model, x_train, y_train, x_val, y_val, epochs=100, patience=4):
    """Train a model with early stopping. Returns history."""
    early_stop = callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        callbacks=[early_stop],
        validation_data=(x_val, y_val),
        shuffle=True,
        verbose=1
    )
    return history


def evaluate_model(model, x_test, y_test, label_encoder):
    """Evaluate model and return metrics dict."""
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=label_encoder.classes_,
                                   output_dict=True)

    accuracy = report["accuracy"]
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "predictions": y_pred.tolist(),
    }


def run(data_dir: str, output_dir: str, epochs=100, patience=4) -> dict:
    """
    Full intent classification training step.
    Loads split data, trains BiLSTM + CNN, evaluates both, saves models and metrics.
    """
    split_dir = os.path.join(data_dir, "splits")
    model_dir = os.path.join(output_dir, "models")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Prepare data
    data = prepare_data(
        train_csv=os.path.join(split_dir, "train_set.csv"),
        test_csv=os.path.join(split_dir, "test_set.csv"),
        val_csv=os.path.join(split_dir, "val_set.csv"),
    )

    results = {}

    # Train BiLSTM
    bilstm = build_bilstm(num_classes=data["num_classes"])
    train_model(bilstm, data["x_train"], data["y_train"],
                data["x_val"], data["y_val"], epochs=epochs, patience=patience)

    bilstm_path = os.path.join(model_dir, "bilstm.h5")
    bilstm.save(bilstm_path)

    bilstm_metrics = evaluate_model(bilstm, data["x_test"], data["y_test"], data["label_encoder"])
    results["bilstm"] = {"model_path": bilstm_path, "accuracy": bilstm_metrics["accuracy"]}

    # Train CNN
    cnn = build_cnn(num_classes=data["num_classes"])
    train_model(cnn, data["x_train"], data["y_train"],
                data["x_val"], data["y_val"], epochs=epochs, patience=patience)

    cnn_path = os.path.join(model_dir, "cnn.h5")
    cnn.save(cnn_path)

    cnn_metrics = evaluate_model(cnn, data["x_test"], data["y_test"], data["label_encoder"])
    results["cnn"] = {"model_path": cnn_path, "accuracy": cnn_metrics["accuracy"]}

    # Save metrics
    all_metrics = {"bilstm": bilstm_metrics, "cnn": cnn_metrics}
    metrics_path = os.path.join(metrics_dir, "intent_classification_report.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    results["metrics_path"] = metrics_path
    results["label_classes"] = list(data["label_encoder"].classes_)
    return results
