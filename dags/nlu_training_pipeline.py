"""
NLU Training Pipeline DAG.
Orchestrates: data generation → preprocessing → intent classification → NER training.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


# Paths inside the Airflow worker container
DATA_DIR = os.environ.get("PIPELINE_DATA_DIR", "/opt/airflow/data")
OUTPUT_DIR = os.environ.get("PIPELINE_OUTPUT_DIR", "/opt/airflow/output")


default_args = {
    "owner": "amhkhowaja",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def task_data_generation(**kwargs):
    from pipeline.parse_rasa_nlu import run
    nlu_path = "/opt/airflow/rasa_data/nlu.yml"
    result = run(nlu_path=nlu_path, output_dir=OUTPUT_DIR)
    print(f"Parsed Rasa NLU: {result['num_examples']} examples, {result['num_intents']} intents")
    return result


def task_preprocessing(**kwargs):
    from pipeline.preprocessing import split_data, preprocess_tokens, train_word2vec
    import os
    import pandas as pd

    # Load parsed Rasa NLU data (single source of truth)
    csv_path = os.path.join(OUTPUT_DIR, "rasa_nlu_parsed.csv")
    df = pd.read_csv(csv_path)

    # Split
    splits = split_data(df)
    split_dir = os.path.join(OUTPUT_DIR, "splits")
    os.makedirs(split_dir, exist_ok=True)
    for name, split_df in splits.items():
        split_df.to_csv(os.path.join(split_dir, f"{name}_set.csv"), index=False)

    # Train Word2Vec on training tokens
    train_texts = list(splits["train"]["Questions"])
    tokens = preprocess_tokens(train_texts)
    w2v_path = train_word2vec(tokens, os.path.join(OUTPUT_DIR, "models", "w2v.model"))

    print(f"Preprocessing complete: train={len(splits['train'])}, test={len(splits['test'])}, val={len(splits['val'])}")
    return {"split_dir": split_dir, "w2v_path": w2v_path}


def task_intent_classification(**kwargs):
    from pipeline.intent_classification import run
    result = run(data_dir=OUTPUT_DIR, output_dir=OUTPUT_DIR, epochs=100, patience=4)
    print(f"Intent classification complete:")
    print(f"  BiLSTM accuracy: {result['bilstm']['accuracy']:.4f}")
    print(f"  CNN accuracy: {result['cnn']['accuracy']:.4f}")
    return result


def task_ner_training(**kwargs):
    from pipeline.ner import run
    result = run(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, epochs=10, batch_size=5)
    print(f"NER training complete: F1={result['ents_f1']:.4f}")
    return result


with DAG(
    dag_id="nlu_training_pipeline",
    default_args=default_args,
    description="Train NLU intent classification and NER models",
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["nlu", "training", "ml"],
) as dag:

    data_gen = PythonOperator(
        task_id="data_generation",
        python_callable=task_data_generation,
    )

    preprocess = PythonOperator(
        task_id="preprocessing",
        python_callable=task_preprocessing,
    )

    intent_clf = PythonOperator(
        task_id="intent_classification",
        python_callable=task_intent_classification,
    )

    ner = PythonOperator(
        task_id="ner_training",
        python_callable=task_ner_training,
    )

    # DAG dependency chain
    data_gen >> preprocess >> intent_clf
    data_gen >> ner  # NER uses annotated.json directly, parallel with preprocessing
