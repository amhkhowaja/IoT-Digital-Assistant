"""
NLU Self-Learning Pipeline DAG.
Extracts user predictions from MongoDB, re-validates with custom models,
augments accepted examples, and writes Rasa training YAML.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


OUTPUT_DIR = os.environ.get("PIPELINE_OUTPUT_DIR", "/opt/airflow/output")
NLU_OUTPUT_PATH = "/opt/airflow/rasa_data/nlu_generated.yml"
SKIP_CUSTOM_VALIDATION = os.environ.get("SKIP_CUSTOM_VALIDATION", "false").lower() == "true"
RASA_CONFIDENCE_THRESHOLD = float(os.environ.get("RASA_CONFIDENCE_THRESHOLD", "0.85"))


default_args = {
    "owner": "amhkhowaja",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def task_extract(**kwargs):
    from pipeline.etl_extract import extract_unprocessed
    records = extract_unprocessed(limit=500)
    if not records:
        print("No new predictions to process.")
    else:
        print(f"Extracted {len(records)} unprocessed predictions.")
    kwargs["ti"].xcom_push(key="records", value=records)
    return len(records)


def task_predict(**kwargs):
    from pipeline.etl_predict import run as predict_run
    ti = kwargs["ti"]
    records = ti.xcom_pull(task_ids="extract", key="records")
    if not records:
        ti.xcom_push(key="custom_predictions", value=[])
        return 0

    custom_predictions = predict_run(records=records, output_dir=OUTPUT_DIR)
    ti.xcom_push(key="custom_predictions", value=custom_predictions)
    print(f"Generated {len(custom_predictions)} custom predictions.")
    return len(custom_predictions)


def task_compare(**kwargs):
    from pipeline.etl_compare import compare_predictions, format_review_items
    from pipeline.etl_extract import save_review_items
    ti = kwargs["ti"]
    records = ti.xcom_pull(task_ids="extract", key="records")
    custom_predictions = ti.xcom_pull(task_ids="predict", key="custom_predictions")

    if not records:
        ti.xcom_push(key="accepted", value=[])
        return 0

    if SKIP_CUSTOM_VALIDATION:
        # Accept based on Rasa confidence only
        accepted = []
        review = []
        for record in records:
            rasa_conf = record["rasa_prediction"]["intent_confidence"]
            item = {
                "text": record["text"],
                "rasa_intent": record["rasa_prediction"]["intent"],
                "rasa_confidence": rasa_conf,
                "custom_intent": "skipped",
                "custom_confidence": 0,
                "entities": record["rasa_prediction"].get("entities", []),
                "record_id": record["_id"],
            }
            if rasa_conf >= RASA_CONFIDENCE_THRESHOLD:
                item["status"] = "auto_accept"
                accepted.append(item)
            else:
                item["status"] = "low_confidence"
                item["reason"] = f"Rasa confidence {rasa_conf:.2f} below {RASA_CONFIDENCE_THRESHOLD}"
                review.append(item)
        print(f"Skip mode: Accepted {len(accepted)}, Review {len(review)} (threshold: {RASA_CONFIDENCE_THRESHOLD})")
    else:
        accepted, review = compare_predictions(records, custom_predictions)
        print(f"Dual validation: Accepted {len(accepted)}, Review {len(review)}")

    # Save review items to MongoDB
    if review:
        formatted_review = format_review_items(review)
        save_review_items(formatted_review)

    ti.xcom_push(key="accepted", value=accepted)
    return len(accepted)


def task_augment(**kwargs):
    from pipeline.etl_augment import augment_examples
    ti = kwargs["ti"]
    accepted = ti.xcom_pull(task_ids="compare", key="accepted")

    if not accepted:
        ti.xcom_push(key="augmented", value=[])
        return 0

    augmented = augment_examples(accepted, use_bert=True, max_per_example=3)
    ti.xcom_push(key="augmented", value=augmented)
    print(f"Generated {len(augmented)} augmented examples from {len(accepted)} originals.")
    return len(augmented)


def task_transform(**kwargs):
    from pipeline.etl_transform import run as transform_run
    from pipeline.etl_extract import mark_processed
    ti = kwargs["ti"]
    accepted = ti.xcom_pull(task_ids="compare", key="accepted")
    augmented = ti.xcom_pull(task_ids="augment", key="augmented")

    if not accepted:
        print("Nothing to write — no accepted examples.")
        return 0

    result = transform_run(
        accepted_items=accepted,
        augmented_items=augmented or [],
        output_path=NLU_OUTPUT_PATH,
    )

    # Mark all extracted records as processed
    record_ids = [item["record_id"] for item in accepted]
    mark_processed(record_ids)

    print(f"Written {result['total_examples']} examples ({result['intents']} intents) to {result['output_path']}")
    return result["total_examples"]


with DAG(
    dag_id="nlu_self_learning",
    default_args=default_args,
    description="Self-learning pipeline: extract predictions, validate, augment, generate Rasa training YAML",
    schedule_interval="0 */6 * * *",  # Every 6 hours
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["nlu", "etl", "self-learning"],
) as dag:

    extract = PythonOperator(task_id="extract", python_callable=task_extract)
    predict = PythonOperator(task_id="predict", python_callable=task_predict)
    compare = PythonOperator(task_id="compare", python_callable=task_compare)
    augment = PythonOperator(task_id="augment", python_callable=task_augment)
    transform = PythonOperator(task_id="transform", python_callable=task_transform)

    extract >> predict >> compare >> augment >> transform
