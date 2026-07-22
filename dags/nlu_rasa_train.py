"""
Rasa Auto-Training DAG.
Trains a new Rasa model in an isolated container, then hot-reloads it
into the running Rasa server via the REST API. Zero downtime.
"""

import os
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator


ENABLE_AUTO_TRAIN = os.environ.get("ENABLE_AUTO_TRAIN", "false").lower() == "true"
RASA_SERVER_URL = os.environ.get("RASA_SERVER_URL", "http://rasa:5005")
MODELS_DIR = "/opt/airflow/models"
PROJECT_DIR = "/opt/airflow/project"


default_args = {
    "owner": "amhkhowaja",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_auto_train_enabled(**kwargs):
    """Gate: skip entire DAG if auto-training is disabled."""
    if ENABLE_AUTO_TRAIN:
        print("Auto-training is ENABLED.")
        return True
    else:
        print("Auto-training is DISABLED (set ENABLE_AUTO_TRAIN=true to enable).")
        return False


def train_rasa_model(**kwargs):
    """Spin up an x86 Rasa container (QEMU on Mac, native on Linux), train, output to shared volume."""
    project_dir = "/opt/airflow/project"

    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "--platform", "linux/amd64",
            "-v", f"{project_dir}/data:/app/data",
            "-v", f"{project_dir}/models:/app/models",
            "-v", f"{project_dir}/config.yml:/app/config.yml",
            "-v", f"{project_dir}/domain.yml:/app/domain.yml",
            "rasa/rasa:3.4.4-full",
            "train", "--fixed-model-name", "latest"
        ],
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min max (QEMU is slow)
    )

    print(f"STDOUT (last 500): {result.stdout[-500:]}")
    if result.returncode != 0:
        print(f"STDERR (last 500): {result.stderr[-500:]}")
        raise Exception(f"Rasa training failed with exit code {result.returncode}")

    print("Training complete. Model saved as models/latest.tar.gz")
    kwargs["ti"].xcom_push(key="model_path", value="/app/models/latest.tar.gz")
    return "models/latest.tar.gz"


def reload_rasa_model(**kwargs):
    """Hot-reload the new model into the running Rasa server."""
    import requests

    model_path = "/app/models/latest.tar.gz"

    try:
        response = requests.put(
            f"{RASA_SERVER_URL}/model",
            json={"model_file": model_path},
            timeout=120
        )
        if response.status_code == 204:
            print(f"Model loaded successfully. Rasa is now serving the new model.")
        elif response.status_code == 200:
            print(f"Model loaded. Response: {response.text[:200]}")
        else:
            print(f"Rasa responded with {response.status_code}: {response.text[:200]}")
            raise Exception(f"Model reload failed: {response.status_code}")
    except requests.ConnectionError as e:
        print(f"Could not connect to Rasa server at {RASA_SERVER_URL}.")
        print("Model trained successfully but not hot-reloaded. Restart Rasa to use it.")
        raise


with DAG(
    dag_id="nlu_rasa_train",
    default_args=default_args,
    description="Auto-train Rasa model and hot-reload into running server",
    schedule_interval="0 */12 * * *",  # Every 12 hours
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["nlu", "rasa", "training", "deployment"],
) as dag:

    gate = ShortCircuitOperator(
        task_id="check_enabled",
        python_callable=check_auto_train_enabled,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=train_rasa_model,
    )

    reload_model = PythonOperator(
        task_id="reload",
        python_callable=reload_rasa_model,
    )

    gate >> train >> reload_model
