"""
Parse Rasa NLU YAML into training CSV for custom models.
Used by the training DAG to keep custom models aligned with Rasa's intent taxonomy.
"""

import os
import re
import yaml
import pandas as pd


def parse_nlu_yaml(nlu_path: str) -> pd.DataFrame:
    """Parse nlu.yml → DataFrame of (Questions, Intent) pairs."""
    with open(nlu_path, "r") as f:
        data = yaml.safe_load(f)

    records = []
    for item in data.get("nlu", []):
        intent = item.get("intent")
        examples = item.get("examples", "")
        if not intent or not examples:
            continue

        for line in examples.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                text = line[2:].strip()
                # Strip [value](entity) annotations → keep value
                text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
                text = re.sub(r'\{[^}]+\}', '', text).strip()
                if text:
                    records.append({"Questions": text, "Intent": intent})

    return pd.DataFrame(records)


def run(nlu_path: str, output_dir: str) -> dict:
    """Parse Rasa YAML and save as training CSV."""
    df = parse_nlu_yaml(nlu_path)
    output_path = os.path.join(output_dir, "rasa_nlu_parsed.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return {
        "output_path": output_path,
        "num_examples": len(df),
        "num_intents": df["Intent"].nunique(),
    }
