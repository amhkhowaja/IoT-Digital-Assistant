"""
Data loading and format conversion utilities.
Reads the source CSV and produces JSON, YAML, and intent-CSV outputs.
"""

import os
import json
import yaml
import pandas as pd


def import_dataframe(file_name: str) -> pd.DataFrame:
    """Load CSV and clean up columns for downstream use."""
    df = pd.read_csv(file_name)
    df["Questions"] = df["Questions"].map(
        lambda x: "".join([a for a in x if not (a == '(' or a == ')')])
    )
    df["Sub-Entities"] = df["Sub-Entities"].map(
        lambda x: x.strip("]['").split("', '"))
    df["Sub-Entities"] = df["Sub-Entities"].map(
        lambda x: [a.strip() for a in x])
    df["Main_Entities"] = df["Main_Entities"].map(
        lambda x: x.strip("]['").split("', '"))
    df["Main_Entities"] = df["Main_Entities"].map(
        lambda x: [a.strip() for a in x])
    df = df[["Questions", "Intent", "Sub-Entities", "Main_Entities"]]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def _add_entities(question, sub, ent):
    """Replace entity values in question with [value](entity) annotation."""
    count = 0
    for i, j in enumerate(sub):
        if j in question.lower():
            question = question.lower().replace(
                j, "[{sube}]({enti})".format(sube=j, enti=ent[i])
            )
            count = 1
    return question, count


def entities_into_questions(df: pd.DataFrame):
    """Annotate questions with entity markup for YAML generation."""
    que = []
    count = 0
    for _, row in df.iterrows():
        questions = row["Questions"]
        sub = row["Sub-Entities"]
        ents = row["Main_Entities"]
        add = _add_entities(questions, sub, ents)
        count += add[1]
        que.append(add[0])
    return que, count


def generate_json(df: pd.DataFrame, output_path: str) -> str:
    """Generate structured JSON grouped by intent and entities."""
    data = {}
    for _, entry in df.iterrows():
        intent = entry["Intent"]
        entities = entry["Sub-Entities"]
        question = entry["Questions"]

        if intent not in data:
            data[intent] = {"entities": {}}
        for e in entities:
            if e in data[intent]["entities"]:
                data[intent]["entities"][e].append(question)
            else:
                data[intent]["entities"][e] = [question]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    return output_path


def generate_yaml(df: pd.DataFrame, output_path: str) -> str:
    """Generate Rasa-compatible NLU YAML with entity annotations."""
    df = df.copy()
    df.insert(4, "Questions_with_entities", entities_into_questions(df)[0])

    yamls = {}
    for _, entry in df.iterrows():
        intent = entry["Intent"]
        if intent not in yamls:
            yamls[intent] = [entry["Questions_with_entities"]]
        else:
            yamls[intent].append(entry["Questions_with_entities"])

    nlus = [{"intent": intent, "examples": examples} for intent, examples in yamls.items()]
    output = {"nlu": nlus}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode="w") as f:
        yaml.dump(output, f, indent=1, sort_keys=False, default_flow_style=False)
    return output_path


def generate_intent_csv(df: pd.DataFrame, output_path: str) -> str:
    """Generate simple question-intent CSV for classification training."""
    fl = df[["Questions", "Intent"]]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fl.to_csv(output_path, index=False)
    return output_path


def run(data_dir: str, output_dir: str) -> dict:
    """
    Full data generation step.
    Loads source CSV, produces JSON + YAML + intent CSV.
    Returns paths to generated files.
    """
    source_csv = os.path.join(data_dir, "final_data.csv")
    df = import_dataframe(source_csv)

    json_path = generate_json(df, os.path.join(output_dir, "data.json"))
    yaml_path = generate_yaml(df, os.path.join(output_dir, "data.yaml"))
    csv_path = generate_intent_csv(df, os.path.join(output_dir, "ques_int.csv"))

    return {
        "source_csv": source_csv,
        "json_path": json_path,
        "yaml_path": yaml_path,
        "csv_path": csv_path,
        "num_samples": len(df),
        "num_intents": df["Intent"].nunique(),
    }
