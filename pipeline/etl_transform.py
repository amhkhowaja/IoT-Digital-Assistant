"""
ETL Transform: Convert predicted/augmented examples to Rasa NLU YAML format.
"""

import os
from collections import defaultdict

import yaml


def annotate_text_with_entities(text, entities):
    """
    Convert text + entity list into Rasa annotation format.
    Example: "show connectivity of 12345678901"
          → "show connectivity of [12345678901](msisdn)"
    """
    if not entities:
        return text

    # Sort entities by start position (reverse) to avoid offset issues
    sorted_ents = sorted(entities, key=lambda e: e.get("start", 0), reverse=True)

    annotated = text
    for ent in sorted_ents:
        start = ent.get("start")
        end = ent.get("end")
        entity_type = ent.get("entity", "")
        value = ent.get("value", "")

        if start is not None and end is not None and start < end and end <= len(annotated):
            # Replace the text span with [value](entity) format
            original_span = annotated[start:end]
            annotated = annotated[:start] + f"[{original_span}]({entity_type})" + annotated[end:]
        elif value and entity_type:
            # Fallback: try to find the value in text and annotate it
            if value in annotated:
                annotated = annotated.replace(value, f"[{value}]({entity_type})", 1)

    return annotated


def group_by_intent(examples):
    """Group examples by intent for YAML structure."""
    grouped = defaultdict(list)
    for ex in examples:
        intent = ex.get("intent", "unknown")
        text = ex.get("text", "")
        entities = ex.get("entities", [])
        annotated = annotate_text_with_entities(text, entities)
        grouped[intent].append(annotated)
    return grouped


def to_rasa_yaml(examples):
    """
    Convert a list of examples to Rasa NLU YAML string.
    Output format:
      nlu:
      - intent: fetch
        examples: |
          - show [connectivity](network_connectivity) of [12345678901](msisdn)
          - what is the [billing state](billing_state)
    """
    grouped = group_by_intent(examples)

    lines = ["version: \"3.1\"", "nlu:"]
    for intent in sorted(grouped.keys()):
        lines.append(f"- intent: {intent}")
        lines.append("  examples: |")
        for example in grouped[intent]:
            lines.append(f"    - {example}")
        lines.append("")

    return "\n".join(lines)


def write_nlu_file(yaml_content, output_path):
    """Append generated NLU examples to existing file, or create if not exists."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        # Parse existing file and merge
        with open(output_path, "r") as f:
            existing = yaml.safe_load(f) or {}

        # Parse new content
        new_data = yaml.safe_load(yaml_content) or {}

        # Merge: combine examples per intent
        existing_intents = {}
        for item in existing.get("nlu", []):
            intent = item.get("intent", "")
            examples = item.get("examples", "").strip().split("\n")
            existing_intents[intent] = set(e.strip() for e in examples if e.strip())

        for item in new_data.get("nlu", []):
            intent = item.get("intent", "")
            examples = item.get("examples", "").strip().split("\n")
            if intent not in existing_intents:
                existing_intents[intent] = set()
            for ex in examples:
                ex = ex.strip()
                if ex:
                    existing_intents[intent].add(ex)

        # Rebuild YAML
        lines = ['version: "3.1"', "nlu:"]
        for intent in sorted(existing_intents.keys()):
            lines.append(f"- intent: {intent}")
            lines.append("  examples: |")
            for example in sorted(existing_intents[intent]):
                lines.append(f"    {example}")
            lines.append("")

        merged_content = "\n".join(lines)
        with open(output_path, "w") as f:
            f.write(merged_content)
    else:
        with open(output_path, "w") as f:
            f.write(yaml_content)

    return output_path


def run(accepted_items, augmented_items, output_path, **kwargs):
    """
    Transform step entry point.
    Combines original accepted + augmented examples and writes Rasa YAML.
    """
    # Combine original accepted examples
    all_examples = []
    for item in accepted_items:
        all_examples.append({
            "text": item["text"],
            "intent": item["rasa_intent"],
            "entities": item.get("entities", []),
        })

    # Add augmented examples
    all_examples.extend(augmented_items)

    # Generate YAML
    yaml_content = to_rasa_yaml(all_examples)

    # Write to file
    written_path = write_nlu_file(yaml_content, output_path)

    return {
        "output_path": written_path,
        "total_examples": len(all_examples),
        "intents": len(group_by_intent(all_examples)),
    }
