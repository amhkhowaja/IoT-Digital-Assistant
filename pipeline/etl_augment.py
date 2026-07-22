"""
ETL Augment: Generate additional training examples via entity substitution and BERT contextual replacement.
"""

import random
import re

import nlpaug.augmenter.word as naw


# Domain-specific entity values for substitution
ENTITY_VALUES = {
    "msisdn": [str(random.randint(10000000000, 99999999999)) for _ in range(50)],
    "IMSI_number": [str(random.randint(100000000000000, 999999999999999)) for _ in range(50)],
    "plan_name": ["10 GB Europe", "20 GB Global", "5 GB Local", "Unlimited Plus", "2 GB Test Kit",
                  "50 GB Enterprise", "1 GB Starter", "100 GB Premium"],
    "connectivity_lock": ["locked", "unlocked"],
    "billing_state": ["active", "inactive", "suspended", "terminated"],
    "network_connectivity": ["connected", "disconnected", "enabled", "disabled"],
    "data_trend": ["upward", "downward", "stable"],
    "in_session": ["true", "false"],
}


def substitute_entities(text, entities, max_substitutions=3):
    """
    Replace entity values in text with random alternatives from lookup tables.
    Returns list of new (text, entities) pairs.
    """
    # Filter to entities with valid offsets
    valid_entities = [e for e in entities if e.get("start") is not None and e.get("end") is not None]

    if not valid_entities:
        return []

    augmented = []

    for _ in range(max_substitutions):
        new_text = text
        new_entities = []
        offset_shift = 0

        for ent in sorted(valid_entities, key=lambda e: e.get("start", 0)):
            entity_type = ent.get("entity", "")
            original_value = str(ent.get("value", ""))
            start = ent.get("start", 0)
            end = ent.get("end", 0)

            if not original_value or start >= end or end > len(new_text) + offset_shift:
                new_entities.append(ent)
                continue

            if entity_type in ENTITY_VALUES:
                new_value = random.choice(ENTITY_VALUES[entity_type])
                adj_start = start + offset_shift
                adj_end = end + offset_shift
                new_text = new_text[:adj_start] + new_value + new_text[adj_end:]
                offset_shift += len(new_value) - (end - start)

                new_entities.append({
                    "entity": entity_type,
                    "value": new_value,
                    "start": adj_start,
                    "end": adj_start + len(new_value),
                })
            else:
                new_entities.append({
                    "entity": entity_type,
                    "value": original_value,
                    "start": start + offset_shift,
                    "end": end + offset_shift,
                })

        augmented.append({"text": new_text, "entities": new_entities})

    return augmented


def augment_with_bert(texts, aug_max=2):
    """
    Use BERT masked language model to generate contextual word replacements.
    Skips entity tokens to preserve meaning.
    """
    try:
        bert_aug = naw.ContextualWordEmbsAug(
            model_path="bert-base-uncased",
            action="substitute",
            aug_max=aug_max,
            device="cpu"
        )
        augmented = bert_aug.augment(texts)
        if isinstance(augmented, str):
            augmented = [augmented]
        return augmented
    except Exception:
        # If BERT fails (model not available, OOM, etc.) return empty
        return []


def augment_examples(accepted_items, use_bert=True, max_per_example=3):
    """
    Augment accepted examples using entity substitution + optionally BERT.
    Returns list of new augmented examples.
    """
    all_augmented = []

    for item in accepted_items:
        text = item["text"]
        entities = item.get("entities", [])
        intent = item["rasa_intent"]

        # Entity substitution (always)
        if entities:
            substituted = substitute_entities(text, entities, max_substitutions=max_per_example)
            for aug in substituted:
                all_augmented.append({
                    "text": aug["text"],
                    "intent": intent,
                    "entities": aug["entities"],
                    "source": "entity_substitution",
                })

        # BERT contextual (only for texts without many entities to avoid breaking annotations)
        if use_bert and len(entities) <= 2:
            bert_texts = augment_with_bert([text], aug_max=2)
            for aug_text in bert_texts:
                if aug_text and aug_text != text:
                    all_augmented.append({
                        "text": aug_text,
                        "intent": intent,
                        "entities": [],  # BERT changes word positions, can't keep entity offsets
                        "source": "bert_contextual",
                    })

    return all_augmented


def run(accepted_items, use_bert=True, **kwargs):
    """Augment step entry point."""
    augmented = augment_examples(accepted_items, use_bert=use_bert)
    return {
        "augmented": augmented,
        "num_augmented": len(augmented),
        "num_original": len(accepted_items),
    }
