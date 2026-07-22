"""
ETL Compare: Compare Rasa predictions vs custom model predictions.
Filter into accepted (high confidence, agreement) and review (disagreement or low confidence).
"""

from datetime import datetime


def compare_predictions(records, custom_predictions, threshold_high=0.85, threshold_low=0.70):
    """
    Compare Rasa and custom model predictions.

    Returns:
        accepted: list of examples where both models agree with high confidence
        review: list of examples that need human review
    """
    accepted = []
    review = []

    for record, custom in zip(records, custom_predictions):
        rasa_intent = record["rasa_prediction"]["intent"]
        rasa_confidence = record["rasa_prediction"]["intent_confidence"]
        custom_intent = custom["custom_intent"]
        custom_confidence = custom["custom_intent_confidence"]

        # Use Rasa's entities (they have character offsets needed for YAML)
        entities = record["rasa_prediction"].get("entities", [])

        item = {
            "text": record["text"],
            "rasa_intent": rasa_intent,
            "rasa_confidence": rasa_confidence,
            "custom_intent": custom_intent,
            "custom_confidence": custom_confidence,
            "entities": entities,
            "record_id": record["_id"],
        }

        # Both models agree on intent
        if rasa_intent == custom_intent:
            if rasa_confidence >= threshold_high and custom_confidence >= threshold_high:
                item["status"] = "auto_accept"
                accepted.append(item)
            elif rasa_confidence >= threshold_low or custom_confidence >= threshold_low:
                item["status"] = "low_confidence"
                item["reason"] = "Agreement but low confidence"
                review.append(item)
            else:
                item["status"] = "very_low_confidence"
                item["reason"] = f"Both below {threshold_low}"
                review.append(item)
        else:
            # Models disagree
            item["status"] = "disagreement"
            item["reason"] = f"Rasa: {rasa_intent} ({rasa_confidence:.2f}) vs Custom: {custom_intent} ({custom_confidence:.2f})"
            review.append(item)

    return accepted, review


def format_review_items(review_items):
    """Format review items for MongoDB storage."""
    formatted = []
    for item in review_items:
        formatted.append({
            "text": item["text"],
            "predicted_by_rasa": item["rasa_intent"],
            "predicted_by_custom": item["custom_intent"],
            "rasa_confidence": item["rasa_confidence"],
            "custom_confidence": item["custom_confidence"],
            "reason": item["reason"],
            "status": "pending",
            "created_at": datetime.utcnow(),
        })
    return formatted


def run(records, custom_predictions, **kwargs):
    """Compare step entry point."""
    accepted, review = compare_predictions(records, custom_predictions)
    return {
        "accepted": accepted,
        "review": review,
        "num_accepted": len(accepted),
        "num_review": len(review),
    }
