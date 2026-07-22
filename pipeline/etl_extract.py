"""
ETL Extract: Pull unprocessed predictions from MongoDB.
"""

import os
from pymongo import MongoClient


def get_db():
    uri = os.environ.get("MONGODB_URI", "mongodb://mongodb:27017")
    db_name = os.environ.get("MONGODB_DB", "IOTA")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client[db_name]


def extract_unprocessed(limit=500):
    """Pull predictions that haven't been processed yet."""
    db = get_db()
    cursor = db["predictions"].find(
        {"processed": False},
        {"_id": 1, "text": 1, "sender_id": 1, "timestamp": 1, "rasa_prediction": 1}
    ).sort("timestamp", 1).limit(limit)

    records = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string for serialization
        if "timestamp" in doc:
            doc["timestamp"] = str(doc["timestamp"])
        records.append(doc)
    return records


def mark_processed(record_ids):
    """Mark records as processed after ETL completes."""
    from bson import ObjectId as BsonObjectId
    db = get_db()
    object_ids = [BsonObjectId(rid) for rid in record_ids]
    db["predictions"].update_many(
        {"_id": {"$in": object_ids}},
        {"$set": {"processed": True}}
    )


def save_review_items(items):
    """Save low-confidence or disagreement items for human review."""
    if not items:
        return
    db = get_db()
    db["reviewed"].insert_many(items)


def run(**kwargs):
    """Extract step entry point."""
    records = extract_unprocessed()
    return {
        "records": records,
        "count": len(records),
    }
