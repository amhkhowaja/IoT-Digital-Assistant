# Feature Study: Self-Learning NLU Pipeline

> **Date:** 2026-07-22  
> **Status:** Implementation  
> **Goal:** Bot automatically learns from user interactions via a dual-model validation pipeline

---

## Concept

Every user message gets predicted by Rasa (live) and then re-checked by custom models (batch). When both agree with high confidence, the example is automatically added to training data. When they disagree, it's flagged for review.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        LIVE (per message)                          │
│                                                                    │
│  User Message → Rasa NLU (intent + entities + confidence)         │
│       │                                                            │
│       ▼                                                            │
│  Custom Action: save prediction to MongoDB "predictions" collection│
│       │                                                            │
│       ▼                                                            │
│  Normal bot response to user                                       │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     BATCH (ETL DAG — runs every 6h)               │
│                                                                    │
│  1. EXTRACT                                                        │
│     └── Pull new predictions from MongoDB (since last run)         │
│                                                                    │
│  2. RE-PREDICT                                                     │
│     └── Run custom models on same text:                            │
│         • BiLSTM/CNN → intent prediction + confidence              │
│         • spaCy NER → entity extraction                            │
│                                                                    │
│  3. COMPARE & FILTER                                               │
│     ├── Rasa intent == Custom intent AND both confidence > 0.85    │
│     │   └── → AUTO-ACCEPT                                          │
│     ├── Rasa intent == Custom intent AND one confidence < 0.85     │
│     │   └── → REVIEW (low confidence)                              │
│     └── Rasa intent != Custom intent                               │
│         └── → REVIEW (disagreement)                                │
│                                                                    │
│  4. AUGMENT (auto-accepted only)                                   │
│     ├── Entity substitution (swap MSISDNs, plan names, etc.)       │
│     └── BERT contextual word replacement                           │
│                                                                    │
│  5. TRANSFORM                                                      │
│     └── Convert to Rasa NLU YAML format:                           │
│         - intent: fetch                                            │
│           examples: |                                              │
│             - show [connectivity](network_connectivity) of [123](msisdn)│
│                                                                    │
│  6. LOAD                                                           │
│     └── Append to data/nlu_generated.yml                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## MongoDB Collections

### `predictions` (written by custom action, read by ETL)

```json
{
  "_id": "ObjectId",
  "text": "show me connectivity of msisdn 55512349876",
  "sender_id": "user_001",
  "timestamp": "2026-07-22T01:30:00Z",
  "rasa_prediction": {
    "intent": "fetch",
    "intent_confidence": 0.94,
    "entities": [
      {"entity": "msisdn", "value": "55512349876", "start": 31, "end": 42, "confidence": 0.97}
    ]
  },
  "processed": false
}
```

### `reviewed` (ETL writes flagged examples here)

```json
{
  "_id": "ObjectId",
  "text": "why is my sim not working",
  "rasa_prediction": {"intent": "Troubleshoot", "confidence": 0.72},
  "custom_prediction": {"intent": "information_seek", "confidence": 0.68},
  "reason": "disagreement",
  "status": "pending",
  "created_at": "2026-07-22T02:00:00Z"
}
```

---

## Components to Build

### 1. Custom Action — Prediction Logger

**File:** `actions/actions.py` (add new action)

Runs after every user message. Saves Rasa's prediction to MongoDB.

```python
class ActionLogPrediction(Action):
    def name(self):
        return "action_log_prediction"

    def run(self, dispatcher, tracker, domain):
        prediction = tracker.latest_message
        db = get_db()
        db["predictions"].insert_one({
            "text": prediction["text"],
            "sender_id": tracker.sender_id,
            "timestamp": datetime.utcnow(),
            "rasa_prediction": {
                "intent": prediction["intent"]["name"],
                "intent_confidence": prediction["intent"]["confidence"],
                "entities": prediction["entities"],
            },
            "processed": False
        })
        return []
```

**Trigger:** Add as first action in every story via a rule.

### 2. ETL DAG — `nlu_self_learning`

**File:** `dags/nlu_self_learning.py`

| Task | Input | Output |
|------|-------|--------|
| `extract_new_predictions` | MongoDB `predictions` (processed=false) | List of unprocessed messages |
| `predict_with_custom_models` | Raw texts | Custom model predictions (intent + entities) |
| `compare_and_filter` | Rasa predictions + Custom predictions | accepted[] + review[] |
| `save_review_items` | review[] | Write to MongoDB `reviewed` collection |
| `augment_accepted` | accepted[] | Augmented examples |
| `write_rasa_yaml` | Augmented + original accepted | Append to `data/nlu_generated.yml` |
| `mark_processed` | All record IDs | Update MongoDB `predictions.processed = true` |

### 3. Pipeline Modules

**File:** `pipeline/etl_extract.py`
- `extract_unprocessed(db) → List[dict]`

**File:** `pipeline/etl_predict.py`
- `predict_intents(texts, model_path) → List[dict]`
- `predict_entities(texts, ner_model_path) → List[dict]`

**File:** `pipeline/etl_compare.py`
- `compare_predictions(rasa_preds, custom_preds, threshold=0.85) → (accepted, review)`

**File:** `pipeline/etl_augment.py`
- `augment_examples(examples, entity_values, use_bert=True) → List[dict]`

**File:** `pipeline/etl_transform.py`
- `to_rasa_yaml(examples) → str` (Rasa NLU YAML format)
- `write_nlu_file(yaml_str, output_path)`

---

## Confidence Filter Logic

```
BOTH agree on intent:
  ├── Both confidence > 0.85 → AUTO-ACCEPT
  ├── One confidence 0.70-0.85 → ACCEPT with flag "low_confidence"
  └── One confidence < 0.70 → REVIEW

Models DISAGREE on intent:
  └── Always → REVIEW (regardless of confidence)
```

---

## Augmentation Strategy

Only applied to **auto-accepted** examples:

| Technique | Library | When to apply |
|-----------|---------|---------------|
| Entity substitution | Custom | Always — swap entity values from lookup tables |
| BERT contextual | `nlpaug` (bert-base-uncased) | Only for intents with < 50 existing examples |

**Entity lookup tables:**

```python
ENTITY_VALUES = {
    "msisdn": [generate 50 random 11-digit numbers],
    "imsi": [generate 50 random 15-digit numbers],
    "plan_name": ["10 GB Europe", "20 GB Global", "5 GB Local", "Unlimited", "2 GB Test Kit"],
    "connectivity_lock": ["locked", "unlocked"],
    "billing_state": ["active", "inactive", "suspended"],
    "network_connectivity": ["connected", "disconnected", "enabled", "disabled"],
}
```

---

## Rasa Rule (trigger prediction logging)

```yaml
# rules.yml — add this
- rule: Log every prediction
  steps:
    - action: action_log_prediction
```

Wait — this won't work as a rule (rules need an intent trigger). Instead, use a **custom action in the pipeline** that runs at the start of every action sequence. Better approach:

**Option:** Add logging inside the existing actions (at the top of each `run()` method) — or create a lightweight middleware.

Simplest: add to `get_db()` area as a utility called from each action.

---

## Output Format

`data/nlu_generated.yml`:

```yaml
nlu:
- intent: fetch
  examples: |
    - show me [connectivity](network_connectivity) of [44770090012](msisdn)
    - what is the [billing state](billing_state) of [55512349876](msisdn)
    - get [plan name](plan_name) for msisdn [99887766554](msisdn)

- intent: Troubleshoot
  examples: |
    - why is [connectivity](network_connectivity) [disconnected](network_connectivity) for [12345678901](msisdn)
    - my device is showing [locked](connectivity_lock) status
```

---

## DAG Schedule

| DAG | Schedule | Trigger |
|-----|----------|---------|
| `nlu_self_learning` | Every 6 hours | Automatic + manual via `make pipeline-trigger-etl` |

---

## Success Criteria

- [ ] Every user message prediction is logged to MongoDB
- [ ] ETL DAG extracts, re-predicts, compares, filters
- [ ] Agreed predictions are augmented and written to `nlu_generated.yml`
- [ ] Disagreements saved to `reviewed` collection
- [ ] `rasa train` picks up `nlu_generated.yml` automatically
- [ ] No manual annotation needed for high-confidence examples

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Echo chamber (model validates itself) | Two different model architectures must agree |
| Bad augmentation corrupts data | Only augment accepted examples, entity substitution uses valid values only |
| nlu_generated.yml grows unbounded | Cap at 200 examples per intent, oldest get rotated out |
| BERT augmentation changes meaning | Exclude entity tokens from BERT replacement |

---

## Implementation Order

1. Prediction logger action + rule wiring
2. ETL extract + predict modules
3. Compare & filter logic
4. Augmentation (entity sub + BERT)
5. Transform to Rasa YAML
6. Airflow DAG wiring
7. Makefile target
