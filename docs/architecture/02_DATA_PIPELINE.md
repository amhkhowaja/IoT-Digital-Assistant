# Data Pipeline Architecture — Apache Airflow ETL

> **Version:** 2.0  
> **Date:** 2026-07-21  
> **Component:** `iot-data-pipeline`  
> **Orchestrator:** Apache Airflow 2.8+

---

## 1. Pipeline Overview

### 1.1 Purpose

The data pipeline transforms raw conversational data into production-ready NLU training datasets through a series of automated, auditable stages. It handles data ingestion, quality validation, PII masking, annotation, augmentation, model training, evaluation, and deployment — all orchestrated by Apache Airflow.

### 1.2 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        APACHE AIRFLOW ORCHESTRATOR                        │
│                                                                           │
│  ┌──────┐   ┌──────────┐   ┌─────────┐   ┌────────────┐   ┌─────────┐ │
│  │INGEST│──▶│VALIDATION│──▶│CLEANUP  │──▶│ANNOTATION  │──▶│AUGMENT  │ │
│  └──────┘   └──────────┘   └─────────┘   └────────────┘   └─────────┘ │
│                                                                    │     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐     │     │
│  │DEPLOYMENT│◀──│EVALUATION│◀──│ TRAINING │◀──│SPLIT/PREP │◀────┘     │
│  └──────────┘   └──────────┘   └──────────┘   └───────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
         │                              │                    │
         ▼                              ▼                    ▼
┌─────────────────┐          ┌──────────────────┐   ┌──────────────┐
│ Model Registry  │          │ MinIO (Artifacts)│   │ MongoDB      │
│ (MLflow/MinIO)  │          │ Raw/Processed    │   │ (Metadata)   │
└─────────────────┘          └──────────────────┘   └──────────────┘
```

### 1.3 DAG Summary

| DAG | Schedule | Purpose | SLA |
|-----|----------|---------|-----|
| `nlu_data_ingestion` | Every 6h | Collect new conversation data | 30 min |
| `nlu_data_processing` | Daily 02:00 | Validate → Clean → Mask → Annotate | 2h |
| `nlu_augmentation` | Daily 04:00 | Data augmentation + balancing | 1h |
| `nlu_model_training` | Weekly Sun 06:00 | Train + evaluate + register model | 4h |
| `nlu_model_promotion` | On-demand | Promote model to production | 15 min |
| `data_quality_report` | Daily 08:00 | Generate data quality dashboards | 30 min |

---

## 2. Stage 1: Data Ingestion

### 2.1 Sources

| Source | Format | Frequency | Volume |
|--------|--------|-----------|--------|
| Rasa Tracker Store (MongoDB) | JSON conversations | Real-time → batch | ~10K msgs/day |
| Manual Upload (CSV/YAML) | Rasa NLU YAML / CSV | On-demand | Variable |
| Feedback Loop (user corrections) | JSON events | Real-time | ~500/day |
| External Corpus (scraped IoT docs) | Text/HTML | Weekly | ~1K docs |

### 2.2 Ingestion DAG Tasks

```python
# DAG: nlu_data_ingestion
@dag(schedule="0 */6 * * *", catchup=False, tags=["ingestion"])
def nlu_data_ingestion():

    @task
    def extract_conversations():
        """Pull new conversations from Rasa tracker store since last run."""
        # MongoDB query: conversations modified since last_successful_run
        # Output: raw JSON to MinIO staging bucket

    @task
    def extract_feedback():
        """Pull user correction events from feedback queue."""
        # RabbitMQ consumer: iot.feedback.corrections queue
        # Output: correction records to MinIO staging

    @task
    def extract_uploads():
        """Check upload bucket for new manual data files."""
        # MinIO: iot-uploads/ prefix scan
        # Validate file format (YAML/CSV/JSON)

    @task
    def merge_sources(conversations, feedback, uploads):
        """Merge all sources into unified raw format."""
        # Deduplicate, assign batch_id, write to raw_training_data collection
        # Output: batch_id for downstream processing

    @task
    def notify_ingestion_complete(batch_id):
        """Publish event for downstream DAGs."""
        # RabbitMQ publish: iot.pipeline.triggers
```

### 2.3 Raw Data Schema

```json
{
  "batch_id": "batch_20260721_060000",
  "source": "tracker_store | feedback | manual_upload",
  "timestamp": "2026-07-21T06:00:00Z",
  "records": [
    {
      "record_id": "uuid",
      "text": "what is the connectivity status of msisdn 12345678901",
      "metadata": {
        "sender_id": "hashed_user_id",
        "channel": "rest",
        "conversation_id": "conv_uuid",
        "timestamp": "2026-07-21T05:30:00Z"
      },
      "existing_labels": {
        "intent": "fetch",
        "entities": [{"entity": "msisdn", "value": "12345678901", "start": 43, "end": 54}]
      },
      "confidence": 0.92
    }
  ]
}
```

---

## 3. Stage 2: Data Validation

### 3.1 Validation Rules

| Rule | Check | Action on Failure |
|------|-------|-------------------|
| Schema validation | JSON conforms to expected structure | Reject + alert |
| Language detection | Text is English (langdetect) | Quarantine |
| Min length | Text ≥ 3 tokens | Skip (too short) |
| Max length | Text ≤ 200 tokens | Truncate + flag |
| Encoding check | Valid UTF-8, no control chars | Clean or reject |
| Duplicate detection | MinHash/SimHash against existing data | Flag duplicates |
| Toxicity filter | No hate speech/profanity (detoxify) | Quarantine |
| Bot-generated filter | Exclude bot responses, system messages | Skip |

### 3.2 Validation Implementation

```python
@task
def validate_batch(batch_id: str) -> dict:
    """Run all validation rules on ingested batch."""
    from pipeline.validators import (
        SchemaValidator, LanguageValidator, LengthValidator,
        EncodingValidator, DuplicateDetector, ToxicityFilter
    )

    validators = [
        SchemaValidator(),
        LanguageValidator(expected="en", threshold=0.8),
        LengthValidator(min_tokens=3, max_tokens=200),
        EncodingValidator(),
        DuplicateDetector(method="minhash", threshold=0.85),
        ToxicityFilter(model="detoxify-base", threshold=0.7),
    ]

    results = {"passed": [], "failed": [], "quarantined": []}
    for record in get_batch_records(batch_id):
        status = "passed"
        for validator in validators:
            result = validator.validate(record)
            if result.status == "reject":
                status = "failed"
                break
            elif result.status == "quarantine":
                status = "quarantined"
                break
        results[status].append(record)

    # Store results + generate quality report
    store_validation_results(batch_id, results)
    return {
        "batch_id": batch_id,
        "total": len(results["passed"]) + len(results["failed"]) + len(results["quarantined"]),
        "passed": len(results["passed"]),
        "failed": len(results["failed"]),
        "quarantined": len(results["quarantined"]),
    }
```

### 3.3 Quality Metrics Tracked

- Pass rate per batch (target: >90%)
- Failure reasons distribution
- Duplicate rate trend
- Language distribution
- Token length histogram

---

## 4. Stage 3: Industrial Cleanup

### 4.1 Cleanup Operations

| Operation | Library | Description |
|-----------|---------|-------------|
| Unicode normalization | `unicodedata` (NFKC) | Normalize unicode chars to canonical form |
| Whitespace normalization | `regex` | Collapse multiple spaces, trim, fix newlines |
| Spelling correction | `symspellpy` / `textblob` | Fix common typos (with IoT domain dictionary) |
| Contraction expansion | Custom dict | "what's" → "what is", "can't" → "cannot" |
| Number normalization | Custom rules | Standardize phone/IMSI number formats |
| HTML/Markup stripping | `beautifulsoup4` | Remove any HTML/markdown from text |
| Special char cleanup | `regex` | Remove invisible chars, zero-width spaces |
| Lowercasing (optional) | Built-in | Configurable per pipeline run |
| Sentence boundary detection | `spacy` | Split multi-sentence inputs for annotation |

### 4.2 Cleanup Pipeline

```python
@task
def cleanup_batch(batch_id: str, config: dict) -> str:
    """Apply industrial text cleanup pipeline."""
    from pipeline.cleanup import (
        normalize_unicode, normalize_whitespace, expand_contractions,
        correct_spelling, normalize_numbers, strip_markup,
        remove_special_chars, detect_sentences
    )

    # Configurable pipeline (order matters)
    cleanup_steps = [
        strip_markup,
        normalize_unicode,
        remove_special_chars,
        normalize_whitespace,
        expand_contractions,
        normalize_numbers,          # IMSI: 15 digits, MSISDN: 10-15 digits
        correct_spelling,           # Uses IoT domain dictionary
    ]

    # Optional steps based on config
    if config.get("lowercase", False):
        cleanup_steps.append(str.lower)
    if config.get("sentence_split", True):
        cleanup_steps.append(detect_sentences)

    records = get_validated_records(batch_id)
    cleaned = []
    for record in records:
        original_text = record["text"]
        cleaned_text = original_text
        changes = []

        for step in cleanup_steps:
            result = step(cleaned_text)
            if result != cleaned_text:
                changes.append({"step": step.__name__, "before": cleaned_text, "after": result})
                cleaned_text = result

        cleaned.append({
            **record,
            "text": cleaned_text,
            "original_text": original_text,
            "cleanup_changes": changes,
            "cleanup_version": "1.0.0"
        })

    store_cleaned_records(batch_id, cleaned)
    return batch_id
```

### 4.3 IoT Domain Dictionary

```yaml
# domain_dictionary.yml — custom terms that should NOT be spell-corrected
preserve_terms:
  - msisdn
  - imsi
  - iota
  - cpi
  - sim
  - puk
  - apn
  - lte
  - 5g
  - mqtt
  - coap
  - nbiot
  - lorawan
  - zigbee

number_patterns:
  imsi: '\d{15}'
  msisdn: '\d{10,15}'
  agreement_number: '\d{14}'
  ip_address: '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
```

---

## 5. Stage 4: PII Masking

### 5.1 PII Categories & Detection

| PII Type | Detection Method | Mask Pattern | Example |
|----------|-----------------|--------------|---------|
| Phone/MSISDN | Regex + context | `[MSISDN_XXX]` | 12345678901 → [MSISDN_001] |
| IMSI | Regex (15 digits) | `[IMSI_XXX]` | 234150999999999 → [IMSI_001] |
| Email | Regex | `[EMAIL_XXX]` | user@example.com → [EMAIL_001] |
| IP Address | Regex | `[IP_XXX]` | 192.168.1.1 → [IP_001] |
| Person Name | spaCy NER (PERSON) | `[PERSON_XXX]` | John Smith → [PERSON_001] |
| Location | spaCy NER (GPE/LOC) | `[LOCATION_XXX]` | Budapest → [LOCATION_001] |
| Agreement Number | Regex (14 digits) | `[AGREEMENT_XXX]` | 12345678901234 → [AGREEMENT_001] |
| PIN/PUK | Context + regex | `[PIN_XXX]` | PIN: 1234 → PIN: [PIN_001] |
| Organization | spaCy NER (ORG) | `[ORG_XXX]` | China Telecom → [ORG_001] |

### 5.2 PII Masking Implementation

```python
@task
def mask_pii(batch_id: str) -> str:
    """Apply PII masking while preserving entity annotation alignment."""
    import spacy
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    # Custom recognizers for IoT domain
    imsi_pattern = Pattern(name="imsi", regex=r"\b\d{15}\b", score=0.9)
    msisdn_pattern = Pattern(name="msisdn", regex=r"\b\d{10,15}\b", score=0.7)
    agreement_pattern = Pattern(name="agreement", regex=r"\b\d{14}\b", score=0.85)

    imsi_recognizer = PatternRecognizer(
        supported_entity="IMSI", patterns=[imsi_pattern]
    )
    msisdn_recognizer = PatternRecognizer(
        supported_entity="MSISDN", patterns=[msisdn_pattern]
    )

    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(imsi_recognizer)
    analyzer.registry.add_recognizer(msisdn_recognizer)

    anonymizer = AnonymizerEngine()

    records = get_cleaned_records(batch_id)
    masked_records = []
    pii_mapping = {}  # Reversible mapping for authorized access

    for record in records:
        # Analyze text for PII
        results = analyzer.analyze(
            text=record["text"],
            language="en",
            entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON",
                     "LOCATION", "IMSI", "MSISDN", "IP_ADDRESS"]
        )

        # Anonymize with consistent placeholders
        anonymized = anonymizer.anonymize(
            text=record["text"],
            analyzer_results=results,
            operators={
                "IMSI": OperatorConfig("replace", {"new_value": "[IMSI]"}),
                "MSISDN": OperatorConfig("replace", {"new_value": "[MSISDN]"}),
                "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
                "DEFAULT": OperatorConfig("replace", {"new_value": "[PII]"}),
            }
        )

        # Update entity offsets after masking
        updated_entities = realign_entities(
            original_text=record["text"],
            masked_text=anonymized.text,
            entities=record.get("existing_labels", {}).get("entities", []),
            pii_results=results
        )

        masked_records.append({
            **record,
            "text": anonymized.text,
            "text_before_masking": record["text"],  # Stored encrypted
            "entities": updated_entities,
            "pii_detected": [{"type": r.entity_type, "start": r.start, "end": r.end} for r in results],
            "masking_version": "1.0.0"
        })

    store_masked_records(batch_id, masked_records)
    store_pii_mapping(batch_id, pii_mapping, encrypted=True)
    return batch_id
```

### 5.3 PII Governance

| Control | Implementation |
|---------|---------------|
| Reversibility | Encrypted mapping stored in Vault, accessible only by `iot-admin` role |
| Audit trail | Every masking operation logged with actor + timestamp |
| Retention | Raw (unmasked) data auto-deleted after 30 days |
| Access control | Masked data: ML engineers; Unmasked: admin only via Keycloak RBAC |
| Compliance | GDPR Article 25 (data protection by design) |

---

## 6. Stage 5: Data Annotation

### 6.1 Annotation Pipeline

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌────────────┐
│ Auto-Annotate│────▶│ Confidence    │────▶│ Human Review │────▶│ Consensus  │
│ (Model-based)│     │ Filtering     │     │ (Label Studio)│    │ Merge      │
└──────────────┘     └───────────────┘     └──────────────┘     └────────────┘
```

### 6.2 Auto-Annotation

```python
@task
def auto_annotate(batch_id: str) -> str:
    """Use current production model to pre-annotate data."""
    from rasa.nlu.model import Interpreter

    interpreter = load_production_model()
    records = get_masked_records(batch_id)
    annotated = []

    for record in records:
        prediction = interpreter.parse(record["text"])

        annotated.append({
            **record,
            "auto_annotation": {
                "intent": prediction["intent"]["name"],
                "intent_confidence": prediction["intent"]["confidence"],
                "entities": prediction["entities"],
                "intent_ranking": prediction["intent_ranking"][:3],
            },
            "annotation_status": classify_confidence(prediction),
            # high_confidence: auto-accept
            # medium_confidence: flag for review
            # low_confidence: mandatory human review
        })

    store_annotated_records(batch_id, annotated)
    return batch_id

def classify_confidence(prediction: dict) -> str:
    """Classify prediction confidence for review routing."""
    intent_conf = prediction["intent"]["confidence"]
    margin = intent_conf - prediction["intent_ranking"][1]["confidence"] if len(prediction["intent_ranking"]) > 1 else intent_conf

    if intent_conf >= 0.95 and margin >= 0.3:
        return "auto_accept"      # ~60% of data
    elif intent_conf >= 0.7:
        return "review_optional"  # ~25% of data
    else:
        return "review_required"  # ~15% of data
```

### 6.3 Human-in-the-Loop (Label Studio Integration)

```python
@task.branch
def route_for_review(batch_id: str) -> str:
    """Route low-confidence samples to Label Studio for human annotation."""
    stats = get_annotation_stats(batch_id)

    if stats["review_required_count"] > 0:
        return "create_label_studio_tasks"
    else:
        return "skip_human_review"

@task
def create_label_studio_tasks(batch_id: str):
    """Push review-required records to Label Studio project."""
    from label_studio_sdk import Client

    ls_client = Client(url=Config.LABEL_STUDIO_URL, api_key=Config.LABEL_STUDIO_KEY)
    project = ls_client.get_project(Config.NLU_ANNOTATION_PROJECT_ID)

    review_records = get_records_by_status(batch_id, "review_required")

    tasks = [{
        "data": {"text": record["text"]},
        "predictions": [{
            "model_version": "current_production",
            "result": format_as_label_studio(record["auto_annotation"])
        }]
    } for record in review_records]

    project.import_tasks(tasks)
```

### 6.4 Annotation Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Inter-annotator agreement (Cohen's κ) | > 0.85 | Subset reviewed by 2+ annotators |
| Auto-annotation acceptance rate | > 60% | High-confidence auto-accepts |
| Time to annotate (per sample) | < 30s | Label Studio timing |
| Review backlog | < 500 samples | Queue size monitoring |

---

## 7. Stage 6: Data Augmentation

### 7.1 Augmentation Techniques

| Technique | Library | Purpose | Multiplier |
|-----------|---------|---------|------------|
| Synonym replacement | `nlpaug` (WordNet) | Vocabulary diversity | 2-3x |
| Back-translation | `nlpaug` (MarianMT) | Paraphrase generation | 2x |
| Random insertion | `nlpaug` | Robustness to noise | 1.5x |
| Random swap | `nlpaug` | Word order robustness | 1.5x |
| Entity substitution | Custom (lookup tables) | Entity value diversity | 3-5x |
| Contextual word embedding | `nlpaug` (BERT) | Semantic augmentation | 2x |
| Template generation | Custom Jinja2 | Pattern-based examples | Variable |
| Class balancing (SMOTE-text) | `imblearn` + custom | Fix class imbalance | Target-based |

### 7.2 Augmentation Pipeline

```python
@task
def augment_training_data(batch_id: str, config: dict) -> str:
    """Apply configurable augmentation strategies."""
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas

    # Augmentation config per intent (balance-aware)
    intent_counts = get_intent_distribution(batch_id)
    target_count = config.get("target_examples_per_intent", 100)

    augmenters = {
        "synonym": naw.SynonymAug(aug_src="wordnet", aug_max=2),
        "backtranslation": nas.BackTranslationAug(
            from_model_name="facebook/wmt19-en-de",
            to_model_name="facebook/wmt19-de-en"
        ),
        "contextual": naw.ContextualWordEmbsAug(
            model_path="bert-base-uncased", action="substitute", aug_max=2
        ),
        "random_swap": naw.RandomWordAug(action="swap", aug_max=2),
    }

    # Entity substitution with domain-specific values
    entity_substitutions = {
        "msisdn": generate_msisdn_variants(count=50),
        "imsi": generate_imsi_variants(count=50),
        "plan_name": ["10 GB Europe", "20 GB Global", "5 GB Local", "Unlimited"],
        "connectivity_lock": ["locked", "unlocked"],
        "billing_state": ["active", "inactive", "suspended"],
        "enterprise_name": load_enterprise_names(),
    }

    augmented_records = []
    for intent, count in intent_counts.items():
        if count >= target_count:
            continue  # Already sufficient

        deficit = target_count - count
        intent_records = get_records_by_intent(batch_id, intent)

        # Apply augmentation strategies
        new_samples = []
        for record in cycle(intent_records):
            if len(new_samples) >= deficit:
                break

            # Choose augmentation strategy
            strategy = select_strategy(record, augmenters)
            augmented_text = strategy.augment(record["text"])

            # Entity substitution (preserves annotation structure)
            if record.get("entities"):
                augmented_text, new_entities = substitute_entities(
                    augmented_text, record["entities"], entity_substitutions
                )
            else:
                new_entities = []

            new_samples.append({
                "text": augmented_text,
                "intent": intent,
                "entities": new_entities,
                "source": "augmentation",
                "augmentation_method": strategy.__class__.__name__,
                "original_record_id": record["record_id"],
            })

        augmented_records.extend(new_samples)

    store_augmented_records(batch_id, augmented_records)
    log_augmentation_stats(batch_id, intent_counts, augmented_records)
    return batch_id
```

### 7.3 Augmentation Quality Controls

| Control | Implementation |
|---------|---------------|
| Semantic preservation | Cosine similarity check (augmented vs original > 0.7) |
| Entity alignment | Verify entities still present and correctly positioned after augmentation |
| Deduplication | MinHash against existing + augmented data |
| Human spot-check | Random 5% sample flagged for manual review |
| Balance verification | Post-augmentation intent distribution within 2x of target |

---

## 8. Stage 7: ML Training Pipeline

### 8.1 Training DAG

```python
@dag(schedule="0 6 * * 0", catchup=False, tags=["training"])
def nlu_model_training():

    @task
    def prepare_training_data():
        """Merge validated + augmented data into Rasa NLU format."""
        # Output: training_data.yml, test_data.yml (80/20 stratified split)
        # Upload to MinIO: iot-training/batch_{date}/

    @task
    def train_model(training_data_path: str) -> dict:
        """Train Rasa NLU model with current config."""
        # rasa train nlu --config config.yml --nlu training_data.yml
        # Output: model tarball to MinIO
        # Return: model_path, training_duration, config_hash

    @task
    def evaluate_model(model_path: str, test_data_path: str) -> dict:
        """Run full evaluation suite on trained model."""
        # rasa test nlu --model model_path --nlu test_data.yml
        # Generate: intent_report, entity_report, confusion_matrix
        # Return: metrics dict

    @task.branch
    def quality_gate(metrics: dict) -> str:
        """Check if model meets promotion criteria."""
        thresholds = {
            "intent_f1_micro": 0.92,
            "entity_f1_micro": 0.88,
            "intent_f1_min_per_class": 0.75,
            "confidence_calibration": 0.85,
        }

        passed = all(
            metrics.get(k, 0) >= v for k, v in thresholds.items()
        )

        if passed:
            return "register_model"
        else:
            return "notify_failure"

    @task
    def register_model(model_path: str, metrics: dict):
        """Register model in MLflow/model registry."""
        # Tag with: version, metrics, training_data_hash, config_hash
        # State: "staging" (not yet production)

    @task
    def notify_failure(metrics: dict):
        """Alert ML team about failed quality gate."""
        # Send to Slack/email with failure details
        # Create Jira ticket for investigation
```

### 8.2 Training Configuration

```yaml
# training_config.yml — versioned alongside model
pipeline:
  - name: SpacyNLP
    model: en_core_web_lg          # Upgraded from _md to _lg
    case_sensitive: false
  - name: SpacyTokenizer
  - name: SpacyFeaturizer
  - name: RegexFeaturizer
  - name: RegexEntityExtractor     # NEW: for IMSI, MSISDN patterns
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: char_wb
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 150
    batch_strategy: balanced        # Handle class imbalance
    early_stopping: true
    patience: 10
    learning_rate: 0.001
    transformer_size: 256
    embedding_dimension: 20
    number_of_attention_heads: 4
  - name: EntitySynonymMapper
  - name: FallbackClassifier       # NEW: graceful degradation
    threshold: 0.6
    ambiguity_threshold: 0.1
  - name: ResponseSelector
    epochs: 100
    early_stopping: true
    patience: 10

policies:
  - name: MemoizationPolicy
  - name: RulePolicy
    core_fallback_threshold: 0.4
    core_fallback_action_name: action_default_fallback
  - name: UnexpecTEDIntentPolicy
    max_history: 8
    epochs: 100
  - name: TEDPolicy
    max_history: 8
    epochs: 100
    constrain_similarities: true
    split_entities_by_comma: true
```

### 8.3 Model Evaluation Metrics

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Intent F1 (micro) | ≥ 0.92 | Overall intent classification |
| Intent F1 (per-class min) | ≥ 0.75 | No intent below this |
| Entity F1 (micro) | ≥ 0.88 | Overall entity extraction |
| Confidence calibration | ≥ 0.85 | Predicted vs actual accuracy correlation |
| Fallback trigger rate | 5-15% | Not too high (unusable) or low (overconfident) |
| Response time (P95) | < 200ms | Inference latency |
| Model size | < 500MB | Deployable in container |

### 8.4 Model Registry Schema

```json
{
  "model_id": "nlu_v2.1.0_20260721",
  "version": "2.1.0",
  "stage": "staging | production | archived",
  "created_at": "2026-07-21T10:00:00Z",
  "training_data": {
    "batch_ids": ["batch_20260714", "batch_20260721"],
    "total_examples": 3200,
    "intent_count": 31,
    "entity_count": 25
  },
  "metrics": {
    "intent_f1_micro": 0.95,
    "entity_f1_micro": 0.91,
    "fallback_rate": 0.08,
    "inference_p95_ms": 145
  },
  "artifacts": {
    "model_path": "s3://iot-models/nlu_v2.1.0_20260721.tar.gz",
    "config_path": "s3://iot-models/config_v2.1.0.yml",
    "report_path": "s3://iot-models/reports/nlu_v2.1.0_20260721/"
  },
  "promoted_by": null,
  "promoted_at": null
}
```

---

## 9. Stage 8: Model Promotion & Deployment

### 9.1 Promotion Flow

```
Model Trained → Quality Gate Pass → Register (staging)
                                         │
                    ┌────────────────────┘
                    ▼
         Shadow Deployment (canary)
         ├── Route 10% traffic to new model
         ├── Compare metrics vs production model
         └── Auto-promote if metrics ≥ production + margin
                    │
                    ▼
         Full Promotion (production)
         ├── Update model reference in NLU service config
         ├── Rolling restart of NLU pods
         └── Archive previous model (keep N-2 for rollback)
```

### 9.2 Rollback Strategy

| Trigger | Action | Time to Recovery |
|---------|--------|-----------------|
| Metrics degradation > 5% | Auto-rollback to N-1 | < 2 min |
| Error rate spike > 2% | Auto-rollback to N-1 | < 2 min |
| Manual trigger | Operator selects target version | < 5 min |
| Canary failure | Cancel promotion, keep current | Immediate |

---

## 10. Airflow Infrastructure

### 10.1 Airflow Architecture

```
┌──────────────────────────────────────────────────────┐
│                 AIRFLOW CLUSTER                        │
├──────────────────────────────────────────────────────┤
│ Webserver (2 replicas) ── UI + API                   │
│ Scheduler (2 replicas) ── DAG parsing + scheduling   │
│ Workers (2-5, KEDA) ──── Task execution              │
│ Triggerer (1 replica) ── Deferred task management    │
│ PostgreSQL ─────────────  Metadata DB                │
│ Redis ──────────────────  Celery broker              │
└──────────────────────────────────────────────────────┘
```

### 10.2 Airflow Configuration

```yaml
# airflow.cfg (key settings)
executor: CeleryKubernetesExecutor   # Celery for light tasks, K8s for heavy (training)
parallelism: 16
dag_concurrency: 8
max_active_runs_per_dag: 2

# Resource pools
[pools]
data_processing: 8 slots
model_training: 2 slots    # GPU-heavy, limited
annotation: 4 slots

# Connections (managed via Vault)
mongodb_conn: mongodb+srv://...
minio_conn: s3://...
rabbitmq_conn: amqp://...
label_studio_conn: https://...
```

### 10.3 Monitoring & Alerting

| Metric | Alert Condition | Channel |
|--------|----------------|---------|
| DAG failure | Any task fails | Slack + PagerDuty |
| SLA miss | Exceeds defined SLA | Slack |
| Data quality drop | Pass rate < 80% | Email to data team |
| Training failure | Quality gate not met | Slack + Jira ticket |
| Backlog growth | > 1000 unprocessed records | Slack |

---

*Next: See [03_API_DESIGN.md](./03_API_DESIGN.md) for complete API specifications.*
