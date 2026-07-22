# System Sequence Diagrams — IoT Digital Assistant

> **Version:** 2.0  
> **Date:** 2026-07-21  
> **Format:** PlantUML  
> **Flows Covered:** 8 major system interactions

---

## 1. User Chat Message Flow (Core Conversation)

The primary flow: user sends a message → NLU → Dialogue → Action → Business Service → Response.

```plantuml
@startuml Chat_Message_Flow
!theme plain
skinparam sequenceMessageAlign center
skinparam responseMessageBelowArrow true

title User Chat Message — End-to-End Flow

actor User as user
participant "Channel\nConnector" as channel
participant "API\nGateway" as gateway
participant "Keycloak" as keycloak
participant "Dialogue\nManager" as dialogue
participant "NLU\nService" as nlu
participant "Action\nServer" as action
participant "Inventory\nService" as inventory
participant "MongoDB" as mongo
participant "Redis\nCache" as redis
participant "RabbitMQ" as mq

user -> channel: "show connectivity of msisdn 12345678901"
activate channel

channel -> gateway: POST /conversations/{id}/messages
activate gateway

gateway -> keycloak: Validate JWT
activate keycloak
keycloak --> gateway: Token valid (roles: iot-user)
deactivate keycloak

gateway -> dialogue: Forward message
activate dialogue

dialogue -> nlu: POST /parse (text)
activate nlu
nlu -> redis: Check cache (text hash)
redis --> nlu: Cache miss
nlu -> nlu: SpaCy + DIET inference
nlu -> redis: Store result (TTL: 5min)
nlu --> dialogue: {intent: "fetch", entities: [{msisdn: "12345678901"}], confidence: 0.94}
deactivate nlu

dialogue -> dialogue: Policy prediction\n(TEDPolicy + RulePolicy)
dialogue -> dialogue: Predicted action:\naction_fetch_inventory

dialogue -> action: Execute action_fetch_inventory
activate action

action -> inventory: GET /inventory/12345678901
activate inventory

inventory -> keycloak: Validate service token
keycloak --> inventory: Valid (role: iot-system)

inventory -> mongo: db.inventory.findOne({msisdn: 12345678901})
activate mongo
mongo --> inventory: {connectivity: "connected", plan: "10GB Europe"...}
deactivate mongo

inventory --> action: 200 OK {data: inventory_record}
deactivate inventory

action -> action: Format response message
action --> dialogue: [SlotSet, BotUttered]
deactivate action

dialogue -> mongo: Store tracker events
dialogue -> mq: Publish conversation.event

dialogue --> gateway: {responses: [{text: "MSISDN 12345678901 is connected..."}]}
deactivate dialogue

gateway --> channel: 200 OK
deactivate gateway

channel --> user: "MSISDN 12345678901 is connected\nPlan: 10GB Europe\nStatus: Active"
deactivate channel

@enduml
```

---

## 2. User Authentication Flow

```plantuml
@startuml Authentication_Flow
!theme plain
skinparam sequenceMessageAlign center

title User Authentication — Login + Token Refresh

actor User as user
participant "Web App\n(SPA)" as spa
participant "API\nGateway" as gateway
participant "Keycloak" as keycloak
participant "LDAP/AD" as ldap

== Initial Login (Authorization Code + PKCE) ==

user -> spa: Click "Login"
spa -> keycloak: GET /auth (redirect)\nclient_id=iot-web\nresponse_type=code\ncode_challenge=...
activate keycloak

keycloak -> user: Show login page
user -> keycloak: username + password + MFA

keycloak -> ldap: Verify credentials
activate ldap
ldap --> keycloak: Valid user + groups
deactivate ldap

keycloak -> keycloak: Generate authorization code
keycloak --> spa: Redirect with ?code=AUTH_CODE
deactivate keycloak

spa -> keycloak: POST /token\ngrant_type=authorization_code\ncode=AUTH_CODE\ncode_verifier=...
activate keycloak
keycloak --> spa: {\n  access_token (5min),\n  refresh_token (30min),\n  id_token\n}
deactivate keycloak

spa -> spa: Store tokens (memory only)
spa --> user: Logged in ✓

== API Request with Token ==

user -> spa: "Show my subscriptions"
spa -> gateway: GET /subscriptions/\nAuthorization: Bearer <access_token>
activate gateway

gateway -> gateway: Validate JWT signature\n(cached JWKS keys)
gateway -> gateway: Check claims:\n- exp not expired\n- roles contain iot-user

gateway --> spa: 200 OK {subscriptions: [...]}
deactivate gateway

== Token Refresh (before expiry) ==

spa -> spa: Access token expires in < 30s
spa -> keycloak: POST /token\ngrant_type=refresh_token\nrefresh_token=...
activate keycloak
keycloak --> spa: New access_token + refresh_token
deactivate keycloak

== Session Expiry ==

spa -> keycloak: POST /token (refresh)
activate keycloak
keycloak --> spa: 401 Refresh token expired
deactivate keycloak
spa --> user: Redirect to login

@enduml
```

---

## 3. Customer Onboarding Flow

```plantuml
@startuml Onboarding_Flow
!theme plain
skinparam sequenceMessageAlign center

title Customer Onboarding — Conversational + API

actor Admin as admin
participant "Chat\nInterface" as chat
participant "Dialogue\nManager" as dialogue
participant "Action\nServer" as action
participant "Onboarding\nService" as onboard
participant "MongoDB" as mongo
participant "RabbitMQ" as mq
participant "Notification\nService" as notif

admin -> chat: "I want to onboard a new enterprise customer"
chat -> dialogue: Parse + predict

dialogue --> chat: "What type of customer?\n[Enterprise] [Advanced Reseller]"
chat --> admin: Show buttons

admin -> chat: Clicks [Enterprise]
chat -> dialogue: "enterprise"

dialogue --> chat: "What is the enterprise name?"
chat --> admin: Ask enterprise name

admin -> chat: "IoT Solutions Ltd"
chat -> dialogue: entity: enterprise_name = "IoT Solutions Ltd"

dialogue --> chat: "Please provide the 14-digit agreement number"
chat --> admin: Ask agreement

admin -> chat: "12345678901234"
chat -> dialogue: Validate: len == 14 ✓

dialogue --> chat: "Select parent organization:\n[China Telecom Mongolia Branch]"
chat --> admin: Show options

admin -> chat: Selects parent org
chat -> dialogue: Form complete

dialogue -> action: action_submit_onboarding
activate action

action -> onboard: POST /onboarding/
activate onboard

onboard -> onboard: Validate:\n- Agreement format ✓\n- No duplicate ✓\n- Parent org exists ✓

onboard -> mongo: Insert customer record\nstatus: "pending_verification"
activate mongo
mongo --> onboard: Inserted (onboarding_id)
deactivate mongo

onboard -> mq: Publish: onboarding.created
activate mq
mq -> notif: Route to notification
deactivate mq

notif -> notif: Send email to\napproval team

onboard --> action: 201 Created {onboarding_id, status: "pending"}
deactivate onboard

action --> dialogue: Response + SlotReset
deactivate action

dialogue --> chat: "✓ Onboarding request created!\nID: onb_12345\nStatus: Pending verification\nNext: Agreement verification (~5min)"
chat --> admin: Show confirmation

@enduml
```

---

## 4. ML Training Pipeline Flow

```plantuml
@startuml ML_Pipeline_Flow
!theme plain
skinparam sequenceMessageAlign center

title ML Training Pipeline — Airflow Orchestrated

actor "ML Engineer" as mle
participant "Airflow\nWebserver" as airflow
participant "Ingestion\nTask" as ingest
participant "Validation\nTask" as validate
participant "Cleanup\nTask" as cleanup
participant "PII Masking\nTask" as pii
participant "Annotation\nTask" as annotate
participant "Augmentation\nTask" as augment
participant "Training\nTask" as train
participant "MongoDB" as mongo
participant "MinIO" as minio
participant "Label\nStudio" as ls
participant "Model\nRegistry" as registry
participant "RabbitMQ" as mq

== Pipeline Trigger ==

mle -> airflow: Trigger DAG: nlu_model_training\n(or scheduled: Sun 06:00)
activate airflow

== Stage 1: Ingestion ==

airflow -> ingest: Execute
activate ingest
ingest -> mongo: Query new conversations\n(since last_run)
mongo --> ingest: 1200 new records
ingest -> minio: Store raw batch\n(batch_20260721)
ingest --> airflow: batch_id = "batch_20260721"
deactivate ingest

== Stage 2: Validation ==

airflow -> validate: Execute (batch_id)
activate validate
validate -> minio: Load raw records
validate -> validate: Schema check\nLanguage detect\nDuplicate filter\nToxicity filter
validate -> mongo: Store validation results
validate --> airflow: {passed: 1100, failed: 50, quarantined: 50}
deactivate validate

== Stage 3: Cleanup ==

airflow -> cleanup: Execute (batch_id)
activate cleanup
cleanup -> minio: Load validated records
cleanup -> cleanup: Unicode normalize\nWhitespace fix\nSpell correct\nNumber normalize
cleanup -> minio: Store cleaned records
cleanup --> airflow: 1100 records cleaned
deactivate cleanup

== Stage 4: PII Masking ==

airflow -> pii: Execute (batch_id)
activate pii
pii -> minio: Load cleaned records
pii -> pii: Presidio analyze\n(MSISDN, IMSI, PERSON, EMAIL)
pii -> pii: Anonymize with\nconsistent placeholders
pii -> minio: Store masked records
pii -> mongo: Store PII mapping (encrypted)
pii --> airflow: {pii_found: 320 records, types: [MSISDN:180, PERSON:90]}
deactivate pii

== Stage 5: Annotation ==

airflow -> annotate: Execute (batch_id)
activate annotate
annotate -> minio: Load masked records
annotate -> annotate: Auto-annotate with\ncurrent production model
annotate -> annotate: Confidence routing:\n- auto_accept: 660\n- review_optional: 275\n- review_required: 165

alt Low confidence samples exist
    annotate -> ls: Push 165 tasks\nto Label Studio
    ls --> annotate: Tasks created
    note right of ls: Human annotators\nreview & correct\n(async)
end

annotate -> minio: Store annotated records
annotate --> airflow: {auto: 660, review: 275, human: 165}
deactivate annotate

== Stage 6: Augmentation ==

airflow -> augment: Execute (batch_id)
activate augment
augment -> minio: Load annotated records
augment -> augment: Analyze class distribution
augment -> augment: Apply per-intent:\n- Synonym replace\n- Back-translation\n- Entity substitution\n- BERT contextual
augment -> augment: Balance to target:\n100 examples/intent
augment -> minio: Store augmented dataset\n(1100 → 3100 records)
augment --> airflow: {original: 1100, augmented: 3100, multiplier: 2.8x}
deactivate augment

== Stage 7: Training ==

airflow -> train: Execute (K8s Pod with GPU)
activate train
train -> minio: Download training data + config
train -> train: rasa train nlu\n(DIET 150 epochs, early stopping)
train -> train: Evaluate on held-out test set
train -> train: Quality gate check:\n- intent_f1 ≥ 0.92 ✓\n- entity_f1 ≥ 0.88 ✓\n- fallback_rate 5-15% ✓

alt Quality gate PASSED
    train -> minio: Upload model artifact
    train -> registry: Register model (stage: "staging")
    train -> mq: Publish: model.trained.success
else Quality gate FAILED
    train -> mq: Publish: model.trained.failed
    train -> mle: Alert: model below threshold
end

train --> airflow: {model_id: "nlu_v2.1.0", status: "registered"}
deactivate train

airflow --> mle: Pipeline complete ✓\nModel nlu_v2.1.0 in staging
deactivate airflow

@enduml
```

---

## 5. Inventory Update Flow (via Chat)

```plantuml
@startuml Inventory_Update_Flow
!theme plain
skinparam sequenceMessageAlign center

title Inventory Update — Conversational CRUD

actor Operator as op
participant "Chat UI" as chat
participant "API Gateway" as gw
participant "Dialogue\nManager" as dialogue
participant "NLU Service" as nlu
participant "Action\nServer" as action
participant "Inventory\nService" as inv
participant "MongoDB" as mongo
participant "RabbitMQ" as mq
participant "Audit Log" as audit

op -> chat: "lock the connectivity of msisdn 44770090012"
chat -> gw: POST /conversations/{id}/messages\nAuth: Bearer <jwt>
activate gw

gw -> gw: Validate JWT\nRole: iot-operator ✓
gw -> dialogue: Forward
activate dialogue

dialogue -> nlu: Parse message
activate nlu
nlu --> dialogue: {\n  intent: "update_inventory",\n  entities: [\n    {entity: "connectivity_lock", value: "locked", role: "update_value"},\n    {entity: "msisdn", value: "44770090012", role: "fetch_value"}\n  ]\n}
deactivate nlu

dialogue -> dialogue: Policy: action_update_inventory
dialogue -> action: Execute action
activate action

action -> action: Validate:\n- MSISDN format ✓\n- Field is updateable ✓\n- Value is valid enum ✓

action -> inv: PATCH /inventory/44770090012\n{connectivity_lock: "locked"}
activate inv

inv -> inv: Input validation\n- msisdn exists ✓\n- field whitelisted ✓\n- value sanitized ✓

inv -> mongo: findOneAndUpdate(\n  {msisdn: 44770090012},\n  {$set: {connectivity_lock: "locked", updated_at: now}}\n)
activate mongo
mongo --> inv: {previous: "unlocked", current: "locked"}
deactivate mongo

inv -> mq: Publish: inventory.updated\n{msisdn, field, old_val, new_val, actor}
inv -> audit: Log: {actor: "operator@ex.com",\naction: "inventory.update",\nresource: "44770090012",\nchanges: {connectivity_lock: "unlocked" → "locked"}}

inv --> action: 200 OK {updated: true, previous: "unlocked"}
deactivate inv

action -> action: Format confirmation
action --> dialogue: [BotUttered: "Updated successfully"]
deactivate action

dialogue --> gw: Response
deactivate dialogue
gw --> chat: 200 OK
deactivate gw
chat --> op: "✓ Connectivity lock updated to 'locked'\nfor MSISDN 44770090012\n(was: unlocked)"

@enduml
```

---

## 6. Model Deployment (Canary) Flow

```plantuml
@startuml Model_Deployment_Flow
!theme plain
skinparam sequenceMessageAlign center

title Model Promotion — Canary Deployment

actor "ML Engineer" as mle
participant "Admin API" as api
participant "Model\nRegistry" as registry
participant "ArgoCD" as argo
participant "NLU Service\n(v2.0 - current)" as nlu_old
participant "NLU Service\n(v2.1 - canary)" as nlu_new
participant "Traefik\nIngress" as ingress
participant "Prometheus" as prom
participant "Alertmanager" as alert

== Initiate Promotion ==

mle -> api: POST /models/nlu_v2.1.0/promote\n{strategy: "canary", percentage: 10}
activate api

api -> registry: Get model nlu_v2.1.0\nVerify stage == "staging"
registry --> api: Model metadata + metrics

api -> argo: Update Helm values:\n- canary.enabled: true\n- canary.image.tag: nlu_v2.1.0\n- canary.weight: 10
activate argo

argo -> argo: Sync: deploy canary pod
argo -> nlu_new: Deploy new version
activate nlu_new
nlu_new -> nlu_new: Load model nlu_v2.1.0\nHealth check: ready
nlu_new --> argo: Pod ready
deactivate nlu_new

argo -> ingress: Update traffic split:\n90% → v2.0, 10% → v2.1
argo --> api: Canary deployed
deactivate argo

api --> mle: Canary active (10% traffic)
deactivate api

== Canary Monitoring (30 min window) ==

loop Every 30 seconds
    prom -> nlu_old: Scrape /metrics
    prom -> nlu_new: Scrape /metrics
    prom -> prom: Compare:\n- latency_p95\n- error_rate\n- intent_confidence_avg\n- fallback_rate
end

alt Canary metrics HEALTHY (30 min)
    prom -> alert: No alerts triggered
    
    mle -> api: Auto-promote (or manual confirm)
    api -> argo: Update: canary.weight: 100\n+ scale down old
    argo -> ingress: 100% → v2.1
    argo -> nlu_old: Scale to 0
    api -> registry: Update stage: "production"\nArchive v2.0

    api --> mle: ✓ Model v2.1.0 is now production

else Canary metrics DEGRADED
    prom -> alert: Alert: canary_error_rate > 2%
    alert -> mle: 🚨 Canary alert!
    
    alert -> argo: Auto-rollback triggered
    argo -> ingress: 100% → v2.0 (rollback)
    argo -> nlu_new: Delete canary pod
    
    argo -> registry: Mark v2.1.0: "failed_canary"
    alert --> mle: ⚠️ Rollback complete.\nModel v2.1.0 failed canary.
end

@enduml
```

---

## 7. Data Pipeline — PII Masking Detail

```plantuml
@startuml PII_Masking_Flow
!theme plain
skinparam sequenceMessageAlign center

title PII Masking — Presidio Pipeline Detail

participant "Airflow\nWorker" as worker
participant "MinIO\n(Object Store)" as minio
participant "Presidio\nAnalyzer" as analyzer
participant "Presidio\nAnonymizer" as anonymizer
participant "spaCy NER\nModel" as spacy
participant "Custom\nRecognizers" as custom
participant "Vault\n(Secrets)" as vault
participant "MongoDB\n(Audit)" as mongo

worker -> minio: Load cleaned batch records
activate worker
minio --> worker: 1100 records

loop For each record

    worker -> analyzer: analyze(text, language="en")
    activate analyzer
    
    analyzer -> spacy: NER inference\n(PERSON, ORG, GPE, LOC)
    spacy --> analyzer: Named entities found
    
    analyzer -> custom: IMSI recognizer (regex: \\d{15})
    custom --> analyzer: IMSI matches
    
    analyzer -> custom: MSISDN recognizer (regex: \\d{10,15})
    custom --> analyzer: MSISDN matches
    
    analyzer -> custom: Email recognizer
    custom --> analyzer: Email matches
    
    analyzer --> worker: PII results: [\n  {type: MSISDN, start: 35, end: 46, score: 0.95},\n  {type: PERSON, start: 10, end: 20, score: 0.85}\n]
    deactivate analyzer
    
    worker -> anonymizer: anonymize(text, results, operators)
    activate anonymizer
    
    anonymizer -> anonymizer: Replace PII with placeholders:\n"call John at 12345678901"\n→ "call [PERSON_001] at [MSISDN_001]"
    
    anonymizer --> worker: {text: masked_text, items: mapping}
    deactivate anonymizer
    
    worker -> worker: Realign entity annotations\n(adjust offsets for masked text)

end

worker -> minio: Store masked records\n(batch_20260721_masked/)
worker -> vault: Store reversible PII mapping\n(encrypted, access: iot-admin only)
worker -> mongo: Log masking audit:\n{batch_id, pii_count: 320, types: {...}, timestamp}

deactivate worker

@enduml
```

---

## 8. System Health Monitoring & Incident Response

```plantuml
@startuml Monitoring_Flow
!theme plain
skinparam sequenceMessageAlign center

title System Monitoring — Health Check & Auto-Recovery

participant "Prometheus" as prom
participant "NLU Service\n(Pod)" as nlu
participant "Kubernetes\n(kubelet)" as k8s
participant "Alertmanager" as alert
participant "Grafana" as grafana
participant "PagerDuty" as pager
participant "Slack" as slack
actor "On-Call\nEngineer" as oncall

== Normal Operation — Periodic Checks ==

loop Every 15s
    prom -> nlu: GET /metrics
    nlu --> prom: nlu_requests_total: 1500\nnlu_latency_p95: 120ms\nnlu_errors_total: 3\nnlu_model_confidence_avg: 0.91
end

loop Every 10s
    k8s -> nlu: GET /health/live
    nlu --> k8s: 200 OK
    k8s -> nlu: GET /health/ready
    nlu --> k8s: 200 OK
end

== Incident: High Latency Detected ==

prom -> prom: Rule triggered:\nnlu_latency_p95 > 500ms\nfor 2 minutes

prom -> alert: Fire alert:\n"NLU_HIGH_LATENCY"
activate alert

alert -> alert: Route by severity:\nWARNING → Slack\nCRITICAL → PagerDuty

alert -> slack: ⚠️ NLU latency P95 = 650ms\n(threshold: 500ms)\nDuration: 2min
alert -> grafana: Create annotation

deactivate alert

== Incident: Pod Crash ==

nlu -> nlu: OOM Kill / Exception
nlu --> k8s: /health/live → 503

k8s -> k8s: Liveness probe failed\n(3 consecutive failures)
k8s -> k8s: Restart pod
k8s -> nlu: New pod starting...

nlu -> nlu: Load model (30s)
nlu --> k8s: /health/ready → 200

k8s -> prom: Pod restart event

prom -> alert: Fire alert:\n"NLU_POD_RESTART"
alert -> slack: 🔄 NLU pod restarted\n(OOMKilled, memory: 4Gi limit)

== Critical Incident: All Pods Down ==

k8s -> k8s: All NLU replicas\nfailing readiness

prom -> alert: CRITICAL:\n"NLU_SERVICE_DOWN"\n(0 ready pods)
activate alert

alert -> pager: 🚨 PAGE: NLU Service DOWN\nAll replicas unhealthy
alert -> slack: 🚨 CRITICAL: NLU Service unavailable\nImpact: All chat functionality broken

pager -> oncall: Phone call + SMS
deactivate alert

oncall -> grafana: Check dashboards
oncall -> oncall: Investigate:\n- kubectl logs\n- Model load failure?\n- MongoDB connection?\n- Memory/CPU?

oncall -> k8s: kubectl rollout restart\ndeployment/iot-nlu-service

k8s -> nlu: Fresh pods deployed
nlu --> k8s: Ready ✓

oncall -> slack: ✅ Resolved: NLU pods recovered\nRCA: Model file corruption\nAction: Redeploy from registry

@enduml
```

---

## 9. Subscription Query Flow (IMSI Lookup)

```plantuml
@startuml Subscription_Query_Flow
!theme plain
skinparam sequenceMessageAlign center

title Subscription Query — Secure IMSI Lookup

actor User as user
participant "Chat UI" as chat
participant "Gateway" as gw
participant "Dialogue Mgr" as dm
participant "NLU" as nlu
participant "Action Server" as action
participant "Subscription\nService" as sub
participant "MongoDB" as mongo
participant "Redis Cache" as redis

user -> chat: "what is the status of IMSI 234150999999999"
chat -> gw: POST /conversations/{id}/messages
activate gw
gw -> gw: JWT valid, role: iot-operator
gw -> dm: Forward
activate dm

dm -> nlu: Parse
nlu --> dm: {intent: "fetch", entities: [{entity: "imsi", value: "234150999999999"}]}

dm -> dm: Set slot: IMSI_number = "234150999999999"
dm -> dm: Policy: action_IMSI_Stats

dm -> action: Execute
activate action

action -> action: Validate IMSI format:\n- 15 digits ✓\n- Numeric only ✓

action -> sub: GET /subscriptions/234150999999999
activate sub

sub -> redis: Check cache (key: sub:234150999999999)
alt Cache HIT
    redis --> sub: Cached subscription data
else Cache MISS
    sub -> mongo: db.subscriptions.findOne({imsi: "234150999999999"})
    mongo --> sub: Document found
    sub -> redis: SET sub:234150999999999 (TTL: 5min)
end

sub -> sub: Filter response:\n- Remove PIN/PUK (NEVER exposed)\n- Include: status, plan, dates

sub --> action: 200 {imsi, msisdn, state, plan, dates}
deactivate sub

action -> action: Format safe response\n(no sensitive fields)
action --> dm: BotUttered
deactivate action

dm --> gw: Response
deactivate dm
gw --> chat: Response
deactivate gw

chat --> user: "IMSI 234150999999999:\n• State: Active\n• MSISDN: 447700900***\n• Plan: 10 GB Europe\n• Installed: 2024-01-15\n• Last active: 2026-07-20"

note right of user
  PIN/PUK are NEVER shown.
  MSISDN is partially masked.
  Full access requires MFA
  via admin portal.
end note

@enduml
```

---

## 10. CI/CD Deployment Flow

```plantuml
@startuml CICD_Flow
!theme plain
skinparam sequenceMessageAlign center

title CI/CD Pipeline — Code to Production

actor Developer as dev
participant "GitHub" as git
participant "GitHub\nActions" as ci
participant "SonarQube" as sonar
participant "JFrog\nArtifactory" as jfrog
participant "Trivy" as trivy
participant "ArgoCD" as argo
participant "K8s Dev" as k8s_dev
participant "K8s Staging" as k8s_stg
participant "K8s Prod" as k8s_prod

== Development ==

dev -> git: Push to feature/xyz
git -> ci: Trigger CI (PR)
activate ci

ci -> ci: 1. Lint (ruff + mypy)
ci -> ci: 2. Unit tests (pytest)\n   Coverage: 85%
ci -> sonar: 3. Static analysis
sonar --> ci: Quality gate: PASSED

ci -> ci: 4. Build Docker image
ci -> trivy: 5. Scan image
trivy --> ci: No HIGH/CRITICAL CVEs ✓

ci -> jfrog: 6. Push image\n(tag: feature-xyz-sha123)
ci --> git: ✓ All checks passed
deactivate ci

dev -> git: Merge PR → develop

== Deploy to Dev ==

git -> ci: Trigger (develop branch)
activate ci
ci -> jfrog: Push image (tag: dev-sha456)
ci -> argo: Update dev values\n(image.tag: dev-sha456)
deactivate ci

argo -> k8s_dev: Sync deployment
k8s_dev -> k8s_dev: Rolling update
k8s_dev --> argo: Healthy ✓

== Deploy to Staging ==

dev -> git: Merge develop → main\nTag: v2.1.0
git -> ci: Trigger (main + tag)
activate ci

ci -> ci: Full test suite\n(unit + integration + e2e)
ci -> jfrog: Push image (tag: 2.1.0)
ci -> jfrog: Push Helm chart (v2.1.0)
ci -> jfrog: Xray scan + license check
deactivate ci

ci -> argo: Update staging values
argo -> k8s_stg: Sync deployment
k8s_stg -> k8s_stg: Deploy + smoke tests
k8s_stg --> argo: Healthy ✓

== Deploy to Production (Manual Gate) ==

argo -> dev: 📋 Approval required\nfor production deploy
dev -> argo: ✅ Approve

argo -> k8s_prod: Sync deployment\n(rolling update, maxSurge: 1)
k8s_prod -> k8s_prod: Rolling update\n(zero-downtime)
k8s_prod --> argo: All pods healthy ✓

argo --> dev: 🚀 v2.1.0 deployed to production

@enduml
```

---

## Diagram Index

| # | Diagram | Key Flow | Participants |
|---|---------|----------|--------------|
| 1 | Chat Message Flow | User → NLU → Dialogue → Action → DB → Response | 10 components |
| 2 | Authentication Flow | Login → JWT → API calls → Refresh | 5 components |
| 3 | Customer Onboarding | Conversational form → Validation → DB → Notification | 8 components |
| 4 | ML Training Pipeline | Ingest → Validate → Clean → Mask → Annotate → Train | 13 components |
| 5 | Inventory Update | Chat → NLU → Action → Validate → Update → Audit | 10 components |
| 6 | Model Deployment | Staging → Canary → Monitor → Promote/Rollback | 8 components |
| 7 | PII Masking Detail | Presidio → spaCy → Custom recognizers → Vault | 8 components |
| 8 | System Monitoring | Prometheus → Alerts → PagerDuty → Recovery | 8 components |
| 9 | Subscription Query | Secure IMSI lookup with caching + masking | 8 components |
| 10 | CI/CD Pipeline | Code → Lint → Test → Scan → Build → Deploy (dev→stg→prod) | 9 components |

---

## How to Render

```bash
# Install PlantUML
brew install plantuml

# Render all diagrams to PNG
plantuml docs/architecture/05_SEQUENCE_DIAGRAMS.md -o ./docs/architecture/diagrams/

# Or use the online server
# https://www.plantuml.com/plantuml/uml/
```

---

*This completes the system architecture documentation suite. All files:*
- *[01_SYSTEM_ARCHITECTURE.md](./01_SYSTEM_ARCHITECTURE.md) — System overview*
- *[02_DATA_PIPELINE.md](./02_DATA_PIPELINE.md) — Airflow ETL pipeline*
- *[03_API_DESIGN.md](./03_API_DESIGN.md) — API specifications*
- *[04_INFRASTRUCTURE.md](./04_INFRASTRUCTURE.md) — Deployment architecture*
- *[05_SEQUENCE_DIAGRAMS.md](./05_SEQUENCE_DIAGRAMS.md) — This file*
