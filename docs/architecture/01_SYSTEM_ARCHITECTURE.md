# IoT Digital Assistant — Production System Architecture

> **Version:** 2.0  
> **Date:** 2026-07-21  
> **Status:** Design Phase  
> **Classification:** Production-Ready Microservice Architecture

---

## 1. System Overview

### 1.1 Vision

Transform the IoT Digital Assistant from a monolithic demo into a production-grade, cloud-native conversational AI platform for IoT service management. The system handles natural language understanding, subscription management, inventory operations, customer onboarding, and knowledge retrieval — all with enterprise-grade security, observability, and CI/CD.

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                         │
│   [Web UI]  [Mobile App]  [REST API]  [Slack/Teams]  [Socket.IO]           │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ HTTPS / WSS
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY (Kong / Traefik)                         │
│   Rate Limiting │ TLS Termination │ JWT Validation │ Request Routing        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌─────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  CONVERSATION   │   │   BUSINESS LOGIC    │   │   DATA PIPELINE     │
│   SERVICES      │   │     SERVICES        │   │     SERVICES        │
├─────────────────┤   ├─────────────────────┤   ├─────────────────────┤
│ • NLU Service   │   │ • Subscription Svc  │   │ • Airflow Scheduler │
│ • Dialogue Mgr  │   │ • Inventory Svc     │   │ • Data Ingestion    │
│ • Action Server │   │ • Onboarding Svc    │   │ • PII Masking       │
│ • Channel Conn. │   │ • CPI Knowledge Svc │   │ • Annotation Svc    │
│                 │   │ • News Service      │   │ • Augmentation Svc  │
│                 │   │ • Notification Svc  │   │ • ML Training Svc   │
└────────┬────────┘   └──────────┬──────────┘   │ • Model Registry    │
         │                       │               └──────────┬──────────┘
         └───────────┬───────────┘                          │
                     ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PLATFORM SERVICES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Keycloak (AuthN/AuthZ)     • MongoDB (Data Store)     • Redis (Cache)     │
│ • RabbitMQ (Event Bus)       • MinIO (Object Storage)   • Vault (Secrets)   │
│ • Prometheus + Grafana       • ELK Stack (Logging)      • Jaeger (Tracing)  │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CI/CD & ARTIFACT MANAGEMENT                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ • JFrog Artifactory (Docker images, Python packages, Helm charts)           │
│ • Jenkins / GitHub Actions (Build pipelines)                                │
│ • ArgoCD (GitOps deployment)                                                │
│ • SonarQube (Code quality)    • Trivy (Container scanning)                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Design Principles

| Principle | Application |
|-----------|-------------|
| Single Responsibility | Each microservice owns one domain |
| 12-Factor App | Config from env, stateless processes, port binding |
| API-First | OpenAPI specs before implementation |
| Event-Driven | Async communication via RabbitMQ for non-critical paths |
| Defense in Depth | Auth at gateway + service level, encrypted at rest + transit |
| Observability | Metrics, logs, traces on every service |
| Immutable Infrastructure | Container images versioned in JFrog, never patched in-place |
| GitOps | Desired state in Git, ArgoCD reconciles |

---

## 2. Microservice Decomposition

### 2.1 Service Catalog

| # | Service | Domain | Language | Port | Database |
|---|---------|--------|----------|------|----------|
| 1 | `iot-nlu-service` | NLU inference (intent + entity) | Python | 5010 | Redis (cache) |
| 2 | `iot-dialogue-manager` | Conversation state + policy | Python | 5011 | MongoDB (tracker) |
| 3 | `iot-action-server` | Custom action orchestration | Python | 5055 | — |
| 4 | `iot-channel-connector` | Multi-channel adapter | Python | 5012 | Redis (sessions) |
| 5 | `iot-subscription-service` | IMSI/SIM management | Python | 5020 | MongoDB |
| 6 | `iot-inventory-service` | Device inventory CRUD | Python | 5021 | MongoDB |
| 7 | `iot-onboarding-service` | Customer registration | Python | 5022 | MongoDB |
| 8 | `iot-knowledge-service` | CPI docs + search | Python | 5023 | MongoDB + Elasticsearch |
| 9 | `iot-notification-service` | Alerts + news delivery | Python | 5024 | Redis |
| 10 | `iot-data-pipeline` | Airflow DAGs + workers | Python | 8080 | PostgreSQL (Airflow) |
| 11 | `iot-ml-training-service` | Model training + eval | Python | 5030 | MinIO (artifacts) |
| 12 | `iot-model-registry` | Model versioning + serving | Python | 5031 | MinIO + MongoDB |
| 13 | `iot-api-gateway` | Routing + rate limiting | Go/Lua | 8443 | — |
| 14 | `iot-admin-portal` | System administration UI | TypeScript | 3000 | — |

### 2.2 Service Ownership Matrix

```
┌────────────────────┬──────────────────────────────────────────────┐
│ CONVERSATION LAYER │ NLU → Dialogue Manager → Action Server       │
│                    │ Channel Connector ↔ Dialogue Manager          │
├────────────────────┼──────────────────────────────────────────────┤
│ BUSINESS LAYER     │ Subscription │ Inventory │ Onboarding        │
│                    │ Knowledge │ Notification                      │
├────────────────────┼──────────────────────────────────────────────┤
│ DATA/ML LAYER      │ Airflow Pipeline → ML Training → Registry    │
├────────────────────┼──────────────────────────────────────────────┤
│ PLATFORM LAYER     │ Gateway │ Keycloak │ MongoDB │ Redis │ MQ    │
└────────────────────┴──────────────────────────────────────────────┘
```

---

## 3. Technology Stack

### 3.1 Core Technologies

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| NLU Engine | Rasa Open Source | 3.8+ | Intent classification, entity extraction |
| ML Framework | TensorFlow | 2.16+ | DIET/TED model backends |
| NLP | spaCy | 3.7+ | Tokenization, featurization, NER |
| Language | Python | 3.11+ | All microservices |
| Web Framework | FastAPI | 0.100+ | REST APIs for business services |
| Async | Celery | 5.3+ | Background task processing |
| ETL/Pipeline | Apache Airflow | 2.8+ | Data pipeline orchestration |

### 3.2 Data & Storage

| Technology | Version | Purpose |
|-----------|---------|---------|
| MongoDB | 7.0+ | Primary data store (subscriptions, inventory, customers, tracker) |
| Redis | 7.2+ | Caching, session store, rate limiting |
| PostgreSQL | 16+ | Airflow metadata, structured analytics |
| Elasticsearch | 8.12+ | CPI document search, NLU analytics |
| MinIO | Latest | Object storage (models, training data, artifacts) |

### 3.3 Infrastructure & Security

| Technology | Version | Purpose |
|-----------|---------|---------|
| Keycloak | 24+ | Authentication, authorization, SSO |
| HashiCorp Vault | 1.15+ | Secret management |
| Kong / Traefik | Latest | API Gateway |
| RabbitMQ | 3.13+ | Event bus, async messaging |
| Docker | 25+ | Containerization |
| Kubernetes | 1.29+ | Orchestration |
| Helm | 3.14+ | Package management |

### 3.4 CI/CD & Artifacts

| Technology | Version | Purpose |
|-----------|---------|---------|
| JFrog Artifactory | 7.x | Docker registry, PyPI, Helm repo, generic artifacts |
| Jenkins / GitHub Actions | Latest | CI pipelines |
| ArgoCD | 2.10+ | GitOps continuous deployment |
| SonarQube | 10+ | Static analysis, code quality |
| Trivy | Latest | Container vulnerability scanning |
| DVC | 3.x | Data/model version control |

### 3.5 Observability

| Technology | Purpose |
|-----------|---------|
| Prometheus | Metrics collection |
| Grafana | Dashboards + alerting |
| OpenTelemetry | Distributed tracing instrumentation |
| Jaeger | Trace visualization |
| ELK (Elasticsearch + Logstash + Kibana) | Centralized logging |
| Alertmanager | Alert routing |

---

## 4. Communication Patterns

### 4.1 Synchronous (REST/gRPC)

Used for: real-time user-facing requests where latency matters.

```
Client → Gateway → NLU Service (inference)
Client → Gateway → Dialogue Manager → Action Server → Business Service → MongoDB
```

### 4.2 Asynchronous (RabbitMQ)

Used for: background processing, event propagation, decoupling.

```
Exchanges:
├── iot.conversation.events    (fanout) → Analytics, Logging, Audit
├── iot.pipeline.triggers      (direct) → Airflow DAG triggers
├── iot.model.events           (topic)  → model.trained, model.deployed, model.failed
├── iot.notification.events    (direct) → Email, Slack, SMS notifications
└── iot.inventory.events       (topic)  → inventory.updated, inventory.created
```

### 4.3 Service Discovery

- Kubernetes DNS for internal services: `<service-name>.<namespace>.svc.cluster.local`
- Health checks: `/health/live` (liveness), `/health/ready` (readiness)
- Configuration: ConfigMaps + Vault-injected secrets

---

## 5. Data Model Overview

### 5.1 MongoDB Collections (per database)

```
Database: iot_assistant
├── conversations          (Rasa tracker store — conversation state)
├── subscriptions          (IMSI, SIM details, plans)
├── inventory              (device inventory, connectivity status)
├── customers              (onboarded customer records)
├── cpi_documents          (CPI links + metadata)
├── model_metadata         (trained model versions + metrics)
├── audit_log              (all state-changing operations)
└── user_sessions          (active user session context)

Database: iot_pipeline (Airflow-managed)
├── raw_training_data      (ingested, unprocessed NLU data)
├── annotated_data         (post-annotation pipeline)
├── cleaned_data           (post-cleanup, PII-masked)
├── augmented_data         (post-augmentation, ready for training)
└── evaluation_results     (model performance history)
```

### 5.2 Key Data Flows

```
User Message → NLU → Dialogue Manager → Action Server → Business Service → MongoDB
                                                              │
                                                              ▼
                                                     RabbitMQ (event)
                                                              │
                                                              ▼
                                                  Analytics / Audit Log
```

---

## 6. Security Architecture

### 6.1 Authentication Flow

```
┌────────┐     ┌─────────┐     ┌──────────┐     ┌──────────────┐
│ Client │────▶│ Gateway │────▶│ Keycloak │────▶│ JWT Issued   │
└────────┘     └─────────┘     └──────────┘     └──────────────┘
                                                        │
    ┌───────────────────────────────────────────────────┘
    ▼
┌─────────┐  JWT in header   ┌──────────────┐
│ Gateway │─────────────────▶│ Microservice │ (validates JWT locally)
└─────────┘                  └──────────────┘
```

### 6.2 Authorization Model

| Role | Permissions |
|------|-------------|
| `iot-user` | Chat, view own subscriptions, view CPI docs |
| `iot-operator` | All user perms + inventory CRUD, view all subscriptions |
| `iot-admin` | All operator perms + onboarding, user management, pipeline triggers |
| `iot-ml-engineer` | Pipeline management, model deployment, training data access |
| `iot-system` | Service-to-service communication (mTLS + client credentials) |

### 6.3 Security Controls

| Control | Implementation |
|---------|---------------|
| AuthN | Keycloak OIDC (JWT tokens, refresh tokens) |
| AuthZ | Keycloak realm roles + resource-based permissions |
| Secrets | Vault with Kubernetes auth backend |
| TLS | Cert-manager for internal mTLS, Let's Encrypt for external |
| PII Protection | Masking pipeline in Airflow, field-level encryption in MongoDB |
| Audit | Every state change logged with actor, timestamp, change delta |
| Input Validation | JSON Schema validation at gateway + service level |
| Rate Limiting | Kong/Traefik plugin: 100 req/min per user, 1000 req/min per service |

---

## 7. Deployment Topology

### 7.1 Kubernetes Namespaces

```
Cluster: iot-assistant-prod
├── iot-conversation    (NLU, Dialogue, Action Server, Channel Connector)
├── iot-business        (Subscription, Inventory, Onboarding, Knowledge, Notification)
├── iot-pipeline        (Airflow, ML Training, Model Registry)
├── iot-platform        (Keycloak, MongoDB, Redis, RabbitMQ, Vault)
├── iot-observability   (Prometheus, Grafana, ELK, Jaeger)
├── iot-ingress         (API Gateway, cert-manager)
└── iot-cicd            (ArgoCD, Jenkins agents)
```

### 7.2 Resource Estimates

| Service | Replicas | CPU (req/lim) | Memory (req/lim) |
|---------|----------|---------------|-------------------|
| NLU Service | 2-4 (HPA) | 500m/2000m | 1Gi/4Gi |
| Dialogue Manager | 2-3 | 250m/1000m | 512Mi/2Gi |
| Action Server | 2-3 | 250m/1000m | 512Mi/1Gi |
| Business Services (each) | 2 | 200m/500m | 256Mi/512Mi |
| Airflow Scheduler | 1 | 500m/1000m | 1Gi/2Gi |
| Airflow Worker | 2-5 (KEDA) | 1000m/4000m | 2Gi/8Gi |
| ML Training (burst) | 0-3 (Job) | 4000m/8000m | 8Gi/16Gi |
| MongoDB | 3 (ReplicaSet) | 1000m/2000m | 2Gi/4Gi |
| Redis | 3 (Sentinel) | 200m/500m | 256Mi/512Mi |
| Keycloak | 2 | 500m/1000m | 512Mi/1Gi |

---

*Next: See [02_DATA_PIPELINE.md](./02_DATA_PIPELINE.md) for the complete Airflow ETL architecture.*
