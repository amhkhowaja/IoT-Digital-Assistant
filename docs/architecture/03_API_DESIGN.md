# API Design — IoT Digital Assistant

> **Version:** 2.0  
> **Date:** 2026-07-21  
> **Standard:** OpenAPI 3.1  
> **Auth:** All endpoints require Keycloak JWT unless marked `[PUBLIC]`

---

## 1. API Overview

### 1.1 Base URLs

| Service | Internal URL | External URL |
|---------|-------------|--------------|
| API Gateway | — | `https://iot-assistant.example.com/api/v1` |
| NLU Service | `http://iot-nlu-service:5010` | `/api/v1/nlu/` |
| Dialogue Manager | `http://iot-dialogue-manager:5011` | `/api/v1/conversations/` |
| Channel Connector | `http://iot-channel-connector:5012` | `/api/v1/channels/` |
| Subscription Service | `http://iot-subscription-service:5020` | `/api/v1/subscriptions/` |
| Inventory Service | `http://iot-inventory-service:5021` | `/api/v1/inventory/` |
| Onboarding Service | `http://iot-onboarding-service:5022` | `/api/v1/onboarding/` |
| Knowledge Service | `http://iot-knowledge-service:5023` | `/api/v1/knowledge/` |
| Notification Service | `http://iot-notification-service:5024` | `/api/v1/notifications/` |
| ML Pipeline | `http://iot-data-pipeline:8080` | `/api/v1/pipeline/` |
| Model Registry | `http://iot-model-registry:5031` | `/api/v1/models/` |
| Admin Portal | `http://iot-admin-portal:3000` | `/admin/` |

### 1.2 Common Headers

```
Authorization: Bearer <keycloak_jwt_token>
Content-Type: application/json
X-Request-ID: <uuid>                    # Correlation ID for tracing
X-Tenant-ID: <tenant_id>               # Multi-tenancy support
Accept-Language: en                      # Localization
```

### 1.3 Common Response Format

```json
{
  "status": "success | error",
  "data": { ... },
  "meta": {
    "request_id": "uuid",
    "timestamp": "2026-07-21T10:00:00Z",
    "version": "v1"
  },
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 150,
    "total_pages": 8
  },
  "errors": [
    {
      "code": "VALIDATION_ERROR",
      "message": "Human readable message",
      "field": "msisdn",
      "details": "Must be 10-15 digits"
    }
  ]
}
```

### 1.4 Error Codes

| HTTP Status | Code | Description |
|-------------|------|-------------|
| 400 | `VALIDATION_ERROR` | Input validation failed |
| 401 | `UNAUTHORIZED` | Missing or invalid JWT |
| 403 | `FORBIDDEN` | Insufficient permissions |
| 404 | `NOT_FOUND` | Resource not found |
| 409 | `CONFLICT` | Duplicate or state conflict |
| 422 | `UNPROCESSABLE` | Semantically invalid request |
| 429 | `RATE_LIMITED` | Too many requests |
| 500 | `INTERNAL_ERROR` | Server error (no details exposed) |
| 503 | `SERVICE_UNAVAILABLE` | Downstream dependency down |

---

## 2. Conversation APIs

### 2.1 NLU Service (`/api/v1/nlu/`)

#### Parse Message
```
POST /api/v1/nlu/parse
Role: iot-user, iot-operator, iot-admin
```

**Request:**
```json
{
  "text": "what is the connectivity status of msisdn 12345678901",
  "message_id": "msg_uuid",
  "metadata": {
    "channel": "web",
    "language": "en"
  }
}
```

**Response (200):**
```json
{
  "status": "success",
  "data": {
    "intent": {
      "name": "fetch",
      "confidence": 0.94
    },
    "intent_ranking": [
      {"name": "fetch", "confidence": 0.94},
      {"name": "information_seek", "confidence": 0.03},
      {"name": "Troubleshoot", "confidence": 0.01}
    ],
    "entities": [
      {
        "entity": "msisdn",
        "value": "12345678901",
        "start": 43,
        "end": 54,
        "confidence": 0.98,
        "extractor": "DIETClassifier"
      },
      {
        "entity": "network_connectivity",
        "value": "connectivity status",
        "start": 12,
        "end": 31,
        "confidence": 0.89,
        "extractor": "DIETClassifier"
      }
    ],
    "text": "what is the connectivity status of msisdn 12345678901",
    "model_version": "nlu_v2.1.0",
    "processing_time_ms": 45
  }
}
```

#### Model Info
```
GET /api/v1/nlu/model/info
Role: iot-operator, iot-admin, iot-ml-engineer
```

**Response (200):**
```json
{
  "status": "success",
  "data": {
    "model_version": "nlu_v2.1.0",
    "loaded_at": "2026-07-21T06:00:00Z",
    "pipeline": ["SpacyNLP", "DIETClassifier", "FallbackClassifier"],
    "intents_count": 31,
    "entities_count": 25,
    "training_data_hash": "sha256:abc123...",
    "metrics": {
      "intent_f1": 0.95,
      "entity_f1": 0.91
    }
  }
}
```

### 2.2 Dialogue Manager (`/api/v1/conversations/`)

#### Send Message (Primary Chat Endpoint)
```
POST /api/v1/conversations/{sender_id}/messages
Role: iot-user, iot-operator, iot-admin
```

**Request:**
```json
{
  "text": "show me details of IMSI 234150999999999",
  "channel": "web",
  "metadata": {
    "session_id": "session_uuid",
    "page_context": "dashboard"
  }
}
```

**Response (200):**
```json
{
  "status": "success",
  "data": {
    "responses": [
      {
        "recipient_id": "sender_123",
        "text": "The details of IMSI [MASKED] are:\nInstallation date: 2024-01-15\nSIM subscription state: Active\nSIM status: Enabled",
        "buttons": null,
        "custom": null
      }
    ],
    "tracker": {
      "sender_id": "sender_123",
      "slots": {
        "IMSI_number": "234150999999999"
      },
      "latest_action": "action_IMSI_Stats",
      "active_loop": null
    }
  }
}
```

#### Get Conversation History
```
GET /api/v1/conversations/{sender_id}/tracker
Role: iot-operator, iot-admin
```

**Query Parameters:**
- `include_events`: boolean (default: false)
- `until`: ISO timestamp (events until this time)

#### Reset Conversation
```
POST /api/v1/conversations/{sender_id}/reset
Role: iot-user (own), iot-admin (any)
```

### 2.3 Channel Connector (`/api/v1/channels/`)

#### Webhook — REST Channel `[PUBLIC]`
```
POST /api/v1/channels/rest/webhook
Auth: Channel-specific token
```

#### Webhook — Socket.IO
```
WS /api/v1/channels/socketio
Auth: JWT token in connection params
Events: user_uttered, bot_uttered, session_request
```

#### Webhook — Slack
```
POST /api/v1/channels/slack/webhook
Auth: Slack signing secret verification
```

#### List Active Channels
```
GET /api/v1/channels/
Role: iot-admin
```

---

## 3. Business Service APIs

### 3.1 Subscription Service (`/api/v1/subscriptions/`)

#### Get Subscription by IMSI
```
GET /api/v1/subscriptions/{imsi}
Role: iot-operator, iot-admin
```

**Response (200):**
```json
{
  "status": "success",
  "data": {
    "imsi": "234150999999999",
    "msisdn": "447700900123",
    "installation_date": "2024-01-15",
    "sim_subscription_state": "active",
    "sim_status": "enabled",
    "plan": {
      "name": "10 GB Europe",
      "data_limit_mb": 10240,
      "billing_cycle": "monthly"
    },
    "last_activity": "2026-07-20T14:30:00Z"
  }
}
```

#### Search Subscriptions
```
GET /api/v1/subscriptions/
Role: iot-operator, iot-admin
Query: ?msisdn=447700900123&state=active&page=1&per_page=20
```

#### Get Subscription Attribute
```
GET /api/v1/subscriptions/{imsi}/attributes/{attribute_name}
Role: iot-operator, iot-admin
```

**Allowed attributes:** `installation_date`, `sim_subscription_state`, `msisdn`, `sim_status`, `plan_name`

**Note:** PIN/PUK codes are NOT exposed via API. They require a separate secured flow with MFA.

#### Update Subscription
```
PATCH /api/v1/subscriptions/{imsi}
Role: iot-admin
```

**Request:**
```json
{
  "sim_subscription_state": "suspended",
  "reason": "Customer request"
}
```

### 3.2 Inventory Service (`/api/v1/inventory/`)

#### List Inventory Items
```
GET /api/v1/inventory/
Role: iot-operator, iot-admin
Query: ?msisdn=447700900123&connectivity_lock=locked&billing_state=active&page=1&per_page=20
```

**Response (200):**
```json
{
  "status": "success",
  "data": [
    {
      "id": "inv_uuid",
      "msisdn": 447700900123,
      "plan_name": "10 GB Europe",
      "connectivity_lock": "unlocked",
      "network_connectivity": "connected",
      "in_session": true,
      "billing_state": "active",
      "monthly_data": "7.2 GB / 10 GB",
      "data_trend": "upward",
      "last_updated": "2026-07-21T10:00:00Z"
    }
  ],
  "pagination": {"page": 1, "per_page": 20, "total": 1, "total_pages": 1}
}
```

#### Get Single Inventory Item
```
GET /api/v1/inventory/{msisdn}
Role: iot-operator, iot-admin
```

#### Update Inventory Item
```
PATCH /api/v1/inventory/{msisdn}
Role: iot-operator, iot-admin
```

**Request:**
```json
{
  "connectivity_lock": "locked",
  "reason": "Security measure",
  "operator_note": "Requested by customer support ticket #12345"
}
```

**Response (200):**
```json
{
  "status": "success",
  "data": {
    "msisdn": 447700900123,
    "updated_fields": ["connectivity_lock"],
    "previous_values": {"connectivity_lock": "unlocked"},
    "updated_at": "2026-07-21T10:30:00Z",
    "updated_by": "operator@example.com"
  }
}
```

#### Bulk Query Inventory
```
POST /api/v1/inventory/query
Role: iot-operator, iot-admin
```

**Request:**
```json
{
  "filters": {
    "billing_state": "active",
    "network_connectivity": "disconnected",
    "data_trend": "downward"
  },
  "fields": ["msisdn", "plan_name", "last_activity"],
  "sort": {"field": "last_activity", "order": "desc"},
  "page": 1,
  "per_page": 50
}
```

### 3.3 Onboarding Service (`/api/v1/onboarding/`)

#### Start Onboarding
```
POST /api/v1/onboarding/
Role: iot-admin
```

**Request:**
```json
{
  "customer_type": "enterprise",
  "enterprise_name": "IoT Solutions Ltd",
  "enterprise_agreement_number": "12345678901234",
  "parent_organization": "ChinaTelecomMongoliaBranch",
  "contact": {
    "name": "John Doe",
    "email": "john@iotsolutions.com",
    "phone": "+447700900123"
  }
}
```

**Response (201):**
```json
{
  "status": "success",
  "data": {
    "onboarding_id": "onb_uuid",
    "status": "pending_verification",
    "customer_type": "enterprise",
    "enterprise_name": "IoT Solutions Ltd",
    "created_at": "2026-07-21T10:00:00Z",
    "next_steps": [
      "Agreement verification (auto, ~5min)",
      "Parent organization approval (manual)",
      "Account provisioning"
    ]
  }
}
```

#### Get Onboarding Status
```
GET /api/v1/onboarding/{onboarding_id}
Role: iot-admin
```

#### List Onboarding Requests
```
GET /api/v1/onboarding/?status=pending_verification&page=1
Role: iot-admin
```

#### Approve/Reject Onboarding
```
POST /api/v1/onboarding/{onboarding_id}/approve
POST /api/v1/onboarding/{onboarding_id}/reject
Role: iot-admin
```

### 3.4 Knowledge Service (`/api/v1/knowledge/`)

#### Search CPI Documents
```
GET /api/v1/knowledge/search
Role: iot-user, iot-operator, iot-admin
Query: ?q=connectivity+troubleshooting&category=networking&limit=5
```

**Response (200):**
```json
{
  "status": "success",
  "data": {
    "results": [
      {
        "id": "cpi_uuid",
        "title": "Connectivity Troubleshooting Guide",
        "category": "networking",
        "url": "https://cpi.example.com/docs/connectivity-troubleshoot",
        "snippet": "Step 1: Verify SIM status...",
        "relevance_score": 0.92,
        "last_updated": "2026-06-01"
      }
    ],
    "total_results": 12,
    "search_time_ms": 35
  }
}
```

#### Get CPI Document by Intent+Entity
```
GET /api/v1/knowledge/resolve
Role: iot-user, iot-operator, iot-admin
Query: ?intent=Learn&entity=connectivity_lock
```

#### Create/Update CPI Entry
```
POST /api/v1/knowledge/documents
PUT /api/v1/knowledge/documents/{doc_id}
Role: iot-admin
```

### 3.5 Notification Service (`/api/v1/notifications/`)

#### Get News Feed
```
GET /api/v1/notifications/news
Role: iot-user, iot-operator, iot-admin
Query: ?source=bbc-news&limit=5
```

#### Subscribe to Alerts
```
POST /api/v1/notifications/subscriptions
Role: iot-user, iot-operator, iot-admin
```

**Request:**
```json
{
  "event_types": ["inventory.connectivity_lost", "subscription.state_change"],
  "channels": ["email", "slack"],
  "filters": {
    "msisdn": [447700900123, 447700900124]
  }
}
```

---

## 4. Data Pipeline & ML APIs

### 4.1 Pipeline Management (`/api/v1/pipeline/`)

#### Trigger Pipeline Run
```
POST /api/v1/pipeline/trigger
Role: iot-ml-engineer, iot-admin
```

**Request:**
```json
{
  "dag_id": "nlu_data_processing",
  "config": {
    "batch_id": "batch_20260721",
    "skip_augmentation": false,
    "force_retrain": true
  }
}
```

**Response (202):**
```json
{
  "status": "success",
  "data": {
    "dag_run_id": "nlu_data_processing__2026-07-21T10:00:00",
    "state": "queued",
    "triggered_by": "ml-engineer@example.com",
    "estimated_duration_minutes": 120,
    "monitor_url": "/api/v1/pipeline/runs/nlu_data_processing__2026-07-21T10:00:00"
  }
}
```

#### Get Pipeline Run Status
```
GET /api/v1/pipeline/runs/{dag_run_id}
Role: iot-ml-engineer, iot-admin
```

**Response (200):**
```json
{
  "status": "success",
  "data": {
    "dag_run_id": "nlu_data_processing__2026-07-21T10:00:00",
    "dag_id": "nlu_data_processing",
    "state": "running",
    "start_date": "2026-07-21T10:00:05Z",
    "tasks": [
      {"task_id": "validate_batch", "state": "success", "duration_s": 45},
      {"task_id": "cleanup_batch", "state": "success", "duration_s": 120},
      {"task_id": "mask_pii", "state": "running", "start_date": "2026-07-21T10:02:15Z"},
      {"task_id": "auto_annotate", "state": "scheduled"},
      {"task_id": "augment_training_data", "state": "scheduled"}
    ]
  }
}
```

#### List Recent Pipeline Runs
```
GET /api/v1/pipeline/runs/
Role: iot-ml-engineer, iot-admin
Query: ?dag_id=nlu_model_training&state=success&limit=10
```

#### Get Data Quality Report
```
GET /api/v1/pipeline/quality/{batch_id}
Role: iot-ml-engineer, iot-admin
```

**Response (200):**
```json
{
  "status": "success",
  "data": {
    "batch_id": "batch_20260721",
    "validation": {
      "total_records": 1200,
      "passed": 1100,
      "failed": 50,
      "quarantined": 50,
      "pass_rate": 0.917
    },
    "pii_masking": {
      "records_with_pii": 320,
      "pii_types_found": {"MSISDN": 180, "PERSON": 90, "EMAIL": 30, "IMSI": 20},
      "masking_success_rate": 1.0
    },
    "class_distribution": {
      "fetch": 180,
      "Learn": 160,
      "information_seek": 150,
      "Troubleshoot": 80,
      "greet": 40
    },
    "augmentation": {
      "original_count": 1100,
      "augmented_count": 2800,
      "multiplier": 2.5
    }
  }
}
```

### 4.2 Model Registry (`/api/v1/models/`)

#### List Models
```
GET /api/v1/models/
Role: iot-ml-engineer, iot-admin
Query: ?stage=production&sort=-created_at
```

#### Get Model Details
```
GET /api/v1/models/{model_id}
Role: iot-ml-engineer, iot-admin
```

#### Promote Model
```
POST /api/v1/models/{model_id}/promote
Role: iot-admin
```

**Request:**
```json
{
  "target_stage": "production",
  "deployment_strategy": "canary",
  "canary_percentage": 10,
  "rollback_threshold": {
    "error_rate_max": 0.02,
    "latency_p95_max_ms": 300,
    "intent_f1_min": 0.90
  }
}
```

#### Rollback Model
```
POST /api/v1/models/rollback
Role: iot-admin
```

**Request:**
```json
{
  "target_version": "nlu_v2.0.0",
  "reason": "Degraded entity extraction performance"
}
```

#### Compare Models
```
GET /api/v1/models/compare?model_a={id}&model_b={id}
Role: iot-ml-engineer, iot-admin
```

---

## 5. Platform APIs

### 5.1 Authentication (`/api/v1/auth/`)

#### Login (via Keycloak)
```
POST /api/v1/auth/token
[PUBLIC]
```

**Request:**
```json
{
  "grant_type": "password",
  "username": "operator@example.com",
  "password": "***",
  "client_id": "iot-assistant-app",
  "scope": "openid profile"
}
```

**Response (200):**
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "Bearer",
  "expires_in": 300,
  "refresh_expires_in": 1800,
  "scope": "openid profile"
}
```

#### Refresh Token
```
POST /api/v1/auth/refresh
[PUBLIC]
```

#### Logout
```
POST /api/v1/auth/logout
Role: any authenticated
```

#### Get Current User
```
GET /api/v1/auth/me
Role: any authenticated
```

**Response (200):**
```json
{
  "status": "success",
  "data": {
    "sub": "user-uuid",
    "email": "operator@example.com",
    "name": "Jane Operator",
    "roles": ["iot-operator"],
    "permissions": ["subscriptions:read", "inventory:read", "inventory:write"],
    "tenant_id": "tenant_001"
  }
}
```

### 5.2 Health & Metrics

#### Health Check (per service)
```
GET /health/live     [PUBLIC] — Kubernetes liveness probe
GET /health/ready    [PUBLIC] — Kubernetes readiness probe
```

**Response (200):**
```json
{
  "status": "healthy",
  "checks": {
    "mongodb": "up",
    "redis": "up",
    "nlu_model": "loaded",
    "disk_space": "ok"
  },
  "version": "2.1.0",
  "uptime_seconds": 86400
}
```

#### Prometheus Metrics
```
GET /metrics         [INTERNAL] — Prometheus scrape endpoint
```

### 5.3 Admin APIs (`/api/v1/admin/`)

#### System Status Dashboard
```
GET /api/v1/admin/status
Role: iot-admin
```

**Response (200):**
```json
{
  "status": "success",
  "data": {
    "services": {
      "nlu_service": {"status": "healthy", "replicas": 3, "model_version": "v2.1.0"},
      "dialogue_manager": {"status": "healthy", "replicas": 2},
      "subscription_service": {"status": "healthy", "replicas": 2},
      "inventory_service": {"status": "healthy", "replicas": 2},
      "data_pipeline": {"status": "healthy", "active_dags": 4}
    },
    "metrics": {
      "active_conversations": 42,
      "messages_today": 1250,
      "avg_response_time_ms": 180,
      "nlu_confidence_avg": 0.91,
      "fallback_rate": 0.07
    },
    "pipeline": {
      "last_training": "2026-07-14T06:00:00Z",
      "next_scheduled": "2026-07-21T06:00:00Z",
      "training_data_size": 3200
    }
  }
}
```

#### Audit Log
```
GET /api/v1/admin/audit
Role: iot-admin
Query: ?action=inventory.update&actor=operator@example.com&from=2026-07-20&to=2026-07-21
```

#### User Management (delegates to Keycloak)
```
GET /api/v1/admin/users
POST /api/v1/admin/users
PATCH /api/v1/admin/users/{user_id}/roles
DELETE /api/v1/admin/users/{user_id}
Role: iot-admin
```

---

## 6. Rate Limiting

| Endpoint Group | Rate Limit | Window |
|---------------|-----------|--------|
| `/api/v1/conversations/*/messages` | 30 req/min | Per user |
| `/api/v1/nlu/parse` | 60 req/min | Per user |
| `/api/v1/subscriptions/` | 100 req/min | Per user |
| `/api/v1/inventory/` | 100 req/min | Per user |
| `/api/v1/pipeline/trigger` | 5 req/hour | Per user |
| `/api/v1/models/*/promote` | 3 req/day | Global |
| `/api/v1/auth/token` | 10 req/min | Per IP |
| Internal service-to-service | 1000 req/sec | Per service |

---

## 7. Versioning Strategy

- URL-based versioning: `/api/v1/`, `/api/v2/`
- Breaking changes require version bump
- Deprecation: 6-month sunset period with `Sunset` header
- Non-breaking changes (new fields, new optional params) added to current version

---

*Next: See [04_INFRASTRUCTURE.md](./04_INFRASTRUCTURE.md) for deployment architecture.*
