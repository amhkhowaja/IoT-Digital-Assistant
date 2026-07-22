# Infrastructure & Deployment Architecture

> **Version:** 2.0  
> **Date:** 2026-07-21  
> **Platform:** Kubernetes 1.29+  
> **Registry:** JFrog Artifactory  
> **Auth:** Keycloak 24+

---

## 1. Infrastructure Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KUBERNETES CLUSTER                                    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ INGRESS (Traefik / Kong)                                                │ │
│  │ TLS Termination + JWT Validation + Rate Limiting                        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐ │
│  │ iot-converse  │  │ iot-business  │  │ iot-pipeline  │  │ iot-platform│ │
│  │ namespace     │  │ namespace     │  │ namespace     │  │ namespace   │ │
│  ├───────────────┤  ├───────────────┤  ├───────────────┤  ├─────────────┤ │
│  │ NLU Svc (x3) │  │ Sub Svc (x2) │  │ Airflow Sched │  │ Keycloak    │ │
│  │ Dialogue (x2)│  │ Inv Svc (x2) │  │ Airflow Work. │  │ MongoDB RS  │ │
│  │ Actions (x2) │  │ Onboard (x2) │  │ ML Training   │  │ Redis Sent. │ │
│  │ Channel (x2) │  │ Knowledge(x2)│  │ Model Reg.    │  │ RabbitMQ    │ │
│  │              │  │ Notif. (x2)  │  │               │  │ Vault       │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └─────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ iot-observability namespace                                              │ │
│  │ Prometheus + Grafana + ELK + Jaeger + Alertmanager                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    JFrog ARTIFACTORY (External)                               │
│   Docker Registry │ PyPI Repository │ Helm Repository │ Generic Artifacts    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. JFrog Artifactory — Artifact Management

### 2.1 Repository Structure

| Repository | Type | Purpose | Retention |
|-----------|------|---------|-----------|
| `iot-docker-local` | Docker | Production container images | Last 10 versions |
| `iot-docker-dev` | Docker | Development/feature images | 30 days |
| `iot-pypi-local` | PyPI | Internal Python packages | Permanent |
| `iot-pypi-remote` | PyPI (remote) | Proxy for PyPI.org (cached) | 90 days cache |
| `iot-helm-local` | Helm | Helm charts for all services | Last 10 versions |
| `iot-generic-local` | Generic | ML models, training data snapshots | Governed by ML registry |
| `iot-generic-releases` | Generic | Release bundles | Permanent |

### 2.2 Image Naming Convention

```
jfrog.example.com/iot-docker-local/<service-name>:<version>

Examples:
jfrog.example.com/iot-docker-local/iot-nlu-service:2.1.0
jfrog.example.com/iot-docker-local/iot-nlu-service:2.1.0-rc1
jfrog.example.com/iot-docker-local/iot-inventory-service:1.3.2
jfrog.example.com/iot-docker-local/iot-data-pipeline:2.0.0
```

### 2.3 Versioning Scheme

```
<major>.<minor>.<patch>[-<pre-release>][+<build-metadata>]

Semantic Versioning:
- MAJOR: Breaking API/contract changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, security patches

Branch → Tag Mapping:
- main     → X.Y.Z (release)
- develop  → X.Y.Z-dev.<build>
- feature/ → X.Y.Z-feature.<name>.<build>
- hotfix/  → X.Y.Z-hotfix.<build>
```

### 2.4 JFrog Build Integration

```yaml
# Jenkinsfile / GitHub Actions integration
stages:
  - lint_and_test:
      - ruff check .
      - mypy .
      - pytest --cov=src
      - bandit -r src/

  - build_image:
      - docker build -t $IMAGE_NAME:$VERSION .
      - trivy image --exit-code 1 --severity HIGH,CRITICAL $IMAGE_NAME:$VERSION

  - push_to_jfrog:
      - docker tag $IMAGE_NAME:$VERSION jfrog.example.com/iot-docker-local/$IMAGE_NAME:$VERSION
      - docker push jfrog.example.com/iot-docker-local/$IMAGE_NAME:$VERSION
      - jfrog rt upload "dist/*.whl" iot-pypi-local/  # Python packages

  - publish_helm:
      - helm package charts/$SERVICE_NAME --version $VERSION
      - helm push $SERVICE_NAME-$VERSION.tgz oci://jfrog.example.com/iot-helm-local

  - scan_artifacts:
      - jfrog xray scan --watches "iot-security-watch"
```

### 2.5 ML Model Artifacts in JFrog

```
iot-generic-local/
├── models/
│   ├── nlu/
│   │   ├── nlu_v2.1.0_20260721.tar.gz        (model archive)
│   │   ├── nlu_v2.1.0_20260721.metadata.json  (metrics, config)
│   │   └── nlu_v2.1.0_20260721.report.html    (evaluation report)
│   └── dialogue/
│       └── dialogue_v1.5.0_20260721.tar.gz
├── training-data/
│   ├── snapshots/
│   │   └── training_data_20260721.tar.gz
│   └── augmented/
│       └── augmented_data_20260721.tar.gz
└── configs/
    ├── pipeline_config_v2.1.0.yml
    └── domain_v2.1.0.yml
```

### 2.6 Xray Security Scanning

```yaml
# JFrog Xray policies
policies:
  - name: "iot-security-policy"
    rules:
      - name: "block-critical-cves"
        criteria:
          min_severity: "Critical"
        actions:
          block_download: true
          notify:
            - "security-team@example.com"

      - name: "warn-high-cves"
        criteria:
          min_severity: "High"
        actions:
          block_download: false
          notify:
            - "dev-team@example.com"

      - name: "block-banned-licenses"
        criteria:
          banned_licenses: ["GPL-3.0", "AGPL-3.0"]
        actions:
          block_download: true
```

---

## 3. Keycloak — Authentication & Authorization

### 3.1 Realm Configuration

```
Realm: iot-assistant
├── Clients
│   ├── iot-assistant-web       (public, PKCE flow — frontend SPA)
│   ├── iot-assistant-mobile    (public, PKCE flow — mobile app)
│   ├── iot-assistant-api       (confidential — service accounts)
│   ├── iot-nlu-service         (confidential — service-to-service)
│   ├── iot-dialogue-manager    (confidential — service-to-service)
│   ├── iot-subscription-svc   (confidential — service-to-service)
│   ├── iot-inventory-svc      (confidential — service-to-service)
│   ├── iot-pipeline-svc       (confidential — service-to-service)
│   └── iot-admin-portal       (confidential — admin UI)
├── Roles (Realm)
│   ├── iot-user
│   ├── iot-operator
│   ├── iot-admin
│   ├── iot-ml-engineer
│   └── iot-system
├── Groups
│   ├── /operators
│   ├── /admins
│   ├── /ml-team
│   └── /service-accounts
├── Identity Providers
│   ├── Corporate LDAP/AD (user federation)
│   └── SAML IdP (SSO)
└── Authentication Flows
    ├── Browser flow (MFA for admin)
    ├── Direct grant (service accounts)
    └── Client credentials (service-to-service)
```

### 3.2 Role Permission Matrix

| Permission | iot-user | iot-operator | iot-admin | iot-ml-engineer |
|-----------|----------|--------------|-----------|-----------------|
| Chat (own conversations) | ✅ | ✅ | ✅ | ✅ |
| View own subscriptions | ✅ | ✅ | ✅ | — |
| View all subscriptions | — | ✅ | ✅ | — |
| Search CPI documents | ✅ | ✅ | ✅ | ✅ |
| Inventory read | — | ✅ | ✅ | — |
| Inventory write | — | ✅ | ✅ | — |
| Onboard customers | — | — | ✅ | — |
| Manage users | — | — | ✅ | — |
| Trigger pipelines | — | — | ✅ | ✅ |
| View training data | — | — | — | ✅ |
| Deploy models | — | — | ✅ | ✅ |
| View audit logs | — | — | ✅ | — |
| System configuration | — | — | ✅ | — |

### 3.3 Token Configuration

```yaml
# Keycloak token settings
access_token:
  lifespan: 300          # 5 minutes
  algorithm: RS256
  claims:
    - sub              # User ID
    - email
    - preferred_username
    - realm_access.roles
    - resource_access
    - tenant_id        # Custom claim for multi-tenancy

refresh_token:
  lifespan: 1800        # 30 minutes
  idle_timeout: 600     # 10 minutes idle

service_account_token:
  lifespan: 3600        # 1 hour (long-lived for batch operations)
```

### 3.4 Service-to-Service Authentication

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│ Action Server│─── mTLS ──▶│ Subscription │         │   Keycloak   │
│              │    +JWT   │   Service    │         │              │
└──────────────┘         └──────────────┘         └──────────────┘
        │                                                   ▲
        │  1. Client credentials grant                      │
        └───────────────────────────────────────────────────┘
           (cached, refreshed before expiry)
```

```python
# Service-to-service auth pattern
from authlib.integrations.httpx_client import AsyncOAuth2Client

class ServiceClient:
    def __init__(self, service_url: str, client_id: str, client_secret: str):
        self.oauth = AsyncOAuth2Client(
            client_id=client_id,
            client_secret=client_secret,
            token_endpoint=f"{KEYCLOAK_URL}/realms/iot-assistant/protocol/openid-connect/token"
        )
        self._token = None

    async def get_token(self):
        if not self._token or self._token.is_expired():
            self._token = await self.oauth.fetch_token(
                grant_type="client_credentials"
            )
        return self._token["access_token"]

    async def call(self, method: str, path: str, **kwargs):
        token = await self.get_token()
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(verify=True) as client:
            return await client.request(method, f"{self.service_url}{path}", headers=headers, **kwargs)
```

### 3.5 JWT Validation Middleware

```python
# FastAPI middleware for all business services
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer
from jose import jwt, JWTError
import httpx

security = HTTPBearer()

class JWTAuth:
    def __init__(self):
        self._jwks = None

    async def get_jwks(self):
        """Fetch and cache Keycloak public keys."""
        if not self._jwks:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{KEYCLOAK_URL}/realms/iot-assistant/protocol/openid-connect/certs"
                )
                self._jwks = resp.json()
        return self._jwks

    async def validate_token(self, token: str = Security(security)) -> dict:
        """Validate JWT and extract claims."""
        try:
            jwks = await self.get_jwks()
            payload = jwt.decode(
                token.credentials,
                jwks,
                algorithms=["RS256"],
                audience="iot-assistant-api",
                issuer=f"{KEYCLOAK_URL}/realms/iot-assistant"
            )
            return payload
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

def require_role(role: str):
    """Dependency to require specific Keycloak role."""
    async def check_role(claims: dict = Depends(jwt_auth.validate_token)):
        roles = claims.get("realm_access", {}).get("roles", [])
        if role not in roles:
            raise HTTPException(status_code=403, detail=f"Requires role: {role}")
        return claims
    return check_role
```

---

## 4. MongoDB — Data Store Architecture

### 4.1 Cluster Topology

```
MongoDB Replica Set: iot-rs0
├── Primary   (iot-mongodb-0) — Read/Write
├── Secondary (iot-mongodb-1) — Read replicas (analytics)
└── Secondary (iot-mongodb-2) — Read replicas (backup)

Connection String:
mongodb://iot-user:***@iot-mongodb-0:27017,iot-mongodb-1:27017,iot-mongodb-2:27017/
  ?replicaSet=iot-rs0&authSource=admin&tls=true&readPreference=secondaryPreferred
```

### 4.2 Database & Collection Design

```
Database: iot_conversations
├── tracker_store              (Rasa conversation state)
│   Indexes: sender_id, timestamp
│   TTL: 90 days (auto-cleanup old conversations)
└── user_sessions
    Indexes: user_id, expires_at
    TTL: 24 hours

Database: iot_business
├── subscriptions
│   Indexes: imsi (unique), msisdn, sim_subscription_state
│   Schema validation: enforced
├── inventory
│   Indexes: msisdn (unique), billing_state, network_connectivity
│   Schema validation: enforced
├── customers
│   Indexes: enterprise_agreement_number (unique), customer_type
│   Schema validation: enforced
├── cpi_documents
│   Indexes: intent+sub_entities (compound), text (search index)
│   Full-text search enabled
└── audit_log
    Indexes: timestamp, actor, resource_type
    TTL: 365 days
    Capped: false (important for compliance)

Database: iot_pipeline
├── raw_training_data
│   Indexes: batch_id, source, timestamp
│   TTL: 90 days
├── processed_data
│   Indexes: batch_id, stage, intent
├── model_metadata
│   Indexes: model_id (unique), stage, created_at
└── pipeline_runs
    Indexes: dag_run_id, dag_id, state
```

### 4.3 Schema Validation Example

```javascript
// MongoDB schema validation for subscriptions collection
db.createCollection("subscriptions", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["imsi", "msisdn", "sim_subscription_state", "sim_status"],
      properties: {
        imsi: {
          bsonType: "string",
          pattern: "^[0-9]{15}$",
          description: "15-digit IMSI number"
        },
        msisdn: {
          bsonType: "string",
          pattern: "^[0-9]{10,15}$",
          description: "10-15 digit MSISDN"
        },
        sim_subscription_state: {
          enum: ["active", "inactive", "suspended", "terminated"],
          description: "Current SIM state"
        },
        sim_status: {
          enum: ["enabled", "disabled", "locked"],
          description: "Physical SIM status"
        },
        plan: {
          bsonType: "object",
          properties: {
            name: { bsonType: "string" },
            data_limit_mb: { bsonType: "int", minimum: 0 },
            billing_cycle: { enum: ["monthly", "yearly", "prepaid"] }
          }
        },
        created_at: { bsonType: "date" },
        updated_at: { bsonType: "date" },
        updated_by: { bsonType: "string" }
      }
    }
  }
});
```

### 4.4 MongoDB Security Configuration

```yaml
# MongoDB configuration
security:
  authorization: enabled
  keyFile: /etc/mongodb/keyfile         # Replica set auth
  enableEncryption: true
  encryptionKeyFile: /etc/mongodb/enc-key

net:
  tls:
    mode: requireTLS
    certificateKeyFile: /etc/mongodb/tls/server.pem
    CAFile: /etc/mongodb/tls/ca.pem

# Users and roles
users:
  - username: iot_app_user
    roles: [readWrite@iot_business, readWrite@iot_conversations]
  - username: iot_pipeline_user
    roles: [readWrite@iot_pipeline, read@iot_business]
  - username: iot_readonly
    roles: [read@iot_business, read@iot_conversations]
  - username: iot_admin
    roles: [dbAdmin@iot_business, dbAdmin@iot_conversations, dbAdmin@iot_pipeline]
```

### 4.5 Backup Strategy

| Type | Frequency | Retention | Tool |
|------|-----------|-----------|------|
| Continuous oplog backup | Real-time | 7 days | MongoDB Backup Agent |
| Full snapshot | Daily 03:00 UTC | 30 days | mongodump + MinIO |
| Cross-region replication | Continuous | Live | MongoDB Atlas / manual |
| Point-in-time recovery | Any point within oplog window | 7 days | oplog replay |

---

## 5. Container Architecture

### 5.1 Base Dockerfile Pattern

```dockerfile
# Multi-stage build for all Python services
# Stage 1: Builder
FROM python:3.11-slim AS builder
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry export -f requirements.txt --without-hashes | pip install -r /dev/stdin

# Stage 2: Runtime
FROM python:3.11-slim AS runtime
WORKDIR /app

# Security: non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy only runtime dependencies
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Security hardening
RUN chmod -R 555 /app && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health/live')" || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "src.main:app"]
CMD ["--host", "0.0.0.0", "--port", "5020", "--workers", "2"]
```

### 5.2 Docker Compose (Local Development)

```yaml
# docker-compose.yml — Local development environment
version: "3.9"

services:
  # === CONVERSATION LAYER ===
  nlu-service:
    build: ./services/nlu-service
    ports: ["5010:5010"]
    environment:
      - REDIS_URL=redis://redis:6379/0
      - MODEL_PATH=/models/current
    volumes:
      - ./models:/models:ro
    depends_on: [redis]

  dialogue-manager:
    build: ./services/dialogue-manager
    ports: ["5011:5011"]
    environment:
      - MONGODB_URI=mongodb://root:password@mongodb:27017/?authSource=admin
      - MONGODB_DB=iot_conversations
      - NLU_SERVICE_URL=http://nlu-service:5010
      - ACTION_SERVER_URL=http://action-server:5055
    depends_on: [mongodb, nlu-service, action-server]

  action-server:
    build: ./services/action-server
    ports: ["5055:5055"]
    environment:
      - SUBSCRIPTION_SERVICE_URL=http://subscription-service:5020
      - INVENTORY_SERVICE_URL=http://inventory-service:5021
      - KNOWLEDGE_SERVICE_URL=http://knowledge-service:5023
      - KEYCLOAK_URL=http://keycloak:8080
    depends_on: [subscription-service, inventory-service]

  # === BUSINESS LAYER ===
  subscription-service:
    build: ./services/subscription-service
    ports: ["5020:5020"]
    environment:
      - MONGODB_URI=mongodb://root:password@mongodb:27017/?authSource=admin
      - MONGODB_DB=iot_business
      - KEYCLOAK_URL=http://keycloak:8080
    depends_on: [mongodb, keycloak]

  inventory-service:
    build: ./services/inventory-service
    ports: ["5021:5021"]
    environment:
      - MONGODB_URI=mongodb://root:password@mongodb:27017/?authSource=admin
      - MONGODB_DB=iot_business
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672
    depends_on: [mongodb, rabbitmq]

  onboarding-service:
    build: ./services/onboarding-service
    ports: ["5022:5022"]
    environment:
      - MONGODB_URI=mongodb://root:password@mongodb:27017/?authSource=admin
      - MONGODB_DB=iot_business
    depends_on: [mongodb]

  knowledge-service:
    build: ./services/knowledge-service
    ports: ["5023:5023"]
    environment:
      - MONGODB_URI=mongodb://root:password@mongodb:27017/?authSource=admin
      - ELASTICSEARCH_URL=http://elasticsearch:9200
    depends_on: [mongodb, elasticsearch]

  # === PLATFORM LAYER ===
  mongodb:
    image: mongo:7.0
    ports: ["27017:27017"]
    environment:
      MONGO_INITDB_ROOT_USERNAME: root
      MONGO_INITDB_ROOT_PASSWORD: password
    volumes:
      - mongodb_data:/data/db
      - ./scripts/mongo-init.js:/docker-entrypoint-initdb.d/init.js:ro

  redis:
    image: redis:7.2-alpine
    ports: ["6379:6379"]
    command: redis-server --requirepass password

  rabbitmq:
    image: rabbitmq:3.13-management
    ports: ["5672:5672", "15672:15672"]
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest

  keycloak:
    image: quay.io/keycloak/keycloak:24.0
    ports: ["8080:8080"]
    environment:
      KEYCLOAK_ADMIN: admin
      KEYCLOAK_ADMIN_PASSWORD: admin
    command: start-dev --import-realm
    volumes:
      - ./config/keycloak/realm-export.json:/opt/keycloak/data/import/realm.json:ro

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
    ports: ["9200:9200"]
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"

  # === PIPELINE (optional for local dev) ===
  airflow-webserver:
    image: apache/airflow:2.8.0-python3.11
    ports: ["8081:8080"]
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
    volumes:
      - ./dags:/opt/airflow/dags
    depends_on: [postgres]
    profiles: ["pipeline"]

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    profiles: ["pipeline"]

volumes:
  mongodb_data:
```

---

## 6. Kubernetes Deployment

### 6.1 Helm Chart Structure

```
helm/
├── iot-assistant/                   (umbrella chart)
│   ├── Chart.yaml
│   ├── values.yaml                  (default values)
│   ├── values-dev.yaml              (dev overrides)
│   ├── values-staging.yaml          (staging overrides)
│   ├── values-prod.yaml             (production overrides)
│   └── charts/
│       ├── iot-nlu-service/
│       ├── iot-dialogue-manager/
│       ├── iot-action-server/
│       ├── iot-subscription-service/
│       ├── iot-inventory-service/
│       ├── iot-onboarding-service/
│       ├── iot-knowledge-service/
│       ├── iot-notification-service/
│       ├── iot-data-pipeline/
│       └── iot-model-registry/
├── iot-platform/                    (platform dependencies)
│   ├── mongodb/
│   ├── redis/
│   ├── rabbitmq/
│   ├── keycloak/
│   └── observability/
└── iot-ingress/
    └── traefik/
```

### 6.2 Service Deployment Example

```yaml
# helm/iot-assistant/charts/iot-nlu-service/templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iot-nlu-service
  namespace: iot-conversation
  labels:
    app.kubernetes.io/name: iot-nlu-service
    app.kubernetes.io/version: "{{ .Values.image.tag }}"
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: iot-nlu-service
  template:
    metadata:
      labels:
        app: iot-nlu-service
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "5010"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: iot-nlu-service
      securityContext:
        runAsNonRoot: true
        fsGroup: 1000
      containers:
        - name: nlu-service
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 5010
              protocol: TCP
          env:
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: iot-redis-credentials
                  key: url
            - name: MODEL_VERSION
              value: "{{ .Values.model.version }}"
          envFrom:
            - configMapRef:
                name: iot-nlu-config
          resources:
            requests:
              cpu: "{{ .Values.resources.requests.cpu }}"
              memory: "{{ .Values.resources.requests.memory }}"
            limits:
              cpu: "{{ .Values.resources.limits.cpu }}"
              memory: "{{ .Values.resources.limits.memory }}"
          livenessProbe:
            httpGet:
              path: /health/live
              port: 5010
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 5010
            initialDelaySeconds: 10
            periodSeconds: 5
          volumeMounts:
            - name: model-volume
              mountPath: /models
              readOnly: true
      volumes:
        - name: model-volume
          persistentVolumeClaim:
            claimName: iot-nlu-models
      imagePullSecrets:
        - name: jfrog-registry-secret
```

### 6.3 Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: iot-nlu-service-hpa
  namespace: iot-conversation
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: iot-nlu-service
  minReplicas: 2
  maxReplicas: 6
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: nlu_inference_latency_p95
        target:
          type: AverageValue
          averageValue: "200m"  # 200ms
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120
```

---

## 7. CI/CD Pipeline

### 7.1 Pipeline Architecture

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  COMMIT  │───▶│   LINT   │───▶│  BUILD   │───▶│  TEST    │───▶│  SCAN    │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                                       │
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  PROD    │◀───│ STAGING  │◀───│   DEV    │◀───│ PUBLISH  │◀─────────┘
│ (manual) │    │ (auto)   │    │ (auto)   │    │ (JFrog)  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### 7.2 GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  JFROG_REGISTRY: jfrog.example.com/iot-docker-local
  HELM_REGISTRY: oci://jfrog.example.com/iot-helm-local

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install ruff mypy bandit
      - run: ruff check .
      - run: mypy src/ --strict
      - run: bandit -r src/ -c pyproject.toml

  test:
    needs: lint
    runs-on: ubuntu-latest
    services:
      mongodb: { image: "mongo:7.0", ports: ["27017:27017"] }
      redis: { image: "redis:7.2-alpine", ports: ["6379:6379"] }
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[test]"
      - run: pytest --cov=src --cov-report=xml --cov-fail-under=80
      - uses: codecov/codecov-action@v4

  build-and-scan:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t $JFROG_REGISTRY/${{ matrix.service }}:${{ github.sha }} .
        working-directory: services/${{ matrix.service }}
      - run: trivy image --exit-code 1 --severity HIGH,CRITICAL $IMAGE
      - run: docker push $JFROG_REGISTRY/${{ matrix.service }}:${{ github.sha }}

  deploy-dev:
    needs: build-and-scan
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    steps:
      - run: |
          helm upgrade --install iot-assistant ./helm/iot-assistant \
            --namespace iot-dev \
            --values helm/iot-assistant/values-dev.yaml \
            --set global.image.tag=${{ github.sha }}
```

### 7.3 ArgoCD Application

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: iot-assistant-prod
  namespace: argocd
spec:
  project: iot-assistant
  source:
    repoURL: https://github.com/org/iot-digital-assistant.git
    targetRevision: main
    path: helm/iot-assistant
    helm:
      valueFiles:
        - values-prod.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: iot-production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

---

## 8. Network & Security Policies

### 8.1 Network Policies

```yaml
# Only allow conversation services to talk to business services
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-action-to-business
  namespace: iot-business
spec:
  podSelector:
    matchLabels: {}
  policyTypes: [Ingress]
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: iot-conversation
        - namespaceSelector:
            matchLabels:
              name: iot-ingress
      ports:
        - protocol: TCP
          port: 5020
        - protocol: TCP
          port: 5021
        - protocol: TCP
          port: 5022
        - protocol: TCP
          port: 5023
```

### 8.2 Secret Management with Vault

```yaml
# Vault Secrets Operator — inject secrets into pods
apiVersion: secrets.hashicorp.com/v1beta1
kind: VaultStaticSecret
metadata:
  name: iot-mongodb-credentials
  namespace: iot-business
spec:
  vaultAuthRef: vault-auth
  mount: secret
  path: iot-assistant/mongodb
  type: kv-v2
  destination:
    name: iot-mongodb-credentials
    create: true
  refreshAfter: 1h
```

---

*Next: See [05_SEQUENCE_DIAGRAMS.md](./05_SEQUENCE_DIAGRAMS.md) for system flow diagrams.*
