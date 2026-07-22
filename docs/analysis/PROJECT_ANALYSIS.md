# IoT Digital Assistant — Comprehensive Project Analysis

> **Analysis Date:** 2026-07-21  
> **Project Origin:** 2023 (intern exhibition/demo project)  
> **Purpose:** Deep audit to inform a full renewal/modernization effort  
> **Verdict:** Not production-ready. Requires ground-up rearchitecture.

---

## Executive Summary

| Dimension | Score | Status |
|-----------|-------|--------|
| Dependency Health | 1/10 | 🔴 7 EOL packages, 60+ CVEs |
| Code Quality | 2/10 | 🔴 Runtime bugs, no patterns, hardcoded everything |
| Architecture | 2/10 | 🔴 No separation of concerns, no layers |
| Security | 1/10 | 🔴 Hardcoded API key, NoSQL injection, exposed PINs |
| Testing | 1/10 | 🔴 Near-zero coverage, no CI gates |
| Build System | 0/10 | 🔴 Completely absent |
| CI/CD | 0/10 | 🔴 No pipeline exists |
| Git Health | 2/10 | 🔴 1.75 GB repo, massive binaries in history |
| NLU Data Quality | 3/10 | 🟡 Severe class imbalance, broken annotations |
| Integration Design | 2/10 | 🔴 No timeouts, no pooling, hardcoded URIs |
| Model Performance | 6/10 | 🟡 Good numbers but statistically unreliable |
| Documentation | 1/10 | 🔴 2-line readme, zero docstrings |

**Overall: 1.75/10** — The project served its purpose as a demo but cannot be used as a foundation for production. A controlled rewrite with proper engineering practices is the only viable path forward.

---

## Table of Contents

1. [Dependency Audit](#1-dependency-audit)
2. [Security Vulnerabilities](#2-security-vulnerabilities)
3. [Code Quality & Architecture](#3-code-quality--architecture)
4. [NLU Training Data Quality](#4-nlu-training-data-quality)
5. [Dialogue Management](#5-dialogue-management)
6. [Testing Strategy](#6-testing-strategy)
7. [Build System & CI/CD](#7-build-system--cicd)
8. [Git Repository Health](#8-git-repository-health)
9. [Integration & External Services](#9-integration--external-services)
10. [Configuration Management](#10-configuration-management)
11. [Renewal Roadmap](#11-renewal-roadmap)

---

## 1. Dependency Audit

### Critical EOL/Deprecated Packages

| Package | Pinned Version | Current (2026) | Status | CVE Count |
|---------|---------------|----------------|--------|-----------|
| rasa | 3.4.4 | Rasa Pro 4.x+ | ☠️ EOL (2024) | Multiple |
| tensorflow | 2.8.4 | 2.18+ | ☠️ EOL (Nov 2022) | 40+ CVEs |
| pymongo | 3.10.1 | 4.9+ | ☠️ EOL (3.x deprecated) | Missing TLS defaults |
| cx_Oracle | (imported) | python-oracledb 2.x | ☠️ Renamed/deprecated (2022) | — |
| urllib3 | 1.26.14 | 2.3+ | ☠️ EOL | CVE-2023-43804/45803 |
| SQLAlchemy | 1.4.46 | 2.1+ | ☠️ EOL (Jan 2024) | — |
| protobuf | 3.19.6 | 5.x+ | ☠️ EOL | CVE-2022-3171 |

### Significantly Outdated Packages

| Package | Pinned | Current | Key Vulnerabilities |
|---------|--------|---------|---------------------|
| cryptography | 39.0.1 | 43+ | CVE-2023-23931 (memory corruption) |
| requests | 2.28.1 | 2.32+ | CVE-2023-32681 (header leak) |
| Pillow | 9.4.0 | 11+ | CVE-2023-44271 (DoS), buffer overflows |
| sanic | 21.12.2 | 24+ | Request smuggling fixes |
| Werkzeug | 2.2.2 | 3.1+ | CVE-2023-25577 (DoS), CVE-2023-46136 |
| redis | 4.5.1 | 5.2+ | CVE-2023-28858/28859 (data leak) |
| confluent-kafka | 1.9.2 | 2.6+ | Multiple security fixes |
| certifi | 2022.12.7 | 2024+ | Missing CA revocations |
| PyJWT | 2.4.0 | 2.9+ | Algorithm confusion risks |
| numpy | 1.23.5 | 2.1+ | Compatibility issues |
| scipy | 1.8.1 | 1.14+ | CVE-2023-25399 (DoS) |

### Requirements File Chaos

The project has **3 separate requirements files** with no clear purpose delineation:

- `requirements.txt` (2.9KB) — appears to be the "main" file
- `requirements2.txt` (5.6KB) — pip freeze dump with conda `file://` paths (non-portable)
- `requirements_rasa_iot.txt` (3.2KB) — another variant

**Issues:**
- No single source of truth for dependencies
- `requirements2.txt` contains absolute local file paths — will not work on any other machine
- No `pyproject.toml`, `setup.py`, or `Pipfile` — no modern Python packaging
- No lock file for reproducible builds

---

## 2. Security Vulnerabilities

### 🔴 CRITICAL

#### Hardcoded API Key
```python
# actions/actions.py — ActionNewsFetch class
query_params = {
    "apiKey": "4dbc17e007ab436fb66416009dfb59a8"  # ← Committed to Git
}
```
- NewsAPI key exposed in source control
- Key likely still active and abusable
- **Remediation:** Rotate immediately, move to environment variable

#### NoSQL Injection (2 instances)
```python
# ActionFetchInventory — user input passed directly to $regex
filtered_entities = {
    k: {'$regex': '^' + v + '$', '$options': 'i'}
    for k, v in entities.items() if v is not None
}
```
- User-controlled entity values injected into MongoDB regex operators
- Enables ReDoS (regex denial of service) and filter bypass
- **Remediation:** Use `re.escape()` or exact match instead of `$regex`

### 🟠 HIGH

| Vulnerability | Location | Impact |
|--------------|----------|--------|
| Sensitive data exposure (PIN/PUK in plain text) | ActionIMSIStats | SIM credentials leaked via chat |
| Unencrypted MongoDB connection (no TLS, no auth) | All actions + endpoints.yml | Data in transit vulnerable |
| Unvalidated DB field access | ActionUpdateInventory | Arbitrary field updates possible |
| Commented credentials in source | actions.py line ~35 | `mongodb://admin:password@...` visible |
| Action endpoint without auth | endpoints.yml | Anyone can invoke custom actions |

### 🟡 MEDIUM

| Vulnerability | Location | Impact |
|--------------|----------|--------|
| Bare `except:` clauses (6 instances) | actions.py throughout | Masks security incidents |
| Debug `print()` with user data | actions.py | Sensitive data in logs |
| No request timeout on NewsAPI call | ActionNewsFetch | Indefinite hang |
| Empty action endpoint token | endpoints.yml | No auth between services |
| REST channel without authentication | credentials.yml | Open bot access |

### .gitignore Gaps

Current `.gitignore` (5 lines):
```
.rasa
models
actions/__pycache__
*.jpg
*.png
```

**Missing critical patterns:**
```
.env / .env.*
venv/ / .venv/ / env/
*.pyc / __pycache__/ (global)
*.pem / *.key / *.cert
.idea/ / .vscode/
*.db / *.sqlite3
credentials.yml (when populated)
*.log
mongo-volume/
node_modules/
.DS_Store
dist/ / build/
```

---

## 3. Code Quality & Architecture

### Runtime Bugs (will crash in production)

| Bug | Location | Impact |
|-----|----------|--------|
| Undefined variable `entity_value_1` | ActionCPIlink, line ~50 | `NameError` crash |
| Typo `dispatcher.uttter_message()` | ValidateEnterpriseForm (2 instances) | `AttributeError` crash |
| Missing `return []` | SubmitOnboardingForm.run() | Returns None instead of events list |
| Duplicate story names | stories.yml | Undefined Rasa training behavior |

### Code Duplication (DRY violations)

MongoDB connection boilerplate repeated **6 times**:
```python
client = MongoClient("mongodb://localhost:27017")
db = client["IOTA"]
collection = db["<name>"]
```

Entity extraction pattern repeated **4 times**:
```python
prediction = tracker.latest_message
current_entity_1 = prediction['entities'][0]['entity']
entity_value = next(tracker.get_latest_entity_values(current_entity_1), None)
```

### SOLID Violations

| Principle | Status | Evidence |
|-----------|--------|----------|
| Single Responsibility | ❌ VIOLATED | Each action handles: DB connection + entity extraction + business logic + response formatting |
| Open/Closed | ❌ VIOLATED | Adding a new DB requires editing every action class |
| Dependency Inversion | ❌ VIOLATED | All actions directly instantiate MongoClient |
| Interface Segregation | ✅ OK | Rasa SDK enforces minimal interface |

### Architecture — What Exists vs What Should Exist

**Current (flat, no layers):**
```
actions/
└── actions.py (17KB, 450 lines, 8 classes, everything mixed)
```

**Required (layered):**
```
actions/
├── __init__.py
├── cpi/
│   └── actions.py
├── inventory/
│   └── actions.py
├── subscription/
│   └── actions.py
├── onboarding/
│   └── actions.py
├── common/
│   ├── database.py      (connection pool, config-driven)
│   ├── entity_utils.py  (extraction helpers)
│   └── validators.py    (input sanitization)
└── config.py            (env-based configuration)
```

### Naming Inconsistencies

```
ActionCPIlink      vs  ActionIMSIStats       (casing)
action_CPI_link    vs  action_IMSI_Stats     (underscore style)
action_fetch_inventory vs action_update_inventory (OK)
ValidateCustomerTypeForm vs SubmitOnboardingForm  (verb position)
```

### Dead Code

- `cx_Oracle` imported but never used
- `databases.py` — empty stub with SQL in a NoSQL context
- Commented-out Oracle SQL queries in actions.py
- Entire `Chatbot/NLU/` directory — legacy parallel NLU system (BiLSTM + spaCy custom NER), completely disconnected from the Rasa pipeline
- Commented-out response payload blocks (data2 in ActionCPIlink)

---

## 4. NLU Training Data Quality

### Class Imbalance (Critical)

```
Intent                                          | Examples | Status
================================================|==========|===============
Learn                                           | ~130     | OVERPRESENTED
information_seek                                | ~120     | OVERPRESENTED
fetch                                           | ~100     | OVERPRESENTED
greet                                           | ~60      | High
enterprise_name                                 | ~50      | High
Troubleshoot                                    | ~45      | OK
information_Comparison                          | ~35      | OK
mood_great, mood_unhappy, affirm                | ~20-27   | OK
onboard_new_customer, news_fetch                | ~20      | OK
goodbye, deny, ask_hru, describe_bot            | ~14-20   | Low
bot_challenge, bot_name                         | ~14      | Low
update_inventory                                | ~15      | Low
ask_joke, thank                                 | ~8-10    | Very Low
customer_enterprise                             | ~5       | CRITICAL
parent_organization_china_telecom_mongolia_branch| ~8      | CRITICAL (also bad intent design)
customer_advanced_reseller                      | ~1       | UNUSABLE
bot_where, ask_bot_question, data_enter         | ~2-3     | UNUSABLE
weather_info, bot_age, troubleshooting_time     | ~1       | UNUSABLE
```

**7 intents have ≤3 examples** — ML cannot learn from this. Results are random chance.

### Annotation Quality Issues

1. **Broken annotations:** `[[enabled](toggle)](toggle)` — nested brackets, parser may fail
2. **Typos in entity names:** `"inventory_atttribute"` (triple 't')
3. **Data-as-entity-value:** MSISDN numbers used as synonyms (non-generalizable)
4. **Role confusion:** `plan_name` with role "update_value" contains MSISDN value "77788899911"
5. **Trailing garbage** in training examples
6. **Inconsistent entity labeling:** Same concept annotated as different entity types across examples

### Missing NLU Components

| Component | Impact | Recommendation |
|-----------|--------|----------------|
| RegexEntityExtractor | IMSI/agreement numbers have clear patterns | Add regex for `\d{15}` (IMSI), `\d{14}` (agreement) |
| Lookup tables | Enterprise names, plan names are finite sets | Add lookup tables |
| FallbackClassifier | No graceful degradation for unknown input | Critical addition |
| Synonym standardization | Inconsistent entity values | Clean up entity synonyms |

### Total Training Volume

~900-1000 examples across 31 intents = **~30 examples per intent average**. This is below the recommended minimum of 50-100 per intent for production Rasa models with entity extraction.

---

## 5. Dialogue Management

### Story Issues

- **Duplicate story names:** `fetching_inventory3`, `interactive_story_1`, `msisdn cannot be updated story` — each defined twice with potentially different content
- **No fallback story** — bot has no defined behavior for unrecognized input
- **No error recovery stories** — what happens when DB is down?
- **No session timeout handling** — conversations that resume after hours
- **No multi-turn clarification** — bot never asks "did you mean X or Y?"

### Policy Configuration Gaps

| Missing | Impact |
|---------|--------|
| FallbackClassifier + fallback action | Bot always picks an intent, even for gibberish |
| AugmentedMemoizationPolicy | No story generalization |
| Confidence thresholds | Defaults may cause unpredictable behavior |
| Early stopping | 100 epochs with small data → overfitting risk |

### Rule Coverage

8 rules cover: goodbye, bot_challenge, CPI link, thanks, and form lifecycle. Missing:
- nlu_fallback rule
- out-of-scope rule
- session_start rule
- action failure recovery rule

---

## 6. Testing Strategy

### Current State

| Test Type | Coverage | Status |
|-----------|----------|--------|
| NLU intent tests | 7 chitchat stories only | ❌ No domain feature tests |
| Entity extraction tests | None | ❌ |
| Action unit tests | None | ❌ |
| Integration tests (action↔DB) | None | ❌ |
| End-to-end conversation tests | None | ❌ |
| Load/stress tests | None | ❌ |
| Security tests | None | ❌ |
| Regression tests | None | ❌ |

### Model Performance Results (existing but unreliable)

- **Intent accuracy:** 98.99% (198 samples, 29 intents)
  - BUT: Many intents have only 1 test sample — not statistically valid
- **Entity F1:** 95.5% micro-avg (235 samples, 23 entity types)
  - FAILING: "portal" entity → 0.0% (confused with "page")
  - Weak: "other" (F1: 0.80), subscription_attributes (F1: 0.875)

### What's Needed for Renewal

```
tests/
├── unit/
│   ├── test_actions_cpi.py
│   ├── test_actions_inventory.py
│   ├── test_actions_subscription.py
│   ├── test_actions_onboarding.py
│   └── test_database_layer.py
├── integration/
│   ├── test_action_server.py
│   └── test_mongodb_operations.py
├── nlu/
│   ├── test_stories_core.yml          (domain-specific story tests)
│   ├── test_stories_edge_cases.yml    (error handling, fallbacks)
│   └── nlu_benchmark.yml             (intent/entity regression)
├── e2e/
│   └── test_conversations.py
└── conftest.py
```

---

## 7. Build System & CI/CD

### Current State: NOTHING EXISTS

- ❌ No Dockerfile
- ❌ No docker-compose.yml
- ❌ No Makefile
- ❌ No shell scripts (setup, build, train, test, deploy)
- ❌ No GitHub Actions / GitLab CI / Jenkins pipeline
- ❌ No pre-commit hooks
- ❌ No linting configuration (no .flake8, .pylintrc, pyproject.toml)
- ❌ No code formatter configuration (no black, isort)
- ❌ No type checking (no mypy config)

### What's Needed

```yaml
# Minimum viable build system:
├── Dockerfile.rasa          # Rasa server container
├── Dockerfile.actions       # Action server container
├── docker-compose.yml       # Rasa + Actions + MongoDB + (optional) Rasa X
├── Makefile                 # Orchestration: make train, make test, make build, make up
├── scripts/
│   ├── setup.sh            # First-time setup
│   ├── train.sh            # Model training
│   └── test.sh             # Full test suite
├── .github/workflows/
│   └── ci.yml              # Lint → Train → Test → Threshold gate → Build image
├── .pre-commit-config.yaml  # Pre-commit hooks
├── pyproject.toml           # Single config: black, isort, mypy, pytest
└── .env.example             # Template for environment variables
```

---

## 8. Git Repository Health

### Size Problem: 1.75 GB (pack file)

For a project with ~45 commits and a ~17KB main Python file, 1.75 GB is extreme.

**Root cause — binaries committed to history:**

| File Type | Approximate Size | Source |
|-----------|-----------------|--------|
| MongoDB journal/WAL files | ~314 MB | `mongo-volume/journal/WiredTiger*` |
| Rasa DIET model weights | ~720 MB | `.rasa/cache/tmp*/DIETClassifier.tf_model.data-*` |
| TensorFlow checkpoints | ~500 MB | Various `.rasa/cache/` model files |
| PNG confusion matrices | ~0.5 MB | `results/*.png` (minor) |

These files are in Git history even though `.gitignore` now excludes `.rasa` and `models`.

### Branch Hygiene

14 remote branches including:
- Auto-generated Rasa X branches with timestamps
- Stale feature branches (chitchat, endpoints, inventory, nlu-models)
- No evidence of branch cleanup or archival

### Commit Practices

- No conventional commit format
- No signed commits
- Direct pushes to main (no PR/review workflow)
- Informal commit messages

### Required Remediation

1. **`git-filter-repo`** or BFG to remove ~1.5 GB of binary artifacts from history
2. Set up **Git LFS** if model tracking is needed in future
3. Delete stale branches
4. Fresh clone after history rewrite (force-push with team coordination)
5. Expected clean repo size: **< 5 MB**

---

## 9. Integration & External Services

### MongoDB — Primary Data Store

| Collection | Used By | Operations | Issues |
|-----------|---------|------------|--------|
| IOTA.CPI | ActionCPIlink | find (compound query) | No index hints |
| IOTA.subscription_details | ActionIMSIStats | find by IMSI | PINs exposed |
| IOTA.inventory | ActionFetchInventory | find (dynamic regex) | NoSQL injection |
| IOTA.inventory | ActionUpdateInventory | find + update_one | Unvalidated fields |
| IOTA.customers | SubmitOnboardingForm | insert_one | No duplicate check |

**Connection issues:**
- Hardcoded URI repeated 6 times (5 in actions.py + 1 in endpoints.yml)
- No connection pooling — new MongoClient per request
- No authentication configured
- No TLS configured
- No timeout configured
- "mongod" typo in tracker store type (endpoints.yml)

### NewsAPI — External API

- Using **deprecated v1 API** (v2 is current)
- Hardcoded API key in source
- No request timeout set
- No rate limiting
- No caching of results

### Oracle DB — Dead Integration

- `cx_Oracle` imported but never used
- Commented-out SQL queries suggest original Oracle backend
- Incomplete migration to MongoDB — dead code remains

### Chatbot/NLU/ — Dead Legacy System

A complete parallel NLU system exists under `Chatbot/NLU/`:
- BiLSTM intent classifier (Keras) — 5 intents, vocab=120
- spaCy custom NER with Word2Vec
- NLTK preprocessing pipeline
- Separate training data (JSON, CSV, TXT formats)
- **Completely disconnected from the Rasa pipeline** — dead code

---

## 10. Configuration Management

### Current State: No Configuration Management

Everything is hardcoded:

```python
# Every action class:
client = MongoClient("mongodb://localhost:27017")  # ← hardcoded
db = client["IOTA"]                                # ← hardcoded

# NewsAPI:
"apiKey": "4dbc17e007ab436fb66416009dfb59a8"       # ← hardcoded secret

# endpoints.yml:
url: mongodb://localhost:27017                     # ← hardcoded
url: "http://localhost:5055/webhook"               # ← hardcoded
```

### What's Needed

```python
# config.py — environment-based configuration
import os

class Config:
    MONGODB_URI = os.environ["MONGODB_URI"]
    MONGODB_DB = os.environ.get("MONGODB_DB", "IOTA")
    NEWS_API_KEY = os.environ["NEWS_API_KEY"]
    NEWS_API_URL = os.environ.get("NEWS_API_URL", "https://newsapi.org/v2/top-headlines")
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "10"))
```

```bash
# .env.example
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=IOTA
NEWS_API_KEY=your_key_here
NEWS_API_URL=https://newsapi.org/v2/top-headlines
REQUEST_TIMEOUT=10
```

---

## 11. Renewal Roadmap

### Phase 0: Immediate Actions (Before Any Development)

- [ ] Rotate the exposed NewsAPI key
- [ ] Run `git-filter-repo` to remove 1.5 GB of binaries from history
- [ ] Expand `.gitignore` comprehensively
- [ ] Delete stale branches
- [ ] Choose: **renovate this repo** or **start fresh repo** (recommended: fresh)

### Phase 1: Foundation (Week 1-2)

- [ ] Set up `pyproject.toml` with modern Python tooling (ruff, mypy, pytest)
- [ ] Create `docker-compose.yml` (Rasa + Action Server + MongoDB)
- [ ] Create `Dockerfile.rasa` and `Dockerfile.actions`
- [ ] Create `Makefile` with targets: setup, train, test, lint, build, up, down
- [ ] Set up `.env` pattern with `.env.example`
- [ ] Create CI pipeline (GitHub Actions or equivalent)
- [ ] Add pre-commit hooks (formatting, linting, secrets scanning)

### Phase 2: Architecture (Week 2-4)

- [ ] Upgrade to Rasa 3.8+ or evaluate Rasa Pro / CALM
- [ ] Upgrade TensorFlow to 2.16+ (or switch to PyTorch backend)
- [ ] Upgrade pymongo to 4.x with TLS + auth
- [ ] Replace cx_Oracle with python-oracledb (if Oracle is still needed)
- [ ] Implement layered architecture (repository pattern for data access)
- [ ] Create configuration management (env-based)
- [ ] Add connection pooling for MongoDB
- [ ] Implement input validation/sanitization layer

### Phase 3: NLU Redesign (Week 3-5)

- [ ] Redesign intent taxonomy (remove data-specific intents like `parent_organization_china_telecom_mongolia_branch`)
- [ ] Balance training data (minimum 50 examples per intent)
- [ ] Add FallbackClassifier with confidence thresholds
- [ ] Add RegexEntityExtractor for structured entities (IMSI, agreement numbers)
- [ ] Add lookup tables for finite entity sets
- [ ] Fix all annotation quality issues
- [ ] Configure proper hyperparameters with early stopping
- [ ] Set up NLU evaluation pipeline with pass/fail thresholds

### Phase 4: Testing & Quality (Week 4-6)

- [ ] Write unit tests for all custom actions (pytest + mongomock)
- [ ] Write story tests for all business flows
- [ ] Write entity extraction tests
- [ ] Set up test coverage reporting (target: 80%+)
- [ ] Add integration tests with test MongoDB
- [ ] Set up load testing for action server
- [ ] Add security scanning (pip-audit, bandit)

### Phase 5: Production Readiness (Week 6-8)

- [ ] Add proper logging framework (structlog)
- [ ] Add health check endpoints
- [ ] Add metrics/monitoring (Prometheus)
- [ ] Implement proper error handling with retry logic
- [ ] Add rate limiting for external API calls
- [ ] Set up model versioning (DVC or MLflow)
- [ ] Create deployment documentation
- [ ] Security hardening review

---

## Files in This Analysis

| File | Content |
|------|---------|
| [PROJECT_ANALYSIS.md](./PROJECT_ANALYSIS.md) | This file — full overview |
| [DEPENDENCY_AUDIT.md](./DEPENDENCY_AUDIT.md) | Detailed dependency versions, CVEs, upgrade paths |
| [SECURITY_REPORT.md](./SECURITY_REPORT.md) | All vulnerabilities with remediation steps |
| [CODE_QUALITY.md](./CODE_QUALITY.md) | Detailed code issues, patterns, and refactoring guide |

---

*This analysis was generated by examining all source files, configuration, git history, and test artifacts in the IoT-Digital-Assistant repository.*
