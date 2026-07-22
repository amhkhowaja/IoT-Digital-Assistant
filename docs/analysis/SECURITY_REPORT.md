# Security Report — IoT Digital Assistant

> **Analysis Date:** 2026-07-21  
> **Risk Level:** 🔴 CRITICAL  
> **Findings:** 1 Critical, 5 High, 4 Medium, 4 Low

---

## Vulnerability Summary

| # | Severity | Category | Location | Status |
|---|----------|----------|----------|--------|
| S-01 | 🔴 CRITICAL | Hardcoded Secret | actions.py:140 | Active in source |
| S-02 | 🟠 HIGH | NoSQL Injection | actions.py (ActionFetchInventory) | Exploitable |
| S-03 | 🟠 HIGH | NoSQL Injection | actions.py (ActionUpdateInventory) | Exploitable |
| S-04 | 🟠 HIGH | Sensitive Data Exposure | actions.py (ActionIMSIStats) | Active |
| S-05 | 🟠 HIGH | Unauthenticated Database | endpoints.yml + actions.py | Active |
| S-06 | 🟠 HIGH | Dependency Vulnerabilities | requirements.txt | 60+ CVEs |
| S-07 | 🟡 MEDIUM | Unvalidated Field Access | actions.py (ActionUpdateInventory) | Exploitable |
| S-08 | 🟡 MEDIUM | No Transport Encryption | All MongoDB connections | Active |
| S-09 | 🟡 MEDIUM | Missing Authentication | endpoints.yml (action endpoint) | Active |
| S-10 | 🟡 MEDIUM | Commented Credentials | actions.py:35 | In git history |
| S-11 | 🟢 LOW | Bare Exception Handlers | actions.py (6 instances) | Active |
| S-12 | 🟢 LOW | Debug Data Exposure | actions.py (print statements) | Active |
| S-13 | 🟢 LOW | Incomplete .gitignore | .gitignore | Active |
| S-14 | 🟢 LOW | Deprecated API Version | actions.py (NewsAPI v1) | Active |

---

## Detailed Findings

### S-01: Hardcoded API Key 🔴 CRITICAL

**Location:** `actions/actions.py`, class `ActionNewsFetch`

```python
query_params = {
    "source": "bbc-news",
    "sortBy": "top",
    "apiKey": "4dbc17e007ab436fb66416009dfb59a8"  # ← HARDCODED SECRET
}
```

**Risk:**
- API key is committed to Git and visible in history permanently
- Can be extracted by anyone with repo access (or if repo becomes public)
- Key can be used to exhaust API quotas or perform unauthorized requests
- Violates every security standard (OWASP, CWE-798)

**Impact:** Moderate (API key abuse, quota exhaustion)  
**Likelihood:** High (trivially extractable)

**Remediation:**
1. **Immediately** rotate the NewsAPI key at https://newsapi.org/account
2. Move to environment variable: `os.environ["NEWS_API_KEY"]`
3. Add `.env` to `.gitignore`
4. Use `git-filter-repo` to remove key from Git history
5. Add secret scanning to CI (e.g., `detect-secrets`, `gitleaks`)

---

### S-02: NoSQL Injection — ActionFetchInventory 🟠 HIGH

**Location:** `actions/actions.py`, class `ActionFetchInventory`

```python
# User-controlled entity values passed directly into MongoDB $regex
filtered_entities = {
    k: int(v) if v.isnumeric() else {'$regex': '^' + v + '$', '$options': 'i'}
    for k, v in entities.items() if v is not None
}
query_result = list(inventory.find(filtered_entities))
```

**Attack Vector:**
A user could send a message with an entity value containing regex metacharacters:
- `.*` → matches all documents (data exfiltration)
- `(a+)+$` → ReDoS (denial of service via catastrophic backtracking)
- Crafted patterns to enumerate data

**Impact:** Data exfiltration, denial of service  
**Likelihood:** Medium (requires understanding of entity extraction)

**Remediation:**
```python
import re
# Option 1: Escape regex metacharacters
safe_value = re.escape(v)
{'$regex': '^' + safe_value + '$', '$options': 'i'}

# Option 2: Use exact match with case-insensitive collation (preferred)
inventory.find(filtered_entities).collation({'locale': 'en', 'strength': 2})
```

---

### S-03: NoSQL Injection — ActionUpdateInventory 🟠 HIGH

**Location:** `actions/actions.py`, class `ActionUpdateInventory`

```python
# User-controlled values used in update operations
update = {}
for i, row in df[df["role"]=="update_value"].reset_index().iterrows():
    update[row["entity"]] = df[df["role"]=="update_value"].iloc[i]["value"]

result = inventory.update_one(fetch, {"$set": update})
```

**Attack Vector:**
- User could craft entity values with MongoDB operator syntax (`$set`, `$unset`, `$rename`)
- Could potentially modify fields beyond the intended `attributes` whitelist
- The `attributes` filter only checks entity names, not the update values

**Impact:** Unauthorized data modification  
**Likelihood:** Medium

**Remediation:**
```python
# Whitelist allowed fields for update
UPDATEABLE_FIELDS = {"plan_name", "connectivity_lock", "network_connectivity", 
                     "in_session", "billing_state", "monthly_data", "data_trend"}

# Validate update keys AND values
validated_update = {}
for key, value in update.items():
    if key not in UPDATEABLE_FIELDS:
        raise ValueError(f"Field {key} is not updateable")
    if isinstance(value, dict):  # Block operator injection
        raise ValueError(f"Invalid value format for {key}")
    validated_update[key] = value

result = inventory.update_one(fetch, {"$set": validated_update})
```

---

### S-04: Sensitive Data Exposure 🟠 HIGH

**Location:** `actions/actions.py`, class `ActionIMSIStats`

```python
msg = f"""The details of IMSI {slot_value} are:
    The Installation date is {query_result["Installation_date"]}.
    The SIM subscription state is {query_result["sim_subscription_state"]}.
    MSISDN is  {query_result["msisdn"]}.
    Pin is {query_result["pin1"]}.         # ← SIM PIN exposed
    Puk is  {query_result["puk1"]}.        # ← SIM PUK exposed
    And SIM status is {query_result["sim_status"]}."""
```

**Risk:**
- SIM PIN and PUK codes are highly sensitive credentials
- Exposed via unencrypted chat channel to any user who knows an IMSI number
- No access control or identity verification before revealing credentials
- IMSI numbers are not secret — they can be enumerated

**Impact:** Critical (SIM cloning, unauthorized access, account takeover)  
**Likelihood:** High (no authentication required)

**Remediation:**
1. **Never expose PIN/PUK via chat** — require separate secure channel
2. At minimum: require user authentication before revealing sensitive fields
3. Mask sensitive data: `Pin: ****` or only show last 2 digits
4. Add audit logging for all sensitive data access
5. Implement role-based access control for different data fields

---

### S-05: Unauthenticated Database Access 🟠 HIGH

**Location:** All MongoDB connections in `actions/actions.py` and `endpoints.yml`

```python
client = pymongo.MongoClient("mongodb://localhost:27017")  # No auth, no TLS
```

```yaml
tracker_store:
   type: mongod
   url: mongodb://localhost:27017  # No credentials
   db: IOTA
```

**Risk:**
- MongoDB accessible without any authentication
- Any process on the same network can read/write all data
- No TLS — data transmitted in plain text on the network
- Default MongoDB port exposed

**Impact:** Full database compromise  
**Likelihood:** High on shared/cloud infrastructure, Medium on isolated dev

**Remediation:**
```python
# Secure connection with auth + TLS
client = MongoClient(
    os.environ["MONGODB_URI"],  # mongodb://user:pass@host:27017/?tls=true&authSource=admin
    tlsCAFile=os.environ.get("MONGODB_CA_FILE"),
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=5000
)
```

---

### S-06: Known Dependency Vulnerabilities 🟠 HIGH

**60+ known CVEs** across pinned dependency versions. Top critical ones:

| CVE | Package | Severity | Type |
|-----|---------|----------|------|
| CVE-2022-41894 | tensorflow 2.8 | Critical | Code execution |
| CVE-2022-41880 | tensorflow 2.8 | High | DoS |
| CVE-2023-25801 | tensorflow 2.8 | High | Code execution |
| CVE-2023-23931 | cryptography 39.0 | High | Memory corruption |
| CVE-2023-32681 | requests 2.28 | Medium | Header leak |
| CVE-2023-43804 | urllib3 1.26 | Medium | Cookie leak |
| CVE-2023-25577 | Werkzeug 2.2 | High | DoS |
| CVE-2023-44271 | Pillow 9.4 | Medium | DoS |
| CVE-2023-28858 | redis 4.5 | Medium | Data leak |

**Remediation:** Full dependency upgrade (see [DEPENDENCY_AUDIT.md](./DEPENDENCY_AUDIT.md))

---

### S-07: Unvalidated Field Access 🟡 MEDIUM

**Location:** `actions/actions.py`, class `ActionIMSIStats`

```python
# entity_value is user-controlled, used as dictionary key
query_result = list(subscription_details.find(query))[0][entity_value]
```

**Risk:** User could potentially access arbitrary fields in the document by controlling entity_value.

**Remediation:**
```python
ALLOWED_FIELDS = {"Installation_date", "sim_subscription_state", "msisdn", "sim_status"}
if entity_value not in ALLOWED_FIELDS:
    dispatcher.utter_message(text="Sorry, I can't look up that information.")
    return []
```

---

### S-08: No Transport Encryption 🟡 MEDIUM

All MongoDB connections use plain `mongodb://` without TLS. All action endpoint communication uses plain HTTP.

**Remediation:** Enable TLS on MongoDB, use HTTPS for all service-to-service communication.

---

### S-09: Missing Action Endpoint Authentication 🟡 MEDIUM

```yaml
action_endpoint:
   url: "http://localhost:5055/webhook"
   token: ""  # ← Empty token
```

Any network-adjacent service can invoke custom actions without authentication.

**Remediation:** Set a strong token and validate it in the action server.

---

### S-10: Commented Credentials in Source 🟡 MEDIUM

```python
# client = pymongo.MongoClient("mongodb://admin:password@mongo-db:27017/")
```

Credentials visible in source code comments. Even if commented out, they're in Git history.

**Remediation:** Remove from source, use `git-filter-repo` to clean history.

---

### S-11: Bare Exception Handlers 🟢 LOW

6 instances of bare `except:` that catch ALL exceptions:

```python
except:
    dispatcher.utter_message(text="Sorry! we can not build the connection with the database")
    return []
```

**Risk:** Masks security-relevant errors, prevents proper incident detection.

**Remediation:** Catch specific exceptions, add structured logging.

---

### S-12: Debug Data Exposure 🟢 LOW

```python
print(str(prediction))
print("Entities" + str(entities))
print("Filtered_entities: " + str(filtered_entities))
```

User input and query details printed to stdout — may be captured in logs.

**Remediation:** Remove debug prints, use proper logging with log levels.

---

### S-13: Incomplete .gitignore 🟢 LOW

Only 5 rules. Missing `.env`, `venv/`, `*.pem`, `*.key`, credential files, IDE files, OS artifacts.

**Remediation:** Expand to comprehensive .gitignore (see PROJECT_ANALYSIS.md).

---

### S-14: Deprecated API Version 🟢 LOW

```python
main_url = "https://newsapi.org/v1/articles"  # v1 is deprecated
```

Using deprecated API version that may have unfixed security issues.

**Remediation:** Upgrade to NewsAPI v2 (`/v2/top-headlines`).

---

## Security Posture Summary

```
┌─────────────────────────────────────────────────────┐
│                  SECURITY POSTURE                     │
├─────────────────────────────────────────────────────┤
│ Authentication:     ❌ None (DB, API, action server) │
│ Authorization:      ❌ None (any user sees any data) │
│ Encryption:         ❌ None (plain MongoDB, HTTP)    │
│ Input Validation:   ❌ None (direct user→DB)         │
│ Secret Management:  ❌ None (hardcoded in source)    │
│ Dependency Health:  ❌ 60+ CVEs, 7 EOL packages     │
│ Audit Logging:      ❌ None (only print statements)  │
│ Error Handling:     ❌ Bare except, no monitoring    │
│ Access Control:     ❌ PINs/PUKs exposed freely     │
│ .gitignore:         ⚠️  Minimal (5 rules)           │
└─────────────────────────────────────────────────────┘
```

**This project has zero security controls.** It must not be exposed to any network beyond isolated local development.

---

## Immediate Actions (Before Any Other Work)

1. ⚡ **Rotate** the NewsAPI key (`4dbc17e007ab436fb66416009dfb59a8`)
2. ⚡ **Add** comprehensive `.gitignore`
3. ⚡ **Remove** PIN/PUK exposure from chat responses
4. ⚡ **Clean** Git history of secrets with `git-filter-repo`
5. ⚡ **Enable** MongoDB authentication (even for local dev)

---

*Next: See [CODE_QUALITY.md](./CODE_QUALITY.md) for detailed code analysis and refactoring recommendations.*
