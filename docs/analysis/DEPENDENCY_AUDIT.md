# Dependency Audit — IoT Digital Assistant

> **Analysis Date:** 2026-07-21  
> **Total Dependencies:** 150+ packages pinned  
> **EOL Packages:** 7  
> **Estimated CVEs:** 60+

---

## Critical: End-of-Life Packages

These packages have reached end-of-life and receive NO security patches.

### 1. Rasa 3.4.4 → Rasa Pro 4.x / Rasa Open Source 3.8+

| Aspect | Details |
|--------|---------|
| Pinned | 3.4.4 (released ~2023) |
| Current | Rasa Pro 4.x (commercial), OSS 3.8.x (community) |
| EOL Date | Rasa 3.x OSS maintenance ended 2024 |
| Impact | No security patches, no bug fixes, incompatible with newer Python/TF |
| Migration | Significant — API changes, CALM architecture in Pro, pipeline changes |

### 2. TensorFlow 2.8.4 → 2.18+

| Aspect | Details |
|--------|---------|
| Pinned | 2.8.4 |
| Current | 2.18+ |
| EOL Date | November 2022 |
| Known CVEs | CVE-2022-41894, CVE-2022-41880, CVE-2023-25801, 40+ more |
| Impact | Denial of service, code execution, crashes |
| Migration | Moderate — API mostly compatible, but Rasa version must also support it |

### 3. PyMongo 3.10.1 → 4.9+

| Aspect | Details |
|--------|---------|
| Pinned | 3.10.1 (released 2020) |
| Current | 4.9+ |
| EOL Date | PyMongo 3.x deprecated since PyMongo 4.0 (2022) |
| Impact | Missing TLS defaults, SCRAM-SHA-256, modern auth |
| Migration | Moderate — some breaking API changes (Collection → explicit codecs) |
| Breaking Changes | `count()` removed, cursor behavior changes, connection string parsing |

### 4. cx_Oracle → python-oracledb

| Aspect | Details |
|--------|---------|
| Pinned | Imported (no version in requirements) |
| Current | python-oracledb 2.x |
| Status | Renamed and deprecated since 2022 |
| Impact | Not receiving updates, Oracle dropping support |
| Migration | Simple rename + thin client mode (no Oracle Client needed) |
| Note | Currently imported but UNUSED — can simply remove |

### 5. urllib3 1.26.14 → 2.3+

| Aspect | Details |
|--------|---------|
| Pinned | 1.26.14 |
| Current | 2.3+ |
| EOL Date | 1.26.x maintenance ended |
| CVEs | CVE-2023-43804 (cookie leak), CVE-2023-45803 (header leak) |
| Migration | Breaking changes in 2.0 — but `requests` handles this internally |

### 6. SQLAlchemy 1.4.46 → 2.1+

| Aspect | Details |
|--------|---------|
| Pinned | 1.4.46 |
| Current | 2.1+ |
| EOL Date | January 2024 |
| Migration | Major — 2.0 has new session/query API |
| Note | Used by Rasa internals, not directly by this project |

### 7. protobuf 3.19.6 → 5.x+

| Aspect | Details |
|--------|---------|
| Pinned | 3.19.6 |
| Current | 5.x (protobuf 4+ naming) |
| CVEs | CVE-2022-3171 (DoS via parsing) |
| Migration | Major version jump with breaking changes |
| Note | Used by TensorFlow/gRPC — will upgrade when TF upgrades |

---

## High Priority: Significantly Outdated

### Security-Relevant Updates Needed

| Package | Pinned | Current | Key CVEs/Issues |
|---------|--------|---------|----------------|
| cryptography | 39.0.1 | 43+ | CVE-2023-23931 (memory corruption), CVE-2023-38325 (X.509) |
| requests | 2.28.1 | 2.32+ | CVE-2023-32681 (Proxy-Authorization header leak) |
| Pillow | 9.4.0 | 11+ | CVE-2023-44271 (DoS), multiple buffer overflows |
| Werkzeug | 2.2.2 | 3.1+ | CVE-2023-25577 (multipart DoS), CVE-2023-46136 |
| sanic | 21.12.2 | 24+ | Request smuggling, DoS vulnerabilities |
| redis | 4.5.1 | 5.2+ | CVE-2023-28858/28859 (data leakage in async) |
| confluent-kafka | 1.9.2 | 2.6+ | Multiple security patches |
| certifi | 2022.12.7 | 2024+ | Missing CA revocations (trusts revoked CAs) |
| scipy | 1.8.1 | 1.14+ | CVE-2023-25399 (memory DoS) |
| pyOpenSSL | 23.0.0 | 24+ | Inherits cryptography CVEs |

---

## Requirements File Analysis

### requirements.txt (primary)
- 150+ packages with exact version pins
- Includes Windows-specific packages (`win-inet-pton`, `wincertstore`, `pyreadline3`) — won't install on Linux/Mac
- Contains conflicting platform assumptions
- Rasa and all its transitive dependencies pinned

### requirements2.txt (broken)
- Pip freeze dump with conda `file://` local paths:
  ```
  mkl-fft @ file:///Users/runner/miniforge3/conda-bld/...
  mkl-random @ file:///Users/runner/miniforge3/conda-bld/...
  ```
- **Completely non-portable** — will fail on any other machine
- Mixed conda and pip packages

### requirements_rasa_iot.txt (unclear purpose)
- Another requirements variant
- Overlaps heavily with requirements.txt
- Slightly different version pins for some packages

---

## Dependency Graph Issues

### Conflict Risks
- TensorFlow 2.8 requires numpy < 1.24 → blocks numpy upgrade
- Rasa 3.4 requires TensorFlow ~= 2.8 → blocks TF upgrade
- Rasa 3.4 requires sanic 21.x → blocks sanic upgrade
- **Cascading lock**: Rasa version locks TF, which locks numpy, which locks scipy

### Upgrade Strategy

The dependency chain is so interlocked that incremental upgrades are impractical.

**Recommended approach:**
1. Start with target Rasa version (3.8+ or Pro 4.x)
2. Let Rasa's requirements determine TF/numpy/scipy versions
3. Pin all other packages to latest compatible versions
4. Use `pip-compile` (pip-tools) or `poetry` for proper dependency resolution
5. Test full pipeline after upgrade

---

## Unused Dependencies (Dead Weight)

These are imported or listed but serve no purpose:

| Package | Status |
|---------|--------|
| cx_Oracle | Imported, never used |
| confluent-kafka | In requirements, no code uses it |
| psycopg2-binary | In requirements, no PostgreSQL code |
| boto3/s3transfer | In requirements, no S3 code |
| twilio | In requirements, channel not configured |
| slack-sdk | In requirements, channel commented out |
| rocketchat-API | In requirements, no code uses it |
| webexteamssdk | In requirements, channel not configured |
| mattermostwrapper | In requirements, channel commented out |
| fbmessenger | In requirements, channel commented out |

**Estimated size reduction by removing unused deps:** ~200 MB of installed packages.

---

## Upgrade Priority Matrix

| Priority | Package | Reason | Effort |
|----------|---------|--------|--------|
| 🔴 P0 | rasa | EOL, blocks all other upgrades | High |
| 🔴 P0 | tensorflow | 40+ CVEs, EOL | High (tied to Rasa) |
| 🔴 P0 | cryptography | Memory corruption CVE | Low |
| 🟠 P1 | pymongo | EOL, missing security features | Medium |
| 🟠 P1 | urllib3/requests | Header/cookie leak CVEs | Low |
| 🟠 P1 | Werkzeug | DoS CVEs | Low |
| 🟡 P2 | Pillow | Buffer overflow CVEs | Low |
| 🟡 P2 | redis | Data leak CVE | Low |
| 🟡 P2 | sanic | Request smuggling | Medium (tied to Rasa) |
| ⚪ P3 | Remove cx_Oracle | Dead code | Trivial |
| ⚪ P3 | Remove unused channel deps | Clean up | Trivial |

---

*Next: See [SECURITY_REPORT.md](./SECURITY_REPORT.md) for full vulnerability details and remediation.*
