# IoT Digital Assistant

A conversational AI chatbot for IoT service portal management. Built with [Rasa](https://rasa.com/), it handles subscription queries, device inventory management, customer onboarding, and knowledge base lookups through natural language.

> ⚠️ **All data in this project is dummy/synthetic data for demonstration purposes only. No real customer information, IMSI numbers, MSISDNs, or credentials are included.**

---

## Features

- **Subscription Lookup** — Query SIM/IMSI details by number
- **Inventory Management** — Fetch and update device connectivity, billing state, plans
- **Customer Onboarding** — Guided conversational flow for enterprise onboarding
- **Knowledge Base** — CPI documentation link retrieval by topic
- **News Feed** — Latest news headlines integration
- **Multi-Channel** — REST API, Socket.IO, Slack (configurable)

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Docker | 20+ | [docker.com](https://docs.docker.com/get-docker/) |
| Docker Compose | v2+ | Included with Docker |
| Make | any | Pre-installed on macOS/Linux |

**macOS (with Colima):**
```bash
brew install colima docker docker-compose
colima start --vm-type vz --vz-rosetta --disk 60 --memory 4 --cpu 4
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin make
sudo systemctl start docker
sudo usermod -aG docker $USER
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/IoT-Digital-Assistant.git
cd IoT-Digital-Assistant

# 2. Setup environment
make setup

# 3. Build containers
make build

# 4. Start all services
make up
```

After ~30 seconds, you'll have:
- **Rasa Server** → http://localhost:5005
- **Action Server** → http://localhost:5055
- **MongoDB** → mongodb://localhost:27017

### Train a Model

```bash
make train
```

This trains the NLU + dialogue model inside Docker. Output goes to `./models/`.

---

## Available Commands

```bash
make help           # Show all commands
make setup          # Create .env from template
make build          # Build Docker images
make up             # Start all services
make down           # Stop all services
make train          # Train Rasa model
make test           # Run test stories
make test-nlu       # Run NLU evaluation
make logs           # Tail all service logs
make logs-rasa      # Tail Rasa logs only
make logs-actions   # Tail action server logs
make shell          # Open Rasa shell (interactive chat)
make status         # Health check all services
make clean          # Remove images and volumes
```

---

## API Usage

### Health Check

```bash
curl http://localhost:5055/health
```
```json
{"status": "ok"}
```

### Send a Chat Message

```bash
curl -X POST http://localhost:5005/webhooks/rest/webhook \
  -H "Content-Type: application/json" \
  -d '{"sender": "user123", "message": "show me the connectivity of msisdn 12345678901"}'
```
```json
[
  {
    "recipient_id": "user123",
    "text": "Sure, Here is the data for AND[{'msisdn': '12345678901'}]:\n{msisdn: 12345678901, plan_name: '10 GB Europe', connectivity_lock: 'unlocked', network_connectivity: 'connected', billing_state: 'active'}"
  }
]
```

### Query IMSI Details

```bash
curl -X POST http://localhost:5005/webhooks/rest/webhook \
  -H "Content-Type: application/json" \
  -d '{"sender": "user123", "message": "what is the status of IMSI 234150999999999"}'
```
```json
[
  {
    "recipient_id": "user123",
    "text": "The details of IMSI 234150999999999 are:\nThe Installation date is 2024-01-15.\nThe SIM subscription state is active.\nMSISDN is 12345678901.\nAnd SIM status is enabled."
  }
]
```

### Learn About a Topic (CPI Link)

```bash
curl -X POST http://localhost:5005/webhooks/rest/webhook \
  -H "Content-Type: application/json" \
  -d '{"sender": "user123", "message": "I want to learn about connectivity lock"}'
```
```json
[
  {
    "recipient_id": "user123",
    "text": "Please follow the below link for detailed information:"
  },
  {
    "recipient_id": "user123",
    "custom": {
      "payload": "iFrame",
      "data": [{"title": "connectivity lock", "url": "https://cpi.example.com/docs/connectivity-lock"}]
    }
  }
]
```

### Update Inventory

```bash
curl -X POST http://localhost:5005/webhooks/rest/webhook \
  -H "Content-Type: application/json" \
  -d '{"sender": "user123", "message": "lock the connectivity of msisdn 12345678901"}'
```
```json
[
  {
    "recipient_id": "user123",
    "text": "Hurrah. Updated Successfully."
  }
]
```

### Get Latest News

```bash
curl -X POST http://localhost:5005/webhooks/rest/webhook \
  -H "Content-Type: application/json" \
  -d '{"sender": "user123", "message": "show me the latest news"}'
```
```json
[
  {
    "recipient_id": "user123",
    "text": "Here is some headlines from bbc-news"
  },
  {
    "recipient_id": "user123",
    "custom": {"payload": "cardsCarousel", "data": [{"title": "...", "url": "..."}]}
  }
]
```

> **Note:** News requires a valid `NEWS_API_KEY` in your `.env` file. Get one free at [newsapi.org](https://newsapi.org/register).

---

## Configuration

Copy the template and edit:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection string | `mongodb://mongodb:27017` |
| `MONGODB_DB` | Database name | `IOTA` |
| `NEWS_API_KEY` | NewsAPI key for news feature | (empty) |

---

## Seed Test Data

To populate MongoDB with dummy data for testing:

```bash
docker compose exec mongodb mongosh --quiet --eval '
db = db.getSiblingDB("IOTA");
db.inventory.insertMany([
  {msisdn: 12345678901, plan_name: "10 GB Europe", connectivity_lock: "unlocked", network_connectivity: "connected", in_session: true, billing_state: "active", monthly_data: "7.2 GB", data_trend: "upward"},
  {msisdn: 99887766554, plan_name: "20 GB Global", connectivity_lock: "locked", network_connectivity: "disconnected", in_session: false, billing_state: "inactive", monthly_data: "0 GB", data_trend: "downward"}
]);
db.subscription_details.insertMany([
  {imsi: 234150999999999, msisdn: 12345678901, Installation_date: "2024-01-15", sim_subscription_state: "active", pin1: "0000", puk1: "00000000", sim_status: "enabled"},
  {imsi: 234150888888888, msisdn: 99887766554, Installation_date: "2023-06-20", sim_subscription_state: "suspended", pin1: "0000", puk1: "00000000", sim_status: "disabled"}
]);
db.CPI.insertMany([
  {intent: "Learn", sub_entities: "connectivity lock", enterprise_links: "https://cpi.example.com/docs/connectivity-lock"},
  {intent: "Learn", sub_entities: "billing", enterprise_links: "https://cpi.example.com/docs/billing-guide"},
  {intent: "information_seek", sub_entities: "apn", enterprise_links: "https://cpi.example.com/docs/apn-configuration"}
]);
print("Done: " + db.inventory.countDocuments() + " inventory, " + db.subscription_details.countDocuments() + " subscriptions, " + db.CPI.countDocuments() + " CPI docs");
'
```

---

## NLU Training Pipeline (Airflow)

The project includes an Apache Airflow-based ML pipeline that trains intent classification (BiLSTM + CNN) and NER (spaCy) models from the training data.

```bash
# Start pipeline services (Airflow + Postgres)
make pipeline-up

# Trigger training DAG
make pipeline-trigger

# Open Airflow UI: http://localhost:8080 (admin/admin)

# Check logs
make pipeline-logs

# Stop pipeline
make pipeline-down
```

Trained models and metrics are saved to `./pipeline_output/`:

```
pipeline_output/
├── models/
│   ├── bilstm.h5           # Intent classifier (BiLSTM)
│   ├── cnn.h5              # Intent classifier (CNN)
│   ├── w2v.model           # Word2Vec embeddings
│   └── ner_model/          # spaCy NER model
├── metrics/
│   ├── intent_classification_report.json
│   └── ner_report.json
├── splits/
│   ├── train_set.csv
│   ├── test_set.csv
│   └── val_set.csv
└── data.json, data.yaml, ques_int.csv
```

---

## Project Structure

```
IoT-Digital-Assistant/
├── actions/
│   └── actions.py            # Custom action server logic
├── data/
│   ├── nlu.yml               # NLU training examples
│   ├── stories.yml           # Conversation flows
│   └── rules.yml             # Dialogue rules
├── models/                   # Trained models (gitignored)
├── config.yml                # NLU pipeline & policy config
├── domain.yml                # Intents, entities, slots, responses
├── endpoints.yml             # Local development endpoints
├── endpoints.docker.yml      # Docker service endpoints
├── credentials.yml           # Channel credentials
├── Dockerfile.rasa           # Rasa server image
├── Dockerfile.actions        # Action server image
├── docker-compose.yml        # Full stack orchestration
├── Makefile                  # Build automation
├── .env.example              # Environment template
└── docs/
    ├── analysis/             # Project analysis documents
    └── architecture/         # System design & architecture
```

---

## Troubleshooting

**Rasa server not responding after `make up`:**
- Run `make train` first — Rasa needs a trained model to serve.

**TensorFlow AVX error on Apple Silicon:**
- This is expected when running x86 Rasa images via emulation. Training and inference work but are slower. Use `make train` inside Docker.

**MongoDB connection refused:**
- Wait 20 seconds after `make up` for the health check to pass.
- Check with: `docker compose ps` — mongodb should show `(healthy)`.

**News feature not working:**
- Add a valid API key to `.env`: `NEWS_API_KEY=your_key_here`

---

## License

MIT License

---

## Disclaimer

This project is for demonstration and development purposes. All data including IMSI numbers, MSISDNs, enterprise names, and credentials are entirely fictional and do not represent any real individuals, organizations, or telecom subscribers.
