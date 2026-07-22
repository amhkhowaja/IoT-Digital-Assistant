# Installation Guide

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (20+)
- [Docker Compose](https://docs.docker.com/compose/) (v2+, included with Docker)
- [Make](https://www.gnu.org/software/make/) (pre-installed on macOS/Linux)

## Quick Start (one command)

```bash
git clone https://github.com/amhkhowaja/IoT-Digital-Assistant.git
cd IoT-Digital-Assistant
make start
```

This will:
1. Create `.env` from template
2. Download the latest trained model from GitHub Releases
3. Build all Docker images
4. Start all services
5. Seed MongoDB with demo data

Once complete, open **http://localhost:3000** in your browser.

## What's Running

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:3000 | Portal UI + chatbot widget |
| **Rasa Server** | http://localhost:5005 | NLU + dialogue engine |
| **Action Server** | http://localhost:5055 | Custom actions (MongoDB queries) |
| **MongoDB** | localhost:27017 | Database |

## Portal Pages

- http://localhost:3000 — Home + chatbot widget
- http://localhost:3000/inventory.html — Device inventory table
- http://localhost:3000/customers.html — Customer list
- http://localhost:3000/subscriptions.html — SIM/IMSI subscriptions

## Common Commands

```bash
make start          # First time: full setup
make up             # Start services (after first time)
make down           # Stop everything
make seed           # Re-seed database with demo data
make download-model # Re-download latest model from CI
make logs           # Watch all logs
make status         # Health check
```

## Apple Silicon (M1/M2/M3/M4) Note

Rasa uses TensorFlow which requires AVX CPU instructions not available on Apple Silicon. The chatbot NLU will load very slowly (~5-10 min) or may not work depending on your Docker setup.

**Everything else works fine on Mac** — frontend, portal pages, database, action server.

For full Rasa functionality, use a Linux x86_64 machine or deploy to a cloud instance.

## Troubleshooting

**`make start` fails at download-model:**
→ The CI hasn't created a release yet. Run `make up` without the model — the portal works without Rasa.

**Frontend shows empty tables:**
→ Run `make seed` to populate MongoDB with demo data.

**Port already in use:**
→ Stop other services on ports 3000, 5005, 5055, or 27017. Or check for stale Docker containers: `docker ps -a`.
