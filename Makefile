.PHONY: help setup build up down train test shell logs clean

# Default target
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ─── Setup ───────────────────────────────────────────────────────────

setup: ## First-time setup: create .env from template
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env — edit it with your API keys"; \
	else \
		echo ".env already exists"; \
	fi
	@mkdir -p models

# ─── Docker ──────────────────────────────────────────────────────────

build: ## Build all Docker images
	docker compose build

up: ## Start all services (Rasa + Actions + MongoDB)
	docker compose up -d
	@echo ""
	@echo "Services starting..."
	@echo "  Rasa:     http://localhost:5005"
	@echo "  Actions:  http://localhost:5055"
	@echo "  MongoDB:  mongodb://localhost:27017"
	@echo ""
	@echo "Test with: curl http://localhost:5005/status"

down: ## Stop all services
	docker compose down

logs: ## Tail logs from all services
	docker compose logs -f

logs-rasa: ## Tail Rasa server logs
	docker compose logs -f rasa

logs-actions: ## Tail action server logs
	docker compose logs -f action-server

# ─── Training ────────────────────────────────────────────────────────

train: ## Train the Rasa model (runs inside Docker)
	docker compose run --rm rasa train
	@echo "Model trained. Output in ./models/"

train-local: ## Train model locally (requires rasa installed)
	rasa train

# ─── Testing ─────────────────────────────────────────────────────────

test: ## Run Rasa test stories
	docker compose run --rm rasa test

test-nlu: ## Run NLU evaluation
	docker compose run --rm rasa test nlu --nlu data/nlu.yml

test-local: ## Run tests locally
	rasa test

# ─── Development ─────────────────────────────────────────────────────

shell: ## Open shell in Rasa container
	docker compose run --rm rasa shell

shell-actions: ## Open bash in action server container
	docker compose exec action-server bash

status: ## Check service health
	@echo "Rasa server:"
	@curl -s http://localhost:5005/status | python3 -m json.tool 2>/dev/null || echo "  Not running"
	@echo ""
	@echo "Action server:"
	@curl -s http://localhost:5055/health 2>/dev/null || echo "  Not running"
	@echo ""
	@echo "MongoDB:"
	@docker compose exec mongodb mongosh --eval "db.adminCommand('ping')" --quiet 2>/dev/null || echo "  Not running"

# ─── Cleanup ─────────────────────────────────────────────────────────

clean: ## Remove built images and volumes
	docker compose down -v --rmi local
	@echo "Cleaned up containers, images, and volumes"

clean-models: ## Remove trained models
	rm -rf models/*.tar.gz
	@echo "Removed model files"
