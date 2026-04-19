# ==============================================================================
# KAIROSYN-1 Makefile
# Author: Dustin Groves — Or4cl3 AI Solutions
# ==============================================================================
# Usage:
#   make help          — List all targets
#   make install       — Install package + dev deps
#   make docker-build  — Build the Docker image
#   make deploy        — Deploy via Docker Compose
#   make up            — Start full stack (server + monitoring)
#   make down          — Stop all containers
#   make logs          — Tail server logs
#   make test          — Run full test suite
#   make lint          — Run linters
#   make train-sft     — Run Phase 1 SFT training
# ==============================================================================

.PHONY: help install install-dev lint format test \
        docker-build docker-push docker-run \
        deploy up down restart logs health \
        train-sft train-ppo train-maml \
        evaluate infer clean

# ── Config ────────────────────────────────────────────────────────────────────
SHELL          := /bin/bash
PYTHON         := python3
PIP            := pip
REGISTRY       := ghcr.io
IMAGE_OWNER    := or4cl3-ai-1
IMAGE_NAME     := kairosyn-1
IMAGE_TAG      ?= latest
FULL_IMAGE     := $(REGISTRY)/$(IMAGE_OWNER)/$(IMAGE_NAME):$(IMAGE_TAG)
COMPOSE_FILE   := docker/docker-compose.yml
MODEL_CONFIG   := configs/model/kairosyn_e4b.yaml
CHECKPOINT_DIR := ./checkpoints
PORT           := 8080

# Colours
CYAN  := \033[0;36m
GREEN := \033[0;32m
YELLOW:= \033[0;33m
RESET := \033[0m

# ── Help ─────────────────────────────────────────────────────────────────────
help: ## Show this help
	@echo ""
	@echo "$(CYAN)KAIROSYN-1 — Build & Deploy Commands$(RESET)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Installation ─────────────────────────────────────────────────────────────
install: ## Install package in editable mode
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev: ## Install package + dev/test dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PIP) install black isort flake8 mypy pytest pytest-cov

install-cpu: ## Install with CPU-only PyTorch (for CI/dev without GPU)
	$(PIP) install --upgrade pip
	$(PIP) install torch==2.3.0 torchvision torchaudio \
	  --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -e ".[dev]" --no-deps
	$(PIP) install pyyaml loguru einops rich click transformers accelerate peft

# ── Code Quality ─────────────────────────────────────────────────────────────
lint: ## Run all linters
	@echo "$(CYAN)Running linters...$(RESET)"
	flake8 kairosyn/ scripts/ --max-line-length=100 --extend-ignore=E203,W503,E501
	isort --check-only kairosyn/ scripts/
	black --check kairosyn/ scripts/
	@echo "$(GREEN)All checks passed.$(RESET)"

format: ## Auto-format code
	@echo "$(CYAN)Formatting code...$(RESET)"
	isort kairosyn/ scripts/
	black kairosyn/ scripts/
	@echo "$(GREEN)Formatting complete.$(RESET)"

typecheck: ## Run mypy type checking
	mypy kairosyn/ --ignore-missing-imports --no-strict-optional

# ── Testing ──────────────────────────────────────────────────────────────────
test: ## Run unit tests (CPU)
	@echo "$(CYAN)Running tests...$(RESET)"
	pytest tests/ -v \
	  --cov=kairosyn \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov \
	  -m "not gpu and not slow" \
	  --timeout=120
	@echo "$(GREEN)Tests passed. Coverage report: htmlcov/index.html$(RESET)"

test-gpu: ## Run all tests including GPU tests (requires CUDA)
	pytest tests/ -v -m "gpu or not slow" --timeout=300

test-fast: ## Run only fast unit tests
	pytest tests/ -v -m "not slow and not gpu and not integration" --timeout=60

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build: ## Build the Docker image
	@echo "$(CYAN)Building Docker image: $(FULL_IMAGE)$(RESET)"
	docker build \
	  -f docker/Dockerfile \
	  -t $(FULL_IMAGE) \
	  -t $(IMAGE_NAME):latest \
	  --build-arg GIT_SHA=$(shell git rev-parse --short HEAD) \
	  --build-arg BUILD_DATE=$(shell date -u +%Y-%m-%dT%H:%M:%SZ) \
	  .
	@echo "$(GREEN)Build complete: $(FULL_IMAGE)$(RESET)"

docker-push: docker-build ## Build and push to GHCR
	@echo "$(CYAN)Pushing to registry...$(RESET)"
	docker push $(FULL_IMAGE)
	@echo "$(GREEN)Pushed: $(FULL_IMAGE)$(RESET)"

docker-run: ## Run a single container locally (no compose)
	docker run --rm \
	  --gpus all \
	  -p $(PORT):$(PORT) \
	  -e KAIROSYN_CONFIG=/app/configs/model/kairosyn_e4b.yaml \
	  -v $(CHECKPOINT_DIR):/app/checkpoints \
	  $(IMAGE_NAME):latest

docker-shell: ## Open a shell inside the container
	docker run --rm -it \
	  --gpus all \
	  -v $(shell pwd):/app \
	  $(IMAGE_NAME):latest bash

# ── Compose / Full Stack ──────────────────────────────────────────────────────
up: ## Start full stack (server + Prometheus + Grafana)
	@echo "$(CYAN)Starting KAIROSYN full stack...$(RESET)"
	@mkdir -p $(CHECKPOINT_DIR) results
	docker compose -f $(COMPOSE_FILE) up -d
	@echo ""
	@echo "$(GREEN)Stack running:$(RESET)"
	@echo "  API Server : http://localhost:$(PORT)"
	@echo "  API Docs   : http://localhost:$(PORT)/docs"
	@echo "  Prometheus : http://localhost:9090"
	@echo "  Grafana    : http://localhost:3000  (admin / kairosyn)"
	@echo ""

deploy: docker-build up ## Build image then start full stack

down: ## Stop all containers
	docker compose -f $(COMPOSE_FILE) down
	@echo "$(GREEN)Stack stopped.$(RESET)"

restart: ## Restart the KAIROSYN server container only
	docker compose -f $(COMPOSE_FILE) restart kairosyn
	@echo "$(GREEN)Server restarted.$(RESET)"

logs: ## Tail the server logs
	docker compose -f $(COMPOSE_FILE) logs -f kairosyn

logs-all: ## Tail all container logs
	docker compose -f $(COMPOSE_FILE) logs -f

ps: ## Show running containers
	docker compose -f $(COMPOSE_FILE) ps

# ── Health & Status ──────────────────────────────────────────────────────────
health: ## Check server health endpoint
	@echo "$(CYAN)Checking health...$(RESET)"
	@curl -s http://localhost:$(PORT)/health | python3 -m json.tool || \
	  echo "$(YELLOW)Server not responding on port $(PORT)$(RESET)"

status: ## Show server status + active metrics
	@echo "$(CYAN)Server info:$(RESET)"
	@curl -s http://localhost:$(PORT)/v1/model/info | python3 -m json.tool 2>/dev/null || \
	  echo "Not available"
	@echo ""
	@echo "$(CYAN)Active containers:$(RESET)"
	@docker compose -f $(COMPOSE_FILE) ps 2>/dev/null || echo "Compose not running"

ping: ## Quick connectivity test against all endpoints
	@echo "Testing endpoints..."
	@curl -sf http://localhost:$(PORT)/health       > /dev/null && echo "  ✅ /health" || echo "  ❌ /health"
	@curl -sf http://localhost:$(PORT)/v1/model/info > /dev/null && echo "  ✅ /v1/model/info" || echo "  ❌ /v1/model/info"
	@curl -sf http://localhost:$(PORT)/docs          > /dev/null && echo "  ✅ /docs (Swagger UI)" || echo "  ❌ /docs"
	@curl -sf http://localhost:9090                  > /dev/null && echo "  ✅ Prometheus" || echo "  ❌ Prometheus"
	@curl -sf http://localhost:3000                  > /dev/null && echo "  ✅ Grafana" || echo "  ❌ Grafana"

# ── Training ──────────────────────────────────────────────────────────────────
train-sft: ## Run Phase 1: Supervised Fine-Tuning
	@echo "$(CYAN)Starting SFT training...$(RESET)"
	$(PYTHON) scripts/train_sft.py \
	  --model_config $(MODEL_CONFIG) \
	  --output_dir $(CHECKPOINT_DIR)/kairosyn-sft-v1

train-ppo: ## Run Phase 2: PPO Introspective RL
	@echo "$(CYAN)Starting PPO training...$(RESET)"
	$(PYTHON) scripts/train_ppo.py \
	  --config configs/training/ppo_config.yaml \
	  --sft_checkpoint $(CHECKPOINT_DIR)/kairosyn-sft-v1 \
	  --output_dir $(CHECKPOINT_DIR)/kairosyn-ppo-v1

train-maml: ## Run Phase 3: MAML Meta-Learning
	@echo "$(CYAN)Starting MAML training...$(RESET)"
	$(PYTHON) scripts/train_maml.py \
	  --config configs/training/maml_config.yaml \
	  --ppo_checkpoint $(CHECKPOINT_DIR)/kairosyn-ppo-v1 \
	  --output_dir $(CHECKPOINT_DIR)/kairosyn-maml-v1

train-all: train-sft train-ppo train-maml ## Run full 3-phase training pipeline

# ── Inference ────────────────────────────────────────────────────────────────
infer: ## Launch interactive inference CLI
	$(PYTHON) scripts/inference.py \
	  --config $(MODEL_CONFIG) \
	  --checkpoint $(CHECKPOINT_DIR)/kairosyn-maml-v1

infer-prompt: ## Run single-prompt inference (set PROMPT=... env var)
	$(PYTHON) scripts/inference.py \
	  --config $(MODEL_CONFIG) \
	  --prompt "$(PROMPT)" \
	  --introspection true

# ── Evaluation ───────────────────────────────────────────────────────────────
evaluate: ## Run full evaluation suite
	$(PYTHON) scripts/evaluate.py \
	  --config configs/evaluation/eval_config.yaml \
	  --checkpoint $(CHECKPOINT_DIR)/kairosyn-maml-v1 \
	  --output_dir ./results/eval-v1

# ── Kubernetes ───────────────────────────────────────────────────────────────
k8s-apply: ## Apply all Kubernetes manifests
	kubectl apply -f deploy/kubernetes/ -n kairosyn

k8s-status: ## Show Kubernetes deployment status
	kubectl get all -n kairosyn

k8s-logs: ## Show Kubernetes pod logs
	kubectl logs -f deployment/kairosyn -n kairosyn

k8s-rollback: ## Rollback the Kubernetes deployment
	kubectl rollout undo deployment/kairosyn -n kairosyn

k8s-port-forward: ## Port-forward the Kubernetes service locally
	kubectl port-forward svc/kairosyn-service $(PORT):$(PORT) -n kairosyn

# ── Utilities ─────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml .pytest_cache/ .mypy_cache/
	@echo "$(GREEN)Clean complete.$(RESET)"

clean-docker: ## Remove local Docker images for this project
	docker rmi $(IMAGE_NAME):latest $(FULL_IMAGE) 2>/dev/null || true
	docker system prune -f
	@echo "$(GREEN)Docker images removed.$(RESET)"

clean-checkpoints: ## Remove training checkpoints (IRREVERSIBLE)
	@read -p "Delete all checkpoints? [y/N] " confirm && \
	  [ "$$confirm" = "y" ] && rm -rf $(CHECKPOINT_DIR)/* && \
	  echo "$(GREEN)Checkpoints removed.$(RESET)" || echo "Cancelled."

version: ## Show versions of key tools
	@echo "Python  : $(shell python3 --version)"
	@echo "pip     : $(shell pip --version | cut -d' ' -f2)"
	@echo "Docker  : $(shell docker --version)"
	@echo "Compose : $(shell docker compose version)"
	@echo "Git SHA : $(shell git rev-parse --short HEAD)"
	@echo "Image   : $(FULL_IMAGE)"
