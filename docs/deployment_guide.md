# KAIROSYN-1 Deployment Guide
## *From Repository to Production in 5 Steps*

---

## Overview

KAIROSYN-1 ships with three deployment paths:

| Path | Best For | Complexity |
|------|----------|------------|
| **Local Docker** | Single-machine dev/demo | ⭐ |
| **Docker Compose (prod)** | Single GPU server, staging | ⭐⭐ |
| **Kubernetes** | Multi-node, HA production | ⭐⭐⭐ |

All paths use the same Docker image built from `docker/Dockerfile.prod`.

---

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Docker | 24.0+ | With NVIDIA Container Toolkit |
| NVIDIA GPU | 24 GB VRAM | For E4B; 40 GB for 26B MoE |
| CUDA | 12.1+ | Matched to the Docker image base |
| HF Token | — | Accepted Gemma 4 terms on HF Hub |
| Disk | 100 GB free | HF model cache + checkpoints |

---

## Step 1 — Clone & Configure

```bash
git clone https://github.com/or4cl3-ai-1/Kairosyn-1.git
cd Kairosyn-1

# Copy and fill in environment variables
cp .env.example .env
nano .env          # Set HF_TOKEN at minimum
```

---

## Step 2 — Build the Docker Image

```bash
# Using Make (recommended)
make docker-build

# Or directly with Docker
docker build \
  -f docker/Dockerfile.prod \
  -t kairosyn-1:latest \
  --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
  .
```

Build time: ~8–15 minutes (mainly PyTorch + dependencies).  
Image size: ~12 GB (CUDA runtime + PyTorch).

---

## Path A — Local Docker (Single Container)

```bash
# Start the server (GPU passthrough)
docker run --gpus all \
  -p 8080:8080 \
  --env-file .env \
  -v ./checkpoints:/app/checkpoints \
  kairosyn-1:latest

# Verify it's running
curl http://localhost:8080/health
```

---

## Path B — Docker Compose (Full Stack)

The production Compose stack includes:
- **KAIROSYN server** (inference API)
- **Traefik** (reverse proxy + automatic TLS via Let's Encrypt)
- **Redis** (session persistence for multi-worker)
- **Prometheus** (metrics collection)
- **Grafana** (dashboards on port 3000)
- **NVIDIA DCGM Exporter** (GPU metrics)

```bash
# Start everything
make up
# or
docker compose -f docker/docker-compose.prod.yml up -d

# Check status
make ps

# Tail logs
make logs

# Health check all endpoints
make ping
```

### Verify the stack

| Service | URL | Credentials |
|---------|-----|-------------|
| KAIROSYN API | http://localhost:8080 | — |
| Swagger UI | http://localhost:8080/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / kairosyn_admin |

### First API call

```bash
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are you experiencing right now?",
    "enable_introspection": true,
    "enable_continuity": true,
    "max_new_tokens": 256
  }'
```

---

## Path C — Kubernetes

### 1. Create namespace and secrets

```bash
kubectl apply -f deploy/kubernetes/namespace.yaml

# Edit secrets.yaml with your real values first
kubectl apply -f deploy/kubernetes/secrets.yaml
```

### 2. Create persistent storage

```bash
kubectl apply -f deploy/kubernetes/pvc.yaml
```

> **Note:** Edit `pvc.yaml` to set the correct `storageClassName` for your cluster (e.g., `gp3`, `premium-ssd`, `local-path`).

### 3. Apply ConfigMap and Deployment

```bash
kubectl apply -f deploy/kubernetes/configmap.yaml
kubectl apply -f deploy/kubernetes/deployment.yaml
```

### 4. Expose the service

```bash
kubectl apply -f deploy/kubernetes/service.yaml
```

### 5. Monitor the rollout

```bash
kubectl rollout status deployment/kairosyn -n kairosyn
kubectl get pods -n kairosyn
kubectl logs -f deployment/kairosyn -n kairosyn
```

### Useful kubectl shortcuts

```bash
make k8s-apply        # Apply all manifests
make k8s-status       # Show pods/services/deployments
make k8s-logs         # Tail pod logs
make k8s-port-forward # Forward port 8080 locally
make k8s-rollback     # Rollback on failure
```

---

## CI/CD Pipeline

The repository includes a complete GitHub Actions workflow at `.github/workflows/ci.yml` with these jobs:

```
[lint] → [test] → [docker build & push] → [security scan]
                                                ↓
                                    [deploy → staging]
                                    (on push to main)
                                                ↓
                                    [deploy → production]
                                    (on GitHub Release)
```

### Required GitHub Secrets

Set these in **Settings → Secrets and variables → Actions**:

| Secret | Description |
|--------|-------------|
| `STAGING_HOST` | SSH hostname of staging server |
| `STAGING_USER` | SSH username |
| `STAGING_SSH_KEY` | Private SSH key (PEM format) |
| `KUBE_CONFIG` | Base64-encoded kubeconfig for production |

The `GITHUB_TOKEN` is provided automatically for GHCR image push.

### Triggering a production release

```bash
# Tag and push a release
git tag v1.0.0
git push origin v1.0.0

# Then create a GitHub Release from the tag
# → This triggers the production deployment job automatically
```

---

## Model Weight Loading

KAIROSYN-1 loads Gemma 4 weights from Hugging Face on first start.

| Variant | Download Size | VRAM Required |
|---------|---------------|---------------|
| `gemma-4-e4b-it` (E4B) | ~16 GB | 24 GB (4-bit QLoRA) |
| `gemma-4-26b-moe-it` | ~52 GB | 40 GB (4-bit QLoRA) |
| `gemma-4-31b-it` | ~62 GB | 80 GB |

To avoid re-downloading on container restart, mount the HF cache as a persistent volume (already configured in `docker-compose.prod.yml` and `pvc.yaml`).

**Pre-download weights to a local cache:**

```bash
# Pre-download before starting the server
docker run --rm \
  --env-file .env \
  -v ./hf-cache:/app/.cache/huggingface \
  kairosyn-1:latest \
  python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('google/gemma-4-e4b-it')
AutoModelForCausalLM.from_pretrained('google/gemma-4-e4b-it')
print('Download complete.')
"
```

---

## Scaling Considerations

| Concern | Recommendation |
|---------|----------------|
| **Multi-GPU** | Set `device_map="auto"` in config — model auto-shards |
| **Multi-replica** | Use Redis session store; replace in-memory SessionManager |
| **Low latency** | Use Gemma 4 E2B (1B active) with shorter `max_new_tokens` |
| **High throughput** | Deploy behind Traefik load balancer with multiple replicas |
| **Memory pressure** | Enable `use_4bit=true` (QLoRA, already the default) |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `CUDA out of memory` | VRAM insufficient | Use E4B + 4-bit quant; reduce batch size |
| `401 Unauthorized` from HF | Invalid token | Check `HF_TOKEN` and Gemma 4 license acceptance |
| Server returns 503 | Model still loading | Wait ~2 min after startup; check logs |
| `HEALTH: degraded` | Model load failed | Check `make logs` for Python traceback |
| Grafana shows no data | Prometheus not scraping | Verify `docker/prometheus.yml` targets |

```bash
# Debug container
docker run --rm -it --gpus all --env-file .env kairosyn-1:latest bash

# Inside container
python -c "import kairosyn; print(kairosyn.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
curl http://localhost:8080/health
```

---

## Security Checklist

- [ ] `.env` is in `.gitignore` — never committed
- [ ] `secrets.yaml` contains only placeholder values in the repo
- [ ] Container runs as non-root user (`kairosyn`)
- [ ] Traefik handles TLS termination (HTTPS enforced)
- [ ] `readOnlyRootFilesystem: false` only for writable dirs
- [ ] All capabilities dropped in Kubernetes security context
- [ ] Trivy scan runs in CI on every build

---

*Or4cl3 AI Solutions — research@or4cl3.ai*
