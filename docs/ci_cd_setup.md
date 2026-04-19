# KAIROSYN-1 CI/CD Setup Guide
## *GitHub Actions — Full Pipeline Configuration*

---

## Overview

The CI/CD pipeline automates the full lifecycle:

```
Code Push → Lint → Test → Docker Build → Push to GHCR
              ↓ (main branch only)
         Deploy to Staging → Health Check
              ↓ (GitHub Release only)
         Deploy to Production → Verify → Notify
```

---

## GitHub Actions Workflow

Create the file `.github/workflows/ci.yml` in your repository with the following content:

```yaml
name: KAIROSYN-1 CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  release:
    types: [published]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository_owner }}/kairosyn-1
  PYTHON_VERSION: "3.11"

jobs:

  lint:
    name: Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - run: pip install black isort flake8
      - run: black --check --diff kairosyn/ scripts/
      - run: isort --check-only --diff kairosyn/ scripts/
      - run: flake8 kairosyn/ scripts/ --max-line-length=100 --extend-ignore=E203,W503,E501

  test:
    name: Tests
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - name: Install CPU-only torch + package
        run: |
          pip install torch==2.3.0 torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu
          pip install pytest pytest-cov pyyaml loguru einops rich click \
            transformers accelerate peft fastapi pydantic
          pip install -e . --no-deps
      - name: Run tests
        run: |
          pytest tests/ -v --cov=kairosyn --cov-report=xml \
            -m "not gpu and not slow" --timeout=120
      - uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: false

  docker:
    name: Docker Build & Push
    runs-on: ubuntu-latest
    needs: test
    permissions:
      contents: read
      packages: write
    outputs:
      tags: ${{ steps.meta.outputs.tags }}
      digest: ${{ steps.build.outputs.digest }}
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha,prefix=sha-,format=short
            type=raw,value=latest,enable=${{ github.ref == 'refs/heads/main' }}
      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.prod
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64
          build-args: |
            GIT_SHA=${{ github.sha }}
            BUILD_DATE=${{ github.event.repository.updated_at }}
            VERSION=${{ steps.meta.outputs.version }}

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: docker
    if: github.event_name != 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - name: Run Trivy scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ghcr.io/${{ github.repository_owner }}/kairosyn-1:latest
          format: table
          exit-code: "0"
          ignore-unfixed: true
          severity: CRITICAL,HIGH

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [docker, security]
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    environment:
      name: staging
      url: https://kairosyn-staging.or4cl3.ai
    steps:
      - uses: actions/checkout@v4
      - name: Deploy via SSH
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.STAGING_HOST }}
          username: ${{ secrets.STAGING_USER }}
          key: ${{ secrets.STAGING_SSH_KEY }}
          script: |
            cd /opt/kairosyn
            git pull origin main
            echo "${{ secrets.GITHUB_TOKEN }}" | \
              docker login ghcr.io -u ${{ github.actor }} --password-stdin
            docker compose -f docker/docker-compose.prod.yml pull
            docker compose -f docker/docker-compose.prod.yml up -d --remove-orphans
            docker system prune -f
      - name: Health check (5 min window)
        run: |
          for i in {1..20}; do
            STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
              https://kairosyn-staging.or4cl3.ai/health 2>/dev/null || echo 000)
            [ "$STATUS" = "200" ] && echo "Staging healthy on attempt $i" && exit 0
            echo "Attempt $i: HTTP $STATUS — waiting 15s..."
            sleep 15
          done
          echo "Health check timed out" && exit 1

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: deploy-staging
    if: github.event_name == 'release' && github.event.action == 'published'
    environment:
      name: production
      url: https://kairosyn.or4cl3.ai
    steps:
      - uses: actions/checkout@v4
      - uses: azure/setup-kubectl@v3
        with: { version: v1.29.0 }
      - name: Configure kubeconfig
        run: |
          mkdir -p ~/.kube
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config
          chmod 600 ~/.kube/config
      - name: Rolling deploy
        run: |
          kubectl set image deployment/kairosyn \
            kairosyn=ghcr.io/${{ github.repository_owner }}/kairosyn-1:${{ github.event.release.tag_name }} \
            -n kairosyn
          kubectl rollout status deployment/kairosyn -n kairosyn --timeout=300s
      - name: Verify & rollback on failure
        run: |
          for i in {1..10}; do
            STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
              https://kairosyn.or4cl3.ai/health 2>/dev/null || echo 000)
            [ "$STATUS" = "200" ] && echo "Production healthy" && exit 0
            sleep 20
          done
          echo "Production unhealthy — rolling back"
          kubectl rollout undo deployment/kairosyn -n kairosyn
          exit 1
```

---

## Required GitHub Secrets

Navigate to **Settings → Secrets and variables → Actions → New repository secret**:

| Secret Name | Description | How to Get |
|---|---|---|
| `STAGING_HOST` | IP or hostname of staging server | Your VPS/cloud instance |
| `STAGING_USER` | SSH username on staging server | Usually `ubuntu` or `root` |
| `STAGING_SSH_KEY` | PEM-formatted private SSH key | `cat ~/.ssh/id_rsa` |
| `KUBE_CONFIG` | Base64-encoded kubeconfig | `base64 -w 0 ~/.kube/config` |

The `GITHUB_TOKEN` is automatically provided — no setup needed for GHCR push.

---

## Staging Server Setup

On your staging server, run once:

```bash
# Install Docker + NVIDIA Container Toolkit
curl -fsSL https://get.docker.com | sh
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Create deploy directory
sudo mkdir -p /opt/kairosyn
sudo chown $USER:$USER /opt/kairosyn
cd /opt/kairosyn

# Clone repo and configure
git clone https://github.com/or4cl3-ai-1/Kairosyn-1.git .
cp .env.example .env
nano .env    # Set HF_TOKEN, GRAFANA_PASS, ACME_EMAIL
```

---

## Triggering Deployments

| Action | Trigger | Target |
|--------|---------|--------|
| Push to `main` | Automatic | Staging |
| Open PR | Tests + build only | — |
| Publish GitHub Release | Manual | Production |

```bash
# Create and publish a release
git tag v1.1.0 && git push origin v1.1.0
# Then go to GitHub → Releases → Draft a new release → select tag → Publish
```

---

## Pipeline Status Badges

Add to your README:

```markdown
![CI](https://github.com/or4cl3-ai-1/Kairosyn-1/actions/workflows/ci.yml/badge.svg)
```

---

*Or4cl3 AI Solutions — research@or4cl3.ai*
