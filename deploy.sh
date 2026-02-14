#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# LabPulse AI — Docker Deployment (safe for shared servers)
# Runs in its own container — won't touch your system Python,
# nginx, or any other services.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

APP_DIR="/opt/labpulse-ai"
REPO_URL="https://github.com/Fliegenbart/LabPulse-AI.git"

echo "═══════════════════════════════════════════"
echo "  🧬 LabPulse AI — Docker Deployment"
echo "═══════════════════════════════════════════"

# ── 1. Check Docker ──────────────────────────────────────────
echo "[1/4] Checking Docker …"
if ! command -v docker &> /dev/null; then
    echo "  → Docker not found. Installing …"
    curl -fsSL https://get.docker.com | sh
fi

if ! command -v docker compose &> /dev/null; then
    echo "  → docker compose plugin not found. Installing …"
    apt-get update -qq && apt-get install -y -qq docker-compose-plugin
fi

echo "  → Docker $(docker --version | awk '{print $3}') ✓"

# ── 2. Clone / Update Repo ──────────────────────────────────
echo "[2/4] Setting up application …"
if [ -d "$APP_DIR" ]; then
    echo "  → Pulling latest changes …"
    cd "$APP_DIR"
    git pull --ff-only
else
    echo "  → Cloning repository …"
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# ── 3. Build & Start Container ──────────────────────────────
echo "[3/4] Building and starting container …"
docker compose down --remove-orphans 2>/dev/null || true
docker compose up -d --build

# ── 4. Verify ────────────────────────────────────────────────
echo "[4/4] Waiting for health check …"
sleep 5

if docker ps --filter "name=labpulse-ai" --filter "status=running" -q | grep -q .; then
    SERVER_IP=$(hostname -I | awk '{print $1}')
    echo ""
    echo "═══════════════════════════════════════════"
    echo "  ✅ LabPulse AI is live!"
    echo "  → http://${SERVER_IP}:8080"
    echo ""
    echo "  Your other services are untouched:"
    echo "    voxdrop.live         — no changes"
    echo "    controlling-engine.de — no changes"
    echo ""
    echo "  Useful commands:"
    echo "    docker compose -f ${APP_DIR}/docker-compose.yml logs -f"
    echo "    docker compose -f ${APP_DIR}/docker-compose.yml restart"
    echo "    docker compose -f ${APP_DIR}/docker-compose.yml down"
    echo "═══════════════════════════════════════════"
else
    echo "⚠️  Container did not start. Check logs:"
    echo "    docker compose -f ${APP_DIR}/docker-compose.yml logs"
    exit 1
fi
