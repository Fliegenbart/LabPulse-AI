#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# LabPulse AI — Deployment Script for Ubuntu (Hetzner GEX44)
# Usage: ssh into your server, clone the repo, run this script.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

APP_DIR="/opt/labpulse-ai"
SERVICE_NAME="labpulse-ai"
REPO_URL="https://github.com/Fliegenbart/LabPulse-AI.git"
PYTHON_MIN="3.10"

echo "═══════════════════════════════════════════"
echo "  🧬 LabPulse AI — Server Deployment"
echo "═══════════════════════════════════════════"

# ── 1. System dependencies ───────────────────────────────────
echo "[1/5] Installing system dependencies …"
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git

# ── 2. Clone / update repo ───────────────────────────────────
echo "[2/5] Setting up application directory …"
if [ -d "$APP_DIR" ]; then
    echo "  → Updating existing installation …"
    cd "$APP_DIR"
    git pull --ff-only
else
    echo "  → Cloning repository …"
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# ── 3. Python virtual environment ────────────────────────────
echo "[3/5] Creating Python virtual environment …"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
deactivate

# ── 4. Systemd service ──────────────────────────────────────
echo "[4/5] Installing systemd service …"
cat > /etc/systemd/system/${SERVICE_NAME}.service <<EOF
[Unit]
Description=LabPulse AI — Streamlit Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/venv/bin/streamlit run app.py
Restart=on-failure
RestartSec=5
Environment="PATH=${APP_DIR}/venv/bin:/usr/bin"

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${SERVICE_NAME}
systemctl restart ${SERVICE_NAME}

# ── 5. Firewall ─────────────────────────────────────────────
echo "[5/5] Opening port 8501 …"
if command -v ufw &> /dev/null; then
    ufw allow 8501/tcp
    echo "  → ufw rule added."
else
    echo "  → No ufw detected. Make sure port 8501 is open in your Hetzner firewall."
fi

# ── Done ─────────────────────────────────────────────────────
SERVER_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ LabPulse AI is live!"
echo "  → http://${SERVER_IP}:8501"
echo ""
echo "  Useful commands:"
echo "    systemctl status  ${SERVICE_NAME}"
echo "    journalctl -u ${SERVICE_NAME} -f"
echo "    systemctl restart ${SERVICE_NAME}"
echo "═══════════════════════════════════════════"
