"""
LabPulse AI — Alert Engine
============================
Configurable alert system with webhook (Slack/Teams compatible)
and optional email notifications.
"""

import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Environment-based config
SMTP_SERVER = os.getenv("SMTP_SERVER", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", "alerts@labpulse.ai")
WEBHOOK_TIMEOUT = 10


class AlertRule:
    """Single alert rule definition."""

    def __init__(
        self,
        name: str,
        alert_type: str,
        threshold: float,
        enabled: bool = True,
    ):
        self.name = name
        self.alert_type = alert_type  # stockout_risk, virus_surge, low_confidence
        self.threshold = threshold
        self.enabled = enabled

    def evaluate(self, kpis: dict, model_info: dict = None) -> Optional[str]:
        """
        Evaluate this rule against current KPIs.
        Returns alert message or None if not triggered.
        """
        if not self.enabled:
            return None

        if self.alert_type == "stockout_risk":
            risk = kpis.get("risk_eur", 0)
            if risk >= self.threshold:
                stockout = kpis.get("stockout_day")
                so_str = (
                    f" (Stockout: {stockout.strftime('%d.%m.%Y')})"
                    if stockout else ""
                )
                return (
                    f"STOCKOUT-RISIKO: Revenue at Risk EUR {risk:,.0f} "
                    f"uebersteigt Schwellwert EUR {self.threshold:,.0f}"
                    f"{so_str}. Sofortige Nachbestellung empfohlen."
                )

        elif self.alert_type == "virus_surge":
            trend = abs(kpis.get("trend_pct", 0))
            if trend >= self.threshold:
                direction = "Anstieg" if kpis.get("trend_pct", 0) > 0 else "Rueckgang"
                return (
                    f"VIRUS-SIGNAL: {direction} von {kpis.get('trend_pct', 0):+.1f}% WoW "
                    f"uebersteigt Schwellwert {self.threshold:.0f}%. "
                    f"Pathogen: {kpis.get('pathogen', 'N/A')}."
                )

        elif self.alert_type == "low_confidence" and model_info:
            confidence = model_info.get("confidence_score", 100)
            if confidence < self.threshold:
                return (
                    f"MODELL-WARNUNG: Prognosequalitaet ({confidence:.0f}%) "
                    f"unter Schwellwert ({self.threshold:.0f}%). "
                    f"Modell: {model_info.get('model_type', 'N/A')}."
                )

        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "alert_type": self.alert_type,
            "threshold": self.threshold,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AlertRule":
        return cls(
            name=d["name"],
            alert_type=d["alert_type"],
            threshold=d["threshold"],
            enabled=d.get("enabled", True),
        )


# ── Default Alert Rules ──────────────────────────────────────────────────────
DEFAULT_RULES = [
    AlertRule("Stockout-Risiko", "stockout_risk", threshold=5000, enabled=True),
    AlertRule("Virus-Surge", "virus_surge", threshold=20.0, enabled=True),
    AlertRule("Modell-Konfidenz", "low_confidence", threshold=50.0, enabled=False),
]


class AlertManager:
    """Manages alert rules and notification delivery."""

    def __init__(self, rules: Optional[List[AlertRule]] = None):
        self.rules = rules or [AlertRule(**r.to_dict()) for r in DEFAULT_RULES]
        self._last_alerts: List[str] = []

    def evaluate_all(
        self,
        kpis: dict,
        model_info: dict = None,
    ) -> List[str]:
        """
        Evaluate all active rules against current KPIs.
        Returns list of triggered alert messages.
        """
        triggered = []
        for rule in self.rules:
            msg = rule.evaluate(kpis, model_info)
            if msg:
                triggered.append(msg)

        self._last_alerts = triggered
        return triggered

    def send_webhook(
        self,
        url: str,
        alerts: List[str],
        pathogen: str = "",
    ) -> bool:
        """
        Send alerts to a webhook URL (Slack/Teams/generic).
        Returns True on success.
        """
        if not url or not alerts:
            return False

        try:
            # Format for Slack-compatible webhook
            text = (
                f"*LabPulse AI Alert* — {pathogen}\n"
                f"_{datetime.now().strftime('%d.%m.%Y %H:%M')}_\n\n"
                + "\n".join(f"• {a}" for a in alerts)
            )

            payload = {"text": text}
            resp = requests.post(url, json=payload, timeout=WEBHOOK_TIMEOUT)
            resp.raise_for_status()
            logger.info("Webhook sent successfully to %s", url[:50])
            return True

        except Exception as exc:
            logger.warning("Webhook failed: %s", exc)
            return False

    def send_email(
        self,
        to_email: str,
        alerts: List[str],
        pathogen: str = "",
    ) -> bool:
        """
        Send alerts via SMTP email.
        Returns True on success.
        """
        if not SMTP_SERVER or not SMTP_USER or not to_email or not alerts:
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"LabPulse AI Alert — {pathogen}"
            msg["From"] = SMTP_FROM
            msg["To"] = to_email

            body_text = (
                f"LabPulse AI Alert — {pathogen}\n"
                f"{datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n"
                + "\n".join(f"- {a}" for a in alerts)
                + "\n\n---\nLabPulse AI"
            )

            body_html = (
                f"<h2>LabPulse AI Alert — {pathogen}</h2>"
                f"<p style='color:#94a3b8;'>{datetime.now().strftime('%d.%m.%Y %H:%M')}</p>"
                f"<ul>"
                + "".join(f"<li style='color:#fca5a5;'>{a}</li>" for a in alerts)
                + "</ul>"
                f"<hr><p style='color:#64748b;font-size:12px;'>LabPulse AI</p>"
            )

            msg.attach(MIMEText(body_text, "plain"))
            msg.attach(MIMEText(body_html, "html"))

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)

            logger.info("Email alert sent to %s", to_email)
            return True

        except Exception as exc:
            logger.warning("Email failed: %s", exc)
            return False

    def test_webhook(self, url: str) -> bool:
        """Send a test message to verify webhook connectivity."""
        return self.send_webhook(
            url,
            ["Dies ist eine Testnachricht von LabPulse AI."],
            pathogen="Test",
        )

    @property
    def last_alerts(self) -> List[str]:
        return self._last_alerts

    def get_rules_config(self) -> List[dict]:
        """Return rules as serializable dicts."""
        return [r.to_dict() for r in self.rules]

    def update_rule(self, index: int, **kwargs):
        """Update a rule by index."""
        if 0 <= index < len(self.rules):
            for key, val in kwargs.items():
                if hasattr(self.rules[index], key):
                    setattr(self.rules[index], key, val)
