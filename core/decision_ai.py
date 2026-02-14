"""
LabPulse AI — Ollama Decision Layer
===================================
Small, robust wrapper for local Ollama endpoint with fallback comments.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class OllamaSettings:
    host: str = "http://host.docker.internal:11434"
    model: str = "qwen2.5:7b"
    timeout: int = 30


def _env_value(name: str, fallback: str) -> str:
    import os

    return os.getenv(name, fallback).strip()


def _build_settings() -> OllamaSettings:
    return OllamaSettings(
        host=_env_value("OLLAMA_HOST", "http://host.docker.internal:11434"),
        model=_env_value("OLLAMA_MODEL", "qwen2.5:7b"),
        timeout=int(_env_value("OLLAMA_TIMEOUT", "30")),
    )


class DecisionNarrator:
    def __init__(self) -> None:
        self.settings = _build_settings()

    @property
    def endpoint(self) -> str:
        return f"{self.settings.host.rstrip('/')}/api/generate"

    def ping(self) -> bool:
        try:
            resp = requests.get(
                f"{self.settings.host.rstrip('/')}/api/tags",
                timeout=min(5, self.settings.timeout),
            )
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("Ollama ping failed: %s", exc)
            return False

    def _query(self, prompt: str, temperature: float = 0.25, max_tokens: int = 512) -> Optional[str]:
        payload = {
            "model": self.settings.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.settings.timeout)
            response.raise_for_status()
            data = response.json()
            return (data.get("response") or "").strip()
        except Exception as exc:
            logger.warning("Ollama generation failed: %s", exc)
            return None

    def summarize(self, context: Dict[str, object]) -> str:
        if not self.ping():
            return ""

        if not isinstance(context, dict):
            return ""

        summary_lines = []
        for key, value in context.items():
            summary_lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        context_block = "\n".join(summary_lines)

        system_prompt = (
            "Du bist ein Senior Analytics-Lead in der Labordiagnostik.\n"
            "Formuliere eine klare Management-Einschätzung in 5 kurzen Sätzen.\n"
            "Nutze konkrete Zahlen aus den KPIs und gib konkrete nächste Schritte.\n"
            "Keine Aufzählung, reiner Fließtext in professionellem Ton."
        )
        user_prompt = (
            "Bitte interpretiere den aktuellen Zustand.\n\n"
            f"{context_block}\n\n"
            "Antworte mit Handlungsempfehlung, Risiko-Bewertung und Entscheidungslogik für heute."
        )
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        answer = self._query(full_prompt, temperature=0.3, max_tokens=720)
        return answer.strip() if answer else ""

    @staticmethod
    def fallback_summary(context: Dict[str, object]) -> str:
        risk = float(context.get("risk_eur", 0.0) or 0.0)
        trend = float(context.get("trend_pct", 0.0) or 0.0)
        stock = int(context.get("stock_on_hand", 0) or 0)
        order_sum = int(context.get("recommended_order_sum", 0) or 0)
        pathogen = str(context.get("pathogen", "Pathogen"))
        return (
            f"Für {pathogen} bleibt der Trend {trend:+.1f}% gegenüber Vorwoche, "
            f"Lagerbestand beträgt aktuell {stock:,} Einheiten. "
            f"Der prognostizierte Fehlbestand-Risiko-Wert liegt bei EUR {risk:,.2f}. "
            f"Empfehlung: {order_sum:,} Tests als Bestellempfehlung priorisieren und "
            f"die Verfügbarkeit der kritischen Verbrauchsmaterialien bis Ende Woche prüfen."
        )
