"""
LabPulse AI — Ollama AI Client
================================
Connects to a local Ollama instance to generate natural-language
insights about dashboard data. Falls back to rule-based German
commentary when Ollama is unreachable.
"""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))


class OllamaClient:
    """Lightweight wrapper around the Ollama REST API."""

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    # ── Health ────────────────────────────────────────────────────────────────
    def health_check(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return list of available model names."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ── Generation ────────────────────────────────────────────────────────────
    def _generate(self, prompt: str, system: str = "") -> Optional[str]:
        """Call Ollama /api/generate and return the full response text."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": 512,
                },
            }
            resp = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as exc:
            logger.warning("Ollama generation failed: %s", exc)
            return None

    # ── Dashboard Insight ─────────────────────────────────────────────────────
    def generate_insight(self, kpis: dict, pathogen: str) -> str:
        """
        Generate a short (3-5 sentence) German insight about the current
        dashboard state. Falls back to rule-based text if Ollama is down.
        """
        system_prompt = (
            "Du bist ein erfahrener Labordiagnostik-Analyst. "
            "Deine Aufgabe: KPI-Daten in 3-5 klare, handlungsorientierte Saetze auf Deutsch zusammenfassen. "
            "Kein Markdown, keine Aufzaehlungen. Schreibe Fliesstext. "
            "Fokussiere auf: Trend, Risiko, empfohlene Massnahmen."
        )

        data_summary = (
            f"Pathogen: {pathogen}\n"
            f"Prognostizierte Tests (7 Tage): {kpis.get('predicted_tests_7d', 'N/A'):,}\n"
            f"Umsatzprognose (7 Tage): EUR {kpis.get('revenue_forecast_7d', 0):,.0f}\n"
            f"Reagenzstatus: {kpis.get('reagent_status', 'N/A')}\n"
            f"Revenue at Risk: EUR {kpis.get('risk_eur', 0):,.0f}\n"
            f"WoW-Trend: {kpis.get('trend_pct', 0):+.1f}%\n"
            f"Lagerbestand: {kpis.get('stock_on_hand', 0):,} Einheiten\n"
            f"Gesamtbedarf (Horizont): {kpis.get('total_demand', 0):,} Einheiten\n"
            f"Stockout-Tag: {kpis.get('stockout_day', 'keiner')}\n"
            f"Test-Kit: {kpis.get('test_name', 'N/A')}\n"
            f"Kosten/Test: EUR {kpis.get('cost_per_test', 45)}"
        )

        prompt = (
            f"Analysiere diese aktuellen Labor-KPIs und gib eine kompakte Einschaetzung:\n\n"
            f"{data_summary}\n\n"
            f"Fasse zusammen: Was ist der aktuelle Status? Welches Risiko besteht? "
            f"Was sollte das Labor jetzt tun?"
        )

        result = self._generate(prompt, system=system_prompt)
        if result:
            return result
        return self._fallback_insight(kpis, pathogen)

    # ── PDF Commentary ────────────────────────────────────────────────────────
    def generate_pdf_commentary(self, kpis: dict, pathogen: str) -> str:
        """
        Generate a longer (5-8 sentence) German commentary for the PDF report.
        """
        system_prompt = (
            "Du bist ein Labordiagnostik-Analyst und schreibst einen Abschnitt fuer "
            "einen Management-Report. Schreibe professionell, praegnant, auf Deutsch. "
            "5-8 Saetze Fliesstext. Kein Markdown."
        )

        data_summary = (
            f"Pathogen: {pathogen}\n"
            f"Prognostizierte Tests (7d): {kpis.get('predicted_tests_7d', 'N/A'):,}\n"
            f"Umsatzprognose (7d): EUR {kpis.get('revenue_forecast_7d', 0):,.0f}\n"
            f"Reagenzstatus: {kpis.get('reagent_status', 'N/A')}\n"
            f"Revenue at Risk: EUR {kpis.get('risk_eur', 0):,.0f}\n"
            f"WoW-Trend: {kpis.get('trend_pct', 0):+.1f}%\n"
            f"Lagerbestand: {kpis.get('stock_on_hand', 0):,}\n"
            f"Gesamtbedarf: {kpis.get('total_demand', 0):,}\n"
            f"Stockout: {kpis.get('stockout_day', 'keiner')}\n"
            f"Kit: {kpis.get('test_name', 'N/A')} (EUR {kpis.get('cost_per_test', 45)}/Test)"
        )

        prompt = (
            f"Schreibe einen Management-Report-Absatz zu diesen KPIs:\n\n"
            f"{data_summary}\n\n"
            f"Inhalt: Zusammenfassung, Trend-Einordnung, Risikobewertung, Handlungsempfehlung."
        )

        result = self._generate(prompt, system=system_prompt)
        if result:
            return result
        return self._fallback_insight(kpis, pathogen)

    # ── Fallback (Rule-Based) ─────────────────────────────────────────────────
    @staticmethod
    def _fallback_insight(kpis: dict, pathogen: str) -> str:
        """Rule-based German commentary when Ollama is unavailable."""
        parts = []

        trend = kpis.get("trend_pct", 0)
        if trend > 10:
            parts.append(
                f"Das Testvolumen fuer {pathogen} zeigt einen deutlichen Anstieg "
                f"von {trend:+.1f}% gegenueber der Vorwoche."
            )
        elif trend < -10:
            parts.append(
                f"Das Testvolumen fuer {pathogen} ist um {trend:+.1f}% "
                f"gegenueber der Vorwoche zurueckgegangen."
            )
        else:
            parts.append(
                f"Das Testvolumen fuer {pathogen} bleibt stabil "
                f"({trend:+.1f}% WoW)."
            )

        risk = kpis.get("risk_eur", 0)
        if risk > 0:
            stockout = kpis.get("stockout_day")
            so_str = (
                f" Der prognostizierte Stockout-Tag ist "
                f"{stockout.strftime('%d.%m.%Y')}."
                if stockout else ""
            )
            parts.append(
                f"Revenue at Risk: EUR {risk:,.0f}.{so_str} "
                f"Eine sofortige Nachbestellung von {kpis.get('test_name', 'Reagenzien')} "
                f"wird empfohlen."
            )
        else:
            parts.append(
                f"Der Lagerbestand von {kpis.get('stock_on_hand', 0):,} Einheiten "
                f"deckt den prognostizierten Bedarf vollstaendig ab."
            )

        pred = kpis.get("predicted_tests_7d", 0)
        rev = kpis.get("revenue_forecast_7d", 0)
        parts.append(
            f"Fuer die naechsten 7 Tage werden ca. {pred:,} Tests prognostiziert "
            f"(Umsatzprognose: EUR {rev:,.0f})."
        )

        return " ".join(parts)


# ── Module-level convenience ──────────────────────────────────────────────────
_client: Optional[OllamaClient] = None


def get_client() -> OllamaClient:
    """Return a module-level singleton OllamaClient."""
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
