"""
LabPulse AI — Dashboard (Zen Mode)
====================================
Progressive Disclosure: Chart first, everything else on demand.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import uuid
import io
import zipfile
import hmac
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from data_engine import (
    fetch_rki_raw, fetch_rki_wastewater, get_available_pathogens,
    generate_lab_volume, build_forecast, AVG_REVENUE_PER_TEST,
    PATHOGEN_REAGENT_MAP, PATHOGEN_GROUPS,
)
from modules.ollama_client import get_client as get_ollama_client
from modules.pdf_export import generate_report as generate_pdf_report
from modules.lab_data_merger import validate_csv, merge_with_synthetic
from modules.regional_aggregation import (
    aggregate_by_bundesland, create_scatter_map_data
)
from modules.ml_forecaster import LabVolumeForecaster
from modules.alert_engine import AlertManager
from modules.external_data import (
    fetch_grippeweb, fetch_are_konsultation, fetch_google_trends,
    PATHOGEN_SEARCH_TERMS,
)
from modules.signal_fusion import fuse_all_signals, SIGNAL_CONFIG

st.set_page_config(
    page_title="LabPulse Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

AUDIT_LOG_PATH = "/app/data/audit_events.jsonl"
DECISION_LEDGER_PATH = "/app/data/decision_ledger.jsonl"
FORECAST_META_PATH = "/app/data/forecast_metadata.jsonl"
OPERATIONS_PAYLOAD_PATH = "/app/data/forecast_operations.jsonl"
SIGNING_SECRET = os.getenv("LABPULSE_SIGNING_SECRET")
APPROVAL_MATRIX = {
    "Niedrig": {
        "required_signatures": 1,
        "sla_hours": 24,
        "narrative": "Standard-Freigabe für operativen Routinelauf.",
    },
    "Mittel": {
        "required_signatures": 1,
        "sla_hours": 12,
        "narrative": "Zusätzliche Kontrolle durch fachliche Freigabe.",
    },
    "Hoch": {
        "required_signatures": 2,
        "sla_hours": 8,
        "narrative": "Hochpriorität: zwei Unterschriften, sofortige Verantwortungszuordnung.",
    },
    "Krisenmodus": {
        "required_signatures": 2,
        "sla_hours": 2,
        "narrative": "Kritisch: zweistufige Freigabe, strikter Audit-Trail.",
    },
}
DEFAULT_RETENTION_DAYS = 30


def _is_valid_webhook_url(url: str | None) -> bool:
    try:
        parsed = urlparse((url or "").strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False
SCENARIO_PROFILES = {
    "Basis": {
        "description": "Ausgeglichen – operatives Basisszenario.",
        "safety_buffer_add": 0,
        "uplift_add": 0.0,
        "lead_time_days": 3,
    },
    "Pessimistisch": {
        "description": "Vorsichtig – erhöhte Reserve für kurzfristige Signalanstiege.",
        "safety_buffer_add": 5,
        "uplift_add": 12.0,
        "lead_time_days": 2,
    },
    "Krisenfall": {
        "description": "Maximal vorsichtig – Krisenmodus für belastbare Handlung.",
        "safety_buffer_add": 10,
        "uplift_add": 28.0,
        "lead_time_days": 1,
    },
}


def _resolve_scenario(
    scenario_name: str,
    safety_buffer_base: int,
    scenario_uplift_base: float,
) -> tuple[int, float, dict]:
    profile = SCENARIO_PROFILES.get(scenario_name, SCENARIO_PROFILES["Basis"])
    return (
        int(max(0, safety_buffer_base + profile["safety_buffer_add"])),
        float(max(0.0, scenario_uplift_base + profile["uplift_add"])),
        profile,
    )


def _fmt_money(value: float) -> str:
    try:
        return f"EUR {float(value):,.0f}"
    except Exception:
        return "EUR 0"


def _risk_band_by_confidence(risk_eur: float, confidence_pct: float | None) -> dict[str, float]:
    conf = max(0.0, min(float(confidence_pct or 0), 100.0))
    c = conf / 100.0
    base = float(risk_eur or 0)
    return {
        "optimistic": int(base * c),
        "base": int(base),
        "stressed": int(base * (1.0 + (1.0 - c))),
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_to_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _iso_for_file_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _approval_policy(urgency: str) -> dict:
    return APPROVAL_MATRIX.get(urgency, APPROVAL_MATRIX["Mittel"]).copy()


def _decision_state_store() -> dict:
    if "decision_approval_states" not in st.session_state:
        st.session_state.decision_approval_states = {}
    return st.session_state.decision_approval_states


def _deterministic_payload(value) -> str:
    normalized = value if value is not None else {}
    return json.dumps(
        _to_json_safe(normalized),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest_payload(value) -> str:
    return hashlib.sha256(_deterministic_payload(value).encode("utf-8")).hexdigest()


def _sign_payload(value, secret: str | None = None) -> str:
    secret_value = (secret or SIGNING_SECRET or "").encode("utf-8")
    if not secret_value:
        return "unsigned"
    return hmac.new(secret_value, _deterministic_payload(value).encode("utf-8"), hashlib.sha256).hexdigest()


def _build_decision_id(
    selected_pathogen: str,
    forecast_horizon: int,
    scenario_name: str,
    safety_buffer: float,
    scenario_uplift: float,
    stock_on_hand: int,
    lab_rows: int,
    wastewater_rows: int,
) -> str:
    fingerprint = {
        "pathogen": _safe_text(selected_pathogen, "unknown"),
        "horizon": int(forecast_horizon),
        "scenario": _safe_text(scenario_name, "Basis"),
        "buffer": float(safety_buffer),
        "uplift": float(scenario_uplift),
        "stock": int(stock_on_hand),
        "lab_rows": int(lab_rows),
        "ww_rows": int(wastewater_rows),
        "day": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    return _digest_payload(fingerprint)[:24]


def _get_approval_state(
    decision_id: str,
    urgency: str,
    context_signature: str,
) -> tuple[dict, bool]:
    policy = _approval_policy(urgency)
    store = _decision_state_store()
    state = store.get(decision_id)
    is_new_state = False

    if not state or state.get("context_signature") != context_signature:
        state = {
            "decision_id": decision_id,
            "context_signature": context_signature,
            "status": "Ausstehend",
            "urgency": urgency,
            "required_signatures": int(policy["required_signatures"]),
            "sla_hours": int(policy["sla_hours"]),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "approvals": [],
            "notes": [],
            "chain_hash": None,
            "narrative": policy.get("narrative", ""),
            "policy": policy,
        }
        is_new_state = True
    else:
        # Keep policy in-sync if urgency mapping changed.
        state["policy"] = policy
        state["required_signatures"] = int(policy["required_signatures"])
        state["sla_hours"] = int(policy["sla_hours"])
        state["narrative"] = policy.get("narrative", state.get("narrative", ""))
        state["urgency"] = urgency

    if state["required_signatures"] <= 0:
        state["status"] = "Freigegeben"
    elif len(state.get("approvals", [])) >= state["required_signatures"]:
        state["status"] = "Freigegeben"
    elif len(state.get("approvals", [])) > 0:
        state["status"] = "Teilweise freigegeben"
    else:
        state["status"] = "Ausstehend"

    store[decision_id] = state
    return state, is_new_state


def _actor_already_signed(state: dict, actor: str) -> bool:
    for entry in state.get("approvals", []):
        if entry.get("actor") == actor:
            return True
    return False


def _can_approve(state: dict) -> bool:
    return state.get("status") != "Freigegeben"


def _append_approval_signature(
    decision_id: str,
    state: dict,
    actor: str,
    comment: str,
) -> bool:
    if _actor_already_signed(state, actor):
        return False
    now = _now_iso()
    state["approvals"] = state.get("approvals", [])
    state["approvals"].append({
        "actor": actor,
        "approved_at": now,
        "comment": _safe_text(comment),
    })
    state["status"] = "Freigegeben" if len(state["approvals"]) >= state["required_signatures"] else "Teilweise freigegeben"
    state["updated_at"] = now

    event = {
        "event_type": "decision_signature",
        "event_id": str(uuid.uuid4()),
        "decision_id": decision_id,
        "state_status": state["status"],
        "signed_count": int(len(state["approvals"])),
        "required_signatures": int(state["required_signatures"]),
        "actor": {"name": actor},
        "comment": _safe_text(comment),
        "session_id": st.session_state.get("audit_session_id", ""),
        "at": now,
    }
    event["prev_chain_hash"] = _latest_chain_hash()
    event["event_hash"] = _digest_payload(event)
    event["event_signature"] = _sign_payload(event, SIGNING_SECRET)
    _append_jsonl(DECISION_LEDGER_PATH, event)
    state["chain_hash"] = event["event_hash"]
    return True


def _latest_chain_hash() -> str:
    rows = _read_jsonl(DECISION_LEDGER_PATH, max_rows=1)
    if not rows:
        return "genesis"
    return rows[-1].get("event_hash", "genesis")


def _ensure_decision_chain_anchor(decision_id: str, state: dict, context_payload: dict) -> str | None:
    if state.get("chain_hash"):
        return state.get("chain_hash")
    anchor = {
        "event_type": "decision_started",
        "event_id": str(uuid.uuid4()),
        "decision_id": decision_id,
        "session_id": st.session_state.get("audit_session_id", ""),
        "pathogen": state.get("pathogen", "unknown"),
        "pathogen_context": state.get("context_signature"),
        "at": _now_iso(),
        "state_status": state.get("status", "Ausstehend"),
        "required_signatures": int(state.get("required_signatures", 1)),
        "payload_hash": _digest_payload(context_payload),
        "payload_signature": _sign_payload(context_payload, SIGNING_SECRET),
    }
    anchor["prev_chain_hash"] = _latest_chain_hash()
    anchor["event_hash"] = _digest_payload(anchor)
    anchor["event_signature"] = _sign_payload(anchor, SIGNING_SECRET)
    _append_jsonl(DECISION_LEDGER_PATH, anchor)
    state["chain_hash"] = anchor["event_hash"]
    state["updated_at"] = anchor["at"]
    return state["chain_hash"]


def _approval_overdue(state: dict) -> bool:
    created = _iso_to_datetime(state.get("created_at"))
    if not created:
        return False
    elapsed_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600.0
    return elapsed_hours > float(state.get("sla_hours", 24))


def _roi_snapshot(
    selected_pathogen: str,
    kpis: dict,
    forecast_df: pd.DataFrame,
    decision_pack: dict,
    what_if_df: pd.DataFrame,
    readiness_score: tuple[int, str, str],
    composite,
) -> dict:
    forecast_cost = float(kpis.get("cost_per_test", 0) or 0) * float(forecast_df["Predicted Volume"].sum()) if not forecast_df.empty and "Predicted Volume" in forecast_df.columns else 0.0
    order_cost = float(decision_pack.get("estimated_order_cost", 0) or 0)
    risk_current = float(kpis.get("risk_eur", 0) or 0)
    basis_risk = risk_current
    if not what_if_df.empty and "Risk at Risk (EUR)" in what_if_df.columns:
        basis_r = what_if_df[what_if_df["Szenario"] == "Basis"]
        if not basis_r.empty:
            basis_risk = float(basis_r["Risk at Risk (EUR)"].iloc[0])
    prevented = max(0.0, basis_risk - risk_current)
    confidence = float(getattr(composite, "confidence_pct", 0))
    approvals_weight = 20 if readiness_score[0] >= 80 else 10 if readiness_score[0] >= 60 else 0
    compliance_score = max(
        0,
        min(
            100,
            int(
                round(
                    (0.55 * confidence)
                    + (0.25 * (100 if len(forecast_df) > 0 else 0))
                    + approvals_weight
                )
            ),
        ),
    )
    if order_cost > 0:
        roi_pct = ((prevented - order_cost) / order_cost) * 100
    else:
        roi_pct = 100.0 if prevented > 0 else 0.0
    return {
        "pathogen": _safe_text(selected_pathogen, "unknown"),
        "forecast_horizon": int(kpis.get("forecast_horizon", 0) or 0),
        "demand_total": float(forecast_df["Predicted Volume"].sum()) if "Predicted Volume" in forecast_df.columns and not forecast_df.empty else 0.0,
        "order_cost_eur": float(order_cost),
        "revenue_projection_7d": float(kpis.get("revenue_forecast_7d", 0)),
        "risk_current_eur": float(risk_current),
        "risk_baseline_eur": float(basis_risk),
        "prevented_risk_eur": float(prevented),
        "roi_estimate_pct": float(round(roi_pct, 2)),
        "compliance_score": int(compliance_score),
        "forecast_to_stock_ratio": float(kpis.get("stock_on_hand", 0)) / max(1.0, float(kpis.get("total_demand", 0))),
        "confidence_pct": float(confidence),
        "readiness_score": int(readiness_score[0]),
    }


def _roi_snapshot_signature(
    selected_pathogen: str,
    kpis: dict,
    forecast_df: pd.DataFrame,
    decision_pack: dict,
    what_if_df: pd.DataFrame,
    readiness_score: tuple[int, str, str],
    composite,
) -> str:
    return _digest_payload(
        {
            "pathogen": _safe_text(selected_pathogen, "unknown"),
            "horizon": int(kpis.get("forecast_horizon", 0) or 0),
            "forecast_signature": _hash_df(forecast_df),
            "decision_pack_signature": _digest_payload(decision_pack),
            "kpis_signature": _digest_payload(kpis),
            "what_if_rows": int(len(what_if_df) if isinstance(what_if_df, pd.DataFrame) else 0),
            "readiness_signature": _safe_text(readiness_score[1], "") + f"|{int(readiness_score[0])}",
            "composite_signature": _safe_text(getattr(composite, "direction", "unknown"), ""),
            "composite_confidence": float(getattr(composite, "confidence_pct", 0)),
            "composite_trend": float(getattr(composite, "weighted_trend", 0)),
        }
    )


def _compliance_export_payload(
    selected_pathogen: str,
    forecast_horizon: int,
    decision_pack: dict,
    forecast_payload: dict,
    decision_state: dict,
    roi_data: dict,
    kpis: dict,
    composite,
) -> dict:
    return {
        "package_type": "labpulse_compliance_bundle_v1",
        "generated_at": _now_iso(),
        "pathogen": _safe_text(selected_pathogen, "unknown"),
        "forecast_horizon": int(forecast_horizon),
        "session_id": st.session_state.get("audit_session_id", ""),
        "decision_pack": _to_json_safe(decision_pack),
        "decision_approvals": _to_json_safe(decision_state),
        "kpis": _to_json_safe(kpis),
        "roi": _to_json_safe(roi_data),
        "signal_state": {
            "confidence_pct": float(getattr(composite, "confidence_pct", 0)),
            "direction": getattr(composite, "direction", "unknown"),
            "weighted_trend": float(getattr(composite, "weighted_trend", 0)),
        },
        "operations_payload_ref": forecast_payload.get("document_type"),
        "operations_payload_signature": forecast_payload.get("decision_signature"),
        "chain_hash": forecast_payload.get("decision_chain_hash"),
        "blueprints": {
            "approval_policy": _approval_policy(decision_pack.get("urgency", "Mittel")),
        },
    }


def _compliance_package_zip_bytes(
    compliance_payload: dict,
    operations_payload: dict,
    decision_id: str,
    what_if_df: pd.DataFrame,
) -> bytes:
    ledger_rows = [
        row for row in _read_jsonl(DECISION_LEDGER_PATH)
        if row.get("decision_id") == decision_id
    ]
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "compliance_package.json",
            json.dumps(_to_json_safe(compliance_payload), ensure_ascii=False, indent=2),
        )
        zf.writestr(
            "operations_payload.json",
            json.dumps(_to_json_safe(operations_payload), ensure_ascii=False, indent=2),
        )
        zf.writestr(
            "decision_ledger.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=False) for row in ledger_rows),
        )
        zf.writestr(
            "audit_events.jsonl",
            "\n".join(
                json.dumps(row, ensure_ascii=False)
                for row in _read_jsonl(AUDIT_LOG_PATH)
            ),
        )
        zf.writestr(
            "what_if_comparison.json",
            what_if_df.to_json(orient="records"),
        )
    return zip_buffer.getvalue()


def _readiness_score(
    decision_pack: dict,
    confidence_pct: float,
    active_alerts: int,
) -> tuple[int, str, str]:
    score = 100.0
    score -= 45 * (1.0 - (float(confidence_pct) / 100.0))
    days_to_stockout = decision_pack.get("days_to_stockout")
    stockout_day = decision_pack.get("stockout_day")
    if days_to_stockout is None and not stockout_day:
        stockout_penalty = 0.0
        stockout_label = "Puffer ausreichend"
    elif days_to_stockout is not None:
        if days_to_stockout <= 0:
            stockout_penalty = 38.0
            stockout_label = "Sofort handeln"
        elif days_to_stockout <= 3:
            stockout_penalty = 24.0
            stockout_label = "Kritischer Puffer"
        elif days_to_stockout <= 7:
            stockout_penalty = 14.0
            stockout_label = "Enger Puffer"
        elif days_to_stockout <= 14:
            stockout_penalty = 8.0
            stockout_label = "Beobachtung nötig"
        else:
            stockout_penalty = 0.0
            stockout_label = "Robust im Horizont"
    else:
        stockout_penalty = 24.0
        stockout_label = "Niedriger Datensatz-Confidence"

    score -= stockout_penalty
    score -= min(35.0, active_alerts * 5.0)
    score = max(0.0, min(100.0, score))
    if score >= 80:
        status = "Operativ stabil"
        desc = f"{stockout_label} · Bereitschaft hoch"
    elif score >= 60:
        status = "Handlung erforderlich"
        desc = f"{stockout_label} · Kontrolle empfohlen"
    elif score >= 40:
        status = "Eskalation vorbereitet"
        desc = f"{stockout_label} · Beschaffungs-/Lead-Time-Puffer erhöhen"
    else:
        status = "Krisenmodus"
        desc = f"{stockout_label} · Manuelle Freigabe dringend empfohlen"
    return int(round(score)), status, desc


def _build_decision_pack(
    selected_pathogen: str,
    forecast_horizon: int,
    kpis: dict,
    scenario: str,
    confidence_pct: float,
    risk_bands: dict[str, float],
    reagent_orders: float,
    cost_per_test: float,
) -> dict:
    stockout_day = kpis.get("stockout_day")
    stockon_hand = int(kpis.get("stock_on_hand", 0))
    total_demand = int(kpis.get("total_demand", 0))
    today = pd.Timestamp(datetime.today()).normalize()

    if stockout_day:
        stockout_day_ts = pd.Timestamp(stockout_day)
        days_to_stockout = int((stockout_day_ts.normalize() - today).days)
    else:
        days_to_stockout = None

    profile = SCENARIO_PROFILES.get(scenario, SCENARIO_PROFILES["Basis"])
    if days_to_stockout is None:
        action = "Monitoring halten"
        urgency = "Niedrig"
        urgency_class = "low"
        reason = "Bestand reicht im gewählten Szenario über den gesamten Horizont."
    elif days_to_stockout <= 0:
        action = "SOFORT ordern"
        urgency = "Kritisch"
        urgency_class = "critical"
        reason = "Deckungsgrenze ist heute erreicht oder überschritten."
    elif days_to_stockout <= profile["lead_time_days"]:
        action = "Heute anstoßen"
        urgency = "Hoch"
        urgency_class = "high"
        reason = f"Bestand reicht nur noch ca. {days_to_stockout} Tage – Puffer für Beschaffung liegt eng."
    elif stockout_day:
        action = "In 24h vorbereiten"
        urgency = "Mittel"
        urgency_class = "medium"
        reason = f"Erwartete Engpassgrenze in {days_to_stockout} Tagen."
    else:
        action = "Beobachten"
        urgency = "Niedrig"
        urgency_class = "low"
        reason = "Keine unmittelbare Maßnahme erforderlich."

    extra_order_cost = int(reagent_orders * max(0.0, float(cost_per_test or 0)))

    return {
        "pathogen": selected_pathogen,
        "horizon": int(forecast_horizon),
        "scenario": scenario,
        "action": action,
        "urgency": urgency,
        "urgency_class": urgency_class,
        "reason": reason,
        "days_to_stockout": days_to_stockout,
        "stockout_day": None if stockout_day is None else pd.Timestamp(stockout_day).strftime("%Y-%m-%d"),
        "stock_on_hand": stockon_hand,
        "total_demand": total_demand,
        "reagent_orders": float(reagent_orders),
        "estimated_order_cost": extra_order_cost,
        "risk_bands": risk_bands,
        "confidence_pct": confidence_pct,
        "scenario_description": profile.get("description", ""),
    }


def _get_session_id() -> str:
    if "audit_session_id" not in st.session_state:
        st.session_state.audit_session_id = str(uuid.uuid4())
    return st.session_state.audit_session_id


def _safe_text(value, fallback=""):
    if value is None:
        return fallback
    return str(value)


def _normalize_terms(terms: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not terms:
        return tuple()
    deduped: list[str] = []
    for value in terms:
        text = str(value or "").strip()
        if not text or text in deduped:
            continue
        deduped.append(text)
    return tuple(deduped[:5])


def _perf_start() -> float:
    return time.perf_counter()


def _perf_mark(name: str, start: float) -> None:
    if "dashboard_perf_ms" not in st.session_state:
        st.session_state.dashboard_perf_ms = {}
    st.session_state.dashboard_perf_ms[name] = round((time.perf_counter() - start) * 1000, 1)


def _perf_summary_text() -> str:
    if os.getenv("LABPULSE_DEBUG_PERF", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return ""
    timings = st.session_state.get("dashboard_perf_ms")
    if not timings:
        return ""
    return " · ".join(
        f"{name}: {value}ms"
        for name, value in timings.items()
    )


def _append_jsonl(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _get_query_param(name: str, default: str | None = None) -> str | None:
    try:
        if hasattr(st, "query_params"):
            query = st.query_params
            values = query.get(name)
            if isinstance(values, list):
                return values[0] if values else default
            if values is not None:
                return str(values)
        params = st.experimental_get_query_params()
        values = params.get(name)
        if isinstance(values, list):
            return values[0] if values else default
        return values
    except Exception:
        return default


def _is_query_true(name: str, default: bool = False) -> bool:
    value = _get_query_param(name, "1" if default else None)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _to_json_safe(value):
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return [_to_json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    return value


def _log_event(event: str, details: dict | None = None, level: str = "info") -> None:
    try:
        payload = {
            "event": event,
            "level": level,
            "at": datetime.now(timezone.utc).isoformat(),
            "session_id": _get_session_id(),
            "path": "Dashboard",
            "details": details or {},
        }
        _append_jsonl(AUDIT_LOG_PATH, payload)
    except Exception:
        pass


def _hash_df(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {"rows": 0, "cols": 0, "head_sig": None, "tail_sig": None}
    head = df.head(5).to_csv(index=False).encode("utf-8")
    tail = df.tail(5).to_csv(index=False).encode("utf-8")
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "head_sig": hashlib.sha256(head).hexdigest()[:16],
        "tail_sig": hashlib.sha256(tail).hexdigest()[:16],
    }


def _record_forecast_metadata(
    selected_pathogen: str,
    forecast_horizon: int,
    lab_df: pd.DataFrame,
    wastewater_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    safety_buffer: float,
    scenario_uplift: float,
    scenario_name: str,
    safety_buffer_base: int | None = None,
    scenario_uplift_base: float | None = None,
) -> dict:
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pathogen": _safe_text(selected_pathogen, "unknown"),
        "inputs": {
            "forecast_horizon_days": int(forecast_horizon),
            "safety_buffer_pct": float(safety_buffer),
            "scenario_uplift_pct": float(scenario_uplift),
            "scenario_name": _safe_text(scenario_name, "Basis"),
            "safety_buffer_base_pct": int(safety_buffer_base) if safety_buffer_base is not None else int(safety_buffer),
            "scenario_uplift_base_pct": float(scenario_uplift_base) if scenario_uplift_base is not None else float(scenario_uplift),
        },
        "data": {
            "lab_data": _hash_df(lab_df),
            "wastewater_data": _hash_df(wastewater_df),
            "result": {"rows": int(forecast_df.shape[0]), "cols": int(forecast_df.shape[1])},
        },
        "session_id": _get_session_id(),
    }
    _append_jsonl(FORECAST_META_PATH, metadata)
    st.session_state.forecast_metadata = metadata
    return metadata


def _forecast_metadata_signature(
    selected_pathogen: str,
    forecast_horizon: int,
    lab_df: pd.DataFrame,
    wastewater_df: pd.DataFrame,
    safety_buffer: float,
    scenario_uplift: float,
    scenario_name: str,
) -> str:
    return _digest_payload(
        {
            "pathogen": _safe_text(selected_pathogen, "unknown"),
            "forecast_horizon_days": int(forecast_horizon),
            "scenario_name": _safe_text(scenario_name, "Basis"),
            "safety_buffer": float(safety_buffer),
            "scenario_uplift": float(scenario_uplift),
            "lab_rows": int(getattr(lab_df, "shape", (0, 0))[0]),
            "wastewater_rows": int(getattr(wastewater_df, "shape", (0, 0))[0]),
            "lab_signature": _hash_df(lab_df),
            "ww_signature": _hash_df(wastewater_df),
        }
    )


def _record_forecast_metadata_if_changed(
    selected_pathogen: str,
    forecast_horizon: int,
    lab_df: pd.DataFrame,
    wastewater_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    safety_buffer: float,
    scenario_uplift: float,
    scenario_name: str,
    safety_buffer_base: int | None = None,
    scenario_uplift_base: float | None = None,
) -> tuple[dict, str]:
    signature = _forecast_metadata_signature(
        selected_pathogen=selected_pathogen,
        forecast_horizon=forecast_horizon,
        safety_buffer=safety_buffer,
        scenario_uplift=scenario_uplift,
        scenario_name=scenario_name,
        lab_df=lab_df,
        wastewater_df=wastewater_df,
    )
    if st.session_state.get("forecast_metadata_signature") == signature and st.session_state.get("forecast_metadata"):
        return st.session_state.forecast_metadata, signature

    metadata = _record_forecast_metadata(
        selected_pathogen=selected_pathogen,
        forecast_horizon=forecast_horizon,
        safety_buffer=safety_buffer,
        scenario_uplift=scenario_uplift,
        scenario_name=scenario_name,
        safety_buffer_base=safety_buffer_base,
        scenario_uplift_base=scenario_uplift_base,
        lab_df=lab_df,
        wastewater_df=wastewater_df,
        forecast_df=forecast_df,
    )
    st.session_state.forecast_metadata_signature = signature
    return metadata, signature


def _append_retention_cleanup(retention_days: int) -> None:
    if retention_days:
        _prune_jsonl(AUDIT_LOG_PATH, retention_days)
        _prune_jsonl(FORECAST_META_PATH, retention_days)
        _prune_jsonl(OPERATIONS_PAYLOAD_PATH, retention_days)
        _prune_jsonl(DECISION_LEDGER_PATH, retention_days)


def _prune_jsonl(path: str, max_days: int) -> None:
    if not max_days or max_days <= 0:
        return
    if not os.path.exists(path):
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_days)
    kept: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            created = row.get("at")
            if not created:
                continue
            try:
                when = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except Exception:
                continue
            if when >= cutoff:
                kept.append(json.dumps(row, ensure_ascii=False))
    if len(kept) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept))


def _read_jsonl(path: str, max_rows: int | None = None) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if max_rows:
        rows = rows[-max_rows:]
    return rows


def _to_jsonl_download_payload(path: str, max_rows: int = 2000) -> str:
    return _to_jsonl_download_payload_cached(path=path, max_rows=max_rows)


def _to_csv_download_bytes(path: str, max_rows: int = 2000) -> bytes:
    return _to_csv_download_bytes_cached(path=path, max_rows=max_rows)


@st.cache_data(ttl=60 * 2, show_spinner=False)
def _to_jsonl_download_payload_cached(path: str, max_rows: int = 2000) -> str:
    rows = _read_jsonl(path, max_rows=max_rows)
    return json.dumps(rows, ensure_ascii=False, indent=2)


@st.cache_data(ttl=60 * 2, show_spinner=False)
def _to_csv_download_bytes_cached(path: str, max_rows: int = 2000) -> bytes:
    rows = _read_jsonl(path, max_rows=max_rows)
    if not rows:
        return "".encode("utf-8")
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _build_operations_payload(
    selected_pathogen: str,
    forecast_horizon: int,
    safety_buffer: float,
    scenario_uplift: float,
    lab_df: pd.DataFrame,
    wastewater_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    kpis: dict,
    ml_model_info: dict | None,
    triggered_alerts: list,
    composite: object,
    readiness: tuple[int, str, str] | None = None,
    scenario_name: str | None = None,
    safety_buffer_base: float | None = None,
    scenario_uplift_base: float | None = None,
    decision_pack: dict | None = None,
    risk_bands: dict | None = None,
    decision_id: str | None = None,
    decision_state: dict | None = None,
    roi_snapshot: dict | None = None,
    compliance_snapshot: dict | None = None,
) -> dict:
    forecast_payload_cols = [
        col for col in ["Date", "Predicted Volume", "Reagent Order", "Revenue", "Reagent Stock"]
        if col in forecast_df.columns
    ]
    forecast_rows = forecast_df[forecast_payload_cols].copy() if forecast_payload_cols else forecast_df.copy()
    if "Date" in forecast_rows.columns:
        forecast_rows["Date"] = pd.to_datetime(forecast_rows["Date"]).dt.strftime("%Y-%m-%d")
    return {
        "document_type": "labpulse_forecast_payload_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_id": _get_session_id(),
        "pathogen": _safe_text(selected_pathogen, "unknown"),
        "inputs": {
            "forecast_horizon_days": int(forecast_horizon),
            "scenario_name": _safe_text(scenario_name, "Basis"),
            "safety_buffer_base_pct": float(safety_buffer_base) if safety_buffer_base is not None else float(safety_buffer),
            "safety_buffer_pct": float(safety_buffer),
            "scenario_uplift_base_pct": float(scenario_uplift_base) if scenario_uplift_base is not None else float(scenario_uplift),
            "scenario_uplift_pct": float(scenario_uplift),
            "stock_on_hand": int(get_stock_for_pathogen(selected_pathogen)),
        },
        "provenance": st.session_state.get("forecast_metadata"),
        "scenario_description": SCENARIO_PROFILES.get(_safe_text(scenario_name, "Basis"), SCENARIO_PROFILES["Basis"]).get("description", ""),
        "decision_id": _safe_text(decision_id),
        "decision_state": _to_json_safe(decision_state),
        "kpis": _to_json_safe(kpis),
        "model": _to_json_safe(ml_model_info) if ml_model_info else None,
        "alerts": _to_json_safe(triggered_alerts),
        "decision_pack": _to_json_safe(decision_pack),
        "risk_bands": _to_json_safe(risk_bands),
        "readiness": _to_json_safe(readiness),
        "roi_snapshot": _to_json_safe(roi_snapshot),
        "compliance_snapshot": _to_json_safe(compliance_snapshot),
        "decision_signature": None,
        "decision_chain_hash": None,
        "confidence_pct": float(getattr(composite, "confidence_pct", 0)),
        "signal_state": {
            "confidence_pct": float(getattr(composite, "confidence_pct", 0)),
            "direction": getattr(composite, "direction", "unknown"),
            "weighted_trend": float(getattr(composite, "weighted_trend", 0.0)),
        },
        "forecast_series": _to_json_safe(forecast_rows.head(120).to_dict(orient="records")),
        "data": {
            "lab_rows": _hash_df(lab_df),
            "wastewater_rows": _hash_df(wastewater_df),
            "forecast_rows": int(forecast_df.shape[0]),
            "series_columns": list(forecast_df.columns),
        },
    }


def _record_operations_payload(path: str, payload: dict) -> dict:
    _append_jsonl(path, payload)
    st.session_state.operations_payload = payload
    return payload

pio.templates.default = "plotly_white"

_PUBLIC_BASE_URL = os.getenv("LABPULSE_PUBLIC_URL", "").strip().rstrip("/")


def _public_url(path: str = "") -> str:
    """Build app-local links without hard-coded relative routes."""
    normalized = (path or "").strip()
    if not normalized or normalized == "/":
        return _PUBLIC_BASE_URL + "/" if _PUBLIC_BASE_URL else "/"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return f"{_PUBLIC_BASE_URL}{normalized}" if _PUBLIC_BASE_URL else normalized


_DASHBOARD_STYLE_PATH = Path(__file__).resolve().parent.parent / "assets/css/dashboard_core.css"
if _DASHBOARD_STYLE_PATH.exists():
    st.markdown(f"<style>{_DASHBOARD_STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Lade RKI-Daten …")
def load_raw():
    return fetch_rki_raw()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _load_grippeweb_cached(erkrankung: str, region: str = "Bundesweit") -> pd.DataFrame:
    return fetch_grippeweb(erkrankung=erkrankung, region=region)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _load_are_cached(bundesland: str = "Bundesweit") -> pd.DataFrame:
    return fetch_are_konsultation(bundesland=bundesland)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _load_trends_cached(terms: tuple, timeframe: str = "today 3-m", geo: str = "DE") -> pd.DataFrame:
    if not terms:
        return pd.DataFrame()
    return fetch_google_trends(list(terms), timeframe=timeframe, geo=geo)


@st.cache_data(ttl=10 * 60, show_spinner=False)
def _load_surveillance_bundle_cached(
    grippeweb_type: str,
    region: str,
    trends_terms: tuple[str, ...] = (),
    trends_timeframe: str = "today 3-m",
    trends_geo: str = "DE",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with ThreadPoolExecutor(max_workers=3) as executor:
        gw_future = executor.submit(_load_grippeweb_cached, grippeweb_type, region)
        are_future = executor.submit(_load_are_cached, region)
        trends_future = (
            executor.submit(_load_trends_cached, trends_terms, trends_timeframe, trends_geo)
            if trends_terms else None
        )

        grippeweb_df = gw_future.result()
        are_df = are_future.result()
        trends_df = trends_future.result() if trends_future is not None else pd.DataFrame()

    return grippeweb_df, are_df, trends_df


def _build_forecast_with_optional_ml(
    lab_json: str,
    wastewater_json: str,
    forecast_horizon: int,
    safety_buffer_pct: float,
    stock_on_hand: int,
    scenario_uplift_pct: float,
    pathogen: str,
    use_ml: bool,
) -> tuple[pd.DataFrame, dict, pd.DataFrame | None, dict | None]:
    if not use_ml:
        forecast_df, kpis = _build_forecast_cached(
            lab_json,
            forecast_horizon,
            safety_buffer_pct,
            stock_on_hand,
            scenario_uplift_pct,
            pathogen,
        )
        return forecast_df, kpis, None, None

    with ThreadPoolExecutor(max_workers=2) as executor:
        forecast_future = executor.submit(
            _build_forecast_cached,
            lab_json,
            forecast_horizon,
            safety_buffer_pct,
            stock_on_hand,
            scenario_uplift_pct,
            pathogen,
        )
        ml_future = executor.submit(
            _run_ml_forecast_cached,
            lab_json,
            wastewater_json,
            pathogen,
            forecast_horizon,
        )

        forecast_df, kpis = forecast_future.result()
        try:
            ml_forecast, ml_model_info = ml_future.result()
        except Exception:
            ml_forecast, ml_model_info = None, None

    return forecast_df, kpis, ml_forecast, ml_model_info


@st.cache_data(ttl=10 * 60, show_spinner=False)
def _load_merged_lab_data(synthetic_json: str, real_json: str, pathogen: str) -> pd.DataFrame:
    synth = pd.read_json(synthetic_json, orient="split")
    real = pd.read_json(real_json, orient="split")
    return merge_with_synthetic(synth, real, pathogen)


@st.cache_data(ttl=5 * 60, show_spinner=False)
def _build_forecast_cached(
    lab_json: str,
    forecast_horizon: int,
    safety_buffer_pct: float,
    stock_on_hand: int,
    scenario_uplift_pct: float,
    pathogen: str,
) -> tuple[pd.DataFrame, dict]:
    lab_df = pd.read_json(lab_json, orient="split")
    return build_forecast(
        lab_df=lab_df,
        horizon_days=forecast_horizon,
        safety_buffer_pct=safety_buffer_pct,
        stock_on_hand=stock_on_hand,
        scenario_uplift_pct=scenario_uplift_pct,
        pathogen=pathogen,
    )


@st.cache_data(ttl=5 * 60, show_spinner=False)
def _run_ml_forecast_cached(
    lab_json: str,
    wastewater_json: str,
    pathogen: str,
    periods: int,
) -> tuple[pd.DataFrame, dict]:
    lab_df = pd.read_json(lab_json, orient="split")
    wastewater_df = pd.read_json(wastewater_json, orient="split")
    forecaster = LabVolumeForecaster(lab_df, wastewater_df, pathogen)
    forecaster.train()
    return forecaster.forecast(periods=periods), forecaster.model_info


@st.cache_data(ttl=10 * 60, show_spinner=False)
def _load_wastewater_cached(raw_json: str, pathogen: str) -> pd.DataFrame:
    raw_df = pd.read_json(raw_json, orient="split")
    return fetch_rki_wastewater(raw_df, pathogen=pathogen)


@st.cache_data(ttl=10 * 60, show_spinner=False)
def _build_lab_volume_cached(wastewater_json: str, pathogen: str, lag_days: int = 14) -> pd.DataFrame:
    wastewater_df = pd.read_json(wastewater_json, orient="split")
    return generate_lab_volume(wastewater_df, lag_days=lag_days, pathogen=pathogen)


@st.cache_data(ttl=5 * 60, show_spinner=False)
def _fuse_signals_cached(
    wastewater_json: str,
    grippeweb_json: str,
    are_json: str,
    trends_json: str | None = None,
) -> object:
    wastewater_df = pd.read_json(wastewater_json, orient="split")
    grippeweb_df = pd.read_json(grippeweb_json, orient="split") if grippeweb_json else pd.DataFrame()
    are_df = pd.read_json(are_json, orient="split") if are_json else pd.DataFrame()
    trends_df = pd.read_json(trends_json, orient="split") if trends_json else pd.DataFrame()
    return fuse_all_signals(
        wastewater_df=wastewater_df,
        grippeweb_df=grippeweb_df,
        are_df=are_df,
        trends_df=trends_df,
    )


@st.cache_data(ttl=10 * 60, show_spinner=False)
def _aggregate_regional_cached(raw_json: str, pathogen_tuple: tuple[str, ...], days_back: int = 30) -> pd.DataFrame:
    raw_df = pd.read_json(raw_json, orient="split")
    if raw_df.empty or not pathogen_tuple:
        return pd.DataFrame()
    return aggregate_by_bundesland(raw_df, list(pathogen_tuple), days_back=days_back)


@st.cache_data(ttl=30 * 60, show_spinner=False)
def _ollama_health_cached() -> bool:
    return get_ollama_client().health_check()


@st.cache_data(ttl=15 * 60, show_spinner=False)
def _cached_ai_insight(selected_pathogen: str, kpi_snapshot_json: str) -> str:
    return get_ollama_client().generate_insight(json.loads(kpi_snapshot_json), selected_pathogen)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def _cached_ai_fallback_insight(selected_pathogen: str, kpi_snapshot_json: str) -> str:
    return get_ollama_client()._fallback_insight(json.loads(kpi_snapshot_json), selected_pathogen)


@st.cache_data(ttl=30 * 60, show_spinner=False)
def _cached_pdf_commentary(selected_pathogen: str, kpi_snapshot_json: str) -> str:
    return get_ollama_client().generate_pdf_commentary(json.loads(kpi_snapshot_json), selected_pathogen)


@st.cache_data(ttl=5 * 60, show_spinner=False)
def _build_composite_cached(
    wastewater_json: str,
    grippeweb_json: str = "",
    are_json: str = "",
    trends_json: str | None = None,
) -> object:
    """
    Cache signal fusion.
    Default args keep it fast by only requiring wastewater data.
    """
    return _fuse_signals_cached(
        wastewater_json=wastewater_json,
        grippeweb_json=grippeweb_json,
        are_json=are_json,
        trends_json=trends_json,
    )


@st.cache_data(ttl=5 * 60, show_spinner=False)
def _build_main_chart_json(
    wastewater_window_json: str,
    lab_window_json: str,
    forecast_window_json: str,
    ml_forecast_json: str,
    use_ml: bool,
    today_str: str,
    forecast_horizon: int,
) -> str:
    wastewater_df = pd.read_json(wastewater_window_json, orient="split")
    lab_df = pd.read_json(lab_window_json, orient="split")
    forecast_df = pd.read_json(forecast_window_json, orient="split")
    ml_df = pd.read_json(ml_forecast_json, orient="split") if ml_forecast_json else pd.DataFrame()

    if "date" in wastewater_df.columns:
        wastewater_df["date"] = pd.to_datetime(wastewater_df["date"])
    if "date" in lab_df.columns:
        lab_df["date"] = pd.to_datetime(lab_df["date"])
    if "Date" in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={"Date": "date"})
    if "date" in forecast_df.columns:
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    if "date" in ml_df.columns:
        ml_df["date"] = pd.to_datetime(ml_df["date"])

    if wastewater_df.empty or lab_df.empty or "virus_load" not in wastewater_df.columns:
        return go.Figure().to_json()

    today = pd.Timestamp(today_str)

    ww_s = wastewater_df.copy()
    lab_s = lab_df.copy()
    lab_actuals = lab_s[lab_s["date"] <= today]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    ww_s["vl_7d"] = ww_s["virus_load"].rolling(7, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=ww_s["date"],
        y=ww_s["vl_7d"],
        name="Viruslast (7d Ø)",
        line=dict(color="#0284C7", width=2),
        hovertemplate="%{x|%d %b}: %{y:,.0f} Kopien/L<extra></extra>",
    ), secondary_y=False)

    lab_s["vol_7d"] = lab_s["order_volume"].rolling(7, min_periods=1).mean() if "order_volume" in lab_s.columns else None
    if "vol_7d" in lab_s.columns and not lab_s["vol_7d"].empty:
        fig.add_trace(go.Scatter(
            x=lab_s["date"],
            y=lab_s["vol_7d"],
            name="Labortests (7d Ø)",
            line=dict(color="#EA580C", width=2.5),
            hovertemplate="%{x|%d %b}: %{y:,.0f} Ø/Tag<extra></extra>",
        ), secondary_y=True)

    if use_ml and not ml_df.empty and {"upper", "lower", "predicted"}.issubset(set(ml_df.columns)):
        try:
            fig.add_trace(go.Scatter(x=ml_df["date"], y=ml_df["upper"], showlegend=False, line=dict(width=0), mode="lines"), secondary_y=True)
            fig.add_trace(go.Scatter(
                x=ml_df["date"],
                y=ml_df["lower"],
                name="ML-Konfidenzband",
                fill="tonexty",
                fillcolor="rgba(234,88,12,0.08)",
                line=dict(width=0.5, color="rgba(234,88,12,0.2)", dash="dot"),
                mode="lines",
            ), secondary_y=True)
            fig.add_trace(go.Scatter(
                x=ml_df["date"],
                y=ml_df["predicted"],
                name="ML-Prognose",
                line=dict(color="#2563EB", width=3, dash="dash"),
                mode="lines+markers",
                marker=dict(size=3, color="#2563EB"),
                hovertemplate="<b>ML</b> %{x|%d %b}: <b>%{y:,.0f}</b><extra></extra>",
            ), secondary_y=True)
        except Exception:
            pass

    if not forecast_df.empty and "date" in forecast_df.columns and "Predicted Volume" in forecast_df.columns:
        vol_col = "order_volume"
        if not lab_actuals.empty and "order_volume" in lab_actuals.columns:
            connector = pd.DataFrame({"date": [lab_actuals["date"].iloc[-1]], vol_col: [lab_actuals["order_volume"].iloc[-1]]})
            fc_plot = pd.concat([connector, forecast_df[["date", "Predicted Volume"]].rename(columns={"Predicted Volume": vol_col})], ignore_index=True)
        else:
            fc_plot = forecast_df[["date", "Predicted Volume"]].rename(columns={"Predicted Volume": vol_col})
        fig.add_trace(go.Scatter(
            x=fc_plot["date"],
            y=fc_plot[vol_col],
            name="Prognose",
            line=dict(color="#EA580C", width=2, dash="dot"),
            hovertemplate="%{x|%d %b}: %{y:,.0f}<extra></extra>",
        ), secondary_y=True)

    fc_end = today + pd.Timedelta(days=forecast_horizon)
    fig.add_vrect(x0=today_str, x1=fc_end.strftime("%Y-%m-%d"), fillcolor="rgba(234,88,12,0.05)", layer="below", line_width=0)
    fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1, yref="paper", line=dict(color="rgba(234,88,12,0.45)", width=1, dash="dot"))
    fig.add_annotation(x=today_str, y=1.04, yref="paper", text="heute", showarrow=False, font=dict(size=9, color="#EA580C", family="Inter"), bgcolor="rgba(234,88,12,0.08)", borderpad=3)

    x_range = None
    if use_ml and not ml_df.empty:
        x_range = [(today - pd.Timedelta(days=21)).strftime("%Y-%m-%d"), (today + pd.Timedelta(days=forecast_horizon + 3)).strftime("%Y-%m-%d")]

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255, 255, 255, 1)",
        plot_bgcolor="rgba(248, 250, 252, 1)",
        height=520,
        margin=dict(l=0, r=0, t=40, b=10),
        legend=dict(
            orientation="h",
            y=1.08,
            x=0.5,
            xanchor="center",
            font=dict(size=10, color="#475569", family="IBM Plex Mono"),
            bgcolor="rgba(255, 255, 255, 0.9)",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="rgba(219, 228, 239, 0.8)",
            font_size=12,
            font_family="IBM Plex Mono",
            font_color="#111827",
        ),
        **({"xaxis_range": x_range} if x_range else {}),
    )
    fig.update_xaxes(
        gridcolor="rgba(219, 228, 239, 0.5)",
        dtick="M1",
        tickformat="%b '%y",
        tickfont=dict(size=10, color="#475569"),
        rangeslider=dict(visible=False),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=forecast_horizon, label="Prognose", step="day", stepmode="todate"),
                dict(step="all", label="Alles"),
            ],
            bgcolor="rgba(248, 250, 252, 0.95)",
            activecolor="#ea580c",
            bordercolor="rgba(219, 228, 239, 0.6)",
            borderwidth=1,
            font=dict(size=10, color="#334155", family="IBM Plex Mono"),
            x=0, y=1.14,
        ),
    )
    fig.update_yaxes(
        title_text="Viruslast",
        secondary_y=False,
        showgrid=False,
        title_font=dict(color="#0284C7", size=10),
        tickfont=dict(size=9, color="#475569"),
    )
    fig.update_yaxes(
        title_text="Tests/Tag",
        secondary_y=True,
        gridcolor="rgba(219, 228, 239, 0.4)",
        title_font=dict(color="#EA580C", size=10),
        tickfont=dict(size=9, color="#475569"),
    )

    return fig.to_json()


def _build_what_if_df_cached(
    lab_json: str,
    selected_pathogen: str,
    forecast_horizon: int,
    safety_buffer_base: int,
    scenario_uplift_base: float,
    stock_on_hand: int,
    lab_signature: dict[str, int | str | None],
    wastewater_signature: dict[str, int | str | None],
    forecast_signature: str,
    confidence_pct: float,
    cost_per_test: float,
) -> pd.DataFrame:
    cache_key = _digest_payload({
        "pathogen": selected_pathogen,
        "horizon": int(forecast_horizon),
        "buffer": int(safety_buffer_base),
        "uplift": float(scenario_uplift_base),
        "stock": int(stock_on_hand),
        "lab_signature": lab_signature,
        "wastewater_signature": wastewater_signature,
        "forecast_signature": forecast_signature,
        "confidence": float(confidence_pct),
        "cost": float(cost_per_test),
    })

    cached = st.session_state.get("_what_if_cache")
    if cached and cached.get("key") == cache_key:
        cached_df = cached.get("df")
        if isinstance(cached_df, pd.DataFrame):
            return cached_df.copy(deep=True)

    what_if_rows = []
    for scenario_name_cmp in SCENARIO_PROFILES:
        cmp_buffer, cmp_uplift, cmp_profile = _resolve_scenario(
            scenario_name=scenario_name_cmp,
            safety_buffer_base=int(safety_buffer_base),
            scenario_uplift_base=float(scenario_uplift_base),
        )
        cmp_forecast_df, cmp_kpis = _build_forecast_cached(
            lab_json=lab_json,
            forecast_horizon=int(forecast_horizon),
            safety_buffer_pct=cmp_buffer / 100,
            stock_on_hand=int(stock_on_hand),
            scenario_uplift_pct=cmp_uplift / 100,
            pathogen=selected_pathogen,
        )
        cmp_reagent_orders = float(cmp_forecast_df["Reagent Order"].clip(lower=0).sum()) if not cmp_forecast_df.empty else 0.0
        cmp_risk = _risk_band_by_confidence(
            float(cmp_kpis.get("risk_eur", 0)),
            float(confidence_pct),
        )
        cmp_pack = _build_decision_pack(
            selected_pathogen=selected_pathogen,
            forecast_horizon=int(forecast_horizon),
            kpis=cmp_kpis,
            scenario=scenario_name_cmp,
            confidence_pct=float(confidence_pct),
            risk_bands=cmp_risk,
            reagent_orders=cmp_reagent_orders,
            cost_per_test=float(cost_per_test),
        )
        what_if_rows.append({
            "Szenario": scenario_name_cmp,
            "Puffer (%)": int(cmp_buffer),
            "Uplift (%)": round(cmp_uplift, 1),
            "Aktion": cmp_pack["action"],
            "Urgency": cmp_pack["urgency"],
            "Lead-Time (Tage)": int(cmp_profile.get("lead_time_days", 3)),
            "Risk at Risk (EUR)": float(cmp_kpis.get("risk_eur", 0)),
            "Beschaffung in (Tage)": cmp_pack.get("days_to_stockout"),
            "Bestellbedarf": int(cmp_reagent_orders),
        })

    what_if_df = pd.DataFrame(what_if_rows) if what_if_rows else pd.DataFrame()
    st.session_state["_what_if_cache"] = {"key": cache_key, "df": what_if_df}
    return what_if_df.copy(deep=True)


def _ensure_what_if_df_cached() -> pd.DataFrame:
    return _build_what_if_df_cached(
        lab_json=st.session_state.get("_labpulse_what_if_lab_json", ""),
        selected_pathogen=st.session_state.get("_labpulse_what_if_pathogen", "SARS-CoV-2"),
        forecast_horizon=int(st.session_state.get("_labpulse_what_if_horizon", 14)),
        safety_buffer_base=int(st.session_state.get("_labpulse_what_if_buffer_base", 10)),
        scenario_uplift_base=float(st.session_state.get("_labpulse_what_if_uplift_base", 0)),
        stock_on_hand=int(st.session_state.get("_labpulse_what_if_stock", 0)),
        lab_signature=st.session_state.get("_labpulse_what_if_lab_signature", {}),
        wastewater_signature=st.session_state.get("_labpulse_what_if_ww_signature", {}),
        forecast_signature=st.session_state.get("_labpulse_what_if_forecast_signature", ""),
        confidence_pct=float(st.session_state.get("_labpulse_what_if_confidence", 0)),
        cost_per_test=float(st.session_state.get("_labpulse_what_if_cost", 0)),
    )


# Session State
if "ml_enabled" not in st.session_state:
    st.session_state.ml_enabled = False
if "uploaded_lab_data" not in st.session_state:
    st.session_state.uploaded_lab_data = None
if "alert_webhook_url" not in st.session_state:
    st.session_state.alert_webhook_url = ""
if "dashboard_exec_mode" not in st.session_state:
    st.session_state.dashboard_exec_mode = False
if "data_retention_days" not in st.session_state:
    st.session_state.data_retention_days = DEFAULT_RETENTION_DAYS
if "selected_pathogen" not in st.session_state:
    st.session_state.selected_pathogen = "SARS-CoV-2"
if "forecast_horizon" not in st.session_state:
    st.session_state.forecast_horizon = 14
if "safety_buffer" not in st.session_state:
    st.session_state.safety_buffer = 10
if "scenario_name" not in st.session_state:
    st.session_state.scenario_name = "Basis"
if "scenario_uplift" not in st.session_state:
    st.session_state.scenario_uplift = 0
if "decision_approval_states" not in st.session_state:
    st.session_state.decision_approval_states = {}
if "decision_signer_name" not in st.session_state:
    st.session_state.decision_signer_name = "Team Lead"
if "dashboard_controls_open" not in st.session_state:
    st.session_state.dashboard_controls_open = False


def _close_dashboard_controls() -> None:
    st.session_state.dashboard_controls_open = False


def _open_dashboard_controls() -> None:
    st.session_state.dashboard_controls_open = True

UNIQUE_KITS = {
    "SARS-CoV-2 PCR Kit": {"cost": 45, "pathogens": ["SARS-CoV-2"], "default_stock": 5000, "unit": "Tests", "lieferzeit_tage": 5},
    "Influenza A/B PCR Panel": {"cost": 38, "pathogens": ["Influenza A", "Influenza B", "Influenza (gesamt)"], "default_stock": 3000, "unit": "Tests", "lieferzeit_tage": 3},
    "RSV PCR Kit": {"cost": 42, "pathogens": ["RSV"], "default_stock": 2000, "unit": "Tests", "lieferzeit_tage": 4},
}
if "kit_inventory" not in st.session_state:
    st.session_state.kit_inventory = {k: v["default_stock"] for k, v in UNIQUE_KITS.items()}

def get_kit_for_pathogen(p):
    for k, v in UNIQUE_KITS.items():
        if p in v["pathogens"]:
            return k
    return "SARS-CoV-2 PCR Kit"

def get_stock_for_pathogen(p):
    return st.session_state.kit_inventory.get(get_kit_for_pathogen(p), 5000)

def get_kit_info_for_pathogen(p):
    return UNIQUE_KITS.get(get_kit_for_pathogen(p), {"cost": 45, "unit": "Tests"})


# ─────────────────────────────────────────────────────────────────────────────
# Steuerzentrale — kompakt in Dropdown/Expander
# ─────────────────────────────────────────────────────────────────────────────
_raw_load_start = _perf_start()
raw_df = load_raw()
_perf_mark("raw_data", _raw_load_start)
pathogens = get_available_pathogens(raw_df)
if st.session_state.selected_pathogen not in pathogens:
    st.session_state.selected_pathogen = pathogens[0] if pathogens else "SARS-CoV-2"
if st.session_state.scenario_name not in SCENARIO_PROFILES:
    st.session_state.scenario_name = "Basis"
selected_pathogen = st.session_state.selected_pathogen
forecast_horizon = st.session_state.forecast_horizon
safety_buffer = st.session_state.safety_buffer
scenario_name = st.session_state.scenario_name
scenario_uplift = st.session_state.scenario_uplift

st.markdown(
    '<div class="zen-control-rail">',
    unsafe_allow_html=True,
)

topbar_cols = st.columns([9, 1], gap="small")
with topbar_cols[0]:
    _home_url = _public_url("/")
    st.markdown(
        '<div class="zen-topbar-shell">'
        f'<a class="zen-home-link" href="{_home_url}">Home</a>'
        '<span class="zen-top-sep"></span>'
        '<span class="zen-top-label">Dashboard</span>'
        '</div>',
        unsafe_allow_html=True,
    )

with topbar_cols[1]:
    _menu_icon = "☰"
    st.button(
        _menu_icon,
        key="toggle_dashboard_controls",
        width="stretch",
        help="Dashboard-Steuerung öffnen",
        on_click=_open_dashboard_controls,
    )

st.markdown('<div class="zen-pathogen-strip">', unsafe_allow_html=True)
pathogen_cols = st.columns([1.6, 1, 1], gap="small")
with pathogen_cols[0]:
    st.markdown('<div class="zen-focus-rail">', unsafe_allow_html=True)
    st.markdown('<div class="zen-pathogen-label">Erreger</div>', unsafe_allow_html=True)
    selected_pathogen = st.selectbox(
        "Erreger",
        pathogens,
        index=pathogens.index(st.session_state.selected_pathogen) if st.session_state.selected_pathogen in pathogens else 0,
        key="selected_pathogen",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)
with pathogen_cols[1]:
    st.markdown('<div class="zen-focus-rail">', unsafe_allow_html=True)
    st.markdown('<div class="zen-pathogen-label">Szenario</div>', unsafe_allow_html=True)
    scenario_name = st.selectbox(
        "Szenario",
        list(SCENARIO_PROFILES.keys()),
        index=list(SCENARIO_PROFILES.keys()).index(st.session_state.scenario_name)
        if st.session_state.scenario_name in SCENARIO_PROFILES
        else 0,
        key="scenario_name",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)
with pathogen_cols[2]:
    st.markdown('<div class="zen-focus-rail">', unsafe_allow_html=True)
    st.markdown('<div class="zen-pathogen-label">Horizont</div>', unsafe_allow_html=True)
    horizon_labels = {7: "7 Tage", 14: "14 Tage", 21: "21 Tage"}
    horizon_options = [horizon_labels.get(v, f"{v} Tage") for v in [7, 14, 21]]
    current_horizon_key = horizon_labels.get(st.session_state.forecast_horizon, "14 Tage")
    forecast_horizon_label = st.segmented_control(
        "Horizont (Tage)",
        options=horizon_options,
        selection_mode="single",
        default=current_horizon_key,
        help="Schnellauswahl Horizont (Tage)",
        key="forecast_horizon_segmented",
    )
    if forecast_horizon_label:
        forecast_horizon = int(forecast_horizon_label.split()[0])
        st.session_state.forecast_horizon = forecast_horizon
    else:
        forecast_horizon = st.session_state.forecast_horizon
st.markdown("</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.dashboard_controls_open:
    st.markdown('<div class="zen-control-backdrop"></div>', unsafe_allow_html=True)
    st.markdown('<div class="zen-control-drawer">', unsafe_allow_html=True)

    close_cols = st.columns([10, 1], gap="small")
    with close_cols[0]:
        st.markdown('<div class="zen-control-title">Dashboard-Steuerung</div>', unsafe_allow_html=True)
    with close_cols[1]:
        st.button(
            "✕",
            key="close_dashboard_controls",
            on_click=_close_dashboard_controls,
            help="Dashboard-Steuerung schließen",
        )

    with st.expander("Dashboard-Steuerung", expanded=True):
        scenario_uplift = st.slider(
            "Viruslast-Uplift %",
            0,
            50,
            value=st.session_state.scenario_uplift,
            step=5,
            key="scenario_uplift",
        )
        st.caption(SCENARIO_PROFILES.get(st.session_state.scenario_name, SCENARIO_PROFILES["Basis"]).get("description", ""))
        dashboard_exec_mode = st.toggle("Analysemodus", value=st.session_state.dashboard_exec_mode, key="dashboard_exec_mode")

    with st.expander("Modell-Szenario", expanded=not st.session_state.dashboard_exec_mode):
        st.caption(f"{selected_pathogen} · {scenario_name}")
        safety_buffer = st.slider("Sicherheitspuffer %", 0, 30, value=st.session_state.safety_buffer, step=5, key="safety_buffer")

    with st.expander("Datenmanagement", expanded=False):
        retention_days = st.slider("Audit-Aufbewahrung", min_value=7, max_value=365, value=st.session_state.data_retention_days, step=7, key="data_retention_days")
        if st.button("Retentionscleanup ausführen", width='stretch'):
            _prune_jsonl(AUDIT_LOG_PATH, retention_days)
            _prune_jsonl(FORECAST_META_PATH, retention_days)
            _prune_jsonl(OPERATIONS_PAYLOAD_PATH, retention_days)
            _prune_jsonl(DECISION_LEDGER_PATH, retention_days)
            _log_event("retention_cleanup", {"retention_days": retention_days})
            st.success("Alte Protokolle bereinigt.")

        with st.expander("Labordaten-Import", expanded=False):
            uploaded_file = st.file_uploader("CSV hochladen", type=["csv"], label_visibility="collapsed")
            if uploaded_file:
                is_valid, msg, real_df = validate_csv(uploaded_file)
                if is_valid:
                    st.session_state.uploaded_lab_data = real_df
                    st.success(msg)
                    _log_event("upload_csv", {"filename": uploaded_file.name, "rows": int(real_df.shape[0])})
                else:
                    st.error(msg)

        with st.expander("Erweiterte Optionen", expanded=False):
            alert_webhook = st.text_input(
                "Webhook-URL",
                value=st.session_state.alert_webhook_url,
                placeholder="https://hooks.slack.com/...",
                label_visibility="collapsed",
            )
            st.session_state.alert_webhook_url = alert_webhook
            if alert_webhook and not _is_valid_webhook_url(alert_webhook):
                st.error("Webhook-URL blockiert: Nur freigegebene öffentliche Webhooks werden akzeptiert.")

            for kit_name, kit_info in UNIQUE_KITS.items():
                current = st.session_state.kit_inventory.get(kit_name, kit_info["default_stock"])
                new_val = st.number_input(
                    kit_name,
                    min_value=0,
                    value=current,
                    step=250,
                    key=f"stock_{kit_name}",
                )
                st.session_state.kit_inventory[kit_name] = new_val
    st.caption(f"Sync: {datetime.now().strftime('%H:%M')} · RKI AMELAG")

    if st.button("Daten neu laden", width='stretch'):
        _refresh_start = _perf_start()
        st.cache_data.clear()
        st.success("Daten neu geladen.")
        raw_df = load_raw()
        _perf_mark("manual_refresh", _refresh_start)
        _log_event("data_refresh", {"pathogen": selected_pathogen})
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────────────────────────────────────
_compute_start = _perf_start()
use_ml = st.session_state.ml_enabled
stock_on_hand = get_stock_for_pathogen(selected_pathogen)
_raw_json = raw_df.to_json(orient="split", date_format="iso")
_perf_mark("raw_to_json", _compute_start)
wastewater_df = _load_wastewater_cached(_raw_json, selected_pathogen)
_ww_signature = _hash_df(wastewater_df)
_ww_json = wastewater_df.to_json(orient="split", date_format="iso")
lab_df = _build_lab_volume_cached(_ww_json, selected_pathogen, lag_days=14)
_lab_signature = _hash_df(lab_df)
_lab_json = lab_df.to_json(orient="split", date_format="iso")

if st.session_state.uploaded_lab_data is not None:
    _merge_start = _perf_start()
    _real_json = st.session_state.uploaded_lab_data.to_json(orient="split", date_format="iso")
    lab_df = _load_merged_lab_data(_lab_json, _real_json, selected_pathogen)
    _lab_json = lab_df.to_json(orient="split", date_format="iso")
    _lab_signature = _hash_df(lab_df)
    _perf_mark("uploaded_merge", _merge_start)

ml_model_info = None
scenario_buffer_pct, scenario_uplift_pct, scenario_profile = _resolve_scenario(
    scenario_name=scenario_name,
    safety_buffer_base=int(safety_buffer),
    scenario_uplift_base=float(scenario_uplift),
)
_forecast_start = _perf_start()
forecast_df, kpis, ml_forecast, ml_model_info = _build_forecast_with_optional_ml(
    _lab_json,
    _ww_json,
    forecast_horizon,
    scenario_buffer_pct / 100,
    stock_on_hand,
    scenario_uplift_pct / 100,
    selected_pathogen,
    use_ml,
)
_perf_mark("forecast_pipeline", _forecast_start)
rev_col = [c for c in forecast_df.columns if "Revenue" in c]
rev_col_name = rev_col[0] if rev_col else "Est. Revenue"
_metadata_start = _perf_start()
forecast_metadata, forecast_metadata_signature = _record_forecast_metadata_if_changed(
    selected_pathogen=selected_pathogen,
    forecast_horizon=forecast_horizon,
    safety_buffer=scenario_buffer_pct,
    scenario_uplift=scenario_uplift_pct,
    scenario_name=scenario_name,
    safety_buffer_base=int(safety_buffer),
    scenario_uplift_base=float(scenario_uplift),
    lab_df=lab_df,
    wastewater_df=wastewater_df,
    forecast_df=forecast_df,
)
_perf_mark("metadata", _metadata_start)
_forecast_signature = json.dumps(
    {
        "pathogen": selected_pathogen,
        "forecast_horizon": forecast_horizon,
        "scenario_name": scenario_name,
        "safety_buffer": scenario_buffer_pct,
        "scenario_uplift": scenario_uplift_pct,
        "lab_rows": forecast_metadata["data"]["lab_data"]["rows"],
        "wastewater_rows": forecast_metadata["data"]["wastewater_data"]["rows"],
    },
    sort_keys=True,
)
if st.session_state.get("_last_forecast_signature") != _forecast_signature:
    _log_event(
        "forecast_recompute",
        {
            "pathogen": selected_pathogen,
            "forecast_horizon": forecast_horizon,
            "safety_buffer": safety_buffer,
            "scenario_uplift": scenario_uplift,
        },
    )
    st.session_state._last_forecast_signature = _forecast_signature

# Alerts
_alerts_start = _perf_start()
alert_mgr = AlertManager()
triggered_alerts = alert_mgr.evaluate_all(kpis, ml_model_info)
if triggered_alerts and st.session_state.alert_webhook_url:
    _webhook_url = st.session_state.alert_webhook_url.strip()
    _webhook_key = _digest_payload(
        {
            "pathogen": selected_pathogen,
            "horizon": forecast_horizon,
            "alerts": triggered_alerts,
            "scenario": scenario_name,
        }
    )
    if _is_valid_webhook_url(_webhook_url):
        if st.session_state.get("_last_webhook_signature") != _webhook_key:
            _sent = alert_mgr.send_webhook(_webhook_url, triggered_alerts, selected_pathogen)
            if _sent:
                st.session_state._last_webhook_signature = _webhook_key
                _log_event("webhook_dispatched", {"count": len(triggered_alerts), "pathogen": selected_pathogen})
    else:
        st.session_state._last_webhook_signature = ""
_perf_mark("alerts", _alerts_start)

# ── Provenienz/Modell-Metadaten (nicht in der Hauptansicht ausgegeben)
# Signal fusion: fast default path for fast dashboard start.
_composite_start = _perf_start()
composite = _build_composite_cached(_ww_json)
_perf_mark("composite", _composite_start)

risk_bands = _risk_band_by_confidence(
    float(kpis.get("risk_eur", 0)),
    float(getattr(composite, "confidence_pct", 0)),
)
_kit_info = get_kit_info_for_pathogen(selected_pathogen)
reagent_orders_total = float(forecast_df["Reagent Order"].clip(lower=0).sum()) if not forecast_df.empty else 0.0
_decision_start = _perf_start()
decision_pack = _build_decision_pack(
    selected_pathogen=selected_pathogen,
    forecast_horizon=forecast_horizon,
    kpis=kpis,
    scenario=scenario_name,
    confidence_pct=float(getattr(composite, "confidence_pct", 0)),
    risk_bands=risk_bands,
    reagent_orders=reagent_orders_total,
    cost_per_test=float(_kit_info.get("cost", 0)),
)
_perf_mark("decision_pack", _decision_start)
st.session_state["_labpulse_what_if_lab_json"] = _lab_json
st.session_state["_labpulse_what_if_pathogen"] = selected_pathogen
st.session_state["_labpulse_what_if_horizon"] = int(forecast_horizon)
st.session_state["_labpulse_what_if_buffer_base"] = int(safety_buffer)
st.session_state["_labpulse_what_if_uplift_base"] = float(scenario_uplift)
st.session_state["_labpulse_what_if_stock"] = stock_on_hand
st.session_state["_labpulse_what_if_lab_signature"] = _lab_signature
st.session_state["_labpulse_what_if_ww_signature"] = _ww_signature
st.session_state["_labpulse_what_if_forecast_signature"] = _forecast_signature
st.session_state["_labpulse_what_if_confidence"] = float(getattr(composite, "confidence_pct", 0))
st.session_state["_labpulse_what_if_cost"] = float(_kit_info.get("cost", 0))
_what_if_signature = _digest_payload(
    {
        "pathogen": selected_pathogen,
        "horizon": int(forecast_horizon),
        "scenario": scenario_name,
        "safety_buffer": int(safety_buffer),
        "scenario_uplift": float(scenario_uplift),
        "lab_signature": _lab_signature,
        "ww_signature": _ww_signature,
        "forecast_signature": _forecast_signature,
        "confidence": float(getattr(composite, "confidence_pct", 0)),
        "cost": float(_kit_info.get("cost", 0)),
    }
)
if st.session_state.get("_labpulse_what_if_signature") != _what_if_signature:
    st.session_state.pop("_labpulse_what_if_df_visible", None)
    st.session_state["_labpulse_what_if_signature"] = _what_if_signature
what_if_df = pd.DataFrame()
_readiness_start = _perf_start()
readiness = _readiness_score(
    decision_pack=decision_pack,
    confidence_pct=float(getattr(composite, "confidence_pct", 0)),
    active_alerts=len(triggered_alerts),
)
readiness_score_value, readiness_status, readiness_desc = readiness
_perf_mark("readiness", _readiness_start)
_decision_signature_start = _perf_start()
decision_signature = _digest_payload({
    "decision_pack": decision_pack,
    "kpis": {k: kpis.get(k) for k in ["risk_eur", "stockout_day", "stock_on_hand", "total_demand"]},
    "alerts": triggered_alerts,
    "composite": {
        "confidence_pct": float(getattr(composite, "confidence_pct", 0)),
        "direction": getattr(composite, "direction", "unknown"),
    },
})
_perf_mark("decision_signature", _decision_signature_start)
_decision_state_start = _perf_start()
decision_id = _build_decision_id(
    selected_pathogen=selected_pathogen,
    forecast_horizon=forecast_horizon,
    scenario_name=scenario_name,
    safety_buffer=scenario_buffer_pct,
    scenario_uplift=scenario_uplift_pct,
    stock_on_hand=stock_on_hand,
    lab_rows=forecast_df.shape[0],
    wastewater_rows=forecast_metadata["data"]["wastewater_data"]["rows"],
)
_decision_context = {
    "decision_id": decision_id,
    "pathogen": selected_pathogen,
    "horizon": forecast_horizon,
    "scenario": scenario_name,
    "scenario_buffer": scenario_buffer_pct,
    "scenario_uplift": scenario_uplift_pct,
    "stock_on_hand": stock_on_hand,
    "safety_buffer_base": int(safety_buffer),
    "scenario_uplift_base": float(scenario_uplift),
    "decision_signature": decision_signature,
}
approval_state, _is_new_decision_state = _get_approval_state(
    decision_id=decision_id,
    urgency=decision_pack["urgency"],
    context_signature=_digest_payload(_decision_context),
)
approval_state["pathogen"] = selected_pathogen
approval_state["decision_signature"] = decision_signature
if not approval_state.get("created_at"):
    approval_state["created_at"] = _now_iso()
if _ensure_decision_chain_anchor(decision_id, approval_state, _decision_context):
    st.session_state.decision_approval_states[decision_id] = approval_state
_perf_mark("decision_state", _decision_state_start)
_roi_start = _perf_start()
_roi_cache_key = _roi_snapshot_signature(
    selected_pathogen=selected_pathogen,
    kpis=kpis,
    forecast_df=forecast_df,
    decision_pack=decision_pack,
    what_if_df=what_if_df,
    readiness_score=readiness,
    composite=composite,
)
if st.session_state.get("_roi_snapshot_signature") != _roi_cache_key:
    st.session_state["_roi_snapshot_signature"] = _roi_cache_key
    st.session_state["_roi_snapshot_data"] = _roi_snapshot(
        selected_pathogen=selected_pathogen,
        kpis=kpis,
        forecast_df=forecast_df,
        decision_pack=decision_pack,
        what_if_df=what_if_df,
        readiness_score=readiness,
        composite=composite,
    )
roi_data = st.session_state.get("_roi_snapshot_data", {})
if not isinstance(roi_data, dict):
    roi_data = {}
_perf_mark("roi_snapshot", _roi_start)
approval_overdue = _approval_overdue(approval_state)

can_approve_current = _can_approve(approval_state)

_payload_signature = _digest_payload(
    {
        "pathogen": selected_pathogen,
        "horizon": forecast_horizon,
        "scenario": scenario_name,
        "safety_buffer": scenario_buffer_pct,
        "scenario_uplift": scenario_uplift_pct,
        "stock_on_hand": stock_on_hand,
        "decision_signature": decision_signature,
        "decision_chain_hash": approval_state.get("chain_hash"),
        "decision_status": approval_state.get("status"),
        "alerts": triggered_alerts,
        "risk_eur": float(kpis.get("risk_eur", 0) or 0),
        "model_type": ml_model_info.get("model_type") if ml_model_info else None,
        "forecast_rows": int(forecast_df.shape[0]),
        "forecast_metadata_signature": forecast_metadata_signature,
    }
)

def _ensure_operations_payload() -> dict:
    if (
        st.session_state.get("payload_signature") == _payload_signature
        and isinstance(st.session_state.get("operations_payload"), dict)
        and st.session_state.get("operations_payload")
    ):
        return st.session_state.get("operations_payload", {})

    _payload_start = _perf_start()
    operations_payload = _build_operations_payload(
        selected_pathogen=selected_pathogen,
        forecast_horizon=forecast_horizon,
        safety_buffer=scenario_buffer_pct,
        scenario_uplift=scenario_uplift_pct,
        scenario_name=scenario_name,
        safety_buffer_base=int(safety_buffer),
        scenario_uplift_base=float(scenario_uplift),
        decision_pack=decision_pack,
        risk_bands=risk_bands,
        readiness=readiness,
        decision_id=decision_id,
        decision_state=approval_state,
        roi_snapshot=roi_data,
        compliance_snapshot=None,
        lab_df=lab_df,
        wastewater_df=wastewater_df,
        forecast_df=forecast_df,
        kpis=kpis,
        ml_model_info=ml_model_info,
        triggered_alerts=triggered_alerts,
        composite=composite,
    )
    operations_payload["decision_signature"] = decision_signature
    operations_payload["decision_chain_hash"] = approval_state.get("chain_hash")
    operations_payload.pop("compliance_signature", None)
    operations_payload = _record_operations_payload(OPERATIONS_PAYLOAD_PATH, operations_payload)
    st.session_state.payload_signature = _payload_signature
    st.session_state.operations_payload = operations_payload
    st.session_state.compliance_payload = {}
    st.session_state["_labpulse_compliance_payload_signature"] = None
    _perf_mark("payload", _payload_start)
    return operations_payload

_perf_mark("compute_total", _compute_start)

is_machine_mode = (
    _is_query_true("machine")
    or _is_query_true("machine_readable")
    or _is_query_true("api")
)
if is_machine_mode:
    operations_payload = _ensure_operations_payload()
    st.json(operations_payload)
    st.caption("Machine-readable payload via query param.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# VIEW — Zen Mode: One screen, one action
# ═════════════════════════════════════════════════════════════════════════════

# ── Header (minimal) ─────────────────────────────────────────────────────────
dashboard_exec_mode = st.session_state.dashboard_exec_mode
st.markdown(
    f'<div class="zen-header">'
    f'<div class="zen-shell">'
    f'<div class="zen-kicker">Live-Krisensteuerung</div>'
    f'<h1>LabPulse · {selected_pathogen}</h1>'
    f'<div class="zen-header-meta">Szenario: {scenario_name} · Horizont: {forecast_horizon} Tage</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

if dashboard_exec_mode:
    st.markdown(
        '<div class="zen-mode-badge">Analysemodus aktiv — Fokus auf den Entscheidungs-Pfad.</div>',
        unsafe_allow_html=True,
    )

def _risk_level_for_context(risk_eur: float, stock_ratio: float) -> tuple[str, str, str]:
    if risk_eur >= 25000 or stock_ratio <= 0.15:
        return "Hoch", "🧨", "high"
    if risk_eur >= 8000 or stock_ratio <= 0.35:
        return "Mittel", "⚠️", "medium"
    return "Niedrig", "✅", "low"


_view_signature = json.dumps(
    {
        "pathogen": selected_pathogen,
        "horizon": forecast_horizon,
        "scenario": scenario_name,
        "buffer": scenario_buffer_pct,
        "uplift": scenario_uplift_pct,
    },
    sort_keys=True,
)
if st.session_state.get("_last_view_signature") != _view_signature:
    _log_event(
        "dashboard_view",
        {
            "pathogen": selected_pathogen,
            "horizon": forecast_horizon,
        },
    )
    st.session_state._last_view_signature = _view_signature

risk_ratio = float(kpis.get("stock_on_hand", 0) or 0) / max(1.0, float(kpis.get("total_demand", 0) or 0))
risk_level, risk_icon, risk_tone = _risk_level_for_context(float(kpis.get("risk_eur", 0) or 0), risk_ratio)
st.markdown(
    f'''
    <div class="zen-context-strip">
        <span class="zen-context-chip zen-risk-{risk_tone}"><strong>{risk_icon} {risk_level}</strong> Risikostatus</span>
        <span class="zen-context-chip"><strong>Pathogen:</strong> {selected_pathogen}</span>
        <span class="zen-context-chip"><strong>Trend:</strong> {float(getattr(composite, "weighted_trend", 0) or 0):+.1f}%</span>
    </div>
    ''',
    unsafe_allow_html=True,
)

st.markdown('<div class="zen-hero-shell">', unsafe_allow_html=True)
hero_cols = st.columns([1.6, 1], gap="medium")
with hero_cols[0]:
    st.markdown(
        f'''
        <div class="zen-hero-copy">
            <div class="zen-kicker">Operational Readout</div>
            <h2>Signalverlauf + Entscheidung</h2>
            <div class="zen-body">
                Prognoseverlauf für <strong>{selected_pathogen}</strong>.
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

_ml_label = "🧠 ML-Prognose deaktivieren" if use_ml else "🧠 ML-Prognose aktivieren"

with hero_cols[1]:
    if st.button(_ml_label, type="primary", width='stretch', key="ml_primary_toggle"):
        st.session_state.ml_enabled = not st.session_state.ml_enabled
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="zen-hero-chart-shell">', unsafe_allow_html=True)
st.markdown('<div class="zen-section-head">Forecast-Hauptsignal</div>', unsafe_allow_html=True)
st.markdown('<div id="zen-chart"></div>', unsafe_allow_html=True)

# ── THE CHART (hero — the product IS this chart) ────────────────────────────
today = pd.Timestamp(datetime.today()).normalize()
today_str = today.strftime("%Y-%m-%d")
chart_start = today - pd.Timedelta(days=90)
chart_window_ww_json = wastewater_df[wastewater_df["date"] >= chart_start][["date", "virus_load"]].to_json(orient="split", date_format="iso")
chart_window_lab_json = lab_df[lab_df["date"] >= chart_start][["date", "order_volume"]].to_json(orient="split", date_format="iso")
chart_window_forecast_json = (
    forecast_df.rename(columns={"Date": "date"})[["date", "Predicted Volume"]].to_json(orient="split", date_format="iso")
    if not forecast_df.empty
    else ""
)
chart_window_ml_json = ml_forecast.to_json(orient="split", date_format="iso") if (use_ml and ml_forecast is not None) else ""
_main_chart_json = _build_main_chart_json(
    chart_window_ww_json,
    chart_window_lab_json,
    chart_window_forecast_json,
    chart_window_ml_json,
    bool(use_ml and ml_forecast is not None),
    today_str,
    forecast_horizon,
)
try:
    fig = go.Figure.from_json(_main_chart_json)
except Exception:
    fig = go.Figure()

st.markdown('<div class="zen-chart">', unsafe_allow_html=True)
st.plotly_chart(fig, width='stretch', key="main_chart", config={
    "scrollZoom": True, "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "displaylogo": False,
    "toImageButtonOptions": dict(format="png", filename="labpulse_prognose", scale=2),
})
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
correlation_fig_for_pdf = fig
st.markdown('<div id="zen-handlung"></div>', unsafe_allow_html=True)

# ── Ergebnis-Deck (klar priorisierte Interpretation) ─────────────────────────
with st.container():
    risk_val = float(kpis.get("risk_eur", 0) or 0)
    if risk_val > 0:
        stockout_hint = kpis.get("stockout_day")
        stockout_txt = (
            f" Engpass in ca. {decision_pack.get('days_to_stockout')} Tagen."
            if stockout_hint is not None
            else ""
        )
        st.markdown(
            f'<div class="zen-risk-banner"><strong>Revenue at Risk:</strong> EUR {risk_val:,.0f} in {forecast_horizon} Tagen.{stockout_txt}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="zen-summary-head">Schnell-Blick</div>', unsafe_allow_html=True)
    st.markdown('<div class="zen-quick-grid">', unsafe_allow_html=True)
    qc_cols = st.columns(4)
    with qc_cols[0]:
        st.markdown('<div class="zen-quick-metric">', unsafe_allow_html=True)
        st.metric("Tests progn. 7d", f"{float(kpis.get('predicted_tests_7d', 0)):,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with qc_cols[1]:
        st.markdown('<div class="zen-quick-metric">', unsafe_allow_html=True)
        st.metric("Umsatz 7d", f"EUR {float(kpis.get('revenue_forecast_7d', 0)):,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with qc_cols[2]:
        trend = float(kpis.get("trend_pct", 0) or 0)
        st.markdown('<div class="zen-quick-metric">', unsafe_allow_html=True)
        st.metric("Trend WoW", f"{trend:+.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    with qc_cols[3]:
        conf = float(getattr(composite, "confidence_pct", 0))
        st.markdown('<div class="zen-quick-metric">', unsafe_allow_html=True)
        st.metric("Signal-Konfidenz", f"{conf:.0f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("📌 Kennzahlen, Risiko und Empfehlung", expanded=False):
    stockout_text = "Sicher über den Betrachtungszeitraum"
    if decision_pack["days_to_stockout"] is not None:
        stockout_text = f"{decision_pack['days_to_stockout']} Tage verbleibend"
    opt_band = _fmt_money(risk_bands.get("optimistic", 0))
    base_band = _fmt_money(risk_bands.get("base", 0))
    stress_band = _fmt_money(risk_bands.get("stressed", 0))
    approval_status = approval_state.get("status", "Ausstehend")
    approval_needed = int(approval_state.get("required_signatures", 1))
    approval_signed = len(approval_state.get("approvals", []))
    approval_chain = approval_state.get("chain_hash") or "—"
    roi_estimate_pct = float(roi_data.get("roi_estimate_pct", 0))
    compliance_score = int(roi_data.get("compliance_score", 0))
    prevention = _fmt_money(float(roi_data.get("prevented_risk_eur", 0)))
    order_cost = _fmt_money(float(roi_data.get("order_cost_eur", 0)))
    decision_overdue_flag = " · Eskalation offen" if approval_overdue else ""
    lead_time_days = int(scenario_profile.get("lead_time_days", 3))
    lead_time_label = f"{lead_time_days} Tag{'e' if lead_time_days != 1 else ''}"
    lead_time_cost = _fmt_money(int(decision_pack.get("estimated_order_cost", 0))) if decision_pack.get("estimated_order_cost") else "—"

    decision_cards_html = f"""
    <div class="zen-bento-grid">
        <article class="zen-bento-card zen-bento-main">
            <div class="zen-bento-kicker">{decision_pack['scenario']} · {scenario_profile.get('description', 'Szenario aktiv')}</div>
            <div class="zen-decision-urgency zen-urgency-{decision_pack['urgency_class']}">⚡ {decision_pack['urgency']}: {decision_pack['action']}</div>
            <h4>Empfehlung</h4>
            <div class="zen-mini-sub">{decision_pack['reason']}</div>
            <div class="zen-mini-sub">Operations-Readiness: {readiness_status} ({readiness_score_value}/100)</div>
            <div class="zen-mini-sub">{readiness_desc}</div>
            <div class="zen-bento-kicker">Lead-Time-Puffer: {lead_time_label} · geschätzte Zusatzkosten: {lead_time_cost}</div>
        </article>
        <article class="zen-bento-card">
            <h4>Verfügbarkeit</h4>
            <div class="zen-mini">{stockout_text}</div>
            <div class="zen-mini-sub">Restbestand: {int(decision_pack.get("stock_on_hand", 0)):,}</div>
            <div class="zen-mini-sub">Bedarf: {int(decision_pack.get("total_demand", 0)):,} Tests</div>
        </article>
        <article class="zen-bento-card">
            <h4>Digitale Freigabe</h4>
            <div class="zen-mini">{approval_status}{decision_overdue_flag}</div>
            <div class="zen-mini-sub">Signaturen: {approval_signed}/{approval_needed}</div>
            <div class="zen-mini-sub">SLA: {int(approval_state.get("sla_hours", 0))}h</div>
            <div class="zen-mini-sub">Chain: {approval_chain[:10]}…</div>
        </article>
        <article class="zen-bento-card">
            <h4>ROI & Compliance</h4>
            <div class="zen-mini">ROI-Qualität: {roi_estimate_pct:+.0f}%</div>
            <div class="zen-mini-sub">Risikoabwehr: {prevention}</div>
            <div class="zen-mini-sub">Entscheidungskosten: {order_cost}</div>
            <div class="zen-mini-sub">Compliance-Score: {compliance_score}/100</div>
        </article>
        <article class="zen-bento-card">
            <h4>Risiko-Bänder</h4>
            <div class="zen-mini">{opt_band}</div>
            <div class="zen-mini-sub">optimistisch</div>
            <div class="zen-mini">{base_band}</div>
            <div class="zen-mini-sub">Basis-Szenario</div>
            <div class="zen-mini">{stress_band}</div>
            <div class="zen-mini-sub">Stress-Szenario</div>
        </article>
    </div>
    """
    st.markdown(decision_cards_html, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PROGRESSIVE DISCLOSURE — Everything below is on-demand
# ═════════════════════════════════════════════════════════════════════════════

# ── KI-Analyse (expander) ────────────────────────────────────────────────────
ollama_ok = False
ai_kpi_snapshot = json.dumps(
    _to_json_safe(
        {
            "pathogen": selected_pathogen,
            "horizon": forecast_horizon,
            "stock_on_hand": kpis.get("stock_on_hand"),
            "total_demand": kpis.get("total_demand"),
            "risk_eur": kpis.get("risk_eur"),
            "confidence_pct": float(getattr(composite, "confidence_pct", 0) or 0),
            "stockout_day": kpis.get("stockout_day"),
            "scenario": scenario_name,
            "safety_buffer": scenario_buffer_pct,
            "scenario_uplift": scenario_uplift_pct,
        }
    ),
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)
_ai_snapshot_state_key = "_labpulse_ai_snapshot_key"
_ai_insight_state_key = "_labpulse_ai_insight_text"
if st.session_state.get(_ai_snapshot_state_key) != ai_kpi_snapshot:
    st.session_state[_ai_snapshot_state_key] = ai_kpi_snapshot
    st.session_state.pop(_ai_insight_state_key, None)

reagent_series = forecast_df["Reagent Order"] if "Reagent Order" in forecast_df.columns else pd.Series([], dtype=float)
reagent_orders = reagent_orders_total
next_order_row = None
if not reagent_series.empty and (reagent_series > 0).any():
    next_order_idx = reagent_series.idxmax()
    next_order_row = forecast_df.loc[next_order_idx, "Date"]
    if pd.notna(next_order_row):
        next_order_row = pd.Timestamp(next_order_row).strftime("%d.%m.%Y")

stockout_msg = (
    f"Restbestand reicht voraussichtlich bis {pd.Timestamp(kpis['stockout_day']).strftime('%d.%m.%Y')}."
    if kpis.get("stockout_day")
    else "Kein Stockout im Prognosefenster erkennbar."
)
order_msg = (
    f"Geplante Zusatzbestellung im Horizont: {reagent_orders:,.0f} Tests."
    if reagent_orders > 0
    else "Aktuell kein Zusatzbedarf im Prognosefenster."
)
next_order_msg = f"Nächste Bestellung am {next_order_row}." if next_order_row else "Noch kein konkreter Auslösestichtag."

if st.session_state.dashboard_exec_mode:
    st.markdown(
        '<div class="zen-exec-note">'
        f'<strong>Entscheidungs-Readout:</strong> {stockout_msg} {order_msg} {next_order_msg}'
        '</div>',
        unsafe_allow_html=True,
)
else:
    _ai_health_state_key = "_labpulse_ai_health_ok"
    _ai_source_state_key = "_labpulse_ai_source_label"

    if st.session_state.get(_ai_snapshot_state_key) != ai_kpi_snapshot:
        st.session_state[_ai_snapshot_state_key] = ai_kpi_snapshot
        st.session_state.pop(_ai_insight_state_key, None)
        st.session_state.pop(_ai_health_state_key, None)
        st.session_state.pop(_ai_source_state_key, None)

    ai_source = st.session_state.get(_ai_source_state_key, "Regelbasiert")

    with st.expander(f"KI-Analyse — {selected_pathogen}", expanded=False):
        if st.button("KI-Analyse jetzt berechnen", key="render_ai_insight", width='stretch'):
            with st.spinner("Erstelle KI-Analyse ..."):
                ollama_ok = _ollama_health_cached()
                st.session_state[_ai_health_state_key] = bool(ollama_ok)
                st.session_state[_ai_source_state_key] = "Ollama LLM" if bool(ollama_ok) else "Regelbasiert"
                if ollama_ok:
                    st.session_state[_ai_insight_state_key] = _cached_ai_insight(selected_pathogen, ai_kpi_snapshot)
                else:
                    st.session_state[_ai_insight_state_key] = _cached_ai_fallback_insight(selected_pathogen, ai_kpi_snapshot)
                st.rerun()
        st.caption(f"Engine: {st.session_state.get(_ai_source_state_key, ai_source)}")
        if st.session_state.get(_ai_insight_state_key):
            st.markdown(f'<div class="zen-body">{st.session_state[_ai_insight_state_key]}</div>', unsafe_allow_html=True)
        else:
            st.caption("Noch nicht berechnet.")

    # ── Signal-Details (expander) ────────────────────────────────────────────────
    with st.expander("Signal-Details", expanded=False):
        direction_icons = {"rising": "↑", "falling": "↓", "flat": "→", "mixed": "↔"}
        direction_labels = {"rising": "steigend", "falling": "fallend", "flat": "stabil", "mixed": "uneinheitlich"}
        st.caption(f"Konfidenz: {composite.confidence_pct:.0f}% · Richtung: {direction_labels.get(composite.direction, '?')} ({composite.weighted_trend:+.1f}%)")
        for sig in composite.signals:
            cfg = SIGNAL_CONFIG.get(sig.name, {})
            icon = cfg.get("icon", "")
            if sig.available:
                di = direction_icons.get(sig.direction, "?")
                st.caption(f"{icon} {sig.name}: {di} {sig.magnitude:+.1f}%")
            else:
                st.caption(f"{icon} {sig.name}: —")
        st.markdown("")
        for line in composite.narrative_de.split("\n"):
            if line.strip():
                st.caption(line)

    # ── ML-Details (expander, only when active) ──────────────────────────────────
    if use_ml and ml_model_info:
        with st.expander("ML-Modell Details", expanded=False):
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Modell", ml_model_info.get("model_type", "—"))
            mc2.metric("Konfidenz", f"{ml_model_info.get('confidence_score', 0):.0f}%")
            mc3.metric("Features", ml_model_info.get("n_features", "—"))
            mc4.metric("Training", f"{ml_model_info.get('training_days', '—')}d")


# ═════════════════════════════════════════════════════════════════════════════
# VIEWS — 3 instead of 5 (merged)
# ═════════════════════════════════════════════════════════════════════════════
_dashboard_view = st.radio(
    "Ansicht",
    ["Prognose", "Signale & Trends", "Regional"],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
    key="dashboard_view",
)


# ── Ansicht: Prognose ────────────────────────────────────────────────────────
if _dashboard_view == "Prognose":
    burndown_fig_for_pdf = None
    with st.expander("Reagenz-Burndown", expanded=not st.session_state.dashboard_exec_mode):
        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            remaining = kpis.get("remaining_stock", [])
            if remaining:
                bd = forecast_df["Date"].values
                fig_burn = go.Figure()
                fig_burn.add_trace(go.Scatter(x=bd, y=remaining, name="Restbestand", fill="tozeroy", line=dict(color="#0284C7", width=2), fillcolor="rgba(94,234,212,0.06)", hovertemplate="%{x|%d %b}: %{y:,.0f}<extra></extra>"))
                fig_burn.add_trace(go.Scatter(x=bd, y=np.cumsum(forecast_df["Predicted Volume"].values), name="Kum. Bedarf", line=dict(color="#ef4444", width=1.5, dash="dot"), hovertemplate="%{x|%d %b}: %{y:,.0f}<extra></extra>"))
                fig_burn.add_hline(y=stock_on_hand, line_dash="longdash", line_color="rgba(148,163,184,0.2)")

                stockout_day = kpis.get("stockout_day")
                if stockout_day:
                    so_str = pd.Timestamp(stockout_day).strftime("%Y-%m-%d")
                    fig_burn.add_shape(type="line", x0=so_str, x1=so_str, y0=0, y1=1, yref="paper", line=dict(color="rgba(239,68,68,0.4)", width=1, dash="dash"))

                fig_burn.update_layout(template="plotly_white", paper_bgcolor="rgba(255,255,255,1)", plot_bgcolor="rgba(248,250,252,1)", height=300, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center", font=dict(size=9, color="#475569"), bgcolor="rgba(0,0,0,0)"), hovermode="x unified")
                fig_burn.update_xaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#475569"))
                fig_burn.update_yaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#475569"))
                st.plotly_chart(fig_burn, width='stretch', key="burndown")
                burndown_fig_for_pdf = fig_burn
            else:
                st.caption("Kein Burndown-Datensatz verfügbar.")

        with col_r:
            remaining = kpis.get("remaining_stock", [])
            stockout_today = "keine" if not remaining else f"{len(remaining)} Tage"
            st.metric("Forecast-Horizont", f"{forecast_horizon} Tage", stockout_today)
        st.metric("Restbestand aktuell", f"{int(kpis.get('stock_on_hand', stock_on_hand)):,.0f}")
    st.metric("Erwartete Zusatzbestellungen", f"{reagent_orders:,.0f}")

    with st.expander("Bestellempfehlungen", expanded=False):
        def hl(row):
            if row["Reagent Order"] > 0:
                return ["background-color: var(--warning-soft); color: #991b1b; font-weight: 700;"] * len(row)
            return ["color: var(--text-soft);"] * len(row)
        styled = forecast_df.style.apply(hl, axis=1).format({"Predicted Volume": "{:,.0f}", "Reagent Order": "{:,.0f}", rev_col_name: "EUR {:,.0f}"})
        st.dataframe(styled, width='stretch', hide_index=True, height=320)

    if st.button("What-if-Szenarien berechnen", key="compute_what_if", width='stretch', type="secondary"):
        what_if_df = _ensure_what_if_df_cached()
        st.session_state["_labpulse_what_if_df_visible"] = what_if_df

    what_if_df = st.session_state.get("_labpulse_what_if_df_visible", what_if_df)
    if not what_if_df.empty:
        with st.expander("What-if-Schnellvergleich", expanded=True):
            what_if_view = what_if_df.copy()
            if "Beschaffung in (Tage)" in what_if_view.columns:
                what_if_view["Beschaffung in (Tage)"] = what_if_view["Beschaffung in (Tage)"].fillna("—")
            st.caption("Kurzer Vergleich der drei operativen Szenarien inkl. Handlungsempfehlung")
            st.dataframe(
                what_if_view.style.format(
                    {
                        "Uplift (%)": "{:.0f}%",
                        "Risk at Risk (EUR)": "EUR {:,.0f}",
                        "Bestellbedarf": "{:,.0f}",
                    }
                ),
                width='stretch',
                hide_index=True,
                height=210,
            )

    with st.expander("Digitale Freigabe (Chain-of-Custody)", expanded=not st.session_state.dashboard_exec_mode):
        st.caption(f"Decision-ID: {decision_id} · Status: {approval_status} · Signaturen: {approval_signed}/{approval_needed}")
        if approval_state.get("approvals"):
            st.dataframe(
                pd.DataFrame(approval_state.get("approvals", [])),
                width='stretch',
                hide_index=True,
                height=150,
            )
        else:
            st.info("Noch keine Signatur erfasst.")

        if can_approve_current:
            approver_name = st.text_input(
                "Person",
                value=st.session_state.decision_signer_name,
                key=f"approver_person_{decision_id}",
            )
            approver_note = st.text_area(
                "Kommentar (optional)",
                value="",
                key=f"approver_note_{decision_id}",
            )
            if st.button("Jetzt freigeben", key=f"approve_decision_{decision_id}", type="primary", width='stretch'):
                if not approver_name.strip():
                    st.warning("Bitte Name der Person angeben.")
                elif _append_approval_signature(
                    decision_id=decision_id,
                    state=approval_state,
                    actor=approver_name.strip(),
                    comment=approver_note,
                ):
                    st.session_state.decision_signer_name = approver_name.strip()
                    st.success("Freigabe erfasst. Kette aktualisiert.")
                    _log_event(
                        "decision_signature_added",
                        {
                            "decision_id": decision_id,
                            "actor": approver_name.strip(),
                        },
                    )
                    st.rerun()
                else:
                    st.info("Diese Person hat bereits eine Signatur abgegeben.")
        else:
            if approval_state["status"] != "Freigegeben":
                st.caption("Das Entscheidungsobjekt ist bereits vollständig freigegeben.")

    # PDF export
    with st.expander("PDF-Export", expanded=False):
        if st.button("PDF-Bericht generieren", width='stretch', type="primary"):
            with st.spinner("Generiere …"):
                try:
                    ai_commentary = (
                        _cached_pdf_commentary(selected_pathogen, ai_kpi_snapshot)
                        if ollama_ok
                        else st.session_state.get(_ai_insight_state_key)
                        or _cached_ai_fallback_insight(selected_pathogen, ai_kpi_snapshot)
                    )
                    pdf_bytes = generate_pdf_report(kpis=kpis, forecast_df=forecast_df, correlation_fig=correlation_fig_for_pdf, burndown_fig=burndown_fig_for_pdf, ai_commentary=ai_commentary, pathogen=selected_pathogen)
                    _log_event(
                        "pdf_generated",
                        {
                            "pathogen": selected_pathogen,
                            "bytes": len(pdf_bytes) if pdf_bytes else 0,
                            "horizon": forecast_horizon,
                        },
                    )
                    st.download_button("Herunterladen", data=pdf_bytes, file_name=f"LabPulse_{selected_pathogen}_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf", width='stretch')
                except Exception as exc:
                    st.error(f"Fehler: {exc}")

    with st.expander("Operations-Export (Admin)", expanded=False):
        export_signature_key = "_labpulse_export_signature"
        export_keys = [
            "_labpulse_ops_json",
            "_labpulse_ops_json_name",
            "_labpulse_audit_jsonl",
            "_labpulse_audit_jsonl_name",
            "_labpulse_audit_csv",
            "_labpulse_audit_csv_name",
            "_labpulse_forecast_csv",
            "_labpulse_forecast_csv_name",
            "_labpulse_compliance_json",
            "_labpulse_compliance_json_name",
            "_labpulse_compliance_zip",
            "_labpulse_compliance_zip_name",
        ]
        if st.session_state.get(export_signature_key) != _payload_signature:
            for key in export_keys:
                st.session_state.pop(key, None)
            st.session_state[export_signature_key] = _payload_signature

        def _prepare_and_cache(
            cache_key: str,
            name_key: str,
            builder,
            file_name: str,
            label: str,
        ):
            if st.button(label, width='stretch'):
                try:
                    spinner_label = label.replace("vorbereiten", "").replace("Vorbereiten", "").strip()
                    with st.spinner(f"{spinner_label or label} wird vorbereitet ..."):
                        data = builder()
                    st.session_state[cache_key] = data
                    st.session_state[name_key] = file_name
                except Exception as exc:
                    st.error(f"Vorbereitung fehlgeschlagen: {exc}")

            if cache_key in st.session_state:
                return st.session_state[cache_key], st.session_state[name_key]
            return None, None

        def _get_compliance_payload():
            compliance_state_key = "_labpulse_compliance_payload_signature"
            if (
                st.session_state.get(compliance_state_key) != _payload_signature
                or not st.session_state.get("compliance_payload")
            ):
                local_operations_payload = _ensure_operations_payload()
                local_compliance_payload = _compliance_export_payload(
                    selected_pathogen=selected_pathogen,
                    forecast_horizon=forecast_horizon,
                    decision_pack=decision_pack,
                    forecast_payload=local_operations_payload,
                    decision_state=approval_state,
                    roi_data=roi_data,
                    kpis=kpis,
                    composite=composite,
                )
                local_operations_payload["compliance_signature"] = _digest_payload(
                    local_compliance_payload
                )
                st.session_state.operations_payload = local_operations_payload
                st.session_state.compliance_payload = local_compliance_payload
                st.session_state[compliance_state_key] = _payload_signature
            return st.session_state.get("compliance_payload", {})

        st.caption("JSONL/CSV-Artefakte für API- und Compliance-Flows")

        ops_json, ops_json_name = _prepare_and_cache(
            cache_key="_labpulse_ops_json",
            name_key="_labpulse_ops_json_name",
            builder=lambda: json.dumps(_ensure_operations_payload(), ensure_ascii=False, indent=2),
            file_name=f"labpulse_ops_{selected_pathogen}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            label="Forecast-Operations (JSON) vorbereiten",
        )
        if ops_json is not None:
            st.download_button(
                "Forecast-Operations (JSON) herunterladen",
                data=ops_json,
                file_name=ops_json_name,
                mime="application/json",
                width='stretch',
            )

        audit_jsonl, audit_jsonl_name = _prepare_and_cache(
            cache_key="_labpulse_audit_jsonl",
            name_key="_labpulse_audit_jsonl_name",
            builder=lambda: _to_jsonl_download_payload(AUDIT_LOG_PATH),
            file_name=f"labpulse_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            label="Audit-Log (JSON) vorbereiten",
        )
        if audit_jsonl is not None:
            st.download_button(
                "Audit-Log (JSON) herunterladen",
                data=audit_jsonl,
                file_name=audit_jsonl_name,
                mime="application/json",
                width='stretch',
            )

        audit_csv, audit_csv_name = _prepare_and_cache(
            cache_key="_labpulse_audit_csv",
            name_key="_labpulse_audit_csv_name",
            builder=lambda: _to_csv_download_bytes(AUDIT_LOG_PATH),
            file_name=f"labpulse_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            label="Audit-Log (CSV) vorbereiten",
        )
        if audit_csv is not None:
            st.download_button(
                "Audit-Log (CSV) herunterladen",
                data=audit_csv,
                file_name=audit_csv_name,
                mime="text/csv",
                width='stretch',
            )

        forecast_csv, forecast_csv_name = _prepare_and_cache(
            cache_key="_labpulse_forecast_csv",
            name_key="_labpulse_forecast_csv_name",
            builder=lambda: forecast_df.to_csv(index=False).encode("utf-8"),
            file_name=f"labpulse_forecast_{selected_pathogen}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            label="Forecast-Zeitreihe (CSV) vorbereiten",
        )
        if forecast_csv is not None:
            st.download_button(
                "Forecast-Zeitreihe (CSV) herunterladen",
                data=forecast_csv,
                file_name=forecast_csv_name,
                mime="text/csv",
                width='stretch',
            )

        st.divider()
        st.caption("Compliance-Paket (signiert, revisionssicher).")
        compliance_json, compliance_json_name = _prepare_and_cache(
            cache_key="_labpulse_compliance_json",
            name_key="_labpulse_compliance_json_name",
            builder=lambda: json.dumps(
                _to_json_safe(_get_compliance_payload()),
                ensure_ascii=False,
                indent=2,
            ),
            file_name=f"labpulse_compliance_{decision_id}_{_iso_for_file_name()}.json",
            label="Compliance-Paket (JSON) vorbereiten",
        )
        if compliance_json is not None:
            st.download_button(
                "Compliance-Paket (JSON) herunterladen",
                data=compliance_json,
                file_name=compliance_json_name,
                mime="application/json",
                width='stretch',
            )

        _zip_what_if_df = st.session_state.get("_labpulse_what_if_df_visible")
        if not isinstance(_zip_what_if_df, pd.DataFrame) or _zip_what_if_df.empty:
            _zip_what_if_df = _ensure_what_if_df_cached()

        compliance_zip, compliance_zip_name = _prepare_and_cache(
            cache_key="_labpulse_compliance_zip",
            name_key="_labpulse_compliance_zip_name",
            builder=lambda: _compliance_package_zip_bytes(
                compliance_payload=_get_compliance_payload(),
                operations_payload=_ensure_operations_payload(),
                decision_id=decision_id,
                what_if_df=_zip_what_if_df,
            ),
            file_name=f"labpulse_compliance_bundle_{decision_id}_{_iso_for_file_name()}.zip",
            label="Compliance-Paket (ZIP) vorbereiten",
        )
        if compliance_zip is not None:
            st.download_button(
                "Compliance-Paket (ZIP) herunterladen",
                data=compliance_zip,
                file_name=compliance_zip_name,
                mime="application/zip",
                width='stretch',
            )


# ── Ansicht: Signale & Trends (merged from 3 old tabs) ──────────────────────
if _dashboard_view == "Signale & Trends":
    if st.session_state.dashboard_exec_mode:
        st.info("Signalanalyse ist im Analysemodus fokussiert ausgeblendet. Wechseln Sie in den Detailmodus für umfassendere Einsichten.")
    else:
        load_signal_data = st.toggle("Surveillance-Daten laden", value=False, key="load_signal_data")
        if not load_signal_data:
            st.caption("Schalten Sie diesen Bereich ein, wenn Sie GrippeWeb-/ARE-/Trends-Daten laden möchten.")
        else:
            st.caption("Surveillance-Signale — GrippeWeb & ARE")
            gw_type = st.radio("GrippeWeb", ["ARE", "ILI"], horizontal=True, key="gw_type")

            if "trends_custom_terms" not in st.session_state:
                suggested_terms = PATHOGEN_SEARCH_TERMS.get(selected_pathogen, [])
                st.session_state.trends_custom_terms = ", ".join(suggested_terms[:3])
            custom_input = st.text_input(
                "Suchbegriffe (max 5)",
                value=st.session_state.get("trends_custom_terms", ""),
                key="trends_input",
                label_visibility="collapsed",
                placeholder="z.B. Corona Test, Grippe",
            )
            st.session_state.trends_custom_terms = custom_input
            tf = st.selectbox(
                "Zeitraum",
                ["today 3-m", "today 12-m", "today 5-y"],
                index=1,
                format_func=lambda x: {"today 3-m": "3M", "today 12-m": "12M", "today 5-y": "5J"}[x],
                key="trends_tf",
                label_visibility="collapsed",
            )
            user_terms = _normalize_terms(custom_input.split(","))

            with st.status("Surveillance-Signale werden geladen ...", expanded=False):
                gw_df, are_df, trends_df = _load_surveillance_bundle_cached(
                    grippeweb_type=gw_type,
                    region="Bundesweit",
                    trends_terms=user_terms,
                    trends_timeframe=tf,
                    trends_geo="DE",
                )

            sc1, sc2 = st.columns(2, gap="large")
            with sc1:
                st.caption("GrippeWeb")
                if not gw_df.empty:
                    gw_r = gw_df[gw_df["date"] >= (today - pd.Timedelta(days=730))].copy()
                    fig_gw = go.Figure()
                    fig_gw.add_trace(go.Scatter(
                        x=gw_r["date"],
                        y=gw_r["incidence"],
                        fill="tozeroy",
                        line=dict(color="#2563EB", width=2),
                        fillcolor="rgba(37,99,235,0.08)",
                        hovertemplate="%{x|%d %b}: %{y:,.1f}<extra></extra>",
                    ))
                    fig_gw.update_layout(template="plotly_white", paper_bgcolor="rgba(255,255,255,1)", plot_bgcolor="rgba(248,250,252,1)", height=260, margin=dict(l=0, r=0, t=5, b=0), showlegend=False, hovermode="x unified")
                    fig_gw.update_xaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#475569"))
                    fig_gw.update_yaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#475569"))
                    st.plotly_chart(fig_gw, width='stretch', key="gw")
                else:
                    st.caption("Nicht verfuegbar.")

            with sc2:
                st.caption("ARE-Konsultationen")
                if not are_df.empty:
                    ar = are_df[are_df["date"] >= (today - pd.Timedelta(days=730))].copy()
                    fig_are = go.Figure()
                    fig_are.add_trace(go.Scatter(
                        x=ar["date"],
                        y=ar["consultation_incidence"],
                        fill="tozeroy",
                        line=dict(color="#0284C7", width=2),
                        fillcolor="rgba(2,132,199,0.1)",
                        hovertemplate="%{x|%d %b}: %{y:,.0f}<extra></extra>",
                    ))
                    fig_are.update_layout(template="plotly_white", paper_bgcolor="rgba(255,255,255,1)", plot_bgcolor="rgba(248,250,252,1)", height=260, margin=dict(l=0, r=0, t=5, b=0), showlegend=False, hovermode="x unified")
                    fig_are.update_xaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#475569"))
                    fig_are.update_yaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#475569"))
                    st.plotly_chart(fig_are, width='stretch', key="are")
                else:
                    st.caption("Nicht verfuegbar.")

            # Google Trends (collapsed by default)
            with st.expander("Google Trends", expanded=False):
                if trends_df is None or trends_df.empty:
                    st.caption("Keine Daten (Rate-Limit oder zu nischig).")
                else:
                    fig_t = go.Figure()
                    colors = ["#EA580C", "#0284C7", "#2563EB", "#16A34A", "#F97316"]
                    for i, col in enumerate([c for c in trends_df.columns if c != "date"]):
                        fig_t.add_trace(go.Scatter(
                            x=trends_df["date"],
                            y=trends_df[col],
                            name=col,
                            line=dict(color=colors[i % 5], width=2),
                        ))
                    fig_t.update_layout(
                        template="plotly_white",
                        paper_bgcolor="rgba(255,255,255,1)",
                        plot_bgcolor="rgba(248,250,252,1)",
                        height=280,
                        margin=dict(l=0, r=0, t=5, b=0),
                        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center", font=dict(size=9, color="#475569"), bgcolor="rgba(0,0,0,0)"),
                        hovermode="x unified",
                    )
                    fig_t.update_xaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#475569"))
                    fig_t.update_yaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#475569"))
                    st.plotly_chart(fig_t, width='stretch', key="trends")


# ── Ansicht: Regional ────────────────────────────────────────────────────────
if _dashboard_view == "Regional":
    if st.session_state.dashboard_exec_mode:
        st.info("Regionale Details sind im Analysemodus fokussiert ausgeblendet. Wechseln Sie in den Detailmodus.")
    else:
        load_regional_data = st.toggle("Regionale Auswertung laden", value=False, key="load_regional_data")
        if not load_regional_data:
            st.caption("Schalten Sie diesen Bereich ein, um die regionale Kennzahlen-Karte zu berechnen.")
        else:
            if raw_df.empty or "bundesland" not in raw_df.columns:
                st.caption("Keine regionalen Daten verfuegbar.")
            else:
                pathogen_types = PATHOGEN_GROUPS.get(selected_pathogen, [selected_pathogen])
                regional_df = _aggregate_regional_cached(
                    raw_df.to_json(orient="split", date_format="iso"),
                    tuple(pathogen_types),
                    30,
                )

                if regional_df.empty:
                    st.caption(f"Keine Daten fuer {selected_pathogen}.")
                else:
                    with st.expander("Kartensicht", expanded=False):
                        map_df = create_scatter_map_data(regional_df)
                        if not map_df.empty:
                            col_m, col_t = st.columns([2, 1], gap="large")
                            with col_m:
                                vl_min, vl_max = map_df["avg_virus_load"].min(), map_df["avg_virus_load"].max()
                                map_df["bs"] = 25 if vl_max <= vl_min else 12 + (map_df["avg_virus_load"] - vl_min) / (vl_max - vl_min) * 40
                                fig_m = go.Figure()
                                fig_m.add_trace(go.Scatter(
                                    x=map_df["lon"], y=map_df["lat"], mode="markers+text",
                                    marker=dict(size=map_df["bs"], color=map_df["trend_pct"], colorscale=[[0, "#0284C7"], [0.5, "#EA580C"], [1, "#B91C1C"]], opacity=0.85, line=dict(width=1, color="rgba(255,255,255,0.2)")),
                                    text=map_df["bundesland"], textposition="top center", textfont=dict(size=9, color="#475569"),
                                    hovertemplate="<b>%{text}</b><br>Ø %{customdata[0]:,.0f}<br>Trend %{customdata[1]:+.1f}%<extra></extra>",
                                    customdata=map_df[["avg_virus_load", "trend_pct"]].values,
                                ))
                                fig_m.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(255,255,255,1)", plot_bgcolor="rgba(248,250,252,1)",
                                    xaxis=dict(range=[5.5, 15.5], showgrid=True, gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#475569")),
                                    yaxis=dict(range=[47, 55.5], showgrid=True, gridcolor="rgba(255,255,255,0.02)", scaleanchor="x", scaleratio=1.5, tickfont=dict(size=9, color="#475569")),
                                    showlegend=False)
                                st.plotly_chart(fig_m, width='stretch', key="map")

                            with col_t:
                                st.caption(f"Top-Regionen · {selected_pathogen}")
                                disp = regional_df[["bundesland", "avg_virus_load", "trend_pct", "site_count"]].copy()
                                disp.columns = ["Land", "Ø Viruslast", "Trend", "Standorte"]
                                disp["Ø Viruslast"] = disp["Ø Viruslast"].map(lambda x: f"{x:,.0f}")
                                disp["Trend"] = disp["Trend"].map(lambda x: f"{x:+.1f}%")
                                st.dataframe(disp, width='stretch', hide_index=True, height=300)


# ── Footer ───────────────────────────────────────────────────────────────────
cost = kpis.get("cost_per_test", AVG_REVENUE_PER_TEST)
ml_str = f" · ML: {ml_model_info['model_type']} ({ml_model_info['confidence_score']:.0f}%)" if ml_model_info else ""
_perf_footer = _perf_summary_text()
_perf_footer = (f" · Perf: { _perf_footer}" if _perf_footer else "")
st.markdown(
    f'<div class="zen-foot">'
    f'<span>LabPulse AI v2 · {selected_pathogen} · EUR {cost}/Test{ml_str}</span>'
    f'{_perf_footer}'
    f'<span><a href="https://github.com/robert-koch-institut/Abwassersurveillance_AMELAG" target="_blank">RKI AMELAG</a> · {datetime.now().strftime("%Y-%m-%d %H:%M")}</span>'
    f'</div>',
    unsafe_allow_html=True,
)
