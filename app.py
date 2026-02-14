"""
LabPulse AI — Landing Page
===========================
Landing page with a high-contrast, light-first editorial layout and
operationally clear visual hierarchy for decision-making.
"""

import streamlit as st
from pathlib import Path
import os

st.set_page_config(
    page_title="Home",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _get_query_param(key: str) -> str | None:
    try:
        params = st.query_params.to_dict()
        value = params.get(key)
    except Exception:
        params = st.experimental_get_query_params()
        value = params.get(key)
    if isinstance(value, list):
        if not value:
            return None
        return str(value[0])
    if value is None:
        return None
    return str(value)


_open_target = _get_query_param("open")
_page_target = _get_query_param("page")
if _open_target in {"dashboard", "1_Dashboard", "Dashboard", "1_dashboard"} or _page_target in {"1_Dashboard", "1_Dashboard.py", "Dashboard"}:
    st.switch_page("pages/1_Dashboard.py")


_PUBLIC_BASE = os.getenv("LABPULSE_PUBLIC_URL", "").rstrip("/")
_dashboard_open_url = f"{_PUBLIC_BASE}/Dashboard" if _PUBLIC_BASE else "/Dashboard"

LANDING_STYLE_PATH = Path(__file__).resolve().parent / "assets/css/landing.css"
_landing_style = LANDING_STYLE_PATH.read_text(encoding="utf-8")
st.markdown(f"<style>{_landing_style}</style>", unsafe_allow_html=True)

st.markdown('<div class="lp-bg"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="lp-page">

<nav class="lp-nav">
    <div class="lp-nav-brand">
        <div class="lp-nav-mark">🧬</div>
        <span class="lp-nav-name">LabPulse AI</span>
        <span class="lp-nav-ver">v2.0</span>
    </div>
</nav>

<section class="lp-hero">
    <div class="lp-eyebrow">Predictive Lab Intelligence</div>
    <h1>Reagenzbedarf<br>vorhersagen, <strong>bevor<br>die Welle kommt.</strong></h1>
    <div class="lp-hero-line"></div>
    <p class="lp-hero-sub">
        Abwassersurveillance trifft Machine Learning.
        LabPulse prognostiziert Ihren Testbedarf 14&nbsp;Tage im Voraus &mdash;
        mit vier unabh&auml;ngigen Datenquellen und einem Konfidenz-Score.
    </p>
    <div class="lp-cta-row">
        <a class="lp-btn lp-btn-fill" href="@@LP_DASHBOARD_OPEN@@" target="_self">Dashboard öffnen →</a>
        <a class="lp-btn lp-btn-ghost" href="#features">Mehr erfahren</a>
    </div>
</section>

<section class="lp-value-row">
    <div class="lp-value-chip">Frühwarnung</div>
    <div class="lp-value-chip">Vorhersage mit Konfidenz</div>
    <div class="lp-value-chip">Entscheidung in Minuten</div>
</section>

<section class="lp-stats">
    <div class="lp-stat">
        <div class="lp-stat-num">386<span>k</span></div>
        <div class="lp-stat-lbl">RKI Datenpunkte</div>
    </div>
    <div class="lp-stat">
        <div class="lp-stat-num">14<span>d</span></div>
        <div class="lp-stat-lbl">Vorlaufzeit</div>
    </div>
    <div class="lp-stat">
        <div class="lp-stat-num">4<span>&times;</span></div>
        <div class="lp-stat-lbl">Signal-Fusion</div>
    </div>
    <div class="lp-stat">
        <div class="lp-stat-num">5<span>+</span></div>
        <div class="lp-stat-lbl">Pathogene</div>
    </div>
</section>

<section class="lp-sect" id="features">
    <div class="lp-sect-head">
        <h2>Weniger Klicks. Mehr Klarheit.</h2>
        <p>Alle wichtigen Signale für die operative Planung an einem Ort.</p>
    </div>
    <div class="lp-grid">
        <article class="lp-card lp-c-7">
            <span class="lp-card-tag lp-tag-violet">Machine Learning</span>
            <h3>ML-Prognose mit Konfidenz</h3>
            <p>Wetter, Abwasser, Labor und Nachfrage als gemeinsam lernende Signale.</p>
        </article>
        <article class="lp-card lp-c-5">
            <span class="lp-card-tag lp-tag-copper">Entscheidung</span>
            <h3>Klare Handlungslogik</h3>
            <p>Klare Ampellogik von Risiko, Trend und Bestandsauswirkung für schnelle Entscheidungen.</p>
        </article>
        <article class="lp-card lp-c-5">
            <span class="lp-card-tag lp-tag-teal">Relevanz</span>
            <h3>Regulatorischer Kontext</h3>
            <p>Berichte, Freigabeketten und nachvollziehbare Modell-Einstellungen on demand.</p>
        </article>
    </div>
</section>


<footer class="lp-foot">
    <span>LabPulse AI v2.0</span>
    <span>Daten: <a href="https://github.com/robert-koch-institut/Abwassersurveillance_AMELAG" target="_blank">RKI AMELAG</a></span>
</footer>

</div>
""".replace("@@LP_DASHBOARD_OPEN@@", _dashboard_open_url), unsafe_allow_html=True)
