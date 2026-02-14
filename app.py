"""
LabPulse AI — Landing Page
===========================
Landing page with a high-contrast, light-first editorial layout and
operationally clear visual hierarchy for decision-making.
"""

import streamlit as st
import os
from urllib.parse import urlparse

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



def _normalize_public_url(raw_url: str) -> str:
    base = (raw_url or "").strip().rstrip("/")
    if not base:
        return ""
    parsed = urlparse(base)
    if parsed.scheme in {"http", "https"}:
        base = parsed.path or ""
    return base


_dashboard_public_url = _normalize_public_url(os.getenv("LABPULSE_PUBLIC_URL", ""))
_dashboard_open_url = (
    f"{_dashboard_public_url}/dashboard" if _dashboard_public_url else "/dashboard"
).rstrip("/")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&family=IBM+Plex+Mono:wght@500;700&display=swap');

:root {
    --lp-bg: #f4f8fc;
    --lp-surface: #ffffff;
    --lp-line: #d8e2ef;
    --lp-accent: #ea580c;
    --lp-accent-bright: #f97316;
    --lp-accent-soft: rgba(234, 88, 12, 0.10);
    --lp-text: #0f172a;
    --lp-text-soft: #334155;
    --lp-text-muted: #64748b;
    --lp-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
    --lp-shadow-soft: 0 8px 20px rgba(15, 23, 42, 0.05);
    --lp-radius: 16px;
    --signal-teal: #0284c7;
    --signal-violet: #2563eb;
    --signal-rose: #ef4444;
    --signal-lime: #16a34a;
    --font-display: 'Manrope', sans-serif;
    --font-body: 'Inter', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
}

/* ── Reset & Global ──────────────────────────────────────── */

#MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }

html, body, [class*="css"] { font-family: var(--font-body); color: var(--lp-text); color-scheme: light; }

.stApp {
    background:
        radial-gradient(circle at 10% -8%, rgba(234, 88, 12, 0.10), transparent 40%),
        radial-gradient(circle at 88% 2%, rgba(37, 99, 235, 0.08), transparent 42%),
        var(--lp-bg) !important;
    color: var(--lp-text) !important;
    overflow-x: hidden;
}

.stApp::before, .stApp::after { content: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.lp-bg, .lp-bg::before, .lp-bg::after { display: none !important; }
.lp-page { position: relative; z-index: 1; min-height: 100vh; display: flex; flex-direction: column; background: transparent; }

/* ── Animations ──────────────────────────────────────────── */

@keyframes fadeUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeDown { from { opacity: 0; transform: translateY(-12px); } to { opacity: 1; transform: translateY(0); } }
@keyframes lineExpand { from { width: 0; } to { width: 80px; } }

/* ── Nav ─────────────────────────────────────────────────── */

.lp-nav {
    display: flex; justify-content: space-between; align-items: center;
    position: sticky; top: 0; z-index: 15;
    padding: 0.15rem 3.2rem 0.22rem;
    backdrop-filter: blur(14px);
    background: rgba(255, 255, 255, 0.86);
    border-bottom: 1px solid rgba(15, 23, 42, 0.08);
    animation: fadeDown 0.7s ease-out both;
}

.lp-nav-brand { display: flex; align-items: center; gap: 0.8rem; }

.lp-nav-mark {
    width: 34px; height: 34px; border-radius: 9px;
    background: linear-gradient(135deg, var(--lp-accent), var(--lp-accent-bright));
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem;
    box-shadow: 0 10px 30px rgba(234, 88, 12, 0.25);
}

.lp-nav-name { font-family: var(--font-body); font-weight: 700; font-size: 1rem; color: var(--lp-text); letter-spacing: -0.02em; }
.lp-nav-ver { font-family: var(--font-mono); font-size: 0.6rem; font-weight: 500; color: var(--lp-text-muted); margin-left: 0.4rem; padding: 0.15rem 0.5rem; border-radius: 4px; border: 1px solid var(--lp-line); }

/* ── Hero ────────────────────────────────────────────────── */

.lp-hero {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center; text-align: center;
    padding: 0.35rem 2rem 0.75rem; min-height: auto;
}

.lp-eyebrow {
    font-family: var(--font-mono); font-size: 0.65rem; font-weight: 600;
    color: var(--lp-accent); letter-spacing: 0.24em; text-transform: uppercase;
    margin-bottom: 1rem;
    animation: fadeUp 0.6s ease-out 0.15s both;
}

.lp-hero h1 {
    font-family: var(--font-display);
    font-size: clamp(2.0rem, 6vw, 3.5rem); font-weight: 700;
    color: var(--lp-text); line-height: 1.05;
    margin: 0.05rem auto 0.18rem; max-width: 980px;
    letter-spacing: -0.045em;
    animation: fadeUp 0.8s ease-out 0.3s both;
}

.lp-hero h1 strong {
    color: var(--lp-accent);
    position: relative; padding-bottom: 0.15rem;
}

.lp-hero h1 strong::after {
    content: ""; display: block;
    width: 84px; height: 4px; border-radius: 999px;
    margin-top: 0.6rem;
    background: linear-gradient(90deg, rgba(234, 88, 12, 0), var(--lp-accent), rgba(234, 88, 12, 0));
}

.lp-hero-line {
    width: 80px; height: 1px; background: var(--lp-accent);
    margin: 0.68rem auto 0.7rem; opacity: 0.6;
    animation: lineExpand 0.8s ease-out 0.5s both;
}

.lp-hero-sub {
    font-family: var(--font-body);
    font-size: clamp(0.88rem, 1.4vw, 1.05rem); font-weight: 400;
    color: var(--lp-text-soft); max-width: 640px;
    margin: 0 auto 0.55rem; line-height: 1.5;
    animation: fadeUp 0.7s ease-out 0.6s both;
}

/* ── Journey ─────────────────────────────────────────────── */

.lp-journey {
    padding: 2rem 3.2rem 1.5rem;
    max-width: 1100px; margin: 0 auto;
}

.lp-journey-head { text-align: center; margin-bottom: 2.1rem; }

.lp-journey-head h2 {
    font-family: var(--font-display);
    font-size: clamp(1.6rem, 3.4vw, 2.5rem);
    font-weight: 700; color: var(--lp-text); margin: 0;
    letter-spacing: -0.025em;
}

.lp-journey-head p { color: var(--lp-text-soft); margin: 0.85rem auto 0; max-width: 620px; font-size: 0.92rem; }

.lp-journey-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }

.lp-journey-item {
    background: linear-gradient(165deg, #fff, rgba(255,255,255,0.9));
    border: 1px solid var(--lp-line); border-radius: var(--lp-radius);
    padding: 1.1rem 1rem;
    box-shadow: var(--lp-shadow-soft);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.lp-journey-item:hover { transform: translateY(-2px); box-shadow: var(--lp-shadow); }
.lp-journey-item h3 { margin: 0; font-family: var(--font-display); font-size: 1.05rem; color: var(--lp-text); }
.lp-journey-item p { margin: 0.5rem 0 0; color: var(--lp-text-soft); font-size: 0.84rem; }

.lp-journey-step {
    width: 2rem; height: 2rem; border-radius: 999px;
    display: inline-flex; align-items: center; justify-content: center;
    margin-bottom: 0.6rem; font-weight: 700; color: #fff;
    background: linear-gradient(135deg, var(--lp-accent), var(--lp-accent-bright));
}

/* ── CTA ─────────────────────────────────────────────────── */

.lp-cta-row {
    display: flex; gap: 0.7rem; margin-top: 0.6rem;
    flex-wrap: nowrap; justify-content: center;
    animation: fadeUp 0.7s ease-out 0.75s both;
}

.lp-btn {
    font-family: var(--font-body); font-size: 0.82rem; font-weight: 700;
    padding: 0.68rem 1.35rem; border-radius: 12px;
    text-decoration: none; border: 1px solid transparent;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.lp-btn-fill {
    color: #fff !important;
    background: linear-gradient(135deg, var(--lp-accent-bright), var(--lp-accent));
    box-shadow: 0 16px 36px rgba(234, 88, 12, 0.2);
}

.lp-btn-fill:hover { transform: translateY(-2px); box-shadow: 0 20px 40px rgba(234, 88, 12, 0.28); color: #fff; }

.lp-btn-ghost {
    color: var(--lp-text-soft) !important; border-color: rgba(15, 23, 42, 0.16);
    background: transparent;
}

.lp-btn-ghost:hover { border-color: var(--lp-accent); color: var(--lp-text) !important; }

/* ── Stats ───────────────────────────────────────────────── */

.lp-stats {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1.2rem; padding: 1.7rem 1.35rem;
    animation: fadeUp 0.7s ease-out 0.9s both;
}

.lp-stat {
    text-align: center;
    background: linear-gradient(180deg, #fff, rgba(255, 255, 255, 0.9));
    border: 1px solid var(--lp-line); border-radius: var(--lp-radius);
    padding: 1.2rem 1rem 0.95rem;
    box-shadow: var(--lp-shadow);
}

.lp-stat-num {
    font-family: var(--font-display); font-size: 2.6rem; font-weight: 900;
    color: var(--lp-text); line-height: 1; letter-spacing: -0.03em; white-space: nowrap;
}

.lp-stat-num span { color: var(--lp-accent); }

.lp-stat-lbl {
    font-family: var(--font-mono); font-size: 0.6rem; font-weight: 500;
    color: var(--lp-text-muted); text-transform: uppercase;
    letter-spacing: 0.18em; margin-top: 0.6rem;
}

/* ── Feature Bento ───────────────────────────────────────── */

.lp-sect { padding: 1.7rem 1.35rem; max-width: 1120px; margin: 0 auto; }
.lp-sect-head { text-align: center; margin-bottom: 2.5rem; }

.lp-sect-head h2 {
    font-family: var(--font-display); font-size: clamp(1.6rem, 3.5vw, 2.6rem);
    font-weight: 700; color: var(--lp-text); margin: 0; letter-spacing: -0.03em;
    text-wrap: balance;
}

.lp-sect-head h2 strong { color: var(--lp-accent); }
.lp-sect-head p { font-family: var(--font-body); color: var(--lp-text-soft); font-size: 0.92rem; margin-top: 1rem; max-width: 500px; margin-left: auto; margin-right: auto; }

.lp-grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 1rem; }

.lp-card {
    background: linear-gradient(180deg, #fff, rgba(255, 255, 255, 0.88));
    border: 1px solid var(--lp-line); border-radius: var(--lp-radius);
    padding: 2rem; position: relative; overflow: hidden;
    box-shadow: var(--lp-shadow);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}

.lp-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 10%, var(--lp-accent) 50%, transparent 90%);
    opacity: 0; transition: opacity 0.4s ease;
}

.lp-card:hover { transform: translateY(-4px); border-color: rgba(234, 88, 12, 0.35); box-shadow: 0 18px 38px rgba(15, 23, 42, 0.12); }
.lp-card:hover::after { opacity: 0.5; }

.lp-c-7 { grid-column: span 7; }
.lp-c-5 { grid-column: span 5; }
.lp-c-4 { grid-column: span 4; }
.lp-c-8 { grid-column: span 8; }
.lp-c-6 { grid-column: span 6; }

.lp-card-tag {
    font-family: var(--font-mono); font-size: 0.58rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.2em;
    padding: 0.2rem 0.6rem; border-radius: 4px;
    display: inline-block; margin-bottom: 1rem;
}

.lp-tag-teal { background: rgba(2, 132, 199, 0.08); color: var(--signal-teal); }
.lp-tag-copper { background: rgba(234, 88, 12, 0.08); color: var(--lp-accent); }
.lp-tag-violet { background: rgba(37, 99, 235, 0.08); color: var(--signal-violet); }
.lp-tag-rose { background: rgba(239, 68, 68, 0.08); color: var(--signal-rose); }
.lp-tag-lime { background: rgba(22, 163, 106, 0.08); color: var(--signal-lime); }

.lp-card h3 { font-family: var(--font-body); font-size: 1.1rem; font-weight: 700; color: var(--lp-text); margin: 0 0 0.8rem; letter-spacing: -0.02em; }
.lp-card p { font-family: var(--font-body); font-size: 0.84rem; color: var(--lp-text-soft); line-height: 1.7; margin: 0; }
.lp-card-big-num { font-family: var(--font-display); font-size: 3.5rem; font-weight: 900; color: var(--lp-accent); line-height: 1; margin-top: auto; padding-top: 2rem; }

/* ── Signal Pills ────────────────────────────────────────── */

.lp-pills { padding: 1.7rem 1.35rem 3rem; text-align: center; }

.lp-pills h2 {
    font-family: var(--font-display); font-size: 2rem; font-weight: 700;
    color: var(--lp-text); margin-bottom: 0.8rem; letter-spacing: -0.02em;
}

.lp-pills h2 strong { color: var(--lp-accent); }
.lp-pills > p { font-family: var(--font-body); color: var(--lp-text-soft); max-width: 480px; margin: 0 auto 2rem; font-size: 0.88rem; }

.lp-pill-row { display: flex; justify-content: center; gap: 0.8rem; flex-wrap: wrap; max-width: 800px; margin: 0 auto; row-gap: 0.6rem; }

.lp-pill {
    display: flex; align-items: center; gap: 0.6rem;
    background: #fff; border: 1px solid #dbeafe;
    border-radius: 999px; padding: 0.62rem 1.1rem;
    transition: all 0.25s ease;
}

.lp-pill:hover { border-color: var(--lp-accent); transform: translateY(-2px); box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08); }
.lp-pill-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.lp-pill-name { font-family: var(--font-body); font-size: 0.78rem; font-weight: 600; color: var(--lp-text); }
.lp-pill-wt { font-family: var(--font-mono); font-size: 0.58rem; font-weight: 600; color: var(--lp-text-muted); }

/* ── Footer ──────────────────────────────────────────────── */

.lp-foot {
    padding: 1.7rem 1.35rem; border-top: 1px solid var(--lp-line);
    display: flex; justify-content: space-between; align-items: center;
}

.lp-foot span { font-family: var(--font-mono); font-size: 0.6rem; color: var(--lp-text-muted); }
.lp-foot a { color: var(--lp-accent) !important; text-decoration: none; font-weight: 500; }

/* ── Responsive ──────────────────────────────────────────── */

@media (max-width: 980px) {
    .lp-nav { position: static; padding: 0.6rem 1.4rem; }
    .lp-hero { padding: 0.8rem 1.4rem 0.75rem; }
    .lp-hero h1 { font-size: clamp(2rem, 8vw, 3rem); }
    .lp-journey { padding: 1.5rem 1.4rem 1rem; }
    .lp-journey-grid { grid-template-columns: 1fr; }
    .lp-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); padding-left: 1.4rem; padding-right: 1.4rem; }
    .lp-sect, .lp-pills, .lp-foot { padding-left: 1.4rem; padding-right: 1.4rem; }
    .lp-grid { grid-template-columns: 1fr; }
    .lp-c-7, .lp-c-5, .lp-c-4, .lp-c-8, .lp-c-6 { grid-column: span 1; }
    .lp-cta-row { flex-direction: column; align-items: stretch; }
    .lp-cta-row .lp-btn { width: 100%; text-align: center; }
}

@media (max-width: 640px) {
    .lp-stats { grid-template-columns: 1fr; }
    .lp-nav { padding: 0.5rem 1.2rem; }
}
</style>""", unsafe_allow_html=True)

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

<section class="lp-journey" id="prozess">
    <div class="lp-journey-head">
        <h2>Drei Schritte zur Entscheidungssicherheit</h2>
        <p>Schnell vom Signal zur Handlung: ein klarer, auditierbarer Workflow.</p>
    </div>
    <div class="lp-journey-grid">
        <div class="lp-journey-item">
            <div class="lp-journey-step">1</div>
            <h3>KPIs prüfen</h3>
            <p>Gesamtlage, Signalqualität und Engpassrisiko in den nächsten 7 Tagen auf einen Blick.</p>
        </div>
        <div class="lp-journey-item">
            <div class="lp-journey-step">2</div>
            <h3>Prognose lesen</h3>
            <p>Signalverlauf, Unsicherheit und ML-Korrekturkurve visuell vergleichen.</p>
        </div>
        <div class="lp-journey-item">
            <div class="lp-journey-step">3</div>
            <h3>Handlung ableiten</h3>
            <p>Empfehlung, Bericht und Freigabe-Pfad direkt für die operative Entscheidung.</p>
        </div>
    </div>
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
        <h2>Alles in <strong>einem Dashboard.</strong></h2>
        <p>Vier Datenquellen. Ein Konfidenz-Score. Klare Handlungsempfehlungen.</p>
    </div>
    <div class="lp-grid">
        <div class="lp-card lp-c-7">
            <span class="lp-card-tag lp-tag-violet">Machine Learning</span>
            <h3>ML-Prognose mit Konfidenzband</h3>
            <p>Prophet-Zeitreihenmodell mit externer Signal-Einbindung auf historischen Patterns.
            Das lila Konfidenzband zeigt Unsicherheit transparent &mdash;
            damit Sie wissen, wie belastbar die Vorhersage ist.</p>
        </div>
        <div class="lp-card lp-c-5" style="display:flex;flex-direction:column;">
            <span class="lp-card-tag lp-tag-copper">Konfidenz</span>
            <h3>Signal-Fusion</h3>
            <p>Vier unabh&auml;ngige Quellen gewichtet zu einem Score.</p>
            <div class="lp-card-big-num">69.2%</div>
        </div>
        <div class="lp-card lp-c-5">
            <span class="lp-card-tag lp-tag-teal">KI-Analyse</span>
            <h3>LLM-gest&uuml;tzte Interpretation</h3>
            <p>Lokale KI (Ollama) analysiert Trends, vergleicht Signale
            und liefert handlungsrelevante Empfehlungen in nat&uuml;rlicher Sprache.</p>
        </div>
        <div class="lp-card lp-c-7">
            <span class="lp-card-tag lp-tag-teal">Korrelation</span>
            <h3>Abwasser ↔ Laborvolumen</h3>
            <p>Viruslast korreliert mit Testbedarf &mdash; 14 Tage vorher.
            Interaktiver Chart mit Zoom, Range-Slider und Prognose-Overlay.</p>
        </div>
        <div class="lp-card lp-c-4">
            <span class="lp-card-tag lp-tag-lime">Bestand</span>
            <h3>Reichweite &amp; Nachbestellung</h3>
            <p>Echtzeit-Reichweite pro Testkit mit automatischer Warnung bei Engpass.</p>
        </div>
        <div class="lp-card lp-c-4">
            <span class="lp-card-tag lp-tag-rose">Regional</span>
            <h3>16 Bundesl&auml;nder</h3>
            <p>Hotspots erkennen, bevor sie das Laborvolumen treiben.</p>
        </div>
        <div class="lp-card lp-c-4">
            <span class="lp-card-tag lp-tag-copper">Trends</span>
            <h3>Google Suchinteresse</h3>
            <p>Bev&ouml;lkerungsbewusstsein als Fr&uuml;hindikator f&uuml;r Testbedarf.</p>
        </div>
    </div>
</section>

<section class="lp-pills" id="signals">
    <h2>Vier Signale. <strong>Eine Wahrheit.</strong></h2>
    <p>Jede Datenquelle hat St&auml;rken und Schw&auml;chen. Zusammen ergibt sich ein klares Bild.</p>
    <div class="lp-pill-row">
        <div class="lp-pill">
            <div class="lp-pill-dot" style="background:var(--signal-teal);"></div>
            <span class="lp-pill-name">Abwasser (AMELAG)</span>
            <span class="lp-pill-wt">45%</span>
        </div>
        <div class="lp-pill">
            <div class="lp-pill-dot" style="background:var(--signal-violet);"></div>
            <span class="lp-pill-name">GrippeWeb</span>
            <span class="lp-pill-wt">25%</span>
        </div>
        <div class="lp-pill">
            <div class="lp-pill-dot" style="background:var(--signal-lime);"></div>
            <span class="lp-pill-name">ARE-Konsultation</span>
            <span class="lp-pill-wt">20%</span>
        </div>
        <div class="lp-pill">
            <div class="lp-pill-dot" style="background:var(--accent);"></div>
            <span class="lp-pill-name">Google Trends</span>
            <span class="lp-pill-wt">10%</span>
        </div>
    </div>
</section>

<footer class="lp-foot">
    <span>LabPulse AI v2.0</span>
    <span>Daten: <a href="https://github.com/robert-koch-institut/Abwassersurveillance_AMELAG" target="_blank">RKI AMELAG</a></span>
</footer>

</div>
""".replace("@@LP_DASHBOARD_URL@@", _dashboard_public_url).replace("@@LP_DASHBOARD_OPEN@@", _dashboard_open_url), unsafe_allow_html=True)
