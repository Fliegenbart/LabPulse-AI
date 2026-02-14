"""
LabPulse AI — Landing Page
===========================
Landing page with a high-contrast, light-first editorial layout and
operationally clear visual hierarchy for decision-making.
"""

import streamlit as st

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



_dashboard_open_url = "/Dashboard"

st.markdown("""<style>
:root {
    --lp-bg: #f4f8fc;
    --lp-surface: #ffffff;
    --lp-line: #d8e2ef;
    --lp-accent: #ea580c;
    --lp-accent-bright: #f97316;
    --lp-accent-soft: rgba(234, 88, 12, 0.10);
    --lp-accent-soft-strong: rgba(234, 88, 12, 0.22);
    --lp-text: #0f172a;
    --lp-text-soft: #334155;
    --lp-text-muted: #64748b;
    --lp-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
    --lp-shadow-soft: 0 8px 20px rgba(15, 23, 42, 0.05);
    --lp-shadow-hero: 0 16px 34px rgba(15, 23, 42, 0.12);
    --lp-glass-bg: rgba(255, 255, 255, 0.56);
    --lp-glass-border: rgba(255, 255, 255, 0.58);
    --lp-radius: 16px;
    --lp-radius-sm: 12px;
    --signal-teal: #0284c7;
    --signal-violet: #2563eb;
    --signal-rose: #ef4444;
    --signal-lime: #16a34a;
    --font-display: "SF Pro Display", "Inter Tight", "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
    --font-body: "SF Pro Text", "SF Pro Display", -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    --font-mono: "SF Mono", "Menlo", "Consolas", "Monaco", "IBM Plex Mono", monospace;
}

/* ── Reset & Global ──────────────────────────────────────── */

#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stDecoration"], [data-testid="stHeader"], .stToolbar { display: none !important; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }

html, body, [class*="css"] {
    font-family: var(--font-body);
    color: var(--lp-text);
    color-scheme: light;
    background-color: var(--lp-bg) !important;
    margin: 0;
    padding: 0;
}

.stApp {
    background:
        radial-gradient(circle at 10% -8%, rgba(234, 88, 12, 0.11), transparent 40%),
        radial-gradient(circle at 88% 2%, rgba(37, 99, 235, 0.06), transparent 42%),
        var(--lp-bg) !important;
    color: var(--lp-text) !important;
    background-color: var(--lp-bg) !important;
    overflow-x: hidden;
}

.stApp::before, .stApp::after { content: none !important; }
.block-container {
    padding: 0rem 1.05rem 0.25rem !important;
    max-width: 100% !important;
}

[data-testid="stAppViewContainer"], [data-testid="stMain"] {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

.stApp > header, .stApp > .main { margin-top: 0 !important; }
div[data-testid="stAppViewContainer"] > section.main { padding-top: 0 !important; }
div[data-testid="stAppViewContainer"] > section.main > div:first-child { margin-top: 0 !important; }

.lp-bg, .lp-bg::before, .lp-bg::after { display: none !important; }
.lp-page { position: relative; z-index: 1; min-height: 100vh; display: flex; flex-direction: column; background: transparent; }

/* ── Animations ──────────────────────────────────────────── */

@keyframes fadeUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeDown { from { opacity: 0; transform: translateY(-12px); } to { opacity: 1; transform: translateY(0); } }
@keyframes lineExpand { from { width: 0; } to { width: 80px; } }

/* ── Nav ─────────────────────────────────────────────────── */

.lp-nav {
    display: flex; justify-content: space-between; align-items: center;
    position: relative;
    top: 0;
    z-index: 15;
    margin: 0;
    padding: 0.1rem 1.1rem 0.1rem;
    backdrop-filter: blur(14px) saturate(175%);
    background: linear-gradient(110deg, var(--lp-glass-bg), rgba(255, 255, 255, 0.25));
    border-bottom: 0;
    border-top: 0;
    line-height: 1;
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
    justify-content: flex-start;
    padding: 0.12rem 1.1rem 0.34rem; min-height: auto;
}

.lp-value-row {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin: 0 auto 0.34rem;
    max-width: 1100px;
    padding: 0 1.25rem;
}

.lp-value-chip {
    border-radius: 999px;
    border: 1px solid rgba(234, 88, 12, 0.12);
    background: var(--lp-glass-bg);
    backdrop-filter: blur(10px);
    color: var(--lp-text-soft);
    font-family: var(--font-mono);
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    padding: 0.42rem 0.74rem;
    white-space: nowrap;
}

.lp-eyebrow {
    font-family: var(--font-mono); font-size: 0.65rem; font-weight: 600;
    color: var(--lp-accent); letter-spacing: 0.24em; text-transform: uppercase;
    margin-bottom: 0.74rem;
    animation: fadeUp 0.6s ease-out 0.15s both;
}

.lp-hero h1 {
    font-family: var(--font-display);
    font-size: clamp(2.0rem, 6vw, 3.5rem); font-weight: 700;
    color: var(--lp-text); line-height: 1.05;
    margin: 0.02rem auto 0.16rem; max-width: 980px;
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
    margin: 0.58rem auto 0.58rem; opacity: 0.58;
    animation: lineExpand 0.8s ease-out 0.5s both;
}

.lp-hero-sub {
    font-family: var(--font-body);
    font-size: clamp(0.84rem, 1.26vw, 0.99rem); font-weight: 400;
    color: var(--lp-text-soft); max-width: 640px;
    margin: 0 auto 0.44rem; line-height: 1.44;
    max-width: 600px;
    animation: fadeUp 0.7s ease-out 0.6s both;
}

/* ── CTA ─────────────────────────────────────────────────── */

.lp-cta-row {
    display: flex; gap: 0.6rem; margin-top: 0.35rem;
    flex-wrap: nowrap; justify-content: center;
    animation: fadeUp 0.7s ease-out 0.75s both;
}

.lp-btn {
    font-family: var(--font-body); font-size: 0.82rem; font-weight: 700;
    padding: 0.6rem 1.12rem; border-radius: 12px;
    text-decoration: none; border: 1px solid transparent;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.lp-btn-fill {
    color: #fff !important;
    background: linear-gradient(135deg, var(--lp-accent-bright), var(--lp-accent));
    border: 1px solid rgba(255, 255, 255, 0.35);
    backdrop-filter: blur(10px);
    box-shadow: 0 14px 30px rgba(234, 88, 12, 0.24);
}

.lp-btn-fill:hover { transform: translateY(-2px); box-shadow: 0 20px 40px rgba(234, 88, 12, 0.28); color: #fff; }

.lp-btn-ghost {
    color: var(--lp-text-soft) !important; border-color: rgba(15, 23, 42, 0.16);
    background: var(--lp-glass-bg);
    backdrop-filter: blur(10px);
}

.lp-btn-ghost:hover { border-color: var(--lp-accent); color: var(--lp-text) !important; }

/* ── Stats ───────────────────────────────────────────────── */

.lp-stats {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.82rem; padding: 1.16rem 1.25rem 1.04rem;
    animation: fadeUp 0.7s ease-out 0.9s both;
}

.lp-stat {
    text-align: center;
    background: linear-gradient(180deg, var(--lp-glass-bg), rgba(255, 255, 255, 0.52));
    border: 1px solid var(--lp-glass-border);
    backdrop-filter: blur(12px);
    border-radius: var(--lp-radius);
    padding: 1.2rem 1rem 0.95rem;
    box-shadow: var(--lp-shadow-soft);
}

.lp-stat-num {
    font-family: var(--font-display); font-size: 2.6rem; font-weight: 900;
    color: var(--lp-text); line-height: 1; letter-spacing: -0.03em; white-space: nowrap;
}

.lp-number,
.lp-stat-num,
.lp-card-big-num,
.lp-hero-sub,
.lp-card h3,
.lp-card p {
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum" 1, "lnum" 1;
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
    background: linear-gradient(165deg, var(--lp-glass-bg), rgba(255, 255, 255, 0.52));
    border: 1px solid var(--lp-glass-border);
    backdrop-filter: blur(14px);
    border-radius: var(--lp-radius);
    padding: 2rem; position: relative; overflow: hidden;
    box-shadow: var(--lp-shadow-soft);
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

.lp-pills { padding: 1.5rem 1.35rem 2.2rem; text-align: center; }

.lp-pills h2 {
    font-family: var(--font-display); font-size: 2rem; font-weight: 700;
    color: var(--lp-text); margin-bottom: 0.8rem; letter-spacing: -0.02em;
}

.lp-pills h2 strong { color: var(--lp-accent); }
.lp-pills > p { font-family: var(--font-body); color: var(--lp-text-soft); max-width: 480px; margin: 0 auto 2rem; font-size: 0.88rem; }

.lp-pill-row { display: flex; justify-content: center; gap: 0.8rem; flex-wrap: wrap; max-width: 800px; margin: 0 auto; row-gap: 0.6rem; }

.lp-pill {
    display: flex; align-items: center; gap: 0.6rem;
    background: var(--lp-glass-bg); border: 1px solid #dbeafe;
    backdrop-filter: blur(8px);
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
    .lp-nav { position: static; padding: 0.08rem 1.05rem; }
    .lp-hero { padding: 0.02rem 1rem 0.34rem; }
    .lp-hero h1 { font-size: clamp(2rem, 8vw, 3rem); }
    .lp-value-row { gap: 0.5rem; }
    .lp-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); padding-left: 1.12rem; padding-right: 1.12rem; }
    .lp-sect, .lp-pills, .lp-foot { padding-left: 1.4rem; padding-right: 1.4rem; }
    .lp-grid { grid-template-columns: 1fr; }
    .lp-c-7, .lp-c-5, .lp-c-4, .lp-c-8, .lp-c-6 { grid-column: span 1; }
    .lp-cta-row { flex-direction: column; align-items: stretch; }
    .lp-cta-row .lp-btn { width: 100%; text-align: center; }
}

@media (max-width: 640px) {
    .lp-stats { grid-template-columns: 1fr; }
    .lp-nav { padding: 0.06rem 0.75rem; }
    .lp-hero-sub { max-width: 95%; }
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
