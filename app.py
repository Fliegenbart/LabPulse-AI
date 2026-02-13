"""
LabPulse AI — Landing Page
===========================
Apple-inspired landing page with dramatic typography, cinematic animations,
and a clear path to the dashboard.
"""

import streamlit as st

st.set_page_config(
    page_title="LabPulse AI — Predictive Lab Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Full-screen CSS: Apple-level design ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-void: #060a14;
    --bg-deep: #0c1222;
    --bg-surface: #162032;
    --bg-card: #1a2538;
    --accent-cyan: #22d3ee;
    --accent-cyan-dim: rgba(34,211,238,0.08);
    --accent-amber: #fbbf24;
    --accent-green: #34d399;
    --accent-red: #fb7185;
    --accent-purple: #c084fc;
    --text-bright: #f8fafc;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --border: rgba(34,211,238,0.06);
    --font-display: 'Instrument Serif', Georgia, serif;
    --font-body: 'Sora', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* Kill all Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
.stApp {
    background: var(--bg-void);
    overflow-x: hidden;
}

/* ── Animated background ── */
.lp-bg {
    position: fixed; inset: 0; z-index: 0;
    background:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(34,211,238,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 85% 70%, rgba(251,191,36,0.04) 0%, transparent 50%),
        radial-gradient(ellipse 40% 50% at 10% 80%, rgba(192,132,252,0.03) 0%, transparent 50%),
        var(--bg-void);
    pointer-events: none;
}
.lp-bg::before {
    content: '';
    position: absolute; inset: 0;
    background-image: radial-gradient(rgba(34,211,238,0.025) 1px, transparent 1px);
    background-size: 40px 40px;
}
.lp-bg::after {
    content: '';
    position: absolute;
    width: 600px; height: 600px;
    top: -200px; left: 50%;
    transform: translateX(-50%);
    border-radius: 50%;
    background: radial-gradient(circle, rgba(34,211,238,0.06) 0%, transparent 70%);
    animation: orbFloat 12s ease-in-out infinite;
}
@keyframes orbFloat {
    0%, 100% { transform: translateX(-50%) translateY(0); }
    50% { transform: translateX(-50%) translateY(30px); }
}

.lp-page {
    position: relative; z-index: 1;
    min-height: 100vh;
    display: flex; flex-direction: column;
}

/* ── Nav ── */
.lp-nav {
    display: flex; justify-content: space-between; align-items: center;
    padding: 1.5rem 4rem;
    animation: fadeDown 0.8s ease-out both;
}
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-12px); }
    to { opacity: 1; transform: translateY(0); }
}
.lp-nav-brand { display: flex; align-items: center; gap: 0.75rem; }
.lp-nav-brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--accent-cyan), #0891b2);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 0 24px rgba(34,211,238,0.15);
}
.lp-nav-brand-text {
    font-family: var(--font-body); font-weight: 600; font-size: 1rem;
    color: var(--text-primary); letter-spacing: -0.01em;
}
.lp-nav-brand-text span {
    color: var(--text-muted); font-weight: 400; font-size: 0.72rem;
    margin-left: 0.5rem; font-family: var(--font-mono);
}

/* ── Hero ── */
.lp-hero {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center; text-align: center;
    padding: 6rem 2rem 4rem; min-height: 70vh;
}
.lp-hero-eyebrow {
    font-family: var(--font-mono); font-size: 0.72rem; font-weight: 500;
    color: var(--accent-cyan); letter-spacing: 0.25em; text-transform: uppercase;
    margin-bottom: 2rem;
    animation: fadeSlideUp 0.6s ease-out 0.2s both;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
}
.lp-hero h1 {
    font-family: var(--font-display);
    font-size: clamp(3rem, 7vw, 5.5rem); font-weight: 400;
    color: var(--text-bright); line-height: 1.05;
    margin: 0 auto; max-width: 900px; letter-spacing: -0.02em;
    animation: fadeSlideUp 0.7s ease-out 0.35s both;
}
.lp-hero h1 em {
    font-style: italic;
    background: linear-gradient(135deg, var(--accent-cyan), #67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.lp-hero-sub {
    font-family: var(--font-body);
    font-size: clamp(1rem, 1.8vw, 1.2rem); font-weight: 400;
    color: var(--text-secondary); max-width: 620px;
    margin: 2rem auto 0; line-height: 1.7;
    animation: fadeSlideUp 0.7s ease-out 0.5s both;
}
.lp-hero-cta {
    display: flex; gap: 1rem; margin-top: 3.5rem;
    animation: fadeSlideUp 0.7s ease-out 0.65s both;
}
.lp-btn-primary {
    display: inline-flex; align-items: center; gap: 0.5rem;
    font-family: var(--font-body); font-size: 0.9rem; font-weight: 600;
    color: var(--bg-void);
    background: linear-gradient(135deg, var(--accent-cyan), #06b6d4);
    padding: 0.85rem 2.2rem; border-radius: 12px; text-decoration: none;
    transition: all 0.3s ease;
    box-shadow: 0 0 30px rgba(34,211,238,0.2), 0 4px 12px rgba(0,0,0,0.3);
}
.lp-btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 40px rgba(34,211,238,0.3), 0 8px 24px rgba(0,0,0,0.4);
    color: var(--bg-void);
}
.lp-btn-secondary {
    display: inline-flex; align-items: center; gap: 0.5rem;
    font-family: var(--font-body); font-size: 0.9rem; font-weight: 500;
    color: var(--text-secondary); background: transparent;
    padding: 0.85rem 2.2rem; border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08); text-decoration: none;
    transition: all 0.3s ease;
}
.lp-btn-secondary:hover {
    color: var(--text-primary);
    border-color: rgba(34,211,238,0.2);
    background: rgba(34,211,238,0.04);
}

/* ── Stats ── */
.lp-stats {
    display: flex; justify-content: center; gap: 4rem; padding: 3rem 2rem;
    animation: fadeSlideUp 0.7s ease-out 0.8s both;
}
.lp-stat { text-align: center; }
.lp-stat-value {
    font-family: var(--font-mono); font-size: 2rem; font-weight: 700;
    color: var(--text-bright); line-height: 1;
}
.lp-stat-value span { color: var(--accent-cyan); }
.lp-stat-label {
    font-family: var(--font-body); font-size: 0.72rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.12em; margin-top: 0.5rem;
}

/* ── Bento Features ── */
.lp-features { padding: 6rem 4rem; max-width: 1200px; margin: 0 auto; }
.lp-features-title { text-align: center; margin-bottom: 4rem; }
.lp-features-title h2 {
    font-family: var(--font-display); font-size: 2.8rem;
    color: var(--text-bright); font-weight: 400; margin: 0;
}
.lp-features-title p {
    font-family: var(--font-body); color: var(--text-secondary);
    font-size: 1rem; margin-top: 1rem;
}
.lp-bento {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.25rem;
}
.lp-bento-card {
    background: linear-gradient(160deg, var(--bg-card), rgba(22,32,50,0.5));
    border: 1px solid var(--border); border-radius: 16px;
    padding: 2rem 2.2rem; transition: all 0.35s ease;
    position: relative; overflow: hidden;
}
.lp-bento-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    opacity: 0; transition: opacity 0.35s ease;
}
.lp-bento-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.3), 0 0 60px rgba(34,211,238,0.04);
}
.lp-bento-card:hover::before { opacity: 1; }
.lp-bento-wide { grid-column: span 2; }
.lp-bento-tall {
    grid-row: span 2; display: flex; flex-direction: column; justify-content: space-between;
}
.lp-bento-icon {
    width: 40px; height: 40px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; margin-bottom: 1.5rem;
}
.lp-bento-icon.cyan { background: rgba(34,211,238,0.1); }
.lp-bento-icon.amber { background: rgba(251,191,36,0.1); }
.lp-bento-icon.green { background: rgba(52,211,153,0.1); }
.lp-bento-icon.purple { background: rgba(192,132,252,0.1); }
.lp-bento-icon.red { background: rgba(251,113,133,0.1); }
.lp-bento-card h3 {
    font-family: var(--font-body); font-size: 1.1rem; font-weight: 600;
    color: var(--text-primary); margin: 0 0 0.75rem 0;
}
.lp-bento-card p {
    font-family: var(--font-body); font-size: 0.85rem;
    color: var(--text-secondary); line-height: 1.65; margin: 0;
}
.lp-mono-highlight {
    font-family: var(--font-mono); font-size: 2.2rem; font-weight: 700;
    color: var(--accent-cyan); line-height: 1; margin-top: auto; padding-top: 2rem;
}

/* ── Signals ── */
.lp-signals { padding: 5rem 4rem; text-align: center; }
.lp-signals h2 {
    font-family: var(--font-display); font-size: 2.4rem;
    color: var(--text-bright); margin-bottom: 1rem;
}
.lp-signals p {
    font-family: var(--font-body); color: var(--text-secondary);
    max-width: 550px; margin: 0 auto 3rem; font-size: 0.95rem;
}
.lp-signal-row {
    display: flex; justify-content: center; gap: 2rem;
    flex-wrap: wrap; max-width: 900px; margin: 0 auto;
}
.lp-signal-chip {
    display: flex; align-items: center; gap: 0.65rem;
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 99px; padding: 0.7rem 1.4rem;
    transition: all 0.25s ease;
}
.lp-signal-chip:hover {
    border-color: rgba(34,211,238,0.2); transform: scale(1.03);
}
.lp-signal-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.lp-signal-chip span {
    font-family: var(--font-body); font-size: 0.82rem; font-weight: 500;
    color: var(--text-primary);
}
.lp-signal-chip small {
    font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-muted);
}

/* ── Footer ── */
.lp-footer {
    padding: 3rem 4rem; border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
}
.lp-footer-left {
    font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-muted);
}
.lp-footer-right {
    font-family: var(--font-body); font-size: 0.72rem; color: var(--text-muted);
}
.lp-footer-right a { color: var(--accent-cyan); text-decoration: none; }

@media (max-width: 768px) {
    .lp-nav { padding: 1rem 1.5rem; }
    .lp-hero { padding: 4rem 1.5rem 3rem; }
    .lp-bento { grid-template-columns: 1fr; }
    .lp-bento-wide { grid-column: span 1; }
    .lp-bento-tall { grid-row: span 1; }
    .lp-features { padding: 3rem 1.5rem; }
    .lp-stats { gap: 2rem; flex-wrap: wrap; }
    .lp-hero-cta { flex-direction: column; align-items: center; }
}
</style>
""", unsafe_allow_html=True)

# ── Background + Page Content ────────────────────────────────────────────────
st.markdown('<div class="lp-bg"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="lp-page">

<nav class="lp-nav">
    <div class="lp-nav-brand">
        <div class="lp-nav-brand-icon">🧬</div>
        <div class="lp-nav-brand-text">LabPulse AI <span>v2.0</span></div>
    </div>
</nav>

<section class="lp-hero">
    <div class="lp-hero-eyebrow">Predictive Lab Intelligence</div>
    <h1>Reagenzbedarf vorhersagen,<br><em>bevor die Welle kommt.</em></h1>
    <p class="lp-hero-sub">
        LabPulse AI verbindet Abwassersurveillance mit Laborvolumen-Daten
        und prognostiziert den Testbedarf 14&nbsp;Tage im Voraus.
        Entwickelt f&uuml;r Ganzimmun Diagnostics.
    </p>
    <div class="lp-hero-cta">
        <a class="lp-btn-primary" href="/Dashboard" target="_self">Dashboard &ouml;ffnen →</a>
        <a class="lp-btn-secondary" href="#features">Mehr erfahren</a>
    </div>
</section>

<section class="lp-stats">
    <div class="lp-stat">
        <div class="lp-stat-value">386<span>k</span></div>
        <div class="lp-stat-label">RKI Datenpunkte</div>
    </div>
    <div class="lp-stat">
        <div class="lp-stat-value">14<span>d</span></div>
        <div class="lp-stat-label">Vorlaufzeit</div>
    </div>
    <div class="lp-stat">
        <div class="lp-stat-value">4<span>&times;</span></div>
        <div class="lp-stat-label">Signal-Fusion</div>
    </div>
    <div class="lp-stat">
        <div class="lp-stat-value">5<span>+</span></div>
        <div class="lp-stat-label">Pathogene</div>
    </div>
</section>

<section class="lp-features" id="features">
    <div class="lp-features-title">
        <h2>Alles in einem Dashboard.</h2>
        <p>Vier Datenquellen. Ein Konfidenz-Score. Klare Handlungsempfehlungen.</p>
    </div>
    <div class="lp-bento">
        <div class="lp-bento-card lp-bento-wide">
            <div class="lp-bento-icon cyan">📊</div>
            <h3>Korrelations-Analyse</h3>
            <p>Viruslast im Abwasser korreliert mit Laborvolumen &mdash; 14&nbsp;Tage vorher.
            LabPulse zeigt beide Kurven &uuml;berlagert mit ML-Prognose, Konfidenzband
            und Zoom-Navigation f&uuml;r pr&auml;zise Analyse.</p>
        </div>
        <div class="lp-bento-card lp-bento-tall">
            <div>
                <div class="lp-bento-icon amber">🎯</div>
                <h3>Signal-Konfidenz</h3>
                <p>Vier unabh&auml;ngige Quellen gewichtet zu einem Score.
                Je h&ouml;her die &Uuml;bereinstimmung, desto sicherer die Prognose.</p>
            </div>
            <div class="lp-mono-highlight">69.2%</div>
        </div>
        <div class="lp-bento-card">
            <div class="lp-bento-icon green">📦</div>
            <h3>Bestandsverwaltung</h3>
            <p>Echtzeit-Reichweite pro Testkit. Automatische Nachbestell-Empfehlung
            bei drohendem Engpass.</p>
        </div>
        <div class="lp-bento-card">
            <div class="lp-bento-icon purple">🤖</div>
            <h3>ML-Prognose</h3>
            <p>Random Forest &amp; Gradient Boosting trainiert auf historischen Patterns.
            Konfidenzband zeigt Unsicherheit transparent.</p>
        </div>
        <div class="lp-bento-card">
            <div class="lp-bento-icon red">🗺️</div>
            <h3>Regionale Analyse</h3>
            <p>16 Bundesl&auml;nder im Vergleich. Hotspots erkennen, bevor sie
            das Laborvolumen treiben.</p>
        </div>
    </div>
</section>

<section class="lp-signals" id="signals">
    <h2>Vier Signale. Eine Wahrheit.</h2>
    <p>Jede Datenquelle hat St&auml;rken und Schw&auml;chen. Zusammen ergibt sich ein klares Bild.</p>
    <div class="lp-signal-row">
        <div class="lp-signal-chip">
            <div class="lp-signal-dot" style="background:var(--accent-cyan);"></div>
            <span>Abwasser (AMELAG)</span><small>45%</small>
        </div>
        <div class="lp-signal-chip">
            <div class="lp-signal-dot" style="background:var(--accent-purple);"></div>
            <span>GrippeWeb</span><small>25%</small>
        </div>
        <div class="lp-signal-chip">
            <div class="lp-signal-dot" style="background:var(--accent-green);"></div>
            <span>ARE-Konsultation</span><small>20%</small>
        </div>
        <div class="lp-signal-chip">
            <div class="lp-signal-dot" style="background:var(--accent-amber);"></div>
            <span>Google Trends</span><small>10%</small>
        </div>
    </div>
</section>

<footer class="lp-footer">
    <div class="lp-footer-left">LabPulse AI v2.0 &middot; Built for Ganzimmun Diagnostics (Limbach Group)</div>
    <div class="lp-footer-right">
        Daten: <a href="https://github.com/robert-koch-institut/Abwassersurveillance_AMELAG" target="_blank">RKI AMELAG</a>
    </div>
</footer>

</div>
""", unsafe_allow_html=True)
