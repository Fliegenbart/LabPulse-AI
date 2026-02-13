"""
LabPulse AI — Landing Page
===========================
Precision Noir aesthetic: Bricolage Grotesque + Darker Grotesque + Azeret Mono.
Copper/rose-gold accent on near-black. Diagonal line patterns. Cinematic.
"""

import streamlit as st

st.set_page_config(
    page_title="LabPulse AI — Predictive Lab Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,200..800&family=Darker+Grotesque:wght@300;400;500;600;700;900&family=Azeret+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-void: #1e1e2e;
    --bg-deep: #252538;
    --bg-surface: #2f2f45;
    --bg-card: #3a3a54;
    --accent: #d4956a;
    --accent-bright: #e8b08a;
    --accent-dim: rgba(212,149,106,0.10);
    --accent-glow: rgba(212,149,106,0.18);
    --signal-teal: #5eead4;
    --signal-violet: #a78bfa;
    --signal-rose: #fb7185;
    --signal-lime: #a3e635;
    --text-bright: #f5f4f8;
    --text-primary: #ebebf0;
    --text-secondary: #9090a8;
    --text-muted: #6b6b88;
    --border: rgba(176,176,196,0.12);
    --border-hover: rgba(212,149,106,0.25);
    --font-display: 'Darker Grotesque', sans-serif;
    --font-body: 'Bricolage Grotesque', sans-serif;
    --font-mono: 'Azeret Mono', monospace;
}

#MainMenu, footer, header[data-testid="stHeader"] { display:none !important; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display:none !important; }

.block-container { padding:0 !important; max-width:100% !important; }
.stApp { background: var(--bg-void); overflow-x:hidden; }

/* ── Background: diagonal hatching + warm glow ── */
.lp-bg {
    position:fixed; inset:0; z-index:0;
    background:
        radial-gradient(ellipse 70% 50% at 50% 0%, rgba(212,149,106,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 40% 60% at 90% 85%, rgba(167,139,250,0.03) 0%, transparent 50%),
        var(--bg-void);
    pointer-events:none;
}
.lp-bg::before {
    content:''; position:absolute; inset:0;
    background-image: repeating-linear-gradient(
        -45deg,
        transparent,
        transparent 60px,
        rgba(212,149,106,0.015) 60px,
        rgba(212,149,106,0.015) 61px
    );
}
.lp-bg::after {
    content:''; position:absolute;
    width:800px; height:800px;
    top:-300px; left:50%; transform:translateX(-50%);
    border-radius:50%;
    background:radial-gradient(circle, rgba(212,149,106,0.04) 0%, transparent 65%);
    animation:drift 16s ease-in-out infinite;
}
@keyframes drift {
    0%,100% { transform:translateX(-50%) translateY(0) scale(1); }
    50% { transform:translateX(-45%) translateY(20px) scale(1.05); }
}

.lp-page { position:relative; z-index:1; min-height:100vh; display:flex; flex-direction:column; }

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity:0; transform:translateY(32px); }
    to { opacity:1; transform:translateY(0); }
}
@keyframes fadeDown {
    from { opacity:0; transform:translateY(-12px); }
    to { opacity:1; transform:translateY(0); }
}
@keyframes lineExpand {
    from { width:0; }
    to { width:80px; }
}

/* ── Nav ── */
.lp-nav {
    display:flex; justify-content:space-between; align-items:center;
    padding:1.8rem 5rem;
    animation:fadeDown 0.7s ease-out both;
}
.lp-nav-brand { display:flex; align-items:center; gap:0.8rem; }
.lp-nav-mark {
    width:32px; height:32px; border-radius:8px;
    background:linear-gradient(135deg, var(--accent), #b87a4a);
    display:flex; align-items:center; justify-content:center;
    font-size:0.95rem;
    box-shadow:0 0 20px rgba(212,149,106,0.2);
}
.lp-nav-name {
    font-family:var(--font-body); font-weight:700; font-size:0.95rem;
    color:var(--text-bright); letter-spacing:-0.02em;
}
.lp-nav-ver {
    font-family:var(--font-mono); font-size:0.6rem; font-weight:500;
    color:var(--text-muted); margin-left:0.4rem;
    padding:0.15rem 0.5rem; border-radius:4px;
    border:1px solid var(--border);
}

/* ── Hero ── */
.lp-hero {
    flex:1; display:flex; flex-direction:column;
    align-items:center; justify-content:center; text-align:center;
    padding:8rem 2rem 5rem; min-height:75vh;
}
.lp-eyebrow {
    font-family:var(--font-mono); font-size:0.65rem; font-weight:600;
    color:var(--accent); letter-spacing:0.3em; text-transform:uppercase;
    margin-bottom:2.5rem;
    animation:fadeUp 0.6s ease-out 0.15s both;
}
.lp-hero h1 {
    font-family:var(--font-display);
    font-size:clamp(3.5rem, 8vw, 7rem); font-weight:300;
    color:var(--text-bright); line-height:0.95;
    margin:0 auto; max-width:1000px; letter-spacing:-0.04em;
    animation:fadeUp 0.8s ease-out 0.3s both;
}
.lp-hero h1 strong {
    font-weight:900;
    background:linear-gradient(135deg, var(--accent-bright), var(--accent), #c47a4a);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
}
.lp-hero-line {
    width:80px; height:1px; background:var(--accent);
    margin:2.5rem auto; opacity:0.4;
    animation:lineExpand 0.8s ease-out 0.5s both;
}
.lp-hero-sub {
    font-family:var(--font-body);
    font-size:clamp(1rem, 1.6vw, 1.15rem); font-weight:400;
    color:var(--text-secondary); max-width:560px;
    margin:0 auto; line-height:1.75;
    animation:fadeUp 0.7s ease-out 0.6s both;
}
.lp-cta-row {
    display:flex; gap:1rem; margin-top:3.5rem;
    animation:fadeUp 0.7s ease-out 0.75s both;
}
.lp-btn {
    font-family:var(--font-body); font-size:0.85rem; font-weight:600;
    padding:0.9rem 2.4rem; border-radius:10px;
    text-decoration:none; transition:all 0.3s cubic-bezier(0.4,0,0.2,1);
    letter-spacing:-0.01em;
}
.lp-btn-fill {
    color:var(--bg-void);
    background:linear-gradient(135deg, var(--accent-bright), var(--accent));
    box-shadow:0 0 28px rgba(212,149,106,0.2), 0 4px 12px rgba(0,0,0,0.4);
}
.lp-btn-fill:hover {
    transform:translateY(-2px) scale(1.02);
    box-shadow:0 0 40px rgba(212,149,106,0.3), 0 8px 24px rgba(0,0,0,0.5);
    color:var(--bg-void);
}
.lp-btn-ghost {
    color:var(--text-secondary);
    border:1px solid rgba(212,149,106,0.12); background:transparent;
}
.lp-btn-ghost:hover {
    color:var(--text-bright);
    border-color:var(--border-hover);
    background:rgba(212,149,106,0.04);
}

/* ── Stats ── */
.lp-stats {
    display:flex; justify-content:center; gap:5rem; padding:2rem 2rem 4rem;
    animation:fadeUp 0.7s ease-out 0.9s both;
}
.lp-stat { text-align:center; }
.lp-stat-num {
    font-family:var(--font-display); font-size:2.8rem; font-weight:900;
    color:var(--text-bright); line-height:1;
}
.lp-stat-num span { color:var(--accent); }
.lp-stat-lbl {
    font-family:var(--font-mono); font-size:0.6rem; font-weight:500;
    color:var(--text-muted); text-transform:uppercase;
    letter-spacing:0.18em; margin-top:0.6rem;
}

/* ── Feature Bento ── */
.lp-sect { padding:6rem 5rem; max-width:1200px; margin:0 auto; }
.lp-sect-head { text-align:center; margin-bottom:4.5rem; }
.lp-sect-head h2 {
    font-family:var(--font-display); font-size:clamp(2rem,4vw,3.2rem);
    font-weight:300; color:var(--text-bright); margin:0;
    letter-spacing:-0.03em;
}
.lp-sect-head h2 strong { font-weight:900; color:var(--accent-bright); }
.lp-sect-head p {
    font-family:var(--font-body); color:var(--text-secondary);
    font-size:0.95rem; margin-top:1.2rem; max-width:500px;
    margin-left:auto; margin-right:auto;
}

.lp-grid {
    display:grid; grid-template-columns:repeat(12,1fr);
    gap:1rem;
}
.lp-card {
    background:linear-gradient(165deg, rgba(58,58,84,0.5), rgba(47,47,69,0.3));
    backdrop-filter:blur(16px) saturate(1.3);
    -webkit-backdrop-filter:blur(16px) saturate(1.3);
    border:1px solid var(--border); border-radius:14px;
    padding:2.2rem; position:relative; overflow:hidden;
    transition:all 0.4s cubic-bezier(0.4,0,0.2,1);
    box-shadow:0 4px 24px rgba(0,0,0,0.12), inset 0 1px 0 rgba(255,255,255,0.03);
}
.lp-card::after {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg, transparent 10%, var(--accent) 50%, transparent 90%);
    opacity:0; transition:opacity 0.4s ease;
}
.lp-card:hover {
    transform:translateY(-6px);
    border-color:var(--border-hover);
    box-shadow:0 20px 50px rgba(0,0,0,0.4), 0 0 80px rgba(212,149,106,0.03);
}
.lp-card:hover::after { opacity:0.5; }

.lp-c-7 { grid-column:span 7; }
.lp-c-5 { grid-column:span 5; }
.lp-c-4 { grid-column:span 4; }
.lp-c-8 { grid-column:span 8; }
.lp-c-6 { grid-column:span 6; }

.lp-card-tag {
    font-family:var(--font-mono); font-size:0.58rem; font-weight:600;
    text-transform:uppercase; letter-spacing:0.2em;
    padding:0.2rem 0.6rem; border-radius:4px;
    display:inline-block; margin-bottom:1.2rem;
}
.lp-tag-teal { background:rgba(94,234,212,0.08); color:var(--signal-teal); }
.lp-tag-copper { background:rgba(212,149,106,0.08); color:var(--accent); }
.lp-tag-violet { background:rgba(167,139,250,0.08); color:var(--signal-violet); }
.lp-tag-rose { background:rgba(251,113,133,0.08); color:var(--signal-rose); }
.lp-tag-lime { background:rgba(163,230,53,0.08); color:var(--signal-lime); }

.lp-card h3 {
    font-family:var(--font-body); font-size:1.15rem; font-weight:700;
    color:var(--text-bright); margin:0 0 0.8rem; letter-spacing:-0.02em;
}
.lp-card p {
    font-family:var(--font-body); font-size:0.82rem;
    color:var(--text-secondary); line-height:1.7; margin:0;
}
.lp-card-big-num {
    font-family:var(--font-display); font-size:3.5rem; font-weight:900;
    color:var(--accent); line-height:1; margin-top:auto; padding-top:2rem;
}

/* ── Signal pills ── */
.lp-pills { padding:5rem 5rem 6rem; text-align:center; }
.lp-pills h2 {
    font-family:var(--font-display); font-size:2.4rem; font-weight:300;
    color:var(--text-bright); margin-bottom:0.8rem; letter-spacing:-0.02em;
}
.lp-pills h2 strong { font-weight:900; }
.lp-pills > p {
    font-family:var(--font-body); color:var(--text-secondary);
    max-width:480px; margin:0 auto 3rem; font-size:0.88rem;
}
.lp-pill-row {
    display:flex; justify-content:center; gap:0.8rem;
    flex-wrap:wrap; max-width:800px; margin:0 auto;
}
.lp-pill {
    display:flex; align-items:center; gap:0.6rem;
    background:rgba(47,47,69,0.4); backdrop-filter:blur(12px);
    border:1px solid var(--border);
    border-radius:99px; padding:0.6rem 1.3rem;
    transition:all 0.25s ease;
}
.lp-pill:hover {
    border-color:var(--border-hover); transform:translateY(-2px);
    box-shadow:0 6px 20px rgba(0,0,0,0.3);
}
.lp-pill-dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
.lp-pill-name {
    font-family:var(--font-body); font-size:0.78rem; font-weight:600;
    color:var(--text-bright);
}
.lp-pill-wt {
    font-family:var(--font-mono); font-size:0.58rem; font-weight:600;
    color:var(--text-muted);
}

/* ── Footer ── */
.lp-foot {
    padding:2.5rem 5rem; border-top:1px solid var(--border);
    display:flex; justify-content:space-between; align-items:center;
}
.lp-foot span {
    font-family:var(--font-mono); font-size:0.6rem; color:var(--text-muted);
}
.lp-foot a { color:var(--accent); text-decoration:none; }

@media (max-width:768px) {
    .lp-nav { padding:1.2rem 1.5rem; }
    .lp-hero { padding:5rem 1.5rem 3rem; }
    .lp-grid { grid-template-columns:1fr; }
    .lp-c-7,.lp-c-5,.lp-c-4,.lp-c-8,.lp-c-6 { grid-column:span 1; }
    .lp-sect { padding:3rem 1.5rem; }
    .lp-stats { gap:2rem; flex-wrap:wrap; }
    .lp-cta-row { flex-direction:column; align-items:center; }
    .lp-pills { padding:3rem 1.5rem; }
}
</style>
""", unsafe_allow_html=True)

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
        <a class="lp-btn lp-btn-fill" href="/Dashboard" target="_self">Dashboard &ouml;ffnen →</a>
        <a class="lp-btn lp-btn-ghost" href="#features">Mehr erfahren</a>
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
            <p>Random Forest &amp; Gradient Boosting trainiert auf historischen Patterns.
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
""", unsafe_allow_html=True)
