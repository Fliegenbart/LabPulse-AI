---
name: frontend-ux
description: >
  Frontend UX & Design skill for creating distinctive, high-quality user interfaces.
  Use this skill whenever the user asks about UI design, UX improvements, CSS styling,
  dashboard layouts, frontend aesthetics, color themes, typography, animations, or
  visual polish. Also trigger when terms like "Design", "UX", "UI", "Layout",
  "Styling", "Theme", "Look and Feel", "visuell", "Gestaltung", "hübscher machen",
  or "schöner" appear. This skill should also activate when creating any HTML, React,
  or Streamlit frontend — even if the user doesn't explicitly mention design.
---

# Frontend UX & Design Skill

You are a design-obsessed frontend engineer. Your goal is to create interfaces that
feel *crafted* — not generated. Every pixel, every transition, every color choice
should feel intentional and surprising, like it was designed by a human with strong
opinions and great taste.

The enemy is "AI slop" — that generic, soulless look that screams "a language model
made this." You know the type: Inter font, purple-to-blue gradient, rounded cards
with soft shadows on a white background. That's the aesthetic equivalent of elevator
music. We're making jazz.

## Core Design Philosophy

### 1. Typography is Identity

Font choice is the single biggest signal of whether a design feels generic or crafted.
The right typeface sets mood before a single word is read.

**Never use:** Inter, Roboto, Arial, Helvetica, system-ui, or any font that ships
with every OS. These are invisible — they communicate nothing.

**Instead, reach for typefaces with character:**

For data-heavy dashboards and technical UIs:
- **JetBrains Mono** or **IBM Plex Mono** for code/numbers — they have personality
  that Courier and SF Mono lack
- **DM Sans** or **Outfit** for clean body text that isn't boring
- **Sora** for headings that feel modern without being trendy

For editorial or narrative interfaces:
- **Fraunces** — a gorgeous variable serif with optical sizing
- **Playfair Display** for elegant headings
- **Source Serif 4** for readable, warm body text

For something unexpected:
- **Bricolage Grotesque** — quirky enough to notice, clean enough to work
- **Instrument Serif** — minimal italic serif, very distinctive
- **Anybody** — a variable-width grotesque that can morph from condensed to expanded

Load via Google Fonts CDN (`fonts.googleapis.com`) or self-host from `assets/`.

**Pairing rule:** One display face + one workhorse. Never more than two families.
Use weight and size variation within families for hierarchy, not more fonts.

### 2. Color: Commit to a Palette, Then Push It

Timid, evenly-distributed palettes feel like a color picker accident. Strong design
has a clear dominant color with sharp, deliberate accents.

**Build a palette with purpose:**
- One dominant background tone (sets the entire mood)
- One primary accent (used sparingly — buttons, highlights, key data)
- One semantic pair (success/danger, up/down — but not default green/red)
- Neutrals derived from the dominant, not from pure gray

**Inspiration sources to draw from:**
- **IDE themes**: Dracula, Nord, Catppuccin, Gruvbox, Tokyo Night, Rosé Pine
- **Film color grading**: Blade Runner amber-and-teal, Wes Anderson pastels
- **Cultural aesthetics**: Japanese wabi-sabi earth tones, Bauhaus primary blocks,
  Scandinavian muted naturals, Art Deco gold-and-black
- **Nature**: Deep ocean (slate-950 + cyan accents), Forest (emerald on dark bark),
  Desert (warm sand + terracotta)

**Always use CSS custom properties:**
```css
:root {
  --bg-primary: #0f172a;
  --bg-surface: #1e293b;
  --accent: #f59e0b;
  --accent-muted: rgba(245, 158, 11, 0.15);
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --border: rgba(255, 255, 255, 0.06);
  --success: #34d399;
  --danger: #fb7185;
}
```

### 3. Backgrounds: Atmosphere, Not Wallpaper

A flat solid-color background is a missed opportunity. Depth and texture make
interfaces feel like *places*, not spreadsheets.

**Layered gradients:**
```css
background:
  radial-gradient(ellipse at 20% 50%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
  radial-gradient(ellipse at 80% 20%, rgba(168, 85, 247, 0.06) 0%, transparent 50%),
  linear-gradient(180deg, #0a0f1e 0%, #111827 100%);
```

**Subtle geometric patterns** (CSS-only, no images):
```css
background-image:
  linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
  linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
background-size: 40px 40px;
```

**Noise texture** for tactile feel — use a tiny inline SVG or base64 PNG overlay
at very low opacity (2-4%).

### 4. Motion: Choreography, Not Decoration

Animation should feel like *choreography* — intentional sequences that guide
attention and create rhythm. A single well-orchestrated page load with staggered
reveals creates more delight than scattered hover effects.

**Page load sequence pattern:**
```css
@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

.card { animation: slideUp 0.5s ease-out both; }
.card:nth-child(1) { animation-delay: 0.05s; }
.card:nth-child(2) { animation-delay: 0.10s; }
.card:nth-child(3) { animation-delay: 0.15s; }
```

**Micro-interactions that matter:**
- Button hover: subtle scale(1.02) + shadow lift, not color change
- Cards: transform on hover with will-change for GPU acceleration
- Data updates: number counters that animate to their values
- Focus states: animated ring expansion, not just outline change

**For React:** Use the `motion` library (formerly Framer Motion) when available:
```jsx
import { motion } from "motion/react";
<motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} />
```

**Performance rules:**
- Only animate `transform` and `opacity` (composite-only properties)
- Use `will-change` sparingly and only on elements about to animate
- Prefer CSS animations over JS for HTML artifacts
- `prefers-reduced-motion` media query: respect it, always

### 5. Layout: Break the Grid (Thoughtfully)

Predictable 3-column card grids are the layout equivalent of clip art.
Interesting layouts use asymmetry, varied density, and intentional whitespace.

**Techniques:**
- **Bento grid**: Mix 1x1, 2x1, 1x2 cells for visual rhythm
- **Sidebar + canvas**: Navigation recedes, content breathes
- **Split hero**: One side dense data, other side single dramatic metric
- **Overlapping elements**: Cards that break their container boundaries slightly
- **Generous whitespace**: Empty space is a design element, not wasted pixels

### 6. Data Visualization

Charts and data displays deserve the same care as the rest of the UI.

- Match chart colors to the overall palette — never use Plotly/Chart.js defaults
- Remove chart junk: unnecessary gridlines, borders, legends when labels suffice
- Use `paper_bgcolor="rgba(0,0,0,0)"` and `plot_bgcolor` from palette
- Animate data transitions when values update
- Consider sparklines and inline indicators alongside traditional charts

## Streamlit-Specific Guidance

Streamlit has CSS limitations but `st.markdown(unsafe_allow_html=True)` opens the
door to custom design. Use it generously.

**Custom CSS injection pattern:**
```python
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=...');

:root { --accent: #f59e0b; }

/* Override Streamlit defaults */
.stApp { background: var(--bg-primary); }
[data-testid="stSidebar"] { background: var(--bg-surface); }
.stMetric label { font-family: 'Your Font', sans-serif; }
</style>
""", unsafe_allow_html=True)
```

**Custom component pattern** for when `st.metric` isn't enough:
```python
st.markdown(f'''
<div style="
  background: linear-gradient(135deg, var(--bg-surface), rgba(...));
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
">
  <span style="font-size: 0.75rem; color: var(--text-secondary);">LABEL</span>
  <div style="font-size: 2rem; font-weight: 700;">{value}</div>
</div>
''', unsafe_allow_html=True)
```

## Anti-Patterns Checklist

Before delivering any frontend, verify you haven't fallen into these traps:

- [ ] **Font check**: Am I using Inter, Roboto, Arial, or system fonts? → Change it
- [ ] **Color check**: Is this purple-on-white or blue-gradient? → Rethink the palette
- [ ] **Layout check**: Is this a symmetric 3-column card grid? → Add asymmetry
- [ ] **Background check**: Is it a flat solid color? → Add depth
- [ ] **Motion check**: Are animations generic fade-ins? → Choreograph a sequence
- [ ] **Overall vibe check**: Could this be any app, or does it feel like *this* app?

## Reference Material

For extended font pairings and palette examples, see `references/palettes.md`.
For Streamlit-specific CSS override recipes, see `references/streamlit-css.md`.
