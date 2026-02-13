# Streamlit CSS Override Recipes

Streamlit's internal class names change between versions. Use `data-testid`
selectors when possible — they're more stable.

## Global App Background
```css
.stApp {
  background:
    radial-gradient(ellipse at 15% 50%, rgba(34, 211, 238, 0.06) 0%, transparent 50%),
    linear-gradient(180deg, #0c1222 0%, #111827 100%);
}
```

## Sidebar
```css
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #162032 0%, #0c1222 100%);
  border-right: 1px solid rgba(255, 255, 255, 0.04);
}
[data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary); }
```

## Metrics — Override the default grey boxes
```css
[data-testid="stMetric"] {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.2rem;
}
[data-testid="stMetric"] label {
  font-family: 'Sora', sans-serif;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-secondary);
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.6rem;
  font-weight: 700;
}
```

## Tabs
```css
.stTabs [data-baseweb="tab-list"] {
  gap: 0;
  border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.85rem;
  font-weight: 500;
  padding: 0.8rem 1.2rem;
  color: var(--text-secondary);
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
  color: var(--accent);
  border-bottom-color: var(--accent);
}
```

## Buttons
```css
.stButton > button {
  font-family: 'DM Sans', sans-serif;
  font-weight: 600;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--bg-surface);
  color: var(--text-primary);
  transition: all 0.2s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  border-color: var(--accent);
}
```

## Expanders
```css
.streamlit-expanderHeader {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.85rem;
  color: var(--text-secondary);
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 8px;
}
```

## Custom KPI Card (HTML injection)
```python
def kpi_card(label, value, delta=None, icon="", color="var(--accent)"):
    delta_html = ""
    if delta is not None:
        d_color = "var(--success)" if delta >= 0 else "var(--danger)"
        d_arrow = "↑" if delta >= 0 else "↓"
        delta_html = f'<div style="font-size:0.8rem;color:{d_color};margin-top:4px;">{d_arrow} {delta:+.1f}%</div>'
    return f'''
    <div style="
      background: linear-gradient(135deg, var(--bg-surface), rgba(255,255,255,0.02));
      border: 1px solid var(--border);
      border-left: 3px solid {color};
      border-radius: 10px;
      padding: 1.2rem 1.4rem;
      transition: transform 0.2s ease;
    ">
      <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;
                  color:var(--text-secondary);margin-bottom:6px;">
        {icon} {label}
      </div>
      <div style="font-size:1.8rem;font-weight:700;font-family:'JetBrains Mono',monospace;
                  color:var(--text-primary);">
        {value}
      </div>
      {delta_html}
    </div>'''
```

## Hide Streamlit Branding (optional)
```css
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
```
