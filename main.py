"""
LabPulse AI — Enterprise OS (NiceGUI)
=====================================
Features:
- Prophet-ML Forecasting
- Google Trends Signal
- Ollama Decision Loop
- Company Inventory Context
- PDF Decision Report
- Apple-like premium UI with glassmorphism style
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from nicegui import app, ui

from core.decision_ai import DecisionNarrator
from core.forecasting import ForecastOutcome, run_forecast_pipeline
from core.reporting import generate_decision_report
from core.trends import TrendContext
from data_engine import (
    fetch_rki_raw_dataset,
    fetch_rki_wastewater,
    generate_lab_volume,
    get_available_pathogens,
    get_signal_source_id_by_label,
    get_signal_source_label,
    get_signal_source_options,
)


def _fmt_eur(value: float) -> str:
    return f"{float(value):,.0f} €".replace(",", ".")


def _fmt_int(value: float | int | None, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "0"
    return f"{float(value):,.{decimals}f}".replace(",", ".")


def _inject_styles() -> None:
    css_path = Path(__file__).resolve().parent / "assets" / "css" / "landing.css"
    with css_path.open("r", encoding="utf-8") as css_file:
        ui.add_css(css_file.read(), shared=True)


@dataclass
class AppState:
    company_name: str = "LabPulse Enterprise"
    facility_name: str = "Hauptstandort"
    signal_source: str = "rki_amelag_einzelstandorte"
    pathogen: str = "SARS-CoV-2"
    forecast_horizon: int = 14
    virus_uplift_pct: int = 0
    safety_buffer_pct: float = 0.10
    stock_level: int = 7000
    tests_per_fte: int = 90
    use_prophet: bool = True

    pathogen_options: List[str] = field(default_factory=list)
    raw_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    wastewater_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    lab_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    trend_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    kpis: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)

    loading: bool = False
    last_error: Optional[str] = None
    refreshed_at: str = ""
    ai_summary: str = ""
    ai_backend: str = ""

    trend_ctx: TrendContext = field(default_factory=lambda: TrendContext(ttl_minutes=180))

    def refresh_signal_options(self) -> None:
        self.pathogen_options = get_available_pathogens(self.raw_df, dataset_id=self.signal_source)
        if not self.pathogen_options:
            self.pathogen_options = ["SARS-CoV-2"]
            self.pathogen = "SARS-CoV-2"
        if self.pathogen not in self.pathogen_options:
            self.pathogen = self.pathogen_options[0]

    def rebuild(self) -> None:
        self.loading = True
        self.last_error = None
        try:
            self.raw_df = fetch_rki_raw_dataset(self.signal_source)
            self.refresh_signal_options()

            self.wastewater_df = fetch_rki_wastewater(
                raw_df=self.raw_df,
                pathogen=self.pathogen,
                dataset_id=self.signal_source,
            )
            self.lab_df = generate_lab_volume(self.wastewater_df, pathogen=self.pathogen)
            self.trend_df = self.trend_ctx.get(self.pathogen)

            outcome: ForecastOutcome = run_forecast_pipeline(
                history=self.lab_df,
                horizon_days=self.forecast_horizon,
                pathogen=self.pathogen,
                use_prophet=self.use_prophet,
                uplift_pct=self.virus_uplift_pct,
                stock_on_hand=self.stock_level,
                safety_buffer_pct=self.safety_buffer_pct,
                trend_df=self.trend_df,
                wastewater_df=self.wastewater_df,
            )

            self.forecast_df = outcome.df
            self.kpis = outcome.kpis
            self.model_info = outcome.model_info
            self.refreshed_at = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            self.ai_summary = ""
            self.ai_backend = ""
        except Exception as exc:
            self.last_error = str(exc)
            self.forecast_df = pd.DataFrame()
            self.kpis = {
                "predicted_tests_7d": 0,
                "revenue_forecast_7d": 0,
                "risk_eur": 0.0,
                "predicted_peak": 0,
                "trend_pct": 0.0,
                "stock_on_hand": self.stock_level,
                "pathogen": self.pathogen,
                "model": "simple",
            }
            self.model_info = {"model_version": "simple", "success": False}
            self.ai_summary = ""
            self.ai_backend = ""
        finally:
            self.loading = False

    def build_decision(self) -> None:
        if not self.kpis:
            self.ai_summary = "Bitte zuerst Berechnung ausführen."
            self.ai_backend = "N/A"
            return

        narrator = DecisionNarrator()
        context = {
            "company": self.company_name,
            "facility": self.facility_name,
            "pathogen": self.pathogen,
            "horizon_days": self.forecast_horizon,
            "uplift_pct": self.virus_uplift_pct,
            "safety_buffer_pct": self.safety_buffer_pct,
            "model": self.model_info.get("model_version", "simple"),
            **self.kpis,
        }
        summary = narrator.summarize(context)
        if summary:
            self.ai_summary = summary
            self.ai_backend = "Ollama"
        else:
            self.ai_summary = DecisionNarrator.fallback_summary(context)
            self.ai_backend = "Fallback"

    def build_pdf(self) -> bytes:
        return generate_decision_report(
            company=f"{self.company_name} · {self.facility_name}",
            pathogen=self.pathogen,
            kpis={**self.kpis, "model": self.model_info.get("model_version", "simple")},
            forecast_df=self.forecast_df,
            recommendation=(self.ai_summary or DecisionNarrator.fallback_summary(self.kpis)),
            ai_provider=self.ai_backend or "Nicht gestartet",
        )


state = AppState()


def _ensure_data() -> None:
    if state.loading:
        return
    if not state.pathogen_options:
        state.rebuild()


def _kpi_card(title: str, value: str, sub: str, color_class: str = "text-cyan-100") -> None:
    with ui.card().classes("lp-card lp-kpi"):
        ui.label(title).classes("lp-kpi-label")
        ui.label(value).classes(f"lp-kpi-value {color_class}")
        ui.label(sub).classes("lp-kpi-sub")


def _build_chart() -> go.Figure:
    fig = go.Figure()

    if not state.lab_df.empty and {"date", "order_volume"}.issubset(state.lab_df.columns):
        history = state.lab_df.tail(90)
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["order_volume"],
                name="Ist-Nachfrage",
                line={"color": "#67e8f9", "width": 2},
            )
        )

    if not state.forecast_df.empty and "Date" in state.forecast_df.columns:
        forecast = state.forecast_df
        fig.add_trace(
            go.Scatter(
                x=forecast["Date"],
                y=forecast["Predicted Volume"],
                name="Prognose",
                mode="lines+markers",
                line={"color": "#22d3ee", "width": 3},
            )
        )
        if "ML Lower" in forecast.columns and "ML Upper" in forecast.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast["Date"],
                    y=forecast["ML Upper"],
                    mode="lines",
                    line={"color": "rgba(56,189,248,0.00)"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast["Date"],
                    y=forecast["ML Lower"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(56,189,248,0.18)",
                    line={"color": "rgba(56,189,248,0.00)"},
                    name="Unsicherheit",
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Bar(
                x=forecast["Date"],
                y=forecast["Reagent Order"],
                name="Bestellbedarf",
                marker={"color": "#fb7185", "opacity": 0.9},
                yaxis="y2",
            )
        )

    if not state.trend_df.empty and {"date", "trend_score"}.issubset(state.trend_df.columns):
        trend = state.trend_df.sort_values("date").tail(45)
        fig.add_trace(
            go.Scatter(
                x=trend["date"],
                y=trend["trend_score"],
                name="Google Trends",
                mode="lines",
                line={"color": "#c4b5fd", "width": 2, "dash": "dot"},
                opacity=0.8,
                yaxis="y3",
            )
        )

    upper_order = 0.0
    if "Reagent Order" in state.forecast_df.columns and not state.forecast_df.empty:
        upper_order = float(pd.to_numeric(state.forecast_df["Reagent Order"], errors="coerce").max() or 0)
    order_range = [0, max(1.0, upper_order * 1.30 + 1)]
    trend_range = [0, float(max(100, state.trend_df["trend_score"].max() if not state.trend_df.empty else 100))]

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=460,
        margin={"l": 12, "r": 10, "t": 20, "b": 30},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.05},
        xaxis={"showgrid": True, "gridcolor": "rgba(148,163,184,0.16)", "title": ""},
        yaxis={
            "title": "Tests",
            "showgrid": True,
            "gridcolor": "rgba(148,163,184,0.14)",
        },
        yaxis2={"overlaying": "y", "side": "right", "title": "Bestellbedarf", "range": order_range, "showgrid": False},
        yaxis3={"overlaying": "y", "side": "right", "anchor": "free", "position": 0.985, "range": trend_range, "showgrid": False, "showticklabels": False},
    )
    return fig


def _build_forecast_rows() -> List[Dict[str, Any]]:
    if state.forecast_df.empty:
        return []

    cols = ["Date", "Predicted Volume", "Reagent Order", "Remaining Stock"]
    revenue_col = [c for c in state.forecast_df.columns if c.startswith("Est. Revenue")]
    if revenue_col:
        cols.append(revenue_col[0])

    preview = state.forecast_df.head(28).copy()
    preview["Date"] = pd.to_datetime(preview["Date"]).dt.strftime("%d.%m.%Y")
    rows = preview[cols].to_dict("records")

    for row in rows:
        for field in ("Predicted Volume", "Reagent Order", "Remaining Stock"):
            if field in row:
                row[field] = _fmt_int(row[field])
        for field in revenue_col:
            if field in row:
                row["Umsatz"] = _fmt_eur(float(row[field]))
                row.pop(field, None)

    return rows


@ui.refreshable
def dashboard_body() -> None:
    if state.loading:
        with ui.card().classes("lp-card q-pa-md"):
            with ui.row().classes("items-center gap-3"):
                ui.spinner("dots", size="2em")
                ui.label("Daten werden geladen und Prognose wird berechnet ...")
        return

    if state.last_error:
        with ui.card().classes("lp-card q-pa-md"):
            ui.label("Warnung: Berechnung teilweise im Fallback.").classes("text-amber-200")
            ui.label(state.last_error).classes("text-xs lp-muted")

    with ui.row().classes("w-full gap-4"):
        _kpi_card(
            "Revenue 7 Tage",
            _fmt_eur(float(state.kpis.get("revenue_forecast_7d", 0))),
            f"{state.kpis.get('trend_pct', 0):+.1f}% WoW",
            "text-cyan-100",
        )
        _kpi_card(
            "Risikopotenzial",
            _fmt_eur(float(state.kpis.get("risk_eur", 0.0))),
            "Unterdeckungsrisiko bei Bestandslücken",
            "text-rose-200" if float(state.kpis.get("risk_eur", 0.0)) > 0 else "text-emerald-200",
        )
        stock_level = int(state.kpis.get("stock_on_hand", state.stock_level))
        avg_daily = float(state.kpis.get("predicted_tests_7d", 0.0)) / 7 if state.kpis.get("predicted_tests_7d") else 0
        coverage = int(stock_level / max(avg_daily, 1))
        _kpi_card(
            "Bestandsabdeckung",
            f"{coverage} Tage",
            f"{stock_level:,} Tests im Bestand".replace(",", "."),
            "text-sky-100",
        )
        _kpi_card(
            "Peak FTE",
            f"{int(state.kpis.get('predicted_peak', 0) / max(1, state.tests_per_fte))}",
            "max. benötigte FTE pro Tag",
            "text-violet-100",
        )

    with ui.row().classes("w-full gap-3 items-center"):
        ui.select(
            options=get_signal_source_options(),
            value=get_signal_source_label(state.signal_source),
            label="Signalquelle",
            on_change=lambda e: _set_signal_source(e.value),
        ).classes("w-[280px]")
        ui.select(
            options=state.pathogen_options,
            value=state.pathogen,
            label="Erreger",
            on_change=lambda e: _set_pathogen(e.value),
        ).classes("w-[220px]")
        ui.checkbox("Prophet aktiv", value=state.use_prophet, on_change=lambda e: _set_prophet(e.value)).classes("ml-1")
        ui.space()
        ui.chip("Engine: " + ("Prophet" if state.use_prophet else "Rule-Based")).classes("lp-chip")
        if state.model_info.get("confidence") is not None:
            ui.chip(f"Confidence {state.model_info['confidence']:.0f}%").classes("lp-chip")

    with ui.card().classes("lp-card q-pa-sm mt-2"):
        ui.label("Forecast, Trend und Bestellimpuls").classes("lp-title")
        ui.plotly(_build_chart()).classes("w-full mt-2")

    tabs = ui.tabs()
    with tabs:
        t1 = ui.tab("Verlauf")
        t2 = ui.tab("Supply")
        t3 = ui.tab("Workforce")

    with ui.tab_panels(tabs, value=t1).classes("w-full"):
        with ui.tab_panel(t1):
            with ui.card().classes("lp-card"):
                ui.table(
                    columns=[
                        {"name": "Date", "label": "Datum", "field": "Date", "align": "left"},
                        {"name": "Predicted Volume", "label": "Forecast", "field": "Predicted Volume", "align": "right"},
                        {"name": "Reagent Order", "label": "Bestellung", "field": "Reagent Order", "align": "right"},
                        {"name": "Remaining Stock", "label": "Restbestand", "field": "Remaining Stock", "align": "right"},
                    ],
                    rows=_build_forecast_rows(),
                    row_key="Date",
                ).classes("w-full")

        with ui.tab_panel(t2):
            with ui.card().classes("lp-card"):
                ui.label("Order-Sicht: Bestellbedarf pro Tag").classes("text-md")
                ui.table(
                    columns=[
                        {"name": "Date", "label": "Datum", "field": "Date", "align": "left"},
                        {"name": "Reagent Order", "label": "Bestellbedarf", "field": "Reagent Order", "align": "right"},
                        {"name": "Remaining Stock", "label": "Restbestand", "field": "Remaining Stock", "align": "right"},
                    ],
                    rows=_build_forecast_rows(),
                    row_key="Date",
                ).classes("w-full")

        with ui.tab_panel(t3):
            with ui.card().classes("lp-card"):
                if state.forecast_df.empty:
                    ui.label("Workforce-Daten nicht verfügbar. Bitte neu berechnen.")
                else:
                    hr = go.Figure()
                    hr.add_bar(
                        x=state.forecast_df["Date"],
                        y=(state.forecast_df["Reagent Order"] / max(1, state.tests_per_fte)).apply(lambda x: int(x) + (1 if float(x) % 1 else 0)),
                        name="FTE",
                        marker={"color": "#38bdf8"},
                    )
                    hr.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=270,
                        margin={"l": 12, "r": 8, "t": 10, "b": 30},
                    )
                    ui.plotly(hr).classes("w-full")

    with ui.row().classes("w-full gap-4 mt-2"):
        with ui.card().classes("lp-card flex-1"):
            with ui.row().classes("items-center justify-between"):
                ui.label("Decision Loop (Mensch + KI)").classes("text-lg lp-title")
                ui.label(state.ai_backend or "bereit").classes("text-xs lp-muted")
            if state.ai_summary:
                ui.markdown(state.ai_summary).classes("lp-muted")
            else:
                ui.label("Noch keine KI-Einschätzung. Bitte Decision Loop starten.").classes("text-xs lp-muted")

        with ui.card().classes("lp-card w-[320px]"):
            ui.label("Google Trends (Kernausschnitt)").classes("text-lg lp-title")
            if state.trend_df.empty:
                ui.label("Noch keine Trendsignale verfügbar.").classes("text-xs lp-muted")
            else:
                rows = [
                    {
                        "Datum": pd.Timestamp(r["date"]).strftime("%d.%m.%Y"),
                        "Score": f"{float(r['trend_score']):.0f}",
                    }
                    for _, r in state.trend_df.sort_values("date").tail(8).iterrows()
                ]
                ui.table(
                    columns=[
                        {"name": "Datum", "label": "Datum", "field": "Datum", "align": "left"},
                        {"name": "Score", "label": "Score", "field": "Score", "align": "right"},
                    ],
                    rows=rows,
                    row_key="Datum",
                ).classes("w-full")


def _rebuild_and_refresh() -> None:
    state.rebuild()
    dashboard_body.refresh()


def _set_signal_source(label: str) -> None:
    state.signal_source = get_signal_source_id_by_label(label)
    _rebuild_and_refresh()


def _set_pathogen(pathogen: str) -> None:
    state.pathogen = pathogen
    _rebuild_and_refresh()


def _set_horizon(v: int) -> None:
    state.forecast_horizon = int(v)
    _rebuild_and_refresh()


def _set_uplift(v: int) -> None:
    state.virus_uplift_pct = int(v)
    _rebuild_and_refresh()


def _set_buffer(v: int) -> None:
    state.safety_buffer_pct = max(0.0, min(0.4, float(v) / 100.0))
    _rebuild_and_refresh()


def _set_stock(v: float) -> None:
    state.stock_level = max(0, int(v))
    _rebuild_and_refresh()


def _set_fte(v: float) -> None:
    state.tests_per_fte = max(1, int(v))
    _rebuild_and_refresh()


def _set_company(v: str) -> None:
    state.company_name = (v or "").strip() or "LabPulse Enterprise"
    _rebuild_and_refresh()


def _set_facility(v: str) -> None:
    state.facility_name = (v or "").strip() or "Hauptstandort"
    _rebuild_and_refresh()


def _set_prophet(v: bool) -> None:
    state.use_prophet = bool(v)
    _rebuild_and_refresh()


def _run_decision() -> None:
    state.build_decision()
    dashboard_body.refresh()


def _download_pdf() -> None:
    report = state.build_pdf()
    ui.download(report, filename=f"labpulse_decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")


@ui.page("/")
def landing_page() -> None:
    _inject_styles()

    with ui.row().classes("lp-wrap lp-title items-center"):
        ui.label("LabPulse AI").classes("text-2xl font-bold")
        ui.space()
        ui.button("Dashboard starten", icon="trending_up", on_click=lambda: ui.open("/dashboard"))

    with ui.row().classes("lp-wrap lp-landing"):
        with ui.card().classes("lp-card lp-hero lp-wrap q-pa-lg"):
            ui.label("Pharma Signal Intelligence").classes("lp-hero-headline")
            ui.label("Decision OS").classes("lp-hero-headline text-2xl")
            ui.markdown(
                "Beschleunigen Sie Material- und Personalplanung mit einem klaren KI-Loop: \n"
                "Datensignal → Forecast → Risikoanalyse → Maßnahmen-Empfehlung."
            ).classes("lp-muted")
            with ui.row().classes("lp-cta"):
                ui.button("Start Dashboard", icon="dashboard", on_click=lambda: ui.open("/dashboard"))
                ui.button(
                    "Überblick PDF",
                    icon="insights",
                    on_click=lambda: ui.notify("PDF im Dashboard verfügbar."),
                ).props("outline")
            ui.label("Pathogen-Matrix").classes("lp-chip mt-4")


@ui.page("/dashboard")
@ui.page("/dashboard/")
def dashboard_page() -> None:
    _inject_styles()
    _ensure_data()

    with ui.left_drawer(value=False, fixed=True, bordered=True).props("width=360 overlay").classes("lp-drawer") as control_drawer:
        ui.label("Steuerzentrum").classes("text-lg lp-title")
        ui.label("Lager-, Signalkontext, Modellsteuerung").classes("lp-muted")
        ui.separator()

        ui.input("Unternehmen", value=state.company_name, on_change=lambda e: _set_company(e.value)).classes("w-full")
        ui.input("Facility", value=state.facility_name, on_change=lambda e: _set_facility(e.value)).classes("w-full mt-2")
        ui.select(
            label="Signalquelle",
            options=get_signal_source_options(),
            value=get_signal_source_label(state.signal_source),
            on_change=lambda e: _set_signal_source(e.value),
        ).classes("w-full mt-2")
        ui.select(
            label="Erreger",
            options=state.pathogen_options,
            value=state.pathogen,
            on_change=lambda e: _set_pathogen(e.value),
        ).classes("w-full mt-2")
        ui.label("Horizon (Tage)").classes("lp-muted")
        ui.slider(
            min=7,
            max=28,
            step=7,
            value=state.forecast_horizon,
            on_change=lambda e: _set_horizon(e.value),
        ).classes("w-full mt-2")
        ui.label("Anstiegsszenario (%)").classes("lp-muted")
        ui.slider(
            min=0,
            max=50,
            step=5,
            value=state.virus_uplift_pct,
            on_change=lambda e: _set_uplift(e.value),
        ).classes("w-full")
        ui.label("Sicherheits-Puffer (%)").classes("lp-muted")
        ui.slider(
            min=0,
            max=40,
            step=1,
            value=int(state.safety_buffer_pct * 100),
            on_change=lambda e: _set_buffer(e.value),
        ).classes("w-full")
        ui.number(
            "Lagerbestand (Tests)",
            value=state.stock_level,
            min=0,
            step=25,
            on_change=lambda e: _set_stock(e.value),
        ).classes("w-full mt-1")
        ui.number(
            "Tests je FTE / Tag",
            value=state.tests_per_fte,
            min=1,
            step=1,
            on_change=lambda e: _set_fte(e.value),
        ).classes("w-full mt-1")
        ui.switch("Prophet aktiv", value=state.use_prophet, on_change=lambda e: _set_prophet(e.value)).classes("mt-2")
        ui.separator()
        ui.button("Neu berechnen", icon="play_arrow", on_click=_rebuild_and_refresh).classes("w-full mt-2")
        ui.button("AI Entscheidungslogik", icon="auto_awesome", on_click=_run_decision).classes("w-full mt-2")

    with ui.row().classes("lp-wrap items-center gap-2 mt-1"):
        ui.space()
        ui.button("Dashboard aktualisieren", icon="autorenew", on_click=_rebuild_and_refresh)
        ui.button("Decision Loop", icon="auto_awesome", on_click=_run_decision)
        ui.button("PDF exportieren", icon="picture_as_pdf", on_click=_download_pdf)
        ui.label(f"Update: {state.refreshed_at or '—'}").classes("text-xs lp-muted")

        with ui.row().classes("items-center gap-2"):
            ui.label(f"{state.pathogen} · {get_signal_source_label(state.signal_source)}").classes("lp-muted")
            ui.label(f"{state.company_name} · {state.facility_name}").classes("lp-muted")

    with ui.row().classes("lp-wrap"):
        with ui.column().classes("w-full gap-3"):
            with ui.row().classes("w-full items-center"):
                ui.button(
                    icon="menu",
                    on_click=control_drawer.toggle,
                    color=None,
                ).props("flat")
                ui.label("Dashboard").classes("text-2xl lp-title")
                ui.space()
                ui.chip("ML-First Flow").classes("lp-chip")
                ui.chip("App-ready").classes("lp-chip")

            dashboard_body()

    dashboard_body.refresh()


app.on_startup(_ensure_data)


if __name__ in {"__main__", "__mp_main__"}:
    _inject_styles()
    _ensure_data()
    ui.run(
        title="LabPulse AI — Decision OS",
        host="0.0.0.0",
        port=8080,
        show=False,
        dark=True,
    )
