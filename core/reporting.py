"""
LabPulse AI — Report Engine
============================
Creates a PDF Management-Report from current dashboard state.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _eur(value: float) -> str:
    return f"{float(value):,.2f} €".replace(",", ".")


def generate_decision_report(
    company: str,
    pathogen: str,
    kpis: Dict[str, object],
    forecast_df: pd.DataFrame,
    recommendation: Optional[str] = None,
    ai_provider: str = "Ollama (falls verfügbar)",
) -> bytes:
    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
    )

    title = ParagraphStyle(
        "LabPulseTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#f8fafc"),
        leading=26,
        spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "LabPulseSub",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#94a3b8"),
        spaceAfter=15,
    )
    heading = ParagraphStyle(
        "LabPulseHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#38bdf8"),
        spaceBefore=8,
        spaceAfter=6,
        leading=17,
    )
    body = ParagraphStyle(
        "LabPulseBody",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#dbeafe"),
    )

    elements = []
    elements.append(Paragraph("LabPulse AI — Entscheidungsvorlage", title))
    elements.append(
        Paragraph(
            f"{company} · {pathogen} · {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            subtitle,
        )
    )

    rows = [
        ["Kennzahl", "Wert"],
        ["Pathogen", pathogen],
        ["Revenue 7 Tage", _eur(float(kpis.get("revenue_forecast_7d", 0) or 0.0))],
        ["Tests 7 Tage", f"{int(kpis.get('predicted_tests_7d', 0) or 0):,}".replace(",", ".")],
        ["Bestand", f"{int(kpis.get('stock_on_hand', 0) or 0):,}".replace(",", ".")],
        ["Sicherheitsrisiko (€)", _eur(float(kpis.get("risk_eur", 0.0) or 0.0))],
        ["Trend 7d", f"{float(kpis.get('trend_pct', 0.0) or 0.0):+.1f} %"],
        ["Reagenzname", str(kpis.get("test_name", "unbekannt"))],
        ["Kosten/Test", _eur(float(kpis.get("cost_per_test", 0) or 0))],
        ["Gesamtnachfrage Horizont", f"{int(kpis.get('total_demand', 0) or 0):,}".replace(",", ".")],
        ["Modell", str((kpis.get("model") or "Simple"))],
        ["KI-Engine", ai_provider],
    ]
    table = Table(rows, colWidths=[7.5 * cm, 9 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#fb923c")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#93c5fd")),
                ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#334155")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#0f172a"), colors.HexColor("#111827")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 0.45 * cm))

    elements.append(Paragraph("Kurz-Interpretation", heading))
    elements.append(Paragraph((recommendation or "").strip() or "Keine KI-Einschätzung verfügbar.", body))
    elements.append(Spacer(1, 0.45 * cm))

    if forecast_df is not None and not forecast_df.empty:
        elements.append(Paragraph("Top 10 – Order-Horizont", heading))
        disp = forecast_df.head(10).copy()
        if "Date" in disp.columns:
            disp["Date"] = pd.to_datetime(disp["Date"], errors="coerce").dt.strftime("%d.%m.%Y")
        table_headers = [c for c in ["Date", "Predicted Volume", "Reagent Order", "Remaining Stock", f"Est. Revenue (€{float(kpis.get('cost_per_test', 0) or 0):.0f}/test)"] if c in disp.columns]
        if table_headers:
            table_rows = [table_headers]
            for _, row in disp[table_headers].iterrows():
                formatted = []
                for col in table_headers:
                    value = row[col]
                    if "Volume" in col or "Order" in col or "Stock" in col or "Revenue" in col:
                        if pd.isna(value):
                            formatted.append("0")
                        elif "Revenue" in col:
                            formatted.append(_eur(float(value)))
                        else:
                            formatted.append(f"{int(value):,}".replace(",", "."))
                    else:
                        formatted.append(str(value))
                table_rows.append(formatted)

            forecast_table = Table(table_rows, colWidths=[3.2 * cm, 3 * cm, 3 * cm, 3.6 * cm, 4.0 * cm])
            forecast_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#f97316")),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#e2e8f0")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#334155")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#0f172a"), colors.HexColor("#111827")]),
                    ]
                )
            )
            elements.append(forecast_table)

    doc.build(elements)
    return buffer.getvalue()
