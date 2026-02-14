"""
LabPulse AI — PDF Report Export
================================
Generates a professional PDF report with KPIs, charts (as PNG), forecast
table, and optional AI commentary. Uses reportlab for minimal dependencies.
"""

import io
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_import_reportlab():
    """Lazy import to avoid hard dependency at module level."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    return {
        "colors": colors,
        "A4": A4,
        "ParagraphStyle": ParagraphStyle,
        "getSampleStyleSheet": getSampleStyleSheet,
        "cm": cm,
        "mm": mm,
        "Image": Image,
        "Paragraph": Paragraph,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer,
        "Table": Table,
        "TableStyle": TableStyle,
    }


# ── Brand Colors ─────────────────────────────────────────────────────────────
ORANGE = (0.969, 0.498, 0.0)       # #f77f00
DARK_BG = (0.043, 0.055, 0.090)    # #0b0e17
CARD_BG = (0.067, 0.094, 0.153)    # #111827
TEXT_PRIMARY = (0.945, 0.961, 0.976)
TEXT_MUTED = (0.392, 0.455, 0.545)
RED = (0.937, 0.267, 0.267)
GREEN = (0.133, 0.773, 0.369)


def _chart_to_png_bytes(fig, width: int = 900, height: int = 400) -> Optional[bytes]:
    """Convert a Plotly figure to PNG bytes via kaleido."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception as exc:
        logger.warning("Chart to PNG failed: %s", exc)
        return None


def generate_report(
    kpis: dict,
    forecast_df: pd.DataFrame,
    correlation_fig=None,
    burndown_fig=None,
    ai_commentary: str = "",
    pathogen: str = "SARS-CoV-2",
) -> bytes:
    """
    Generate a complete PDF report and return as bytes.

    Parameters
    ----------
    kpis : dict — KPI dictionary from build_forecast()
    forecast_df : pd.DataFrame — Forecast display table
    correlation_fig : plotly.graph_objects.Figure — Wastewater vs Lab chart
    burndown_fig : plotly.graph_objects.Figure — Stock burndown chart
    ai_commentary : str — AI-generated text (from Ollama or fallback)
    pathogen : str — Current pathogen name

    Returns
    -------
    bytes — PDF file content
    """
    rl = _safe_import_reportlab()

    buffer = io.BytesIO()
    doc = rl["SimpleDocTemplate"](
        buffer,
        pagesize=rl["A4"],
        topMargin=1.5 * rl["cm"],
        bottomMargin=1.5 * rl["cm"],
        leftMargin=2 * rl["cm"],
        rightMargin=2 * rl["cm"],
    )

    styles = rl["getSampleStyleSheet"]()

    # Custom styles
    title_style = rl["ParagraphStyle"](
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        spaceAfter=4,
        textColor=rl["colors"].HexColor("#f1f5f9"),
    )
    subtitle_style = rl["ParagraphStyle"](
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=16,
        textColor=rl["colors"].HexColor("#94a3b8"),
    )
    section_style = rl["ParagraphStyle"](
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=18,
        spaceAfter=8,
        textColor=rl["colors"].HexColor("#f77f00"),
    )
    body_style = rl["ParagraphStyle"](
        "BodyText",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=rl["colors"].HexColor("#cbd5e1"),
    )
    small_style = rl["ParagraphStyle"](
        "SmallText",
        parent=styles["Normal"],
        fontSize=7,
        textColor=rl["colors"].HexColor("#64748b"),
    )

    elements = []
    Paragraph = rl["Paragraph"]
    Spacer_ = rl["Spacer"]
    Table_ = rl["Table"]
    TableStyle_ = rl["TableStyle"]
    Image_ = rl["Image"]
    cm_ = rl["cm"]
    colors_ = rl["colors"]

    now = datetime.now()

    # ── Title Block ───────────────────────────────────────────────────────────
    elements.append(Paragraph("LabPulse AI — Report", title_style))
    elements.append(
        Paragraph(
            f"{pathogen} &nbsp;|&nbsp; {now.strftime('%d.%m.%Y %H:%M')}",
            subtitle_style,
        )
    )
    elements.append(Spacer_(1, 0.3 * cm_))

    # ── KPI Summary ───────────────────────────────────────────────────────────
    elements.append(Paragraph("KPI-Zusammenfassung", section_style))

    kpi_data = [
        ["Kennzahl", "Wert"],
        ["Prognostizierte Tests (7d)", f"{kpis.get('predicted_tests_7d', 0):,}"],
        ["Umsatzprognose (7d)", f"EUR {kpis.get('revenue_forecast_7d', 0):,.0f}"],
        ["Reagenzstatus", kpis.get("reagent_status", "N/A")],
        ["Revenue at Risk", f"EUR {kpis.get('risk_eur', 0):,.0f}"],
        ["WoW-Trend", f"{kpis.get('trend_pct', 0):+.1f}%"],
        ["Lagerbestand", f"{kpis.get('stock_on_hand', 0):,} Einheiten"],
        ["Gesamtbedarf (Horizont)", f"{kpis.get('total_demand', 0):,} Einheiten"],
        ["Test-Kit", kpis.get("test_name", "N/A")],
        ["Kosten/Test", f"EUR {kpis.get('cost_per_test', 45)}"],
    ]

    stockout = kpis.get("stockout_day")
    if stockout:
        so_str = pd.Timestamp(stockout).strftime("%d.%m.%Y")
        kpi_data.append(["Stockout-Datum", so_str])

    col_widths = [8 * cm_, 8 * cm_]
    kpi_table = Table_(kpi_data, colWidths=col_widths)
    kpi_table.setStyle(
        TableStyle_(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors_.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors_.HexColor("#f77f00")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 1), (0, -1), colors_.HexColor("#94a3b8")),
                ("TEXTCOLOR", (1, 1), (1, -1), colors_.HexColor("#e2e8f0")),
                ("BACKGROUND", (0, 1), (-1, -1), colors_.HexColor("#111827")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors_.HexColor("#111827"),
                    colors_.HexColor("#0f172a"),
                ]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors_.HexColor("#1e293b")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    elements.append(kpi_table)
    elements.append(Spacer_(1, 0.5 * cm_))

    # ── Charts ────────────────────────────────────────────────────────────────
    if correlation_fig is not None:
        png_bytes = _chart_to_png_bytes(correlation_fig, width=900, height=380)
        if png_bytes:
            elements.append(Paragraph("Wastewater vs. Laborvolumen", section_style))
            img_buf = io.BytesIO(png_bytes)
            img = Image_(img_buf, width=16 * cm_, height=6.8 * cm_)
            elements.append(img)
            elements.append(Spacer_(1, 0.3 * cm_))

    if burndown_fig is not None:
        png_bytes = _chart_to_png_bytes(burndown_fig, width=900, height=340)
        if png_bytes:
            elements.append(Paragraph("Reagenz-Bestandsprognose", section_style))
            img_buf = io.BytesIO(png_bytes)
            img = Image_(img_buf, width=16 * cm_, height=6 * cm_)
            elements.append(img)
            elements.append(Spacer_(1, 0.3 * cm_))

    # ── Forecast Table ────────────────────────────────────────────────────────
    elements.append(Paragraph("Bestellempfehlungen", section_style))

    # Build table data from forecast_df
    header = list(forecast_df.columns)
    table_data = [header]
    for _, row in forecast_df.iterrows():
        formatted_row = []
        for col in header:
            val = row[col]
            if isinstance(val, (int, float)):
                if "Revenue" in col or "revenue" in col.lower():
                    formatted_row.append(f"EUR {val:,.0f}")
                else:
                    formatted_row.append(f"{val:,.0f}")
            elif hasattr(val, "strftime"):
                formatted_row.append(val.strftime("%d.%m.%Y"))
            else:
                formatted_row.append(str(val))
        table_data.append(formatted_row)

    n_cols = len(header)
    fw = 16 * cm_ / n_cols
    forecast_table = Table_(table_data, colWidths=[fw] * n_cols)

    # Highlight rows with orders > 0
    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors_.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors_.HexColor("#f77f00")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors_.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 1), (-1, -1), colors_.HexColor("#111827")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors_.HexColor("#1e293b")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ]

    # Find "Reagent Order" column index
    order_col_idx = None
    for i, col_name in enumerate(header):
        if "Order" in col_name or "order" in col_name.lower():
            order_col_idx = i
            break

    if order_col_idx is not None:
        for row_idx in range(1, len(table_data)):
            raw_val = forecast_df.iloc[row_idx - 1]
            order_val = raw_val.get("Reagent Order", 0)
            if order_val and order_val > 0:
                style_commands.append(
                    ("BACKGROUND", (0, row_idx), (-1, row_idx),
                     colors_.HexColor("#1c1117"))
                )
                style_commands.append(
                    ("TEXTCOLOR", (0, row_idx), (-1, row_idx),
                     colors_.HexColor("#fca5a5"))
                )

    forecast_table.setStyle(TableStyle_(style_commands))
    elements.append(forecast_table)
    elements.append(Spacer_(1, 0.5 * cm_))

    # ── AI Commentary ─────────────────────────────────────────────────────────
    if ai_commentary:
        elements.append(Paragraph("KI-Analyse", section_style))
        elements.append(Paragraph(ai_commentary, body_style))
        elements.append(Spacer_(1, 0.3 * cm_))

    # ── Footer ────────────────────────────────────────────────────────────────
    elements.append(Spacer_(1, 1 * cm_))
    elements.append(
        Paragraph(
            f"LabPulse AI &nbsp;|&nbsp; "
            f"Datenquelle: RKI AMELAG &nbsp;|&nbsp; "
            f"Generiert: {now.strftime('%d.%m.%Y %H:%M')}",
            small_style,
        )
    )

    # Build PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    logger.info("PDF report generated: %d bytes", len(pdf_bytes))
    return pdf_bytes
