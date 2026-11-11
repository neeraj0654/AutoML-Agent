"""
PDF Export — builds a styled PDF report from the pipeline results.
"""
import io
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ── Color palette ─────────────────────────────────────────────────────────────
PURPLE = colors.HexColor("#6C63FF")
PINK   = colors.HexColor("#FF6584")
DARK   = colors.HexColor("#0E1117")
LIGHT  = colors.HexColor("#E0E0E0")
GREY   = colors.HexColor("#A0A0B0")
WHITE  = colors.white

PAGE_W, PAGE_H = A4


def _styles():
    s = getSampleStyleSheet()
    base = dict(fontName="Helvetica", textColor=LIGHT)
    return {
        "title": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=22,
                                 textColor=PURPLE, alignment=TA_CENTER, spaceAfter=4),
        "subtitle": ParagraphStyle("Sub", fontName="Helvetica", fontSize=11,
                                    textColor=GREY, alignment=TA_CENTER, spaceAfter=20),
        "h2": ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=14,
                               textColor=PURPLE, spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("Body", fontName="Helvetica", fontSize=10,
                                textColor=colors.black, spaceAfter=6, leading=16),
        "meta": ParagraphStyle("Meta", fontName="Helvetica-Oblique", fontSize=9,
                                textColor=GREY, spaceAfter=4),
    }


def _table(data: list[list], col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  PURPLE),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F7F7FB"), WHITE]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D0E0")),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    return t


def build_pdf(
    dataset_name: str,
    profile: dict,
    feature_plan: dict,
    leaderboard: list[dict],
    best_model_name: str,
    best_score: float,
    metric_name: str,
    report_text: str,
    chart_bufs: dict,          # {"leaderboard": BytesIO, "importance": BytesIO}
) -> io.BytesIO:

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm,
    )
    st  = _styles()
    els = []

    # ── Title ──────────────────────────────────────────────────────────────────
    els += [
        Paragraph("AutoML Agent Report", st["title"]),
        Paragraph(f"Dataset: {dataset_name}  •  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", st["subtitle"]),
        HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=16),
    ]

    # ── Dataset Overview ───────────────────────────────────────────────────────
    els.append(Paragraph("1. Dataset Overview", st["h2"]))
    overview_data = [
        ["Metric", "Value"],
        ["Rows", str(profile.get("n_rows", "—"))],
        ["Columns", str(profile.get("n_cols", "—"))],
        ["Numeric columns", str(profile.get("n_numeric", "—"))],
        ["Categorical columns", str(profile.get("n_categorical", "—"))],
        ["Missing cells", str(profile.get("total_missing", "—"))],
        ["Duplicate rows", str(profile.get("n_duplicates", "—"))],
        ["Task type", str(profile.get("task_type", "—")).upper()],
        ["Target column", str(profile.get("target_col", "—"))],
    ]
    els += [_table(overview_data, col_widths=[8*cm, 7*cm]), Spacer(1, 12)]

    # ── Feature Engineering ────────────────────────────────────────────────────
    els.append(Paragraph("2. Feature Engineering", st["h2"]))
    steps = feature_plan.get("steps_applied", [])
    for step in steps:
        els.append(Paragraph(f"• {step}", st["body"]))
    if not steps:
        els.append(Paragraph("No feature engineering steps recorded.", st["body"]))
    els.append(Spacer(1, 8))

    # ── Model Leaderboard ──────────────────────────────────────────────────────
    els.append(Paragraph("3. Model Leaderboard", st["h2"]))
    if leaderboard:
        lb_data = [["Rank", "Model", f"CV {metric_name}", "Fit Time (s)"]]
        for i, row in enumerate(leaderboard):
            lb_data.append([
                str(i + 1),
                row["model"],
                f"{row['cv_score']:.4f}",
                f"{row.get('fit_time', 0):.2f}",
            ])
        els += [_table(lb_data, col_widths=[1.5*cm, 8*cm, 4*cm, 3.5*cm]), Spacer(1, 8)]

    if chart_bufs.get("leaderboard"):
        chart_bufs["leaderboard"].seek(0)
        img = RLImage(chart_bufs["leaderboard"], width=14*cm, height=6*cm)
        els += [img, Spacer(1, 12)]

    # ── Feature Importances ────────────────────────────────────────────────────
    if chart_bufs.get("importance"):
        els.append(Paragraph("4. Feature Importances", st["h2"]))
        chart_bufs["importance"].seek(0)
        img = RLImage(chart_bufs["importance"], width=14*cm, height=7*cm)
        els += [img, Spacer(1, 12)]

    # ── LLM Report ────────────────────────────────────────────────────────────
    els.append(Paragraph("5. AI Analysis", st["h2"]))
    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            els.append(Spacer(1, 4))
        else:
            line = line.lstrip("#").lstrip("*").strip()
            els.append(Paragraph(line, st["body"]))

    # ── Footer ─────────────────────────────────────────────────────────────────
    els += [
        Spacer(1, 20),
        HRFlowable(width="100%", thickness=0.5, color=GREY),
        Paragraph("Generated by AutoML Agent  •  Built with scikit-learn, XGBoost & Google Gemini", st["meta"]),
    ]

    doc.build(els)
    buf.seek(0)
    return buf
