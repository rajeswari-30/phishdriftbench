"""Downloadable PDF report for one checked URL.

The page shows only the verdict and what was found in the URL. Everything
else (why it decided, SHAP contributions, every measured signal, and the
domain and certificate lookups) goes into this report so it can be
previewed in the browser, saved, and shared.

Presentation only. Nothing in this module feeds the model.
"""
from __future__ import annotations

import io
import os
import re
import uuid
from datetime import datetime, timezone
from xml.sax.saxutils import escape

import tldextract
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (Flowable, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

INK = colors.HexColor("#111110")
INK_2 = colors.HexColor("#3d3d3a")
MUTED = colors.HexColor("#6f6e69")
FAINT = colors.HexColor("#a3a29c")
LINE = colors.HexColor("#e7e6e2")
SOFT = colors.HexColor("#f4f4f2")
PHISH = colors.HexColor("#c2372b")
LEGIT = colors.HexColor("#1d7d3f")
UNSURE = colors.HexColor("#94670d")

VERDICTS = {
    "PHISHING": (PHISH, colors.HexColor("#fdf1ef"), "Phishing",
                 "Do not open this link or enter any details on the page it leads to."),
    "LEGITIMATE": (LEGIT, colors.HexColor("#eff8f1"), "Legitimate",
                   "Nothing suspicious was found in the link text."),
    "UNCERTAIN": (UNSURE, colors.HexColor("#fcf6e8"), "Uncertain", None),
}

MARGIN_X = 20 * mm
TITLE = "URL analysis report"

_fonts_ready = False


def _register_fonts() -> None:
    """Inter and JetBrains Mono, bundled in demo/fonts/ (OFL). Unlike the PDF
    base-14 fonts they cover the Cyrillic and Greek look-alike letters that
    homoglyph phishing URLs are built from; Helvetica would print those as
    empty boxes, hiding exactly the character that makes the URL dangerous.
    Bundled rather than taken from matplotlib, which the deployed API no
    longer installs."""
    global _fonts_ready
    if _fonts_ready:
        return
    fonts = os.path.join(os.path.dirname(__file__), "fonts")
    pdfmetrics.registerFont(TTFont("PL", os.path.join(fonts, "Inter-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("PL-Bold", os.path.join(fonts, "Inter-SemiBold.ttf")))
    pdfmetrics.registerFont(TTFont("PL-Mono", os.path.join(fonts, "JetBrainsMono-Regular.ttf")))
    addMapping("PL", 0, 0, "PL")
    addMapping("PL", 1, 0, "PL-Bold")
    addMapping("PL", 0, 1, "PL")
    addMapping("PL", 1, 1, "PL-Bold")
    _fonts_ready = True


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def report_host(url: str) -> str:
    ext = tldextract.extract(url)
    host = (ext.fqdn or ext.domain or "").lower()
    return host[4:] if host.startswith("www.") else host


def report_filename(url: str, when: datetime | None = None) -> str:
    """`phishlens-report_<host>_<YYYY-MM-DD>.pdf`.

    Lowercase, hyphens inside a field, underscores between fields, ISO 8601
    date: the convention archives and records guidance agree on, because it
    sorts chronologically in any file browser and survives every file system
    and mail client without escaping."""
    slug = re.sub(r"[^a-z0-9]+", "-", report_host(url)).strip("-")[:60].strip("-") or "url"
    day = (when or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    return f"phishlens-report_{slug}_{day}.pdf"


def page_label(number: int, total: int) -> str:
    return f"Page {number} of {total}"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _t(value) -> str:
    """Escape for Paragraph markup and keep the house style of no em dashes."""
    s = "" if value is None else str(value)
    s = s.replace(" — ", ", ").replace("—", ", ").replace("–", " to ")
    return escape(s)


def _sentence(value) -> str:
    s = str(value or "").strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    return s if s[-1] in ".!?" else s + "."


def _num(value) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(f)) if f.is_integer() else f"{f:.4g}"


def _date(value):
    """WHOIS returns full timestamps; a report reader wants the day."""
    if value is None:
        return None
    s = str(value)
    return s[:10] if re.match(r"\d{4}-\d{2}-\d{2}", s) else s


def _fit(text: str, font: str, size: float, width: float) -> str:
    """Truncate with an ellipsis so a label never runs into the chart."""
    if pdfmetrics.stringWidth(text, font, size) <= width:
        return text
    while text and pdfmetrics.stringWidth(text + "…", font, size) > width:
        text = text[:-1]
    return text.rstrip() + "…"


def _styles() -> dict:
    s = {
        "brand": ParagraphStyle("brand", fontName="PL-Bold", fontSize=10.5, leading=13, textColor=INK),
        "meta": ParagraphStyle("meta", fontName="PL", fontSize=7.5, leading=11, textColor=MUTED,
                               alignment=TA_RIGHT),
        "title": ParagraphStyle("title", fontName="PL-Bold", fontSize=22, leading=26, textColor=INK),
        "host": ParagraphStyle("host", fontName="PL-Mono", fontSize=10.5, leading=14, textColor=MUTED,
                               wordWrap="CJK"),
        "h2": ParagraphStyle("h2", fontName="PL-Bold", fontSize=11.5, leading=15, textColor=INK,
                             spaceBefore=22, spaceAfter=9),
        "body": ParagraphStyle("body", fontName="PL", fontSize=9, leading=13.5, textColor=INK_2),
        "small": ParagraphStyle("small", fontName="PL", fontSize=7.8, leading=11.5, textColor=MUTED),
        "label": ParagraphStyle("label", fontName="PL", fontSize=7, leading=9, textColor=MUTED),
        "big": ParagraphStyle("big", fontName="PL-Bold", fontSize=14, leading=17, textColor=INK),
        "mono": ParagraphStyle("mono", fontName="PL-Mono", fontSize=8.5, leading=12, textColor=INK,
                               wordWrap="CJK"),
        "cell": ParagraphStyle("cell", fontName="PL", fontSize=7.8, leading=10.5, textColor=INK_2),
        "cell_muted": ParagraphStyle("cell_muted", fontName="PL", fontSize=7.8, leading=10.5,
                                     textColor=MUTED),
        "cell_mono": ParagraphStyle("cell_mono", fontName="PL-Mono", fontSize=7.4, leading=10,
                                    textColor=INK, wordWrap="CJK"),
        "cell_right": ParagraphStyle("cell_right", fontName="PL-Mono", fontSize=7.4, leading=10,
                                     textColor=INK, alignment=TA_RIGHT),
        "head": ParagraphStyle("head", fontName="PL-Bold", fontSize=6.8, leading=9, textColor=MUTED),
        "head_right": ParagraphStyle("head_right", fontName="PL-Bold", fontSize=6.8, leading=9,
                                     textColor=MUTED, alignment=TA_RIGHT),
    }
    return s


def _section(number: int, title: str, S: dict) -> Paragraph:
    """Numbered heading that always stays on the same page as its content."""
    p = Paragraph(f'<font name="PL-Mono" color="#a3a29c">{number:02d}</font>'
                  f'&nbsp;&nbsp;&nbsp;{_t(title)}', S["h2"])
    p.keepWithNext = True
    return p


# ---------------------------------------------------------------------------
# Drawn elements
# ---------------------------------------------------------------------------

class _Bar(Flowable):
    """Diverging bar for one signal: right of centre pushes toward phishing,
    left toward legitimate, scaled to the most influential signal."""

    def __init__(self, value: float, max_abs: float, width: float = 64, height: float = 5):
        super().__init__()
        self.value, self.max_abs = value, max_abs or 1e-9
        self.width, self.height = width, height

    def wrap(self, avail_w, avail_h):
        return self.width, self.height

    def draw(self):
        c, w, h = self.canv, self.width, self.height
        mid = w / 2
        c.setFillColor(SOFT)
        c.roundRect(0, 0, w, h, h / 2, stroke=0, fill=1)
        span = min(abs(self.value) / self.max_abs, 1.0) * mid
        if span > 0.3:
            c.setFillColor(PHISH if self.value > 0 else LEGIT)
            c.roundRect(mid if self.value > 0 else mid - span, 0, span, h, h / 2, stroke=0, fill=1)
        c.setStrokeColor(LINE)
        c.setLineWidth(0.6)
        c.line(mid, -1.5, mid, h + 1.5)


class _ShapChart(Flowable):
    """SHAP contributions for the most influential signals, drawn as vector
    shapes. Replaces the shap package's force plot image, whose feature labels
    overlapped one another at report width and could not be read."""

    ROW = 17
    HEAD = 16

    def __init__(self, rows: list, width: float):
        super().__init__()
        self.rows = rows
        self.width = width
        self.max_abs = max((abs(v) for _, v, _ in rows), default=0) or 1e-9
        self.height = self.HEAD + self.ROW * len(rows)

    def wrap(self, avail_w, avail_h):
        return self.width, self.height

    def draw(self):
        c = self.canv
        label_w = self.width * 0.40
        value_w = self.width * 0.11
        bar_x = label_w + 10
        bar_w = self.width - bar_x - value_w - 10
        mid = bar_x + bar_w / 2
        top = self.height - self.HEAD

        c.setFont("PL", 6.6)
        c.setFillColor(FAINT)
        c.drawString(bar_x, top + 5, "toward legitimate")
        c.drawRightString(bar_x + bar_w, top + 5, "toward phishing")
        c.drawRightString(self.width, top + 5, "log odds")

        for i, (label, value, aggregate) in enumerate(self.rows):
            y = top - (i + 1) * self.ROW
            c.setStrokeColor(LINE)
            c.setLineWidth(0.4)
            c.line(0, y, self.width, y)
            c.setFont("PL", 7.8)
            c.setFillColor(MUTED if aggregate else INK_2)
            c.drawString(0, y + 5.5, _fit(label, "PL", 7.8, label_w))
            span = min(abs(value) / self.max_abs, 1.0) * (bar_w / 2)
            if span > 0.4:
                c.setFillColor(PHISH if value > 0 else LEGIT)
                if aggregate:
                    c.setFillAlpha(0.45)
                c.roundRect(mid if value > 0 else mid - span, y + 4, span, 8, 2, stroke=0, fill=1)
                c.setFillAlpha(1)
            c.setFont("PL-Mono", 7.4)
            c.setFillColor(PHISH if value > 0 else LEGIT if value < 0 else MUTED)
            c.drawRightString(self.width, y + 5.5, f"{value:+.3f}")

        c.setStrokeColor(FAINT)
        c.setLineWidth(0.6)
        c.line(mid, top - len(self.rows) * self.ROW, mid, top + 1)


def _canvas_class(host: str, generated: str, report_id: str, title: str = TITLE):
    """Canvas that holds every page until the document is finished, so each
    footer can say "Page 2 of 3". Pages after the first also get a running
    header naming the report and the host it covers."""

    class _Canvas(rl_canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._held = []

        def showPage(self):
            self._held.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._held)
            for state in self._held:
                self.__dict__.update(state)
                self._chrome(total)
                rl_canvas.Canvas.showPage(self)
            rl_canvas.Canvas.save(self)

        def _chrome(self, total: int):
            n = self._pageNumber
            w, h = A4
            left, right = MARGIN_X, w - MARGIN_X
            self.saveState()
            self.setStrokeColor(LINE)
            self.setLineWidth(0.5)
            if n > 1:
                self.setFont("PL-Bold", 8)
                self.setFillColor(INK)
                self.drawString(left, h - 13 * mm, "PhishLens")
                self.setFont("PL", 8)
                self.setFillColor(MUTED)
                self.drawRightString(right, h - 13 * mm,
                                     _fit(f"{title}  ·  {host}", "PL", 8, (right - left) * 0.7))
                self.line(left, h - 15.5 * mm, right, h - 15.5 * mm)
            self.line(left, 16 * mm, right, 16 * mm)
            self.setFont("PL", 7.3)
            self.setFillColor(MUTED)
            self.drawString(left, 11 * mm, f"Report {report_id}")
            self.drawCentredString(w / 2, 11 * mm, page_label(n, total))
            self.drawRightString(right, 11 * mm, generated)
            self.restoreState()

    return _Canvas


def _plain_table(data, col_widths, extra=None, repeat=0):
    t = Table(data, colWidths=col_widths, repeatRows=repeat)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ] + (extra or [])))
    return t


def _kv_block(title: str, rows: list, S: dict, width: float):
    """One titled key/value group. Each row is (label, value, kind) where kind
    is 'text', 'mono' or 'flag'."""
    data = [[Paragraph(title.upper(), S["head"]), ""]]
    for label, value, kind in rows:
        if kind == "flag":
            cell = Paragraph('<font color="#1d7d3f">Yes</font>' if value
                             else '<font color="#c2372b">No</font>', S["cell"])
        else:
            missing = value in (None, "", [])
            cell = Paragraph(_t("Not available" if missing else value),
                             S["cell_mono"] if kind == "mono" and not missing else S["cell"])
        data.append([Paragraph(_t(label), S["cell_muted"]), cell])
    extra = [("LINEBELOW", (0, 0), (-1, 0), 0.6, INK), ("SPAN", (0, 0), (1, 0)),
             ("VALIGN", (0, 1), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (0, -1), 0)]
    extra += [("LINEBELOW", (0, i), (-1, i), 0.3, LINE) for i in range(1, len(data) - 1)]
    return KeepTogether([_plain_table(data, [width * 0.3, width * 0.7], extra), Spacer(1, 12)])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report_pdf(result: dict, context: dict | None = None,
                     generated_at: datetime | None = None) -> bytes:
    """Render the full report for one `predict_and_explain` result. `context`
    is the dict the /api/context endpoint returns; if a live lookup failed the
    report still renders and says so."""
    _register_fonts()
    S = _styles()
    when = generated_at or datetime.now(timezone.utc)
    generated = when.strftime("%d %b %Y, %H:%M UTC")
    report_id = (context or {}).get("job_id") or uuid.uuid4().hex[:12]
    url = result.get("url", "")
    host = report_host(url) or url

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X,
        topMargin=22 * mm, bottomMargin=24 * mm,
        title=f"PhishLens report: {host}", author="PhishLens", creator="PhishLens",
        subject=f"URL analysis of {url}", keywords="phishing, URL, SHAP, PhishLens",
    )
    W = doc.width
    story = []

    # ---- title block -------------------------------------------------------
    head = Table([[Paragraph("PhishLens", S["brand"]),
                   Paragraph(f"Generated {generated}<br/>Report {report_id}", S["meta"])]],
                 colWidths=[W * 0.5, W * 0.5])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                              ("TOPPADDING", (0, 0), (-1, -1), 0),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    story += [head, Spacer(1, 26), Paragraph(TITLE, S["title"]), Spacer(1, 3),
              Paragraph(_t(host), S["host"])]

    # ---- 01 summary --------------------------------------------------------
    verdict = result.get("verdict", "")
    fg, bg, label, advice = VERDICTS.get(verdict, (INK, SOFT, verdict.title(), None))
    # A verdict settled by the trusted-domain list or by "this page is
    # published by a stranger" needs to say so: the SHAP numbers further down
    # are then the model's opinion, not the reason for the answer.
    if result.get("decision_reason"):
        advice = _sentence(result.get("decision_reason"))
    elif result.get("abstained"):
        advice = _sentence(result.get("abstain_reason"))
    confidence = ("No verdict" if result.get("abstained")
                  else f"{float(result.get('confidence', 0)) * 100:.1f}%")
    rows = [
        [Paragraph("VERDICT", S["label"]), Paragraph("CONFIDENCE", S["label"])],
        [Paragraph(_t(label), ParagraphStyle("v", parent=S["big"], textColor=fg)),
         Paragraph(_t(confidence), S["big"])],
        [Paragraph("URL", S["label"]), ""],
        [Paragraph(_t(url), S["mono"]), ""],
    ]
    if advice:
        rows.append([Paragraph(_t(advice), S["body"]), ""])
    summary = Table(rows, colWidths=[W * 0.5, W * 0.5])
    last = len(rows) - 1
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, fg),
        ("SPAN", (0, 2), (1, 2)), ("SPAN", (0, 3), (1, 3))]
        + ([("SPAN", (0, 4), (1, 4)), ("TOPPADDING", (0, 4), (-1, 4), 10)] if advice else [])
        + [("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
           ("TOPPADDING", (0, 0), (-1, 0), 14), ("TOPPADDING", (0, 2), (-1, 2), 10),
           ("BOTTOMPADDING", (0, last), (-1, last), 14)]))
    story += [_section(1, "Summary", S), summary]

    # ---- 02 found in the URL -----------------------------------------------
    brands = result.get("brands_referenced") or []
    words = result.get("suspicious_words") or []
    story.append(_section(2, "Found in the URL", S))
    if brands or words:
        found = []
        if brands:
            found.append([Paragraph("Brand in the link", S["cell_muted"]),
                          Paragraph(_t(", ".join(brands)), S["cell_mono"])])
        if words:
            found.append([Paragraph("Suspicious words", S["cell_muted"]),
                          Paragraph(_t(", ".join(words)), S["cell_mono"])])
        story.append(_plain_table(found, [W * 0.3, W * 0.7],
                                  [("LEFTPADDING", (0, 0), (0, -1), 0),
                                   ("LINEBELOW", (0, 0), (-1, -2), 0.3, LINE)]))
    else:
        story.append(Paragraph("No brand names or suspicious words in the link.", S["body"]))

    # ---- 03 why it decided -------------------------------------------------
    story.append(_section(3, "Why it decided", S))
    decided_by = result.get("decided_by")
    if decided_by and decided_by != "model":
        story.append(Paragraph(
            _t(f"This verdict was settled by the {decided_by}, not by the classifier. "
               f"The signals and contributions below are still the model's reading of the "
               f"URL text, reported in full, but they did not decide the answer."),
            S["note"] if "note" in S else S["body"]))
        story.append(Spacer(1, 8))
    reasons = result.get("top_reasons") or []
    if reasons:
        story.append(_plain_table(
            [[Paragraph(f"{i}", S["cell_muted"]), Paragraph(_t(_sentence(r)), S["body"])]
             for i, r in enumerate(reasons, 1)],
            [W * 0.05, W * 0.95],
            [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (0, -1), 0),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    else:
        story.append(Paragraph("No single signal stood out.", S["body"]))

    # ---- 04 SHAP contributions ---------------------------------------------
    signals = result.get("signals") or []
    if signals:
        top = [s for s in signals if s["contribution"] != 0][:10]
        rest = signals[len(top):]
        rows = [(_sentence(s["label"]).rstrip("."), float(s["contribution"]), False) for s in top]
        rest_sum = sum(float(s["contribution"]) for s in rest)
        if rest and abs(rest_sum) > 1e-6:
            rows.append((f"{len(rest)} other signals combined", rest_sum, True))
        score = float(result.get("ml_score", 0)) * 100
        threshold = float(result.get("decision_threshold", 0.5)) * 100
        story += [
            _section(4, "SHAP contributions", S),
            _ShapChart(rows, W),
            Spacer(1, 8),
            Paragraph(f"Model score {score:.1f}%, against a decision threshold of {threshold:.1f}%. "
                      "Each bar is an exact TreeSHAP contribution for this URL.", S["small"]),
        ]

    # ---- 05 all signals ----------------------------------------------------
    if signals:
        max_abs = max((abs(s["contribution"]) for s in signals), default=0.0) or 1e-9
        data = [[Paragraph("SIGNAL", S["head"]), Paragraph("VALUE", S["head_right"]),
                 Paragraph("EFFECT", S["head"]), Paragraph("WEIGHT", S["head_right"])]]
        for s in signals:
            c = float(s["contribution"])
            tone = PHISH if c > 0 else LEGIT if c < 0 else MUTED
            data.append([
                Paragraph(_t(_sentence(s["label"]).rstrip(".")), S["cell"]),
                Paragraph(_t(_num(s["value"])), S["cell_right"]),
                _Bar(c, max_abs),
                Paragraph(f"{c:+.3f}" if c else "0",
                          ParagraphStyle("w", parent=S["cell_right"], textColor=tone)),
            ])
        extra = [("LINEBELOW", (0, 0), (-1, 0), 0.6, INK), ("ALIGN", (2, 1), (2, -1), "CENTER"),
                 ("LEFTPADDING", (0, 0), (0, -1), 0)]
        extra += [("BACKGROUND", (0, i), (-1, i), SOFT) for i in range(2, len(data), 2)]
        story += [
            _section(5, f"All {len(signals)} signals", S),
            Paragraph("Sorted by influence. Weights are in log odds and sum to the model score.",
                      S["small"]),
            Spacer(1, 6),
            _plain_table(data, [W * 0.52, W * 0.13, W * 0.2, W * 0.15], extra, repeat=1),
        ]

    # ---- 06 domain and certificate -----------------------------------------
    note = Table([[Paragraph(
        "<b>Not used by the detector.</b> Our audit found these lookups did not improve accuracy "
        "over the URL text alone, so the verdict comes only from the URL. They are shown as "
        "background for a person reviewing the case.", S["small"])]], colWidths=[W])
    note.setStyle(TableStyle([("LINEBEFORE", (0, 0), (0, -1), 2, LINE),
                              ("LEFTPADDING", (0, 0), (-1, -1), 10),
                              ("TOPPADDING", (0, 0), (-1, -1), 2),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    note.keepWithNext = True
    note.spaceAfter = 14
    story += [_section(6, "Domain and certificate", S), note]

    if context:
        w_, dns, ssl = context.get("whois") or {}, context.get("dns") or {}, context.get("ssl") or {}
        story.append(_kv_block("Registration", [
            ("Lookup succeeded", w_.get("success"), "flag"),
            ("Registered on", _date(w_.get("registration_date")), "mono"),
            ("Expires on", _date(w_.get("expiry_date")), "mono"),
            ("Registrar country", w_.get("registrar_country"), "text"),
        ] + ([("Lookup error", w_.get("error"), "text")] if w_.get("error") else []), S, W))
        ips = dns.get("ip_addresses") or []
        story.append(_kv_block("Hosting", [
            ("Resolved", dns.get("dns_success"), "flag"),
            ("IP address" + ("es" if len(ips) > 1 else ""), ", ".join(ips), "mono"),
            ("Nameservers", ", ".join(dns.get("nameservers") or []), "mono"),
            ("Mail records", dns.get("mx_present"), "flag"),
        ] + ([("Lookup error", dns.get("dns_error"), "text")] if dns.get("dns_error") else []), S, W))
        cert = [("Certificate present", ssl.get("ssl_present"), "flag")]
        if ssl.get("ssl_present"):
            sans = ssl.get("subject_alt_names") or []
            shown = ", ".join(sans[:14]) + (f", and {len(sans) - 14} more" if len(sans) > 14 else "")
            cert += [("Issued to", ssl.get("subject_cn"), "mono"),
                     ("Issued by", ", ".join(x for x in (ssl.get("issuer_org"), ssl.get("issuer_cn"))
                                             if x), "text"),
                     ("Valid from", ssl.get("not_before"), "mono"),
                     ("Valid until", ssl.get("not_after"), "mono"),
                     ("Not expired", not ssl.get("is_expired"), "flag"),
                     (f"Also covers ({len(sans)})", shown, "mono")]
        elif ssl.get("ssl_error"):
            cert.append(("Lookup error", ssl.get("ssl_error"), "text"))
        story.append(_kv_block("Certificate", cert, S, W))
    else:
        story.append(Paragraph("Live lookups could not be completed for this URL.", S["body"]))

    doc.build(story, canvasmaker=_canvas_class(host, generated, report_id))
    return buf.getvalue()
