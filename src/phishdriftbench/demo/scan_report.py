"""The PDF report for one full scan (scan/engine.py).

Written to read like a security advisory, not a data dump. Page one answers
the only questions most readers have: is this link dangerous, why, and what
should I do. The technical facts follow as an appendix for whoever needs
them, with empty fields left out rather than printed as "Not available".

Typography is Inter and JetBrains Mono, bundled in demo/fonts/ (both OFL).
They are bundled rather than borrowed from an installed package so the
report renders the same on any host, and both cover the Cyrillic and Greek
look-alike letters homograph links are built from, which a base PDF font
would print as empty boxes.
"""
from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (Flowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from phishdriftbench.demo.report import report_host

FONT_DIR = Path(__file__).resolve().parent / "fonts"

# ---- palette (matches the web app) ---------------------------------------------
INK = colors.HexColor("#141413")
INK_2 = colors.HexColor("#3b3a37")
MUTED = colors.HexColor("#6c6b67")
FAINT = colors.HexColor("#a5a4a0")
LINE = colors.HexColor("#e7e6e2")
SOFT = colors.HexColor("#f6f6f4")
ACCENT = colors.HexColor("#5b3fd6")
TONES = {
    "danger": (colors.HexColor("#c2372b"), colors.HexColor("#fcf1ef")),
    "warn": (colors.HexColor("#a16207"), colors.HexColor("#fbf5e6")),
    "ok": (colors.HexColor("#1d7a43"), colors.HexColor("#eff7f1")),
    "neutral": (MUTED, SOFT),
}
VERDICT = {
    "phishing": ("Phishing", "danger"),
    "suspicious": ("Suspicious", "warn"),
    "safe": ("Looks safe", "ok"),
}
ADVICE = {
    "phishing": [
        "Don't open the link, and don't enter anything on the page it leads to.",
        "If you already entered a password, change it now, starting with any account that shares it.",
        "Report the message it came in, then delete it.",
    ],
    "suspicious": [
        "Don't sign in or pay through this link.",
        "Go to the company's site by typing its address yourself, or use its app.",
        "If the message is urgent or unexpected, check with the sender another way.",
    ],
    "safe": [
        "No warning signs were found, but a clean scan is not a guarantee.",
        "Make sure you trust whoever sent the link, especially if it asks you to sign in.",
    ],
}
STRENGTH = ((3.0, "Major"), (1.5, "Strong"), (0.6, "Moderate"), (0.0, "Minor"))

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm

_fonts = False


def _register_fonts() -> None:
    global _fonts
    if _fonts:
        return
    for name, file in (("In", "Inter-Regular.ttf"), ("In-Md", "Inter-Medium.ttf"),
                       ("In-Sb", "Inter-SemiBold.ttf"), ("Mono", "JetBrainsMono-Regular.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
    from reportlab.lib.fonts import addMapping
    addMapping("In", 0, 0, "In")
    addMapping("In", 1, 0, "In-Sb")
    addMapping("In", 0, 1, "In")
    addMapping("In", 1, 1, "In-Sb")
    _fonts = True


def _sentence(text: str) -> str:
    """Capital first letter and a full stop, except that a sentence opening
    with a domain keeps the domain exactly as written: "github.com", not
    "Github.com", and above all not a homograph's Cyrillic "а" turned into
    a capital that hides the trick."""
    text = (text or "").strip()
    if not text:
        return text
    if "." not in text.split(" ", 1)[0]:
        text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def _t(v) -> str:
    return escape(str(v)) if v not in (None, "") else ""


def _styles() -> dict:
    def st(name, font="In", size=9, lead=None, color=INK_2, **kw):
        return ParagraphStyle(name, fontName=font, fontSize=size, leading=lead or size * 1.45,
                              textColor=color, **kw)
    return {
        "reason": st("reason", "In-Md", 13.5, 18.5, INK),
        "h2": st("h2", "In-Sb", 12.5, 16, INK, spaceBefore=0, spaceAfter=0),
        "caption": st("caption", "In", 8.4, 12, MUTED),
        "body": st("body", "In", 9, 13.5, INK_2),
        "title": st("title", "In-Md", 9.6, 13, INK),
        "detail": st("detail", "In", 8.4, 12.2, MUTED),
        "label": st("label", "In", 7.6, 10, MUTED),
        "value": st("value", "In-Md", 10.5, 13, INK),
        "k": st("k", "In", 8.4, 11.5, MUTED),
        "v": st("v", "In", 8.4, 11.5, INK),
        "vm": st("vm", "Mono", 7.8, 11.5, INK, wordWrap="CJK"),
        "link": st("link", "Mono", 8.2, 12, INK_2, wordWrap="CJK"),
        "advice": st("advice", "In", 9, 13.5, INK_2),
        "block": st("block", "In-Sb", 9, 12, INK),
    }


# ---- page chrome ------------------------------------------------------------------

def _logo(c, x, y, size):
    """The PhishLens mark: a violet tile with a lens and a check."""
    c.saveState()
    c.setFillColor(ACCENT)
    c.roundRect(x, y, size, size, size * 0.28, stroke=0, fill=1)
    s = size / 32
    c.setStrokeColor(colors.white)
    c.setLineCap(1)
    c.setLineJoin(1)
    c.setLineWidth(2.6 * s)
    c.circle(x + 14 * s, y + (32 - 14) * s, 7.4 * s, stroke=1, fill=0)
    c.setLineWidth(3 * s)
    c.line(x + 19.6 * s, y + (32 - 19.6) * s, x + 24.5 * s, y + (32 - 24.5) * s)
    c.setLineWidth(2.2 * s)
    p = c.beginPath()
    p.moveTo(x + 10.6 * s, y + (32 - 14.2) * s)
    p.lineTo(x + 12.9 * s, y + (32 - 16.5) * s)
    p.lineTo(x + 17.3 * s, y + (32 - 11.4) * s)
    c.drawPath(p, stroke=1, fill=0)
    c.restoreState()


def _canvas(generated: str, report_id: str, host: str):
    class Canvas(rl_canvas.Canvas):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._pages = []

        def showPage(self):
            self._pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._pages)
            for state in self._pages:
                self.__dict__.update(state)
                self._chrome(total)
                rl_canvas.Canvas.showPage(self)
            rl_canvas.Canvas.save(self)

        def _chrome(self, total):
            n = self._pageNumber
            left, right = MARGIN, PAGE_W - MARGIN
            top = PAGE_H - 13 * mm
            _logo(self, left, top - 2.2 * mm, 6.4 * mm)
            self.setFont("In-Sb", 10.5)
            self.setFillColor(INK)
            self.drawString(left + 8.6 * mm, top - 0.3 * mm, "PhishLens")
            self.setFont("In", 8)
            self.setFillColor(MUTED)
            self.drawRightString(right, top + 0.6 * mm, "Link scan report" if n == 1 else host)
            self.drawRightString(right, top - 3.4 * mm, generated)
            self.setStrokeColor(LINE)
            self.setLineWidth(0.6)
            self.line(left, top - 6 * mm, right, top - 6 * mm)
            # footer
            self.line(left, 14 * mm, right, 14 * mm)
            self.setFont("In", 7.4)
            self.setFillColor(FAINT)
            self.drawString(left, 9.6 * mm, f"Report {report_id}")
            self.drawCentredString(PAGE_W / 2, 9.6 * mm, f"Page {n} of {total}")
            self.drawRightString(right, 9.6 * mm, "Automated scan. Verify before acting on it.")
    return Canvas


# ---- flowables -------------------------------------------------------------------

class Hero(Flowable):
    """The verdict panel: tinted, with the score as the largest thing on
    the page, the reason beside it, and the link underneath."""

    def __init__(self, scan: dict, width: float, S: dict):
        super().__init__()
        self.scan, self.width, self.S = scan, width, S
        label, tone = VERDICT.get(scan.get("verdict"), ("Unknown", "neutral"))
        self.label, (self.fg, self.bg) = label, TONES[tone]
        self.left_w = 58 * mm
        self.pad = 7 * mm
        right_w = width - self.left_w - 2 * self.pad
        self.reason = Paragraph(_t(scan.get("summary")), S["reason"])
        link = _t(scan.get("url"))
        if scan.get("final_url") and scan["final_url"] != scan.get("url"):
            link += f'<br/><font color="#6c6b67">Lands on </font>{_t(scan["final_url"])}'
        self.link = Paragraph(link, S["link"])
        self.rw = right_w
        _, self.reason_h = self.reason.wrap(right_w, 400)
        _, self.link_h = self.link.wrap(right_w - 8 * mm, 400)
        # The findings that pushed hardest toward the verdict, as pills.
        toward = scan.get("verdict") != "safe"
        top = sorted((f for f in scan.get("findings") or [] if (f.get("weight", 0) > 0) == toward and f.get("weight")),
                     key=lambda f: -abs(f["weight"]))[:3]
        self.pills = self._layout_pills([f["title"] for f in top], right_w)
        pills_h = (self.pills[-1][1] + self.PILL_H) if self.pills else 0
        self.height = max(47 * mm, self.reason_h + pills_h + self.link_h + 2 * self.pad + 16 * mm)

    PILL_H = 5.6 * mm

    def _layout_pills(self, titles, width):
        out, x, y = [], 0.0, 0.0
        for t in titles:
            w = pdfmetrics.stringWidth(t, "In-Md", 7.8) + 6 * mm
            if x and x + w > width:
                x, y = 0.0, y + self.PILL_H + 1.6 * mm
            out.append((x, y, w, t))
            x += w + 1.8 * mm
        return out

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c, w, h = self.canv, self.width, self.height
        # panel
        c.setFillColor(colors.white)
        c.setStrokeColor(LINE)
        c.setLineWidth(0.7)
        c.roundRect(0, 0, w, h, 4 * mm, stroke=1, fill=1)
        p = c.beginPath()
        r = 4 * mm
        lw = self.left_w
        p.moveTo(r, 0)
        p.lineTo(lw, 0)
        p.lineTo(lw, h)
        p.lineTo(r, h)
        p.arcTo(0, h - 2 * r, 2 * r, h, startAng=90, extent=90)
        p.lineTo(0, r)
        p.arcTo(0, 0, 2 * r, 2 * r, startAng=180, extent=90)
        p.close()
        c.setFillColor(self.bg)
        c.drawPath(p, stroke=0, fill=1)
        c.setStrokeColor(LINE)
        c.line(lw, 0, lw, h)

        # left: verdict, score, meter
        x = self.pad
        top = h - self.pad
        c.setFillColor(self.fg)
        c.circle(x + 1.3 * mm, top - 2.2 * mm, 1.3 * mm, stroke=0, fill=1)
        c.setFont("In-Sb", 11)
        c.drawString(x + 4.4 * mm, top - 3.3 * mm, self.label)
        risk = int(self.scan.get("risk", 0))
        c.setFont("In-Sb", 46)
        c.drawString(x - 1, top - 22 * mm, str(risk))
        sw = pdfmetrics.stringWidth(str(risk), "In-Sb", 46)
        c.setFont("In", 9)
        c.setFillColor(MUTED)
        c.drawString(x + sw + 1.5 * mm, top - 22 * mm, "/100 risk")
        # meter
        mw = lw - 2 * self.pad
        my = self.pad + 4.2 * mm
        c.setFillColor(colors.Color(0, 0, 0, alpha=0.08))
        c.roundRect(x, my, mw, 1.6 * mm, 0.8 * mm, stroke=0, fill=1)
        c.setFillColor(self.fg)
        c.roundRect(x, my, max(1.6 * mm, mw * risk / 100), 1.6 * mm, 0.8 * mm, stroke=0, fill=1)
        c.setFillColor(self.bg)
        for t in (0.4, 0.7):
            c.rect(x + mw * t - 0.35 * mm, my - 0.2 * mm, 0.7 * mm, 2 * mm, stroke=0, fill=1)
        c.setFont("Mono", 6.4)
        c.setFillColor(MUTED)
        for t, s in ((0, "0"), (0.4, "40"), (0.7, "70"), (1, "100")):
            fn = c.drawString if t == 0 else c.drawRightString if t == 1 else c.drawCentredString
            fn(x + mw * t, my - 3.6 * mm, s)

        # right: reason, then the link in a soft box
        rx = lw + self.pad
        y = h - self.pad - self.reason_h
        self.reason.drawOn(c, rx, y)
        y -= 4 * mm
        for px, py, pw, title in self.pills:
            top_y = y - py - self.PILL_H
            c.setFillColor(self.bg)
            c.setStrokeColor(colors.Color(self.fg.red, self.fg.green, self.fg.blue, alpha=0.28))
            c.setLineWidth(0.6)
            c.roundRect(rx + px, top_y, pw, self.PILL_H, self.PILL_H / 2, stroke=1, fill=1)
            c.setFillColor(self.fg)
            c.setFont("In-Md", 7.8)
            c.drawString(rx + px + 3 * mm, top_y + 1.85 * mm, title)
        box_h = self.link_h + 4.6 * mm
        c.setFillColor(SOFT)
        c.setStrokeColor(LINE)
        c.roundRect(rx, self.pad, self.rw, box_h, 2 * mm, stroke=1, fill=1)
        self.link.drawOn(c, rx + 4 * mm, self.pad + 2.3 * mm)


class Dot(Flowable):
    def __init__(self, color, size=2.2 * mm):
        super().__init__()
        self.color, self.size = color, size

    def wrap(self, aw, ah):
        return self.size, self.size

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.circle(self.size / 2, self.size / 2 - 0.4, self.size / 2.6, stroke=0, fill=1)


class Impact(Flowable):
    """Direction and strength in words, with a small bar under it."""

    def __init__(self, weight: float, max_abs: float, width=26 * mm):
        super().__init__()
        self.w, self.max_abs, self.width, self.height = weight, max_abs or 1, width, 9.5 * mm

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        raises = self.w > 0
        word = next(s for t, s in STRENGTH if abs(self.w) >= t)
        color = TONES["danger" if raises else "ok"][0]
        c.setFont("In-Md", 8)
        c.setFillColor(color)
        c.drawRightString(self.width, self.height - 3.2 * mm, f"{'Raises' if raises else 'Lowers'} risk, {word.lower()}")
        bw = self.width * 0.62
        bx = self.width - bw
        by = self.height - 6.4 * mm
        c.setFillColor(SOFT)
        c.roundRect(bx, by, bw, 1.3 * mm, 0.65 * mm, stroke=0, fill=1)
        span = max(0.12, min(1, abs(self.w) / self.max_abs)) * bw
        c.setFillColor(color)
        c.roundRect(bx if raises else bx + bw - span, by, span, 1.3 * mm, 0.65 * mm, stroke=0, fill=1)


# ---- building blocks -----------------------------------------------------------------

def _facts(scan: dict, width: float, S: dict) -> Table:
    d, p = scan.get("domain") or {}, scan.get("page")
    listed = sum(1 for i in scan.get("intel") or [] if i.get("status") == "hit")
    checked = sum(1 for i in scan.get("intel") or [] if i.get("status") in ("hit", "clean"))
    age = d.get("age_days")
    age_s = ("Hosting platform" if d.get("hosted_platform") else "Unknown" if age is None
             else f"{age // 365} years" if age >= 730 else f"{age} days")
    cells = [
        ("Page", "Not fetched" if not p else f"HTTP {p.get('status')}" if p.get("fetched") else "Offline"),
        ("Domain age", age_s),
        ("Threat lists", f"{listed} listed" if listed else f"{checked} checked, clear"),
        ("Evidence", str(scan.get("evidence", "")).capitalize()),
        ("Scanned in", f"{scan.get('elapsed_ms', 0) / 1000:.1f} s"),
    ]
    data = [[[Paragraph(k, S["label"]), Spacer(1, 1.2 * mm), Paragraph(_t(v), S["value"])] for k, v in cells]]
    t = Table(data, colWidths=[width / 5] * 5)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, LINE), ("LINEBELOW", (0, 0), (-1, 0), 0.6, LINE),
        ("LINEBEFORE", (1, 0), (-1, 0), 0.6, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5 * mm), ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 3 * mm), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2 * mm),
    ]))
    return t


def _findings(scan: dict, width: float, S: dict) -> Table:
    fs = [f for f in scan.get("findings") or [] if f.get("weight")]
    max_abs = max((abs(f["weight"]) for f in fs), default=1)
    sev_tone = {"critical": "danger", "high": "danger", "medium": "warn", "low": "neutral",
                "info": "neutral", "good": "ok"}
    rows = []
    for f in fs:
        fg = TONES[sev_tone.get(f.get("severity"), "neutral")][0]
        rows.append([Dot(fg),
                     [Paragraph(_t(f["title"]), S["title"]), Spacer(1, 0.8 * mm), Paragraph(_t(_sentence(f["detail"])), S["detail"])],
                     Impact(f["weight"], max_abs)])
    if not rows:
        return Table([[Paragraph("No signal moved the score either way.", S["body"])]], colWidths=[width])
    t = Table(rows, colWidths=[5 * mm, width - 5 * mm - 30 * mm, 30 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (1, 0), (1, -1), 5 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5 * mm), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 * mm),
        ("TOPPADDING", (0, 0), (0, -1), 3.3 * mm),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, LINE),
    ]))
    return t


def _advice(scan: dict, width: float, S: dict) -> Table:
    tips = ADVICE.get(scan.get("verdict"), ADVICE["suspicious"])
    _, tone = VERDICT.get(scan.get("verdict"), ("", "neutral"))
    fg, bg = TONES[tone]
    inner = [[Paragraph("What to do", S["block"])]]
    for i, tip in enumerate(tips, 1):
        inner.append([Paragraph(f'<font name="In-Sb" color="{fg.hexval().replace("0x", "#")}">{i}</font>'
                                f'&nbsp;&nbsp;&nbsp;{_t(tip)}', S["advice"])])
    t = Table(inner, colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("ROUNDEDCORNERS", [3 * mm] * 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6 * mm), ("RIGHTPADDING", (0, 0), (-1, -1), 6 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2 * mm), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2 * mm),
        ("TOPPADDING", (0, 0), (-1, 0), 5 * mm), ("BOTTOMPADDING", (0, 0), (-1, 0), 2 * mm),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 5 * mm),
    ]))
    return t


def _kv(title: str, rows: list, width: float, S: dict):
    """A titled block of facts. Rows with no value are dropped, not printed
    as 'Not available'."""
    rows = [(k, v, kind) for k, v, kind in rows if v not in (None, "", [], "None")]
    if not rows:
        return None
    data = [[Paragraph(title, S["block"]), ""]]
    for k, v, kind in rows:
        data.append([Paragraph(_t(k), S["k"]), Paragraph(_t(v), S["vm" if kind == "mono" else "v"])])
    t = Table(data, colWidths=[width * 0.36, width * 0.64])
    t.setStyle(TableStyle([
        ("SPAN", (0, 0), (1, 0)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 1.35 * mm), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.35 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2.4 * mm),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, LINE),
    ]))
    return t


def _two_columns(blocks: list, width: float) -> Table:
    """Pack fact blocks into two columns, filling the shorter one each time."""
    gap = 9 * mm
    col_w = (width - gap) / 2
    cols: list[list] = [[], []]
    heights = [0.0, 0.0]
    for b in blocks:
        if b is None:
            continue
        _, h = b.wrap(col_w, 1000)
        i = 0 if heights[0] <= heights[1] else 1
        cols[i] += [b, Spacer(1, 6 * mm)]
        heights[i] += h + 6 * mm
    t = Table([[cols[0], "", cols[1]]], colWidths=[col_w, gap, col_w])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")] +
                          [(k, (0, 0), (-1, -1), 0) for k in ("LEFTPADDING", "RIGHTPADDING", "TOPPADDING",
                                                               "BOTTOMPADDING")]))
    return t


def _model_bars(scan: dict, width: float, S: dict):
    um = scan.get("url_model") or {}
    sig = [s for s in um.get("signals") or [] if s.get("contribution")][:6]
    if not sig:
        return None
    max_abs = max(abs(s["contribution"]) for s in sig)

    class Bars(Flowable):
        ROW = 5.6 * mm

        def wrap(self, aw, ah):
            return width, self.ROW * len(sig) + 2 * mm

        def draw(self):
            c = self.canv
            lab_w, val_w = width * 0.48, 14 * mm
            bx = lab_w + 4 * mm
            bw = width - bx - val_w - 3 * mm
            mid = bx + bw / 2
            for i, s in enumerate(sig):
                y = (len(sig) - 1 - i) * self.ROW + 2 * mm
                label = s["label"][:1].upper() + s["label"][1:]
                c.setFont("In", 8)
                c.setFillColor(INK_2)
                while pdfmetrics.stringWidth(label, "In", 8) > lab_w and len(label) > 4:
                    label = label[:-2].rstrip() + "…"
                c.drawString(0, y + 0.6 * mm, label)
                c.setFillColor(SOFT)
                c.roundRect(bx, y + 0.9 * mm, bw, 1.4 * mm, 0.7 * mm, stroke=0, fill=1)
                v = s["contribution"]
                span = abs(v) / max_abs * bw / 2
                c.setFillColor(TONES["danger" if v > 0 else "ok"][0])
                c.roundRect(mid if v > 0 else mid - span, y + 0.9 * mm, max(span, 0.8), 1.4 * mm, 0.7 * mm, stroke=0, fill=1)
                c.setStrokeColor(FAINT)
                c.setLineWidth(0.5)
                c.line(mid, y + 0.3 * mm, mid, y + 2.9 * mm)
                c.setFont("Mono", 7.2)
                c.setFillColor(MUTED)
                c.drawRightString(width, y + 0.6 * mm, f"{v:+.2f}")

    score = float(um.get("ml_score", 0)) * 100
    thr = float(um.get("decision_threshold", 0)) * 100
    return KeepTogether([
        Paragraph("Link text model", S["block"]),
        Spacer(1, 1.5 * mm),
        Paragraph(f"The lexical model scores the address {score:.1f}% on its shape alone, against a "
                  f"{thr:.1f}% threshold. Green bars pull toward safe, red toward phishing. In the overall "
                  f"score its influence is capped.", S["caption"]),
        Spacer(1, 3 * mm),
        Bars(),
    ])


# ---- report ----------------------------------------------------------------------------

def build_scan_report_pdf(scan: dict, generated_at: datetime | None = None) -> bytes:
    _register_fonts()
    S = _styles()
    when = generated_at or datetime.now(timezone.utc)
    generated = when.strftime("%-d %B %Y, %H:%M UTC")
    report_id = scan.get("id") or uuid.uuid4().hex[:12]
    host = report_host(scan.get("url", "")) or scan.get("url", "")

    buf = io.BytesIO()
    # Platypus frames pad their content by 6 pt; pulling the margins out by
    # the same amount puts the content edge exactly on the header rule.
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=MARGIN - 6, rightMargin=MARGIN - 6,
                            topMargin=26 * mm, bottomMargin=22 * mm,
                            title=f"PhishLens scan: {host}", author="PhishLens", creator="PhishLens",
                            subject=f"Link scan of {scan.get('url', '')}")
    W = PAGE_W - 2 * MARGIN
    story = [
        Hero(scan, W, S),
        Spacer(1, 5 * mm),
        _facts(scan, W, S),
        Spacer(1, 6 * mm),
        KeepTogether([_advice(scan, W, S)]),
        Spacer(1, 9 * mm),
        KeepTogether([
            Paragraph("Why this score", S["h2"]),
            Spacer(1, 1.5 * mm),
            Paragraph("Each finding raises or lowers the risk. Together they make up the score.", S["caption"]),
            Spacer(1, 2 * mm),
        ]),
        _findings(scan, W, S),
    ]

    # ---- appendix ------------------------------------------------------------------
    d, p = scan.get("domain") or {}, scan.get("page") or {}
    a = p.get("analysis") or {}
    hops = " → ".join(report_host(h["url"]) or h["url"] for h in p.get("hops") or [])
    page_block = _kv("Page", [
        ("Loaded", f"Yes, HTTP {p.get('status')}" if p.get("fetched") else (p.get("error") or None), "text"),
        ("Redirect path", hops if len(p.get("hops") or []) > 1 else None, "mono"),
        ("Encryption", "Plain http" if p.get("fetched") and p.get("tls_valid") is None
         else "Certificate verified" if p.get("tls_valid") else p.get("tls_error"), "text"),
        ("Title", a.get("title"), "text"),
        ("Asks for a password", "Yes" if a.get("has_password") else None, "text"),
        ("Sensitive fields", ", ".join(a.get("sensitive_fields") or []), "mono"),
        ("Forms send to", ", ".join(sorted({f.get("exfil") or f.get("action_domain") for f in a.get("forms") or []
                                             if f.get("exfil") or f.get("action_domain")})), "mono"),
        ("Presents itself as", ", ".join(a.get("brand_claims") or []), "text"),
        ("Most-linked sites", ", ".join(list(a.get("link_domains") or {})[:4]), "mono"),
    ], (W - 9 * mm) / 2, S) if scan.get("page") else None
    domain_block = _kv("Domain", [
        ("Domain", d.get("domain"), "mono"),
        ("Registered", f"{d['registered']} ({d['age_days']:,} days ago)" if d.get("registered") else d.get("rdap_error"), "text"),
        ("Registrar", d.get("registrar"), "text"),
        ("Expires", d.get("expires"), "text"),
        ("IP addresses", ", ".join(d.get("a_records") or []), "mono"),
        ("Nameservers", ", ".join(d.get("nameservers") or []), "mono"),
    ], (W - 9 * mm) / 2, S)
    cert_block = _kv("Certificate", [
        ("Issued to", d.get("cert_subject"), "text"),
        ("Issued by", d.get("cert_issuer") or d.get("cert_error"), "text"),
        ("Valid", f"{d['cert_not_before']} to {d['cert_not_after']}" if d.get("cert_not_before") else None, "text"),
        ("Covers", ", ".join((d.get("cert_sans") or [])[:6]), "mono"),
    ], (W - 9 * mm) / 2, S)
    status = {"hit": "Listed", "clean": "Not listed", "unavailable": "Unavailable", "not_configured": "Not set up"}
    intel_block = _kv("Threat lists", [
        (i["source"], status.get(i["status"], i["status"]) + (f". {i['detail'][:1].upper()}{i['detail'][1:]}" if i.get("detail") else ""), "text")
        for i in scan.get("intel") or []
    ], (W - 9 * mm) / 2, S)

    from reportlab.platypus import CondPageBreak
    story += [
        # A new page only when what's left of this one can't hold a start.
        CondPageBreak(90 * mm),
        Spacer(1, 8 * mm),
        Paragraph("Technical details", S["h2"]),
        Spacer(1, 1.5 * mm),
        Paragraph("The raw facts behind the findings. The page was read as text only; none of its scripts ran. "
                  "Scores of 0 to 39 look safe, 40 to 69 are suspicious, and 70 or more are phishing.",
                  S["caption"]),
        Spacer(1, 6 * mm),
        _two_columns([page_block, domain_block, intel_block, cert_block], W),
    ]
    bars = _model_bars(scan, W, S)
    if bars:
        story += [Spacer(1, 2 * mm), bars]
    doc.build(story, canvasmaker=_canvas(generated, report_id, host))
    return buf.getvalue()
