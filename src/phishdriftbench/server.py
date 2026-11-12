"""PhishLens HTTP API.

    POST /api/scan     {"url": "...", "deep": true}   full scan (scan/engine.py)
    POST /api/check    {"url": "..."}                  URL-text model only, no network
    GET  /api/report   ?url=...&download=1            PDF of a full scan
    GET  /api/health                                  what is configured

Run locally:
    PYTHONPATH=src uvicorn phishdriftbench.server:app --port 8000 --reload
On Vercel, api/index.py imports `app` from here.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from phishdriftbench.demo import model as url_model
from phishdriftbench.demo import report as report_pdf
from phishdriftbench.demo import scan_report
from phishdriftbench.demo import trust
from phishdriftbench.scan import engine, judge, netsafe

app = FastAPI(title="PhishLens API", version="2.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json")

_origins = [o.strip() for o in os.environ.get("PHISHLENS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"^chrome-extension://[a-p]{32}$",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Per-client sliding window. In-process only: on a serverless host each
# instance keeps its own window, which still stops one client hammering
# one instance. Put a real limiter (Vercel firewall, Upstash) in front for
# anything public and busy.
RATE_LIMIT = int(os.environ.get("PHISHLENS_RATE_LIMIT", "30"))  # scans per minute per client
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _client(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


def _rate_limit(request: Request) -> None:
    now = time.monotonic()
    key = _client(request)
    with _hits_lock:
        q = _hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            raise HTTPException(429, "Too many scans. Wait a minute and try again.")
        q.append(now)
        if len(_hits) > 10_000:  # forget idle clients
            for k in [k for k, v in _hits.items() if not v]:
                del _hits[k]


class ScanRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)
    deep: bool = True


def _scan_or_400(url: str, deep: bool) -> dict:
    try:
        return engine.scan(url, deep=deep)
    except trust.BadURL as e:
        raise HTTPException(400, str(e)) from e
    except netsafe.UnsafeTarget as e:
        raise HTTPException(400, f"PhishLens does not scan private or internal addresses ({e}).") from e


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "version": app.version,
        "sources": {
            "openphish": True,
            "safe_browsing": bool(os.environ.get("GOOGLE_SAFE_BROWSING_KEY")),
            "urlhaus": bool(os.environ.get("URLHAUS_AUTH_KEY")),
            "virustotal": bool(os.environ.get("VIRUSTOTAL_API_KEY")),
            "ai": judge.enabled(),
        },
    }


@app.post("/api/scan")
def scan(body: ScanRequest, request: Request) -> dict:
    _rate_limit(request)
    return _scan_or_400(body.url, body.deep)


@app.post("/api/check")
def check(body: ScanRequest, request: Request) -> dict:
    """The original URL-only verdict, kept for the browser extension and
    anyone who wants an answer without contacting the link."""
    _rate_limit(request)
    try:
        return url_model.predict_and_explain(body.url, engine.get_model())
    except trust.BadURL as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/report")
def report(request: Request, url: str = Query(min_length=1, max_length=4096), download: bool = False) -> Response:
    _rate_limit(request)
    result = _scan_or_400(url, True)  # the scan cache makes the usual scan-then-report pair cost one scan
    pdf = scan_report.build_scan_report_pdf(result)
    name = report_pdf.report_filename(result["url"])
    return Response(pdf, media_type="application/pdf", headers={
        "Content-Disposition": f'{"attachment" if download else "inline"}; filename="{name}"',
        "Cache-Control": "private, no-store",
    })
