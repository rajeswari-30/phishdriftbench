"""Minimal local web demo for the phishing URL checker.

Run: PYTHONPATH=../src python app.py   (from inside webapp/)
or:  PYTHONPATH=src python webapp/app.py   (from the repo root)
Then open http://127.0.0.1:5001
"""
import sys
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tldextract
from flask import Flask, Response, jsonify, redirect, render_template, request

from phishdriftbench.demo import model
from phishdriftbench.demo import report as report_pdf
from phishdriftbench.demo import trust
from phishdriftbench.features import lexical
from phishdriftbench.provenance.taxonomy import LiveLookupClient

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True  # otherwise template edits need a process restart
_model = None
_lookup_client = LiveLookupClient(timeout_s=4.0, rate_limit_per_s=1.0)

# matplotlib (used for the SHAP force plot endpoint) is not thread-safe and
# Flask's dev server is threaded, so plotting is serialised.
_plot_lock = threading.Lock()

# Preview then Download is the normal sequence, and a report costs a few
# seconds of live WHOIS/DNS/TLS round trips. Keeping recent PDFs briefly makes
# the second click instant instead of repeating every lookup.
_REPORT_TTL_S = 600
_REPORT_MAX = 32
_report_cache: "OrderedDict[str, tuple[float, bytes, str]]" = OrderedDict()
_report_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        _model = model.load()
    return _model


@app.route("/")
def index():
    return render_template("home.html", page="home")


@app.route("/how-it-works")
def how_it_works():
    # The explanation now lives on the landing page, directly under the input.
    return redirect("/#how", code=302)


@app.route("/results")
def results():
    return render_template("results.html", page="results")


@app.route("/testing")
def testing():
    return render_template("testing.html", page="testing")


@app.route("/about")
def about():
    return render_template("about.html", page="about")


@app.route("/api/check", methods=["POST"])
def check():
    url = (request.get_json(silent=True) or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "Enter a URL to check."}), 400
    try:
        m = get_model()
    except FileNotFoundError:
        return jsonify({"error": "No trained model found. Run scripts/train_demo_model.py first."}), 500
    try:
        result = model.predict_and_explain(url, m)
    except trust.BadURL as e:
        # Unreadable input gets said plainly instead of scored: a verdict on
        # something that is not a URL would be confident nonsense.
        return jsonify({"error": str(e)}), 400
    result["report_name"] = report_pdf.report_filename(result["url"])
    return jsonify(result)


@app.route("/api/shap_plot", methods=["POST"])
def shap_plot():
    """Real SHAP force plot (exact TreeSHAP via the shap package) for the
    given URL, as a base64 PNG."""
    url = (request.get_json(silent=True) or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "Enter a URL to check."}), 400
    try:
        m = get_model()
    except FileNotFoundError:
        return jsonify({"error": "No trained model found. Run scripts/train_demo_model.py first."}), 500
    try:
        with _plot_lock:
            png_b64 = model.shap_force_plot_png(url, m)
    except trust.BadURL as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"png_base64": png_b64})


def _gather_context(raw: str) -> dict:
    """Registration / hosting / certificate context for a URL.

    IMPORTANT FRAMING: none of this is used by the detector. C3's audit
    measured exactly these lookup-based (P2) features and found they did NOT
    improve accuracy over the URL text alone (AUC -0.0051), which is why the
    model is deliberately lexical-only. It exists so a human investigator can
    see the surrounding facts while the verdict stays honestly attributable
    to the URL string. Raises ValueError if no domain can be parsed."""
    ext = tldextract.extract(raw)
    domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ""
    if not domain or "." not in domain:
        raise ValueError(f"Couldn't parse a domain from {raw!r}.")

    CONFIRM = True  # user-initiated action; see taxonomy.py
    whois_info = _lookup_client.whois_lookup(
        domain, i_understand_this_contacts_external_infrastructure=CONFIRM)
    dns_info = _lookup_client.dns_lookup_detail(
        domain, i_understand_this_contacts_external_infrastructure=CONFIRM)
    ssl_info = _lookup_client.ssl_lookup_detail(
        domain, i_understand_this_contacts_external_infrastructure=CONFIRM)

    def _clean(v):
        return None if v is None else str(v)

    return {
        "job_id": uuid.uuid4().hex[:12],
        "checked_at": datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M:%S UTC"),
        "source_url": raw,
        "domain": domain,
        "tld": ext.suffix or None,
        "subdomain": ext.subdomain or None,
        "brands_referenced": lexical.detected_brands(raw),
        "whois": {
            "success": bool(whois_info.get("whois_success")),
            "registration_date": _clean(whois_info.get("whois_registration_date")),
            "expiry_date": _clean(whois_info.get("whois_expiry_date")),
            "registrar_country": whois_info.get("whois_registrar_country"),
            "error": whois_info.get("whois_error"),
        },
        "dns": dns_info,
        "ssl": ssl_info,
    }


@app.route("/api/context", methods=["POST"])
def context():
    raw = (request.get_json(silent=True) or {}).get("url", "").strip()
    if not raw:
        return jsonify({"error": "Enter a URL."}), 400
    try:
        return jsonify(_gather_context(raw))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


def _cached_report(url: str):
    with _report_lock:
        hit = _report_cache.get(url)
        if hit and time.time() - hit[0] < _REPORT_TTL_S:
            _report_cache.move_to_end(url)
            return hit[1], hit[2]
        _report_cache.pop(url, None)
    return None


def _store_report(url: str, pdf: bytes, name: str) -> None:
    with _report_lock:
        _report_cache[url] = (time.time(), pdf, name)
        _report_cache.move_to_end(url)
        while len(_report_cache) > _REPORT_MAX:
            _report_cache.popitem(last=False)


@app.route("/report")
def report():
    """Full PDF report for one URL: why it decided, SHAP contributions, every
    measured signal, and the domain and certificate context. `download=1`
    sends it as an attachment; otherwise it opens in the browser's viewer."""
    url = (request.args.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Missing url."}), 400
    try:
        m = get_model()
    except FileNotFoundError:
        return jsonify({"error": "No trained model found. Run scripts/train_demo_model.py first."}), 500

    cached = _cached_report(url)
    if cached is None:
        try:
            result = model.predict_and_explain(url, m)
        except trust.BadURL as e:
            return jsonify({"error": str(e)}), 400
        try:
            ctx = _gather_context(result["url"])  # the repaired form, not the raw paste
        except Exception:
            ctx = None  # the report still renders; it says the lookups failed
        pdf = report_pdf.build_report_pdf(result, ctx)
        name = report_pdf.report_filename(result["url"])
        _store_report(url, pdf, name)
    else:
        pdf, name = cached

    disposition = "attachment" if request.args.get("download") == "1" else "inline"
    return Response(pdf, mimetype="application/pdf", headers={
        "Content-Disposition": f'{disposition}; filename="{name}"',
        "Cache-Control": "private, no-store",
    })


@app.route("/api/provenance", methods=["POST"])
def provenance():
    """Live demonstration of the C3 provenance audit's lookup capability
    (scripts/run_p2_lookups.py) -- NOT part of the phishing-checker model,
    which is deliberately lexical-only."""
    raw = (request.get_json(silent=True) or {}).get("domain", "").strip()
    if not raw:
        return jsonify({"error": "Enter a domain to look up."}), 400

    ext = tldextract.extract(raw)
    domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else raw
    if not domain or "." not in domain:
        return jsonify({"error": f"Couldn't parse a domain from {raw!r}."}), 400

    CONFIRM = True  # single user-initiated lookup per click; see taxonomy.py docstring
    whois = _lookup_client.whois_lookup(domain, i_understand_this_contacts_external_infrastructure=CONFIRM)
    dns = _lookup_client.dns_lookup(domain, i_understand_this_contacts_external_infrastructure=CONFIRM)
    ssl = _lookup_client.ssl_lookup(domain, i_understand_this_contacts_external_infrastructure=CONFIRM)

    def _clean(v):
        return None if v is None else str(v)

    return jsonify({
        "domain": domain,
        "whois_success": bool(whois.get("whois_success")),
        "whois_registration_date": _clean(whois.get("whois_registration_date")),
        "whois_registrar_country": whois.get("whois_registrar_country"),
        "dns_success": bool(dns.get("dns_success")),
        "dns_a_record_count": dns.get("dns_a_record_count"),
        "dns_mx_record_present": bool(dns.get("dns_mx_record_present")),
        "dns_ns_record_count": dns.get("dns_ns_record_count"),
        "ssl_present": bool(ssl.get("ssl_present")),
        "ssl_certificate_age_days": ssl.get("ssl_certificate_age_days"),
    })


if __name__ == "__main__":
    m = get_model()  # fail fast at startup if the model isn't trained yet
    model.shap_force_plot_png("https://example.com", m)  # warm matplotlib/shap for /api/shap_plot
    report_pdf._register_fonts()
    app.run(host="127.0.0.1", port=5001, debug=False)
