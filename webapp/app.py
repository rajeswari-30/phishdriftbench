"""Minimal local web demo for the phishing URL checker.

Run: PYTHONPATH=../src python app.py   (from inside webapp/)
or:  PYTHONPATH=src python webapp/app.py   (from the repo root)
Then open http://127.0.0.1:5001
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tldextract
from flask import Flask, jsonify, render_template, request

from phishdriftbench.demo import model
from phishdriftbench.provenance.taxonomy import LiveLookupClient

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True  # otherwise template edits need a process restart
_model = None
_lookup_client = LiveLookupClient(timeout_s=4.0, rate_limit_per_s=1.0)


def get_model():
    global _model
    if _model is None:
        _model = model.load()
    return _model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/check", methods=["POST"])
def check():
    url = (request.get_json(silent=True) or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "Enter a URL to check."}), 400
    try:
        m = get_model()
    except FileNotFoundError:
        return jsonify({"error": "No trained model found. Run scripts/train_demo_model.py first."}), 500
    result = model.predict_and_explain(url, m)
    return jsonify(result)


@app.route("/api/shap_plot", methods=["POST"])
def shap_plot():
    """Real SHAP force plot (exact TreeSHAP via the shap package, not just
    XGBoost's own pred_contribs) for the given URL. Separate endpoint from
    /api/check so the main verdict renders immediately and this loads in
    as a progressive enhancement -- the first plot in a fresh server
    process takes ~3s (matplotlib font-cache warmup), which get_model()'s
    startup warmup below avoids for the first real user request."""
    url = (request.get_json(silent=True) or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "Enter a URL to check."}), 400
    try:
        m = get_model()
    except FileNotFoundError:
        return jsonify({"error": "No trained model found. Run scripts/train_demo_model.py first."}), 500
    png_b64 = model.shap_force_plot_png(url, m)
    return jsonify({"png_base64": png_b64})


@app.route("/api/provenance", methods=["POST"])
def provenance():
    """Live demonstration of the C3 provenance audit's lookup capability
    (scripts/run_p2_lookups.py) -- NOT part of the phishing-checker model
    above, which is deliberately lexical-only. This performs one real
    WHOIS lookup, one real DNS resolution, and one real TLS handshake
    against whatever domain the user enters, so the underlying capability
    can be seen working directly rather than trusted from a CSV."""
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
    model.shap_force_plot_png("https://example.com", m)  # warm matplotlib/shap so the first real request isn't slow
    app.run(host="127.0.0.1", port=5001, debug=False)
