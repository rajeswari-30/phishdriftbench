"""Feature-provenance taxonomy (Section IV of main.tex / C3).

Every feature used anywhere in the reimplemented baselines is classified into
exactly one of three provenance classes:

  P1  static-lexical      computable from the URL string alone, no network access
  P2  lookup-at-inference requires a live network operation (WHOIS, DNS, SSL,
                           domain age, shortened-URL expansion)
  P3  third-party reputation requires an external ranking/index service
                           (Alexa/Tranco rank, PageRank, search-engine indexing)

This module does *not* perform live lookups by default. `LiveLookupClient`
below is a network-touching implementation kept separate and opt-in, because
(a) resolving WHOIS/DNS against a corpus collected years earlier reproduces
the retrospective-lookup artifact this project is measuring, not something
that should happen silently, and (b) shortener expansion contacts
attacker-controlled infrastructure in the phishing case (Sec. VI-C /
Li et al.'s cloaking taxonomy) and must never run unattended.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Provenance(str, Enum):
    P1_STATIC_LEXICAL = "P1"
    P2_LOOKUP_AT_INFERENCE = "P2"
    P3_THIRD_PARTY_REPUTATION = "P3"


# All P1 features are exactly the LexicalFeatures fields in features/lexical.py.
from phishdriftbench.features.lexical import LexicalFeatures  # noqa: E402

P1_FEATURES = set(LexicalFeatures.field_names())

P2_FEATURES = {
    "domain_age_days",
    "whois_registration_date",
    "whois_expiry_date",
    "whois_registrar_country",
    "dns_a_record_count",
    "dns_mx_record_present",
    "dns_ns_record_count",
    "ssl_certificate_present",
    "ssl_certificate_issuer",
    "ssl_certificate_age_days",
    "shortener_expanded_target",
    "hosting_asn",
    "hosting_country",
}

P3_FEATURES = {
    "tranco_rank",
    "alexa_rank",
    "pagerank_score",
    "search_engine_indexed",
}


@dataclass
class ProvenanceAudit:
    p1: list[str]
    p2: list[str]
    p3: list[str]
    unknown: list[str]

    @property
    def has_lookup_dependency(self) -> bool:
        return bool(self.p2 or self.p3)


def classify(feature_names: list[str]) -> ProvenanceAudit:
    """Partition a list of feature column names by provenance class."""
    p1, p2, p3, unknown = [], [], [], []
    for name in feature_names:
        if name in P1_FEATURES:
            p1.append(name)
        elif name in P2_FEATURES:
            p2.append(name)
        elif name in P3_FEATURES:
            p3.append(name)
        else:
            unknown.append(name)
    return ProvenanceAudit(p1=p1, p2=p2, p3=p3, unknown=unknown)


def ablate(df, keep: Provenance | tuple[Provenance, ...] = (Provenance.P1_STATIC_LEXICAL,)):
    """Return `df` restricted to columns whose provenance is in `keep`."""
    import pandas as pd

    if isinstance(keep, Provenance):
        keep = (keep,)
    audit = classify(list(df.columns))
    keep_cols = []
    if Provenance.P1_STATIC_LEXICAL in keep:
        keep_cols += audit.p1
    if Provenance.P2_LOOKUP_AT_INFERENCE in keep:
        keep_cols += audit.p2
    if Provenance.P3_THIRD_PARTY_REPUTATION in keep:
        keep_cols += audit.p3
    return df[keep_cols]


class LiveLookupClient:
    """Opt-in, rate-limited P2 lookups. NOT invoked anywhere by default.

    Each method performs a real network call. Call sites must pass
    ``i_understand_this_contacts_external_infrastructure=True`` to make the
    safety implication explicit at every call site, since expanding a
    shortened phishing URL contacts attacker infrastructure directly
    (Sec. Ethical Considerations, main.tex).
    """

    def __init__(self, timeout_s: float = 3.0, rate_limit_per_s: float = 2.0):
        self.timeout_s = timeout_s
        self._min_interval = 1.0 / rate_limit_per_s
        self._last_call = 0.0

    def _throttle(self):
        import time

        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def whois_lookup(self, domain: str, *, i_understand_this_contacts_external_infrastructure: bool = False) -> dict:
        if not i_understand_this_contacts_external_infrastructure:
            raise RuntimeError("Set the confirmation flag to perform a live WHOIS lookup.")
        self._throttle()
        import whois  # python-whois; optional dependency, install only if this path is used

        def _first(value):
            # Some registrars/parsers return a list of dates rather than one.
            if isinstance(value, (list, tuple)):
                return value[0] if value else None
            return value

        try:
            w = whois.whois(domain)
            creation = _first(w.creation_date)
            success = creation is not None
            return {
                "whois_success": int(success),
                "whois_registration_date": creation,
                "whois_expiry_date": _first(w.expiration_date),
                "whois_registrar_country": getattr(w, "country", None),
            }
        except Exception as e:
            return {"whois_success": 0, "whois_registration_date": None, "whois_expiry_date": None,
                    "whois_registrar_country": None, "whois_error": str(e)[:100]}

    def dns_lookup(self, domain: str, *, i_understand_this_contacts_external_infrastructure: bool = False) -> dict:
        if not i_understand_this_contacts_external_infrastructure:
            raise RuntimeError("Set the confirmation flag to perform a live DNS lookup.")
        self._throttle()
        import dns.resolver  # dnspython; optional dependency

        out = {"dns_a_record_count": 0, "dns_mx_record_present": 0, "dns_ns_record_count": 0}
        try:
            out["dns_a_record_count"] = len(dns.resolver.resolve(domain, "A", lifetime=self.timeout_s))
        except Exception:
            pass
        try:
            out["dns_mx_record_present"] = int(bool(dns.resolver.resolve(domain, "MX", lifetime=self.timeout_s)))
        except Exception:
            pass
        try:
            out["dns_ns_record_count"] = len(dns.resolver.resolve(domain, "NS", lifetime=self.timeout_s))
        except Exception:
            pass
        out["dns_success"] = int(out["dns_a_record_count"] > 0)
        return out

    def ssl_lookup(self, domain: str, *, i_understand_this_contacts_external_infrastructure: bool = False) -> dict:
        """A TLS handshake on port 443 to read certificate metadata only --
        no HTTP request is made and no page content is ever fetched."""
        if not i_understand_this_contacts_external_infrastructure:
            raise RuntimeError("Set the confirmation flag to perform a live SSL/TLS lookup.")
        self._throttle()
        import socket
        import ssl as ssl_module

        ctx = ssl_module.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl_module.CERT_NONE
        try:
            with socket.create_connection((domain, 443), timeout=self.timeout_s) as sock:
                with ctx.wrap_socket(sock, server_hostname=domain) as tls:
                    der_cert = tls.getpeercert(binary_form=True)
            # getpeercert()'s parsed dict is only populated under full
            # verification; with CERT_NONE (required here, since many
            # phishing hosts have self-signed/expired/mismatched certs) it
            # comes back empty, so the DER form is parsed directly instead.
            import pandas as pd
            from cryptography import x509

            cert = x509.load_der_x509_certificate(der_cert)
            not_before = getattr(cert, "not_valid_before_utc", None) or cert.not_valid_before
            not_before_ts = pd.Timestamp(not_before)
            if not_before_ts.tzinfo is None:
                not_before_ts = not_before_ts.tz_localize("UTC")
            age_days = (pd.Timestamp.now(tz="UTC") - not_before_ts).days
            return {"ssl_present": 1, "ssl_not_before": str(not_before), "ssl_certificate_age_days": age_days}
        except Exception as e:
            return {"ssl_present": 0, "ssl_not_before": None, "ssl_certificate_age_days": None,
                    "ssl_error": str(e)[:100]}
