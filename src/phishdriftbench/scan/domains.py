"""Facts about a domain that the URL text cannot show: how old it is, who
issued its certificate and when, and whether it resolves at all.

Domain age is the single most useful of these. Phishing domains are
typically registered days before use and abandoned within weeks; a domain
that has existed for ten years is very rarely a phishing kit's own domain
(it may be a compromised site, which the page scan covers instead).

The paper's provenance audit found these lookup features add nothing to a
model trained on archived URLs, because the lookups were made years after
the fact (C3). Made at scan time, on a live link, they measure what they
claim to: that is the difference between a feature leaking label
information from the future and a feature describing the present.
"""
from __future__ import annotations

import socket
import ssl
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import tldextract
import urllib3

from phishdriftbench.scan import netsafe

_extract = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
_http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=3.0, read=4.0), retries=urllib3.Retry(
    total=3, redirect=3, status=0), headers={"Accept": "application/rdap+json, application/json",
                                             "User-Agent": "PhishLens/2.0"})


def registrable(host: str) -> str:
    """Registrable domain, treating hosting platforms (PSL private section)
    as their own registries: `evil.web.app` is its own domain."""
    return _extract(host).top_domain_under_public_suffix.lower()


def icann_domain(host: str) -> str:
    """Registrable domain under the ICANN section only: the thing that has
    a WHOIS/RDAP record (`vercel.app`, not `site.vercel.app`)."""
    return tldextract.extract(host).top_domain_under_public_suffix.lower()


@dataclass
class DomainInfo:
    domain: str
    registered: str | None = None  # ISO date
    age_days: int | None = None
    expires: str | None = None
    registrar: str | None = None
    rdap_error: str | None = None
    a_records: list[str] = field(default_factory=list)
    nameservers: list[str] = field(default_factory=list)
    mx: bool | None = None
    dns_error: str | None = None
    cert_issuer: str | None = None
    cert_subject: str | None = None
    cert_not_before: str | None = None
    cert_not_after: str | None = None
    cert_age_days: int | None = None
    cert_sans: list[str] = field(default_factory=list)
    cert_error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def rdap(domain: str, info: DomainInfo) -> None:
    """Registration date, expiry and registrar from RDAP (the JSON
    successor to WHOIS). rdap.org redirects to the authoritative registry."""
    try:
        r = _http.request("GET", f"https://rdap.org/domain/{domain}")
    except urllib3.exceptions.HTTPError as e:
        info.rdap_error = f"lookup failed ({e.__class__.__name__})"
        return
    if r.status == 404:
        info.rdap_error = "no registration record found"
        return
    if r.status != 200:
        info.rdap_error = f"registry returned HTTP {r.status}"
        return
    try:
        data = r.json()
    except ValueError:
        info.rdap_error = "registry returned an unreadable response"
        return
    for event in data.get("events", []):
        action = event.get("eventAction")
        when = _parse_date(event.get("eventDate"))
        if not when:
            continue
        if action == "registration":
            info.registered = when.date().isoformat()
            info.age_days = (datetime.now(timezone.utc) - when).days
        elif action == "expiration":
            info.expires = when.date().isoformat()
    for ent in data.get("entities", []):
        if "registrar" in (ent.get("roles") or []):
            for item in (ent.get("vcardArray") or [None, []])[1]:
                if item and item[0] == "fn":
                    info.registrar = str(item[3])[:80]
    if info.registered is None and not info.rdap_error:
        info.rdap_error = "registry does not publish a registration date"


def dns(host: str, info: DomainInfo) -> None:
    import dns.resolver

    resolver = dns.resolver.Resolver()
    resolver.lifetime = 3.0
    try:
        info.a_records = [r.address for r in resolver.resolve(host, "A")][:6]
    except Exception as e:  # NXDOMAIN, NoAnswer, Timeout...
        info.dns_error = e.__class__.__name__
    try:
        info.nameservers = sorted(str(r.target).rstrip(".") for r in resolver.resolve(info.domain, "NS"))[:4]
    except Exception:
        pass
    try:
        info.mx = bool(resolver.resolve(info.domain, "MX"))
    except Exception:
        info.mx = False


def certificate(host: str, info: DomainInfo, port: int = 443) -> None:
    """Certificate identity from a TLS handshake to the checked address.
    Handshake only: no HTTP request."""
    from cryptography import x509
    from cryptography.x509.oid import ExtensionOID, NameOID

    try:
        ip = netsafe.resolve_public(host, port)
    except (LookupError, netsafe.UnsafeTarget) as e:
        info.cert_error = str(e)
        return
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((ip, port), timeout=4.0) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as tls:
                der = tls.getpeercert(binary_form=True)
    except (OSError, ssl.SSLError) as e:
        info.cert_error = "no TLS on port 443" if isinstance(e, ConnectionRefusedError) else str(e)[:120]
        return
    cert = x509.load_der_x509_certificate(der)

    def _cn(name):
        attrs = name.get_attributes_for_oid(NameOID.ORGANIZATION_NAME) or \
            name.get_attributes_for_oid(NameOID.COMMON_NAME)
        return attrs[0].value if attrs else None

    info.cert_issuer = _cn(cert.issuer)
    info.cert_subject = _cn(cert.subject)
    nb, na = cert.not_valid_before_utc, cert.not_valid_after_utc
    info.cert_not_before, info.cert_not_after = nb.date().isoformat(), na.date().isoformat()
    info.cert_age_days = (datetime.now(timezone.utc) - nb).days
    try:
        san = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
        info.cert_sans = san.value.get_values_for_type(x509.DNSName)[:12]
    except x509.ExtensionNotFound:
        pass
