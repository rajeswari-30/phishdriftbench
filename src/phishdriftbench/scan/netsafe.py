"""Fetching a URL a stranger handed us, without letting it point us inward.

A scanner that fetches arbitrary links is an SSRF primitive by default: a
"phishing link" of `http://169.254.169.254/latest/meta-data/` or a domain
that resolves to 10.0.0.5 would have the server read its own network on the
attacker's behalf. Three rules close that:

  * every hostname is resolved here, and the request goes to the exact
    address that was checked (no second lookup the attacker can rebind);
  * an address is acceptable only if it is globally routable;
  * every redirect hop goes through the same check, because the first hop
    being public says nothing about the second.

Only GET, only http/https, only the usual web ports, a hard byte cap and a
wall-clock deadline. No script on the fetched page ever runs: this returns
bytes, and the parser downstream reads them as text.
"""
from __future__ import annotations

import ipaddress
import socket
import ssl
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

import certifi
import urllib3

ALLOWED_PORTS = {80, 443, 8080, 8443}
MAX_BYTES = 2_000_000
MAX_REDIRECTS = 6
# A desktop browser's UA: phishing kits routinely serve a blank page or a 404
# to anything that announces itself as a bot.
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


class UnsafeTarget(ValueError):
    """The URL points somewhere a public scanner must not go."""


def _is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_global and not addr.is_multicast


def resolve_public(host: str, port: int) -> str:
    """One public address for `host`, or UnsafeTarget.

    All resolved addresses must be public: a name that returns one public
    and one private record is either misconfigured or an attempt to win a
    race, and neither deserves a fetch."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as e:
        raise LookupError(f"{host} does not resolve") from e
    addrs = list(dict.fromkeys(info[4][0] for info in infos))
    if not addrs:
        raise LookupError(f"{host} does not resolve")
    for a in addrs:
        if not _is_public(a):
            raise UnsafeTarget(f"{host} resolves to a non-public address")
    # Prefer IPv4: many hosting setups have broken v6.
    return next((a for a in addrs if ":" not in a), addrs[0])


@dataclass
class Hop:
    url: str
    status: int | None
    ip: str | None


@dataclass
class FetchResult:
    requested_url: str
    final_url: str
    status: int | None = None
    content_type: str = ""
    body: bytes = b""
    truncated: bool = False
    hops: list[Hop] = field(default_factory=list)
    tls_valid: bool | None = None  # None for plain http
    tls_error: str | None = None
    error: str | None = None
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and self.status is not None

    @property
    def text(self) -> str:
        charset = "utf-8"
        ct = self.content_type.lower()
        if "charset=" in ct:
            charset = ct.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
        try:
            return self.body.decode(charset, errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")


def _target(url: str) -> tuple[str, str, int, str]:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeTarget(f"refusing to follow a {parts.scheme or 'schemeless'} link")
    host = parts.hostname
    if not host:
        raise UnsafeTarget("link has no host")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise UnsafeTarget(f"refusing to connect to port {port}")
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    return parts.scheme, host, port, path


def _pool(scheme: str, host: str, ip: str, port: int, verify: bool, timeout: urllib3.Timeout):
    if scheme == "http":
        return urllib3.HTTPConnectionPool(ip, port, timeout=timeout, retries=False, maxsize=1)
    if verify:
        return urllib3.HTTPSConnectionPool(
            ip, port, timeout=timeout, retries=False, maxsize=1,
            server_hostname=host, assert_hostname=host,
            cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())
    return urllib3.HTTPSConnectionPool(
        ip, port, timeout=timeout, retries=False, maxsize=1,
        server_hostname=host, assert_hostname=False, cert_reqs="CERT_NONE")


def fetch(url: str, *, deadline_s: float = 8.0, max_bytes: int = MAX_BYTES,
          max_redirects: int = MAX_REDIRECTS) -> FetchResult:
    """GET `url`, following redirects by hand so each hop is re-checked.

    Never raises for network trouble: failures land in `.error`, because an
    unreachable page is itself evidence. UnsafeTarget does propagate."""
    started = time.monotonic()
    result = FetchResult(requested_url=url, final_url=url)
    current = url
    for _ in range(max_redirects + 1):
        remaining = deadline_s - (time.monotonic() - started)
        if remaining <= 0.5:
            result.error = "timed out"
            break
        scheme, host, port, path = _target(current)
        try:
            ip = resolve_public(host, port)
        except LookupError as e:
            result.error = str(e)
            result.hops.append(Hop(current, None, None))
            break
        timeout = urllib3.Timeout(connect=min(4.0, remaining), read=min(6.0, remaining))
        headers = {
            "Host": host if port in (80, 443) else f"{host}:{port}",
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        verify = scheme == "https"
        resp = None
        try:
            resp = _pool(scheme, host, ip, port, verify, timeout).urlopen(
                "GET", path, headers=headers, redirect=False, preload_content=False)
            if scheme == "https" and result.tls_valid is None:
                result.tls_valid = True
        except urllib3.exceptions.SSLError as e:
            # A broken certificate is a finding, not a reason to stop
            # looking: read the page anyway, unverified, and say so.
            if result.tls_valid is None:
                result.tls_valid, result.tls_error = False, _short(e)
            try:
                resp = _pool(scheme, host, ip, port, False, timeout).urlopen(
                    "GET", path, headers=headers, redirect=False, preload_content=False)
            except (urllib3.exceptions.HTTPError, OSError, ssl.SSLError) as e2:
                result.error = _short(e2)
        except (urllib3.exceptions.HTTPError, OSError) as e:
            result.error = _short(e)
        if resp is None:
            result.hops.append(Hop(current, None, ip))
            break

        result.hops.append(Hop(current, resp.status, ip))
        location = resp.headers.get("Location")
        if 300 <= resp.status < 400 and location:
            resp.release_conn()
            current = urljoin(current, location.strip())
            result.final_url = current
            continue

        result.status = resp.status
        result.final_url = current
        result.content_type = resp.headers.get("Content-Type", "")
        chunks, size = [], 0
        try:
            for chunk in resp.stream(65536, decode_content=True):
                chunks.append(chunk)
                size += len(chunk)
                if size >= max_bytes:
                    result.truncated = True
                    break
                if time.monotonic() - started > deadline_s:
                    result.truncated = True
                    break
        except (urllib3.exceptions.HTTPError, OSError) as e:
            result.error = result.error or _short(e)
        finally:
            resp.release_conn()
        result.body = b"".join(chunks)[:max_bytes]
        break
    else:
        result.error = f"more than {max_redirects} redirects"

    result.elapsed_ms = int((time.monotonic() - started) * 1000)
    return result


def _short(e: BaseException) -> str:
    text = str(e) or e.__class__.__name__
    if "CERTIFICATE_VERIFY_FAILED" in text:
        return "certificate could not be verified"
    if "Hostname mismatch" in text or "doesn't match" in text:
        return "certificate does not match the host"
    if "timed out" in text.lower():
        return "timed out"
    if "Connection refused" in text:
        return "connection refused"
    return text[:160]
