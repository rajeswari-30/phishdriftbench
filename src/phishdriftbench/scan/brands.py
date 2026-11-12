"""The brands phishing pages pretend to be, and the domains each one really
owns.

Page-level impersonation is the strongest single signal a scan can find: a
page whose title, logo or copyright line says "PayPal", which asks for a
password, and which is not served from a PayPal domain. Detecting the claim
needs the brand's name as people write it; refuting it needs the domains the
brand actually operates. Both live here.

Names are matched as whole words, case-insensitively. Names that are also
ordinary words ("Chase", "Steam", "Apple", "Discord") are flagged `common`
and only count when they appear in a strong position (title, logo, form
context), never from body text alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Brand:
    key: str
    name: str
    domains: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    common: bool = False  # name is also an everyday word

    @property
    def patterns(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


BRANDS: tuple[Brand, ...] = (
    Brand("paypal", "PayPal", ("paypal.com", "paypal.me", "paypalobjects.com")),
    Brand("apple", "Apple", ("apple.com", "icloud.com", "me.com"), ("Apple ID", "iCloud"), common=True),
    Brand("microsoft", "Microsoft", ("microsoft.com", "live.com", "office.com", "microsoftonline.com",
                                     "outlook.com", "office365.com", "sharepoint.com", "onedrive.com",
                                     "msauth.net", "msftauth.net", "windows.net", "azure.com", "bing.com",
                                     "xbox.com", "skype.com", "hotmail.com", "msn.com", "aka.ms"),
          ("Office 365", "Microsoft 365", "Outlook", "OneDrive", "SharePoint", "Hotmail")),
    Brand("google", "Google", ("google.com", "gmail.com", "youtube.com", "googleusercontent.com",
                               "gstatic.com", "googleapis.com", "google.co.in", "google.co.uk"),
          ("Gmail", "Google Drive", "Google Docs")),
    Brand("amazon", "Amazon", ("amazon.com", "amazon.in", "amazon.co.uk", "amazon.de", "amazon.ca",
                               "amazon.co.jp", "amazon.fr", "amazon.es", "amazon.it", "amazonaws.com",
                               "media-amazon.com", "ssl-images-amazon.com", "aws.amazon.com", "primevideo.com"),
          ("AWS", "Prime Video")),
    Brand("meta", "Facebook", ("facebook.com", "fb.com", "meta.com", "messenger.com", "fbcdn.net"), ("Meta",)),
    Brand("instagram", "Instagram", ("instagram.com", "cdninstagram.com")),
    Brand("whatsapp", "WhatsApp", ("whatsapp.com", "whatsapp.net", "wa.me")),
    Brand("netflix", "Netflix", ("netflix.com", "nflxext.com", "nflximg.net")),
    Brand("linkedin", "LinkedIn", ("linkedin.com", "licdn.com")),
    Brand("x", "Twitter", ("twitter.com", "x.com", "twimg.com")),
    Brand("dropbox", "Dropbox", ("dropbox.com", "dropboxstatic.com")),
    Brand("docusign", "DocuSign", ("docusign.com", "docusign.net")),
    Brand("adobe", "Adobe", ("adobe.com", "adobelogin.com", "typekit.net"), ("Adobe Acrobat", "Adobe Sign")),
    Brand("wetransfer", "WeTransfer", ("wetransfer.com", "we.tl")),
    Brand("chase", "Chase", ("chase.com", "jpmorganchase.com", "jpmorgan.com"), ("JPMorgan",), common=True),
    Brand("wellsfargo", "Wells Fargo", ("wellsfargo.com", "wf.com")),
    Brand("bankofamerica", "Bank of America", ("bankofamerica.com", "bofa.com")),
    Brand("citi", "Citibank", ("citi.com", "citibank.com", "citigroup.com")),
    Brand("capitalone", "Capital One", ("capitalone.com",)),
    Brand("amex", "American Express", ("americanexpress.com", "aexp.com", "amex.com")),
    Brand("hsbc", "HSBC", ("hsbc.com", "hsbc.co.uk", "hsbc.co.in")),
    Brand("barclays", "Barclays", ("barclays.co.uk", "barclays.com")),
    Brand("santander", "Santander", ("santander.com", "santander.co.uk")),
    Brand("natwest", "NatWest", ("natwest.com",)),
    Brand("lloyds", "Lloyds Bank", ("lloydsbank.com", "lloydsbank.co.uk")),
    Brand("revolut", "Revolut", ("revolut.com",)),
    Brand("venmo", "Venmo", ("venmo.com",)),
    Brand("zelle", "Zelle", ("zellepay.com",)),
    Brand("cashapp", "Cash App", ("cash.app", "squareup.com")),
    Brand("stripe", "Stripe", ("stripe.com",)),
    Brand("coinbase", "Coinbase", ("coinbase.com",)),
    Brand("binance", "Binance", ("binance.com",)),
    Brand("kraken", "Kraken", ("kraken.com",)),
    Brand("metamask", "MetaMask", ("metamask.io",)),
    Brand("trustwallet", "Trust Wallet", ("trustwallet.com",)),
    Brand("ledger", "Ledger", ("ledger.com",), common=True),
    Brand("dhl", "DHL", ("dhl.com", "dhl.de", "dhl.co.uk", "dhl.co.in")),
    Brand("fedex", "FedEx", ("fedex.com",)),
    Brand("ups", "UPS", ("ups.com",)),
    Brand("usps", "USPS", ("usps.com",)),
    Brand("royalmail", "Royal Mail", ("royalmail.com",)),
    Brand("indiapost", "India Post", ("indiapost.gov.in",)),
    Brand("irs", "IRS", ("irs.gov",)),
    Brand("hmrc", "HMRC", ("gov.uk",)),
    Brand("ebay", "eBay", ("ebay.com", "ebay.co.uk", "ebay.in", "ebayimg.com")),
    Brand("walmart", "Walmart", ("walmart.com",)),
    Brand("spotify", "Spotify", ("spotify.com", "scdn.co")),
    Brand("steam", "Steam", ("steampowered.com", "steamcommunity.com", "steamstatic.com"), common=True),
    Brand("roblox", "Roblox", ("roblox.com", "rbxcdn.com")),
    Brand("discord", "Discord", ("discord.com", "discord.gg", "discordapp.com"), common=True),
    Brand("telegram", "Telegram", ("telegram.org", "t.me")),
    Brand("yahoo", "Yahoo", ("yahoo.com", "yimg.com")),
    Brand("aol", "AOL", ("aol.com",)),
    Brand("att", "AT&T", ("att.com", "att.net")),
    Brand("verizon", "Verizon", ("verizon.com",)),
    Brand("tmobile", "T-Mobile", ("t-mobile.com",)),
    Brand("github", "GitHub", ("github.com", "githubusercontent.com", "github.io")),
    Brand("zoom", "Zoom", ("zoom.us", "zoom.com"), common=True),
    Brand("booking", "Booking.com", ("booking.com",)),
    Brand("airbnb", "Airbnb", ("airbnb.com", "airbnb.co.in")),
    Brand("sbi", "State Bank of India", ("sbi.co.in", "onlinesbi.sbi", "sbi.bank.in"), ("SBI", "YONO")),
    Brand("hdfc", "HDFC Bank", ("hdfcbank.com", "hdfc.com", "hdfcbank.bank.in")),
    Brand("icici", "ICICI Bank", ("icicibank.com", "icici.bank.in")),
    Brand("axis", "Axis Bank", ("axisbank.com", "axis.bank.in")),
    Brand("kotak", "Kotak Mahindra Bank", ("kotak.com", "kotak.bank.in"), ("Kotak",)),
    Brand("paytm", "Paytm", ("paytm.com", "paytm.in")),
    Brand("phonepe", "PhonePe", ("phonepe.com",)),
    Brand("airtel", "Airtel", ("airtel.in", "airtel.com")),
    Brand("jio", "Jio", ("jio.com",)),
    Brand("irctc", "IRCTC", ("irctc.co.in",)),
    Brand("incometax", "Income Tax Department", ("incometax.gov.in", "incometaxindia.gov.in")),
    Brand("uidai", "Aadhaar", ("uidai.gov.in",), ("UIDAI",)),
    Brand("epfo", "EPFO", ("epfindia.gov.in",)),
)

BY_KEY = {b.key: b for b in BRANDS}


@lru_cache(maxsize=1)
def _domain_owner() -> dict[str, str]:
    return {d: b.key for b in BRANDS for d in b.domains}


def owner_of(domain: str) -> str | None:
    """Brand key that owns this registrable domain (or a parent of it)."""
    owners = _domain_owner()
    d = domain.lower()
    while d:
        if d in owners:
            return owners[d]
        if "." not in d:
            return None
        d = d.split(".", 1)[1]
    return None


def belongs_to(brand_key: str, host_or_domain: str) -> bool:
    return owner_of(host_or_domain) == brand_key


@lru_cache(maxsize=1)
def _pattern() -> tuple[re.Pattern, dict[str, str]]:
    lookup = {}
    for b in BRANDS:
        for p in b.patterns:
            lookup[p.lower()] = b.key
    alternation = "|".join(sorted((re.escape(p) for p in lookup), key=len, reverse=True))
    return re.compile(rf"(?<![\w-])({alternation})(?![\w-])", re.IGNORECASE), lookup


def mentions(text: str, *, strong: bool = True) -> list[str]:
    """Brand keys named in `text`, most-mentioned first. With strong=False
    (body text), brands whose name is an everyday word are skipped."""
    if not text:
        return []
    regex, lookup = _pattern()
    counts: dict[str, int] = {}
    for m in regex.finditer(text):
        key = lookup[m.group(1).lower()]
        if not strong and BY_KEY[key].common:
            continue
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts, key=counts.get, reverse=True)


def from_url_token(token: str) -> str | None:
    """Map a brand token from the URL features (lexical.detected_brands,
    e.g. "paypal" or "oracle (misspelled)") to a brand key, if we know it."""
    t = token.split(" ")[0].lower()
    aliases = {"facebook": "meta", "outlook": "microsoft", "onedrive": "microsoft",
               "icloud": "apple", "hdfcbank": "hdfc", "icicibank": "icici",
               "axisbank": "axis", "americanexpress": "amex", "citibank": "citi",
               "twitter": "x", "incometaxindia": "incometax", "aadhaar": "uidai",
               "epfindia": "epfo"}
    t = aliases.get(t, t)
    return t if t in BY_KEY else None
