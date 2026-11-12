import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = { title: "How it works" };

const LAYERS = [
  {
    n: "01",
    title: "Threat lists",
    body: "The link is checked against OpenPhish's live feed, and against Google Safe Browsing, URLhaus and VirusTotal when they are configured. A listing settles it: the score goes to 97 or above. Not being listed counts for very little, because most phishing links live for hours and are never reported.",
  },
  {
    n: "02",
    title: "The page, read safely",
    body: "Our server fetches the page's HTML, re-checking every redirect so a link can never point the scanner at a private network. Nothing on the page runs. We look for what a phishing page cannot do without: a password or card form, a claim to be a brand it is not hosted by, a form that posts to a Telegram bot or another site, links copied from the real brand, and the warning page a host such as Cloudflare puts up once it has confirmed a site is phishing.",
  },
  {
    n: "03",
    title: "The domain",
    body: "Registration age from RDAP, the TLS certificate, and DNS. Phishing domains are usually days old; a domain registered a decade ago is rarely a kit's own. On free hosting platforms like vercel.app or pages.dev the platform's age says nothing, so it is not used.",
  },
  {
    n: "04",
    title: "The link text",
    body: "The lexical XGBoost model from the PhishDriftBench paper scores the URL's shape, with rules ahead of it for look-alike characters, brand squatting, shorteners and pages anyone can publish. Its influence is capped: shape alone should never convict a link.",
  },
  {
    n: "05",
    title: "AI review (optional)",
    body: "When an Anthropic API key is configured, Claude reads the page's title, forms and visible text and says which brand it imitates and why. The page text is treated as untrusted evidence, and the review only adjusts the score: it can never mark a link safe on its own.",
  },
];

export default function HowItWorks() {
  return (
    <div className="mx-auto max-w-3xl px-4 pb-24 pt-16 sm:px-6 sm:pt-20">
      <p className="mb-5 font-mono text-[12.5px] text-muted">How it works</p>
      <h1 className="text-balance text-4xl font-semibold tracking-[-0.03em] sm:text-5xl">
        Five kinds of evidence, one score you can check.
      </h1>
      <p className="mt-5 max-w-xl text-lg leading-relaxed text-muted">
        Every check adds or subtracts a weight. The weights add up to the score, so the findings you see are the whole
        reason for it.
      </p>

      <ol className="mt-12 divide-y divide-rule border-y border-rule">
        {LAYERS.map((l) => (
          <li key={l.n} className="grid gap-2 py-7 sm:grid-cols-[64px_1fr]">
            <span className="font-mono text-sm text-faint">{l.n}</span>
            <div>
              <h2 className="text-lg font-semibold tracking-tight">{l.title}</h2>
              <p className="mt-2 leading-relaxed text-ink-2">{l.body}</p>
            </div>
          </li>
        ))}
      </ol>

      <section className="mt-14">
        <h2 className="text-2xl font-semibold tracking-tight">The score</h2>
        <div className="mt-5 grid gap-3 sm:grid-cols-3">
          {[
            ["0–39", "Looks safe", "bg-legit"],
            ["40–69", "Suspicious", "bg-unsure"],
            ["70–100", "Phishing", "bg-phish"],
          ].map(([range, label, dot]) => (
            <div key={label} className="rounded-xl border border-rule bg-card p-4">
              <div className="flex items-center gap-2 font-medium">
                <span className={`size-2 rounded-full ${dot}`} />
                {label}
              </div>
              <p className="mt-1 font-mono text-sm text-muted">{range}</p>
            </div>
          ))}
        </div>
        <p className="mt-5 leading-relaxed text-ink-2">
          On a live sample of 150 links from OpenPhish and 150 long-tail sites from Tranco, scored with the threat lists
          switched off so the feed could not grade itself, 74% of phishing scored 70 or more while 0.7% of real sites
          did; 88% of phishing scored 40 or more at 2% of real sites. With the lists on, anything already reported is
          caught outright. The script is <code className="font-mono text-sm">scripts/eval_scanner.py</code>.
        </p>
      </section>

      <section className="mt-14">
        <h2 className="text-2xl font-semibold tracking-tight">What it can&apos;t do</h2>
        <ul className="mt-4 list-disc space-y-2 pl-5 leading-relaxed text-ink-2">
          <li>See pages built entirely in JavaScript. Scripts never run, which keeps the scanner safe and blind to them.</li>
          <li>See what a kit shows only to its targets. Kits that check location or device can show scanners something else.</li>
          <li>Promise a link is safe. A low score means no evidence was found, not that there is none.</li>
        </ul>
      </section>

      <section className="mt-14">
        <h2 className="text-2xl font-semibold tracking-tight">API</h2>
        <pre className="mt-4 overflow-x-auto rounded-xl border border-rule bg-card p-4 font-mono text-[13px] leading-relaxed text-ink-2">
{`POST /api/scan    {"url": "https://…", "deep": true}
POST /api/check   {"url": "https://…"}        # link text only, no network
GET  /api/report?url=https://…&download=true  # PDF
GET  /api/health`}
        </pre>
        <p className="mt-3 text-sm text-muted">
          Interactive docs at <Link className="underline underline-offset-4" href="/api/docs">/api/docs</Link>.
        </p>
      </section>
    </div>
  );
}
