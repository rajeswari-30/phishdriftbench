import type { Metadata } from "next";
import { Badge, Card, CardHeader, cx, Metrics, TONE_BG, type Tone } from "@/components/ui";

export const metadata: Metadata = { title: "How it works" };

const CHECKS: { n: string; title: string; tone: Tone; tags: string[] }[] = [
  { n: "01", title: "Threat lists", tone: "danger", tags: ["OpenPhish", "Safe Browsing", "URLhaus", "VirusTotal"] },
  { n: "02", title: "Page scan", tone: "warn", tags: ["Password forms", "Fake brands", "Data drops", "Cloaking", "Host warnings"] },
  { n: "03", title: "Domain", tone: "accent", tags: ["Age", "Certificate", "DNS"] },
  { n: "04", title: "Link text", tone: "neutral", tags: ["XGBoost", "Look-alikes", "Squatting", "Shorteners"] },
  { n: "05", title: "AI review", tone: "ok", tags: ["Optional", "Brand match", "Never clears a link"] },
];

const SCALE: { range: string; label: string; tone: Tone; width: string }[] = [
  { range: "0–39", label: "Looks safe", tone: "ok", width: "40%" },
  { range: "40–69", label: "Suspicious", tone: "warn", width: "30%" },
  { range: "70–100", label: "Phishing", tone: "danger", width: "30%" },
];

export default function HowItWorks() {
  return (
    <div className="mx-auto max-w-3xl space-y-5 px-4 pt-10 pb-24 sm:px-6 sm:pt-16">
      <h1 className="enter text-center text-3xl font-semibold tracking-[-0.035em] sm:text-5xl">How it works</h1>

      <div className="grid gap-3 sm:grid-cols-2">
        {CHECKS.map((c, i) => (
          <Card key={c.n} className={cx("enter p-4", i === 0 && "sm:col-span-2")} style={{ animationDelay: `${i * 50}ms` }}>
            <div className="flex items-center gap-2.5">
              <span className={cx("size-2 rounded-full", TONE_BG[c.tone])} aria-hidden="true" />
              <h2 className="text-[15px] font-semibold">{c.title}</h2>
              <span className="ml-auto font-mono text-xs text-faint">{c.n}</span>
            </div>
            <ul className="mt-3 flex flex-wrap gap-1.5">
              {c.tags.map((t) => (
                <li key={t}>
                  <Badge>{t}</Badge>
                </li>
              ))}
            </ul>
          </Card>
        ))}
      </div>

      <Card className="enter overflow-hidden">
        <CardHeader title="Score" aside={<span className="font-mono">0–100</span>} />
        <div className="p-4">
          <div className="flex h-2 gap-1" aria-hidden="true">
            {SCALE.map((s) => (
              <span key={s.label} className={cx("rounded-full", TONE_BG[s.tone])} style={{ width: s.width }} />
            ))}
          </div>
          <div className="mt-3 flex gap-1">
            {SCALE.map((s) => (
              <div key={s.label} style={{ width: s.width }} className="min-w-0">
                <p className="truncate text-[13px] font-medium">{s.label}</p>
                <p className="font-mono text-xs text-muted">{s.range}</p>
              </div>
            ))}
          </div>
        </div>
      </Card>

      <Card className="enter overflow-hidden">
        <CardHeader title="Measured live" aside={<Badge>Lists off</Badge>} />
        <Metrics
          items={[
            { label: "Caught ≥70", value: "74%", tone: "ok" },
            { label: "False alarms ≥70", value: "0.7%" },
            { label: "Caught ≥40", value: "88%", tone: "ok" },
            { label: "False alarms ≥40", value: "2.0%" },
            { label: "Phishing links", value: "150" },
            { label: "Real sites", value: "150" },
          ]}
        />
      </Card>

      <div className="grid gap-3 sm:grid-cols-2">
        <Card className="enter overflow-hidden">
          <CardHeader title="Limits" />
          <ul className="flex flex-wrap gap-1.5 p-4">
            {["JS-only pages", "Targeted cloaking", "Low ≠ guaranteed"].map((t) => (
              <li key={t}>
                <Badge tone="warn">{t}</Badge>
              </li>
            ))}
          </ul>
        </Card>
        <Card className="enter overflow-hidden">
          <CardHeader title="Safety" />
          <ul className="flex flex-wrap gap-1.5 p-4">
            {["Scripts never run", "Private IPs blocked", "Redirects re-checked", "Size & time caps"].map((t) => (
              <li key={t}>
                <Badge tone="ok">{t}</Badge>
              </li>
            ))}
          </ul>
        </Card>
      </div>

      <Card className="enter overflow-hidden">
        <CardHeader title="API" aside={<a className="transition-colors hover:text-fg" href="/api/docs">Docs ↗</a>} />
        <ul className="divide-y divide-line font-mono text-[12.5px]">
          {[
            ["POST", "/api/scan", "Full scan"],
            ["POST", "/api/check", "Link text only"],
            ["GET", "/api/report", "PDF"],
            ["GET", "/api/health", "Sources"],
          ].map(([method, path, label]) => (
            <li key={path} className="flex items-center gap-3 px-4 py-2.5">
              <Badge tone={method === "POST" ? "accent" : "neutral"} className="w-12 justify-center font-mono text-[11px]">
                {method}
              </Badge>
              <span className="text-fg">{path}</span>
              <span className="ml-auto font-sans text-xs text-muted">{label}</span>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
