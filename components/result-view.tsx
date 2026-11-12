"use client";

import { useState } from "react";
import { reportUrl } from "@/lib/api";
import type { Finding, ScanResult, Severity, Verdict } from "@/lib/types";
import { RiskGauge } from "./risk-gauge";

const VERDICT: Record<Verdict, { label: string; box: string; text: string; advice: string }> = {
  phishing: {
    label: "Phishing",
    box: "border-phish/25 bg-phish-soft",
    text: "text-phish",
    advice: "Don't open this link or enter anything on the page it leads to.",
  },
  suspicious: {
    label: "Suspicious",
    box: "border-unsure/25 bg-unsure-soft",
    text: "text-unsure",
    advice: "Don't enter any details unless you can confirm the site another way.",
  },
  safe: {
    label: "Looks safe",
    box: "border-legit/25 bg-legit-soft",
    text: "text-legit",
    advice: "A low score means no warning signs were found, not a guarantee. Make sure you trust the sender.",
  },
};

const SEVERITY: Record<Severity, { dot: string; label: string }> = {
  critical: { dot: "bg-phish", label: "Critical" },
  high: { dot: "bg-phish", label: "High" },
  medium: { dot: "bg-unsure", label: "Medium" },
  low: { dot: "bg-faint", label: "Low" },
  info: { dot: "bg-rule-2", label: "Info" },
  good: { dot: "bg-legit", label: "Reassuring" },
};

const TABS = ["Findings", "Page", "Domain", "Threat lists", "Link text"] as const;
type Tab = (typeof TABS)[number];

export function ResultView({ result, onRescan }: { result: ScanResult; onRescan: () => void }) {
  const v = VERDICT[result.verdict];
  const [tab, setTab] = useState<Tab>("Findings");
  const [copied, setCopied] = useState(false);
  const moved = result.final_url && result.final_url !== result.url;

  return (
    <section className="rise mt-10" aria-label="Scan result">
      <div className={`rounded-2xl border p-5 sm:p-7 ${v.box}`}>
        <div className="flex flex-col gap-6 sm:flex-row sm:items-center">
          <RiskGauge risk={result.risk} verdict={result.verdict} />
          <div className="min-w-0 flex-1">
            <div className={`text-3xl font-semibold tracking-tight sm:text-4xl ${v.text}`}>{v.label}</div>
            <p className="mt-2 text-[15px] leading-relaxed text-ink-2">{result.summary}</p>
            <p className="mt-1 text-sm text-muted">{v.advice}</p>
            <div className="mt-4 space-y-1 font-mono text-[13px] text-muted">
              <p className="break-all">{result.url}</p>
              {moved && <p className="break-all">→ lands on {result.final_url}</p>}
            </div>
          </div>
        </div>
        <div className="mt-6 flex flex-wrap items-center gap-2 border-t border-ink/10 pt-4 text-sm">
          <EvidenceBadge level={result.evidence} />
          <span className="text-muted">
            {(result.elapsed_ms / 1000).toFixed(1)} s · {result.findings.filter((f) => f.weight !== 0).length} signals
          </span>
          <div className="flex w-full flex-wrap gap-2 sm:ml-auto sm:w-auto">
            <a href={reportUrl(result.input)} target="_blank" rel="noopener" className={btnGhost}>
              View report
            </a>
            <a href={reportUrl(result.input, true)} className={btnGhost}>
              Download PDF
            </a>
            <button
              type="button"
              className={btnGhost}
              onClick={async () => {
                try {
                  await navigator.clipboard.writeText(window.location.href);
                  setCopied(true);
                  setTimeout(() => setCopied(false), 1500);
                } catch {
                  /* clipboard blocked */
                }
              }}
            >
              {copied ? "Copied" : "Share"}
            </button>
            <button type="button" onClick={onRescan} className={btnGhost}>
              Rescan
            </button>
          </div>
        </div>
      </div>

      <div className="mt-8">
        <div role="tablist" className="-mx-4 flex gap-1 overflow-x-auto border-b border-rule px-4 sm:mx-0 sm:px-0">
          {TABS.map((t) => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className="relative whitespace-nowrap px-3 py-2.5 text-sm text-muted transition-colors hover:text-ink aria-selected:text-ink aria-selected:after:absolute aria-selected:after:inset-x-3 aria-selected:after:-bottom-px aria-selected:after:h-0.5 aria-selected:after:bg-ink"
            >
              {t}
            </button>
          ))}
        </div>
        <div className="pt-5" role="tabpanel">
          {tab === "Findings" && <Findings findings={result.findings} />}
          {tab === "Page" && <PagePanel result={result} />}
          {tab === "Domain" && <DomainPanel result={result} />}
          {tab === "Threat lists" && <IntelPanel result={result} />}
          {tab === "Link text" && <UrlPanel result={result} />}
        </div>
      </div>
    </section>
  );
}

const btnGhost =
  "inline-flex h-9 items-center rounded-lg border border-ink/15 bg-card/70 px-3 text-sm font-medium text-ink transition hover:bg-card active:scale-[0.97]";

function EvidenceBadge({ level }: { level: ScanResult["evidence"] }) {
  const text = { strong: "Strong evidence", moderate: "Moderate evidence", limited: "Limited evidence" }[level];
  return (
    <span
      title="How many independent checks could run: threat lists, domain age and the page itself."
      className="rounded-full border border-ink/15 bg-card/70 px-2.5 py-0.5 text-xs font-medium text-ink-2"
    >
      {text}
    </span>
  );
}

function Findings({ findings }: { findings: Finding[] }) {
  const shown = findings.filter((f) => f.weight !== 0 || f.severity === "info");
  const max = Math.max(1, ...shown.map((f) => Math.abs(f.weight)));
  if (!shown.length) return <p className="text-muted">No signal moved the score either way.</p>;
  return (
    <ul className="divide-y divide-rule">
      {shown.map((f) => (
        <li key={f.id + f.title} className="grid grid-cols-[auto_1fr_auto] gap-x-3 py-3.5">
          <span className={`mt-2 size-2 rounded-full ${SEVERITY[f.severity].dot}`} aria-hidden="true" />
          <div className="min-w-0">
            <p className="font-medium text-ink">
              {f.title}
              <span className="sr-only"> ({SEVERITY[f.severity].label})</span>
            </p>
            <p className="mt-0.5 text-sm leading-relaxed text-muted">{f.detail}</p>
          </div>
          <WeightBar weight={f.weight} max={max} />
        </li>
      ))}
    </ul>
  );
}

function WeightBar({ weight, max }: { weight: number; max: number }) {
  if (!weight) return <span className="w-20 pt-1 text-right font-mono text-xs text-faint">0</span>;
  const pct = Math.max(8, (Math.abs(weight) / max) * 100);
  return (
    <div className="flex w-20 flex-col items-end gap-1 pt-1" title="Log-odds weight: how far this moved the score">
      <span className={`font-mono text-xs tabular-nums ${weight > 0 ? "text-phish" : "text-legit"}`}>
        {weight > 0 ? "+" : ""}
        {weight.toFixed(1)}
      </span>
      <span className="h-1 w-full rounded-full bg-rule">
        <span
          className={`block h-1 rounded-full ${weight > 0 ? "bg-phish" : "bg-legit"}`}
          style={{ width: `${pct}%`, marginLeft: weight > 0 ? 0 : "auto" }}
        />
      </span>
    </div>
  );
}

function Rows({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className="divide-y divide-rule rounded-xl border border-rule bg-card">
      {rows.map(([k, val]) => (
        <div key={k} className="grid gap-1 px-4 py-2.5 sm:grid-cols-[200px_1fr] sm:gap-4">
          <dt className="text-sm text-muted">{k}</dt>
          <dd className="min-w-0 break-words text-sm text-ink">{val ?? <span className="text-faint">Not available</span>}</dd>
        </div>
      ))}
    </dl>
  );
}

const mono = (s: string | null | undefined) => (s ? <span className="font-mono text-[13px]">{s}</span> : null);
const yesNo = (b: boolean | null | undefined) =>
  b == null ? null : b ? <span className="text-ink">Yes</span> : <span className="text-muted">No</span>;

function PagePanel({ result }: { result: ScanResult }) {
  const p = result.page;
  if (!p) return <p className="text-muted">Link-only scan: the page was not visited.</p>;
  const a = p.analysis;
  const rows: [string, React.ReactNode][] = [
    ["Loaded", p.fetched ? `Yes, HTTP ${p.status}` : `No (${p.error ?? "unknown error"})`],
    ["Redirect path", p.hops.length ? mono(p.hops.map((h) => hostOf(h.url)).join(" → ")) : null],
    ["Certificate", p.tls_valid == null ? "Plain http, no encryption" : p.tls_valid ? "Verified" : `Invalid: ${p.tls_error}`],
  ];
  if (a) {
    rows.push(
      ["Title", a.title || null],
      ["Asks for a password", yesNo(a.has_password)],
      ["Sensitive fields", a.sensitive_fields.length ? mono(a.sensitive_fields.join(", ")) : "None"],
      [
        "Forms send to",
        a.forms.length ? mono([...new Set(a.forms.map((f) => f.exfil ?? f.action_domain).filter(Boolean))].join(", ")) : "No forms",
      ],
      ["Presents itself as", a.brand_claims.length ? a.brand_claims.join(", ") : "No brand"],
      ["Links", `${a.link_count} (${Math.round(a.dead_link_ratio * 100)}% go nowhere)`],
      ["Most-linked sites", Object.keys(a.link_domains).length ? mono(Object.keys(a.link_domains).slice(0, 5).join(", ")) : null],
      ["Script tricks", a.obfuscation.length || a.anti_inspect ? [...a.obfuscation, ...(a.anti_inspect ? ["blocks inspection"] : [])].join(", ") : "None found"],
    );
  }
  return (
    <div className="space-y-3">
      <Rows rows={rows} />
      <p className="text-xs text-faint">
        Read as text only: no script on the page was run, and nothing from it reached your browser.
      </p>
    </div>
  );
}

function DomainPanel({ result }: { result: ScanResult }) {
  const d = result.domain;
  return (
    <Rows
      rows={[
        ["Domain", mono(d.domain)],
        [
          "Registered",
          d.hosted_platform
            ? "Hosting platform: each site is created by a user"
            : d.registered
              ? `${d.registered} (${d.age_days?.toLocaleString()} days ago)`
              : d.rdap_error,
        ],
        ["Registrar", d.registrar],
        ["Expires", d.expires],
        ["IP addresses", mono(d.a_records.join(", "))],
        ["Nameservers", mono(d.nameservers.join(", "))],
        ["Certificate issuer", d.cert_issuer ?? d.cert_error],
        ["Certificate valid", d.cert_not_before ? `${d.cert_not_before} to ${d.cert_not_after}` : null],
        ["Certificate covers", d.cert_sans.length ? mono(d.cert_sans.slice(0, 6).join(", ")) : null],
      ]}
    />
  );
}

const INTEL_STATUS: Record<string, { text: string; cls: string }> = {
  hit: { text: "Listed", cls: "bg-phish-soft text-phish-ink" },
  clean: { text: "Not listed", cls: "bg-legit-soft text-legit-ink" },
  unavailable: { text: "Unavailable", cls: "bg-paper-2 text-muted" },
  not_configured: { text: "Not configured", cls: "bg-paper-2 text-faint" },
};

function IntelPanel({ result }: { result: ScanResult }) {
  return (
    <div className="space-y-3">
      <ul className="divide-y divide-rule rounded-xl border border-rule bg-card">
        {result.intel.map((i) => (
          <li key={i.source} className="flex items-center gap-3 px-4 py-3">
            <span className="flex-1 text-sm text-ink">{i.source}</span>
            {i.detail && <span className="hidden text-xs text-muted sm:block">{i.detail}</span>}
            <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${INTEL_STATUS[i.status].cls}`}>
              {INTEL_STATUS[i.status].text}
            </span>
          </li>
        ))}
      </ul>
      <p className="text-xs text-faint">
        A listing is decisive. Not being listed counts for little: most phishing links live for hours and are never
        reported.
      </p>
    </div>
  );
}

function UrlPanel({ result }: { result: ScanResult }) {
  const m = result.url_model;
  const top = m.signals.filter((s) => s.contribution !== 0).slice(0, 8);
  const max = Math.max(0.01, ...top.map((s) => Math.abs(s.contribution)));
  return (
    <div className="space-y-4">
      <p className="text-sm leading-relaxed text-muted">
        The lexical model from the PhishDriftBench paper scores the address{" "}
        <b className="text-ink">{(m.ml_score * 100).toFixed(1)}%</b> on its shape alone (it flags anything above{" "}
        {(m.decision_threshold * 100).toFixed(1)}%). Here it is one signal among several, with its influence capped.
      </p>
      {(m.brands_referenced.length > 0 || m.suspicious_words.length > 0) && (
        <Rows
          rows={[
            ["Brands in the link", m.brands_referenced.length ? mono(m.brands_referenced.join(", ")) : "None"],
            ["Bait words", m.suspicious_words.length ? mono(m.suspicious_words.join(", ")) : "None"],
          ]}
        />
      )}
      <ul className="space-y-2">
        {top.map((s) => (
          <li key={s.name} className="grid grid-cols-[1fr_120px_56px] items-center gap-3 text-sm">
            <span className="truncate text-ink-2" title={s.label}>
              {s.label}
            </span>
            <span className="relative h-1.5 rounded-full bg-rule">
              <span
                className={`absolute top-0 h-1.5 rounded-full ${s.contribution > 0 ? "left-1/2 bg-phish" : "right-1/2 bg-legit"}`}
                style={{ width: `${(Math.abs(s.contribution) / max) * 50}%` }}
              />
            </span>
            <span className={`text-right font-mono text-xs ${s.contribution > 0 ? "text-phish" : "text-legit"}`}>
              {s.contribution > 0 ? "+" : ""}
              {s.contribution.toFixed(2)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}
