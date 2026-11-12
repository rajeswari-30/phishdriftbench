"use client";

import {
  IconActivity,
  IconAlertTriangle,
  IconBrowser,
  IconCalendar,
  IconCheck,
  IconChevronDown,
  IconClock,
  IconDownload,
  IconFileText,
  IconLink,
  IconListDetails,
  IconRefresh,
  IconScale,
  IconShare,
  IconShieldCheck,
  IconShieldSearch,
  IconShieldX,
  IconSparkles,
  IconWorld,
  IconWorldWww,
  type Icon as TablerIcon,
} from "@tabler/icons-react";
import { useState } from "react";
import { reportUrl } from "@/lib/api";
import type { Finding, ScanResult, Severity, Verdict } from "@/lib/types";
import { Badge, Card, cx, Metrics, pillClass, RollingNumber, TONE_BG, TONE_TEXT, type Tone } from "./ui";

const VERDICT: Record<Verdict, { label: string; tone: Tone; Icon: TablerIcon }> = {
  phishing: { label: "Phishing", tone: "danger", Icon: IconShieldX },
  suspicious: { label: "Suspicious", tone: "warn", Icon: IconAlertTriangle },
  safe: { label: "Looks safe", tone: "ok", Icon: IconShieldCheck },
};

const CATEGORY_ICON: Record<Finding["category"], TablerIcon> = {
  url: IconLink,
  domain: IconWorld,
  page: IconBrowser,
  intel: IconShieldSearch,
  ai: IconSparkles,
};

const SEVERITY: Record<Severity, { tone: Tone; label: string }> = {
  critical: { tone: "danger", label: "Critical" },
  high: { tone: "danger", label: "High" },
  medium: { tone: "warn", label: "Medium" },
  low: { tone: "neutral", label: "Low" },
  info: { tone: "neutral", label: "Info" },
  good: { tone: "ok", label: "Reassuring" },
};

const TABS = ["Findings", "Page", "Domain", "Threat lists", "Link text"] as const;
type Tab = (typeof TABS)[number];
const TAB_ICON: Record<Tab, TablerIcon> = {
  Findings: IconListDetails,
  Page: IconBrowser,
  Domain: IconWorld,
  "Threat lists": IconShieldSearch,
  "Link text": IconLink,
};

export function ResultView({ result, onRescan }: { result: ScanResult; onRescan: () => void }) {
  const v = VERDICT[result.verdict];
  const [tab, setTab] = useState<Tab>("Findings");
  const [copied, setCopied] = useState(false);
  const moved = result.final_url && result.final_url !== result.url;
  const signals = result.findings.filter((f) => f.weight !== 0).length;
  const age = result.domain.age_days;

  return (
    <section className="enter mt-6 space-y-4" aria-label="Scan result">
      <Card className="overflow-hidden">
        {/* header */}
        <div className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-3">
          <v.Icon className={cx("size-[18px]", TONE_TEXT[v.tone])} stroke={1.75} aria-hidden="true" />
          <h2 className="text-sm font-semibold">Scan result</h2>
          <Badge tone={v.tone}>{v.label}</Badge>
          <div className="flex w-full gap-1.5 overflow-x-auto no-scrollbar sm:ml-auto sm:w-auto">
            <a href={reportUrl(result.input)} target="_blank" rel="noopener" className={pillClass("ghost", "sm")}>
              <IconFileText stroke={1.75} aria-hidden="true" />
              Report
            </a>
            <a href={reportUrl(result.input, true)} className={pillClass("ghost", "sm")}>
              <IconDownload stroke={1.75} aria-hidden="true" />
              PDF
            </a>
            <button
              type="button"
              className={pillClass("ghost", "sm")}
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
              {copied ? <IconCheck stroke={1.75} aria-hidden="true" /> : <IconShare stroke={1.75} aria-hidden="true" />}
              {copied ? "Copied" : "Share"}
            </button>
            <button type="button" onClick={onRescan} className={pillClass("ghost", "sm")}>
              <IconRefresh stroke={1.75} aria-hidden="true" />
              Rescan
            </button>
          </div>
        </div>

        {/* verdict */}
        <div className="grid gap-5 px-4 py-5 sm:grid-cols-[auto_1fr] sm:items-center sm:gap-8 sm:px-6">
          <div className="flex items-end gap-2">
            <RollingNumber value={result.risk} className={cx("text-6xl font-semibold leading-none tracking-tighter tabular-nums sm:text-7xl", TONE_TEXT[v.tone])} />
            <span className="pb-1.5 text-sm text-muted">/100</span>
          </div>
          <div className="min-w-0">
            <p className="text-[15px] leading-relaxed text-fg">{result.summary}</p>
            <RiskMeter risk={result.risk} tone={v.tone} />
          </div>
        </div>

        <div className="space-y-1 border-t border-line px-4 py-3 font-mono text-xs text-muted sm:px-6">
          <p className="break-all">{result.url}</p>
          {moved && <p className="break-all text-fg-2">↳ {result.final_url}</p>}
        </div>

        <Metrics
          items={[
            { label: "Verdict", value: v.label, tone: v.tone, icon: <v.Icon stroke={1.75} aria-hidden="true" /> },
            {
              label: "Page",
              icon: <IconWorldWww stroke={1.75} aria-hidden="true" />,
              value: !result.page ? "Skipped" : result.page.fetched ? `HTTP ${result.page.status}` : "Offline",
              tone: result.page && !result.page.fetched ? "warn" : undefined,
            },
            {
              label: "Domain age",
              icon: <IconCalendar stroke={1.75} aria-hidden="true" />,
              value: result.domain.hosted_platform ? "Platform" : age == null ? "Unknown" : age >= 730 ? `${Math.floor(age / 365)} yrs` : `${age} days`,
              tone: age != null && age < 90 && !result.domain.hosted_platform ? "danger" : undefined,
            },
            { label: "Signals", value: signals, icon: <IconActivity stroke={1.75} aria-hidden="true" /> },
            { label: "Evidence", value: result.evidence[0].toUpperCase() + result.evidence.slice(1), icon: <IconScale stroke={1.75} aria-hidden="true" /> },
            { label: "Time", value: `${(result.elapsed_ms / 1000).toFixed(1)}s`, icon: <IconClock stroke={1.75} aria-hidden="true" /> },
          ]}
        />

        {/* tabs */}
        <div className="px-2 pt-2 sm:px-3">
          <div
            role="tablist"
            aria-label="Scan details"
            onKeyDown={(e) => {
              // Arrow keys move between tabs (WAI-ARIA tabs pattern).
              const i = TABS.indexOf(tab);
              const next =
                e.key === "ArrowRight" ? (i + 1) % TABS.length
                : e.key === "ArrowLeft" ? (i - 1 + TABS.length) % TABS.length
                : e.key === "Home" ? 0
                : e.key === "End" ? TABS.length - 1
                : -1;
              if (next < 0) return;
              e.preventDefault();
              setTab(TABS[next]);
              document.getElementById(`tab-${next}`)?.focus();
            }}
            className="no-scrollbar flex gap-1 overflow-x-auto rounded-full border border-line bg-surface-2 p-1"
          >
            {TABS.map((t, i) => (
              <button
                key={t}
                id={`tab-${i}`}
                role="tab"
                type="button"
                aria-selected={tab === t}
                aria-controls="tab-panel"
                tabIndex={tab === t ? 0 : -1}
                onClick={() => setTab(t)}
                className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-full px-3 py-1.5 text-[13px] whitespace-nowrap text-muted transition-[color,background-color,box-shadow] duration-200 hover:bg-surface/60 hover:text-fg aria-selected:bg-surface aria-selected:text-fg aria-selected:shadow-[0_0_0_1px_oklch(0%_0_0/0.08)]"
              >
                {(() => {
                  const I = TAB_ICON[t];
                  return <I className="size-4 shrink-0" stroke={1.75} aria-hidden="true" />;
                })()}
                {t}
                {t === "Findings" && <span className="ml-1.5 font-mono text-[11px] text-faint">{signals}</span>}
              </button>
            ))}
          </div>
        </div>
        <div
          role="tabpanel"
          id="tab-panel"
          aria-labelledby={`tab-${TABS.indexOf(tab)}`}
          tabIndex={0}
          key={tab}
          className="enter p-2 focus-visible:outline-offset-[-2px] sm:p-3"
        >
          {tab === "Findings" && <Findings findings={result.findings} />}
          {tab === "Page" && <PagePanel result={result} />}
          {tab === "Domain" && <DomainPanel result={result} />}
          {tab === "Threat lists" && <IntelPanel result={result} />}
          {tab === "Link text" && <UrlPanel result={result} />}
        </div>
      </Card>
    </section>
  );
}

function RiskMeter({ risk, tone }: { risk: number; tone: Tone }) {
  // Segmented track with the 40 and 70 verdict boundaries marked.
  return (
    <div className="mt-4" aria-hidden="true">
      <div className="relative h-1.5 overflow-hidden rounded-full bg-surface-3">
        <div
          className={cx("absolute inset-y-0 left-0 rounded-full transition-[width] duration-700 ease-[var(--ease-out)]", TONE_BG[tone])}
          style={{ width: `${Math.max(2, risk)}%` }}
        />
        <span className="absolute inset-y-0 left-[40%] w-0.5 bg-surface" />
        <span className="absolute inset-y-0 left-[70%] w-0.5 bg-surface" />
      </div>
      <div className="relative mt-1.5 h-3 font-mono text-[10px] text-faint">
        <span className="absolute left-0">Safe</span>
        <span className="absolute left-[40%] -translate-x-1/2">Suspicious</span>
        <span className="absolute left-[70%] -translate-x-1/2 max-sm:hidden">Phishing</span>
        <span className="absolute right-0">100</span>
      </div>
    </div>
  );
}

function Findings({ findings }: { findings: Finding[] }) {
  const shown = findings.filter((f) => f.weight !== 0 || f.severity === "info");
  const max = Math.max(1, ...shown.map((f) => Math.abs(f.weight)));
  const [open, setOpen] = useState<string | null>(null);
  if (!shown.length) return <Empty>No signals</Empty>;
  return (
    <ul>
      {shown.map((f, i) => {
        const key = f.id + f.title;
        const expanded = open === key;
        return (
          <li key={key} className="enter" style={{ animationDelay: `${Math.min(i, 8) * 35}ms` }}>
            <button
              type="button"
              aria-expanded={expanded}
              aria-controls={`finding-${i}`}
              onClick={() => setOpen(expanded ? null : key)}
              className="group grid w-full grid-cols-[auto_1fr_auto_auto] items-center gap-x-3 rounded-xl px-2 py-2.5 text-left transition-colors duration-200 hover:bg-surface-2 aria-expanded:bg-surface-2"
            >
              {(() => {
                const I = CATEGORY_ICON[f.category];
                return <I className={cx("size-4", TONE_TEXT[SEVERITY[f.severity].tone])} stroke={1.75} aria-label={f.category} />;
              })()}
              <span className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 text-[14px] font-medium text-fg">
                <span className="min-w-0">{f.title}</span>
                <Badge tone={SEVERITY[f.severity].tone} className="text-[11px] max-sm:hidden">
                  {SEVERITY[f.severity].label}
                </Badge>
              </span>
              <WeightBar weight={f.weight} max={max} />
              <IconChevronDown
                className={cx("size-4 text-faint transition-[transform,color] duration-200 group-hover:text-fg", expanded && "rotate-180")}
                stroke={1.75}
                aria-hidden="true"
              />
            </button>
            <div
              id={`finding-${i}`}
              role="region"
              aria-label={f.title}
              inert={!expanded}
              className={cx(
                "grid transition-[grid-template-rows,opacity] duration-300 ease-[var(--ease-out)]",
                expanded ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0",
              )}
            >
              <p className="min-h-0 overflow-hidden px-2 pl-[36px] text-[13px] leading-relaxed text-muted">
                <span className="block pb-2.5">{f.detail}</span>
              </p>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="px-2 py-6 text-center text-[13px] text-faint">{children}</p>;
}

function WeightBar({ weight, max }: { weight: number; max: number }) {
  if (!weight) return <span className="w-14 text-right font-mono text-xs text-faint sm:w-20">0</span>;
  const pct = Math.max(10, (Math.abs(weight) / max) * 100);
  return (
    <div className="flex w-14 flex-col items-end gap-1.5 sm:w-20" title="Log-odds weight: how far this moved the score">
      <span className={cx("font-mono text-xs tabular-nums", weight > 0 ? "text-danger" : "text-ok")}>
        {weight > 0 ? "+" : ""}
        {weight.toFixed(1)}
      </span>
      <span className="h-1 w-full overflow-hidden rounded-full bg-surface-3">
        <span
          className={cx("block h-1 rounded-full", weight > 0 ? "bg-danger" : "ml-auto bg-ok")}
          style={{ width: `${pct}%` }}
        />
      </span>
    </div>
  );
}

function Rows({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className="divide-y divide-line">
      {rows.map(([k, val]) => (
        <div key={k} className="grid gap-0.5 px-2 py-2.5 sm:grid-cols-[180px_1fr] sm:gap-4">
          <dt className="text-[13px] text-muted">{k}</dt>
          <dd className="min-w-0 text-[13px] break-words text-fg">{val ?? <span className="text-faint">Not available</span>}</dd>
        </div>
      ))}
    </dl>
  );
}

const mono = (s: string | null | undefined) => (s ? <span className="font-mono text-[12.5px]">{s}</span> : null);
const yesNo = (b: boolean | null | undefined, bad = false) =>
  b == null ? null : b ? <Badge tone={bad ? "danger" : "ok"}>Yes</Badge> : <Badge>No</Badge>;

function PagePanel({ result }: { result: ScanResult }) {
  const p = result.page;
  if (!p) return <Empty>Deep scan off</Empty>;
  const a = p.analysis;
  const rows: [string, React.ReactNode][] = [
    ["Loaded", p.fetched ? <Badge tone="ok">HTTP {p.status}</Badge> : <span className="text-muted">No ({p.error ?? "unknown error"})</span>],
    ["Redirect path", p.hops.length ? mono(p.hops.map((h) => hostOf(h.url)).join(" → ")) : null],
    [
      "Certificate",
      p.tls_valid == null ? <Badge tone="warn">Plain http</Badge> : p.tls_valid ? <Badge tone="ok">Verified</Badge> : <Badge tone="danger">Invalid: {p.tls_error}</Badge>,
    ],
  ];
  if (a) {
    rows.push(
      ["Title", a.title || null],
      ["Asks for a password", yesNo(a.has_password, true)],
      ["Sensitive fields", a.sensitive_fields.length ? mono(a.sensitive_fields.join(", ")) : "None"],
      ["Forms send to", a.forms.length ? mono([...new Set(a.forms.map((f) => f.exfil ?? f.action_domain).filter(Boolean))].join(", ")) : "No forms"],
      ["Presents itself as", a.brand_claims.length ? a.brand_claims.join(", ") : "No brand"],
      ["Links", `${a.link_count} (${Math.round(a.dead_link_ratio * 100)}% go nowhere)`],
      ["Most-linked sites", Object.keys(a.link_domains).length ? mono(Object.keys(a.link_domains).slice(0, 5).join(", ")) : null],
      [
        "Script tricks",
        a.obfuscation.length || a.anti_inspect ? [...a.obfuscation, ...(a.anti_inspect ? ["blocks inspection"] : [])].join(", ") : "None found",
      ],
    );
  }
  return <Rows rows={rows} />;
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
        ["Certificate valid", d.cert_not_before ? `${d.cert_not_before} → ${d.cert_not_after}` : null],
        ["Certificate covers", d.cert_sans.length ? mono(d.cert_sans.slice(0, 6).join(", ")) : null],
      ]}
    />
  );
}

const INTEL: Record<string, { text: string; tone: Tone }> = {
  hit: { text: "Listed", tone: "danger" },
  clean: { text: "Not listed", tone: "ok" },
  unavailable: { text: "Unavailable", tone: "neutral" },
  not_configured: { text: "Not configured", tone: "neutral" },
};

function IntelPanel({ result }: { result: ScanResult }) {
  return (
    <>
      <ul className="divide-y divide-line">
        {result.intel.map((i) => (
          <li key={i.source} className="flex flex-wrap items-center gap-x-3 gap-y-1 px-2 py-2.5">
            <span className="text-[13.5px] text-fg">{i.source}</span>
            <Badge tone={INTEL[i.status].tone} className="ml-auto">
              {INTEL[i.status].text}
            </Badge>
            {i.detail && <span className="w-full text-xs text-faint">{i.detail}</span>}
          </li>
        ))}
      </ul>
    </>
  );
}

function UrlPanel({ result }: { result: ScanResult }) {
  const m = result.url_model;
  const top = m.signals.filter((s) => s.contribution !== 0).slice(0, 8);
  const max = Math.max(0.01, ...top.map((s) => Math.abs(s.contribution)));
  return (
    <div className="space-y-3 px-2 py-2">
      <div className="flex flex-wrap gap-1.5">
        <Badge tone={m.ml_score >= m.decision_threshold ? "danger" : "ok"}>Model {(m.ml_score * 100).toFixed(1)}%</Badge>
        <Badge>Threshold {(m.decision_threshold * 100).toFixed(1)}%</Badge>
        <Badge>Influence capped</Badge>
      </div>
      {(m.brands_referenced.length > 0 || m.suspicious_words.length > 0) && (
        <div className="flex flex-wrap gap-1.5">
          {m.brands_referenced.map((b) => (
            <Badge key={b} tone="warn">{b}</Badge>
          ))}
          {m.suspicious_words.map((w) => (
            <Badge key={w}>{w}</Badge>
          ))}
        </div>
      )}
      <ul className="space-y-2.5 pt-1">
        {top.map((s) => (
          <li key={s.name} className="grid grid-cols-[1fr_90px_48px] items-center gap-3 text-[13px] sm:grid-cols-[1fr_140px_56px]">
            <span className="truncate text-fg-2" title={s.label}>
              {s.label}
            </span>
            <span className="relative h-1 rounded-full bg-surface-3">
              <span className="absolute inset-y-[-3px] left-1/2 w-px bg-line-2" />
              <span
                className={cx("absolute top-0 h-1 rounded-full", s.contribution > 0 ? "left-1/2 bg-danger" : "right-1/2 bg-ok")}
                style={{ width: `${(Math.abs(s.contribution) / max) * 50}%` }}
              />
            </span>
            <span className={cx("text-right font-mono text-xs", s.contribution > 0 ? "text-danger" : "text-ok")}>
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
