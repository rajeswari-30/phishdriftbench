"use client";

import {
  IconAdjustmentsHorizontal,
  IconAlertTriangle,
  IconBrowser,
  IconCalendar,
  IconCheck,
  IconChevronDown,
  IconClock,
  IconCopy,
  IconCornerDownRight,
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
  IconWand,
  IconTrendingDown,
  IconTrendingUp,
  IconWorld,
  IconWorldWww,
  type Icon as TablerIcon,
} from "@tabler/icons-react";
import { useState } from "react";
import { reportUrl } from "@/lib/api";
import type { Finding, ScanResult, Severity, Verdict } from "@/lib/types";
import { Segmented } from "./segmented";
import { Badge, Card, CardHeader, cx, IconSwap, Metrics, pillClass, RollingNumber, TONE_BG, TONE_TEXT, type Tone } from "./ui";

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
  ai: IconWand,
};

const SEVERITY: Record<Severity, { tone: Tone; label: string }> = {
  critical: { tone: "danger", label: "Critical" },
  high: { tone: "danger", label: "High" },
  medium: { tone: "warn", label: "Medium" },
  low: { tone: "neutral", label: "Low" },
  info: { tone: "neutral", label: "Info" },
  good: { tone: "ok", label: "Reassuring" },
};

const TONE_PANEL: Record<Tone, string> = {
  danger: "bg-danger/[0.06]",
  warn: "bg-warn/[0.07]",
  ok: "bg-ok/[0.06]",
  accent: "bg-accent/[0.06]",
  neutral: "bg-surface-2",
};

type View = "overview" | "advanced";

export function ResultView({ result, onRescan }: { result: ScanResult; onRescan: () => void }) {
  const v = VERDICT[result.verdict];
  const [view, setView] = useState<View>("overview");
  const [copied, setCopied] = useState(false);
  const moved = result.final_url && result.final_url !== result.url;
  const signals = result.findings.filter((f) => f.weight !== 0).length;
  const age = result.domain.age_days;
  const p = result.page;
  const listed = result.intel.filter((i) => i.status === "hit").length;
  const checked = result.intel.filter((i) => i.status === "hit" || i.status === "clean").length;

  return (
    <section className="stagger mt-8 space-y-4" aria-label="Scan result">
      <Card className="overflow-hidden">
        <div className="grid lg:grid-cols-[300px_minmax(0,1fr)]">
          {/* Verdict: the one thing to read first. */}
          <div className={cx("flex flex-col justify-center gap-4 border-b border-line p-5 sm:p-6 lg:border-r lg:border-b-0", TONE_PANEL[v.tone])}>
            <div className={cx("flex items-center gap-2 text-[15px] font-semibold", TONE_TEXT[v.tone])}>
              <v.Icon className="size-5" stroke={1.75} aria-hidden="true" />
              <h2>{v.label}</h2>
            </div>
            <div className="flex items-end gap-2">
              <RollingNumber
                value={result.risk}
                className={cx("text-7xl font-semibold leading-[0.85] tracking-tighter tabular-nums", TONE_TEXT[v.tone])}
              />
              <span className="pb-1 text-sm text-muted">/100 risk</span>
            </div>
            <RiskMeter risk={result.risk} tone={v.tone} />
          </div>

          {/* Reason, link and actions. */}
          <div className="flex min-w-0 flex-col gap-4 p-5 sm:p-6">
            <p className="text-lg leading-snug font-medium text-pretty text-fg">{result.summary}</p>
            <TopReasons findings={result.findings} verdict={result.verdict} />
            <LinkBox url={result.url} finalUrl={moved ? result.final_url : null} />
          </div>
        </div>

        <Metrics
          cols={5}
          items={[
            {
              label: "Page",
              icon: <IconWorldWww stroke={1.75} aria-hidden="true" />,
              value: !p ? "Skipped" : p.fetched ? `HTTP ${p.status}` : "Offline",
              tone: p && !p.fetched ? "warn" : undefined,
            },
            {
              label: "Domain age",
              icon: <IconCalendar stroke={1.75} aria-hidden="true" />,
              value: result.domain.hosted_platform ? "Platform" : age == null ? "Unknown" : age >= 730 ? `${Math.floor(age / 365)} yrs` : `${age} days`,
              tone: age != null && age < 90 && !result.domain.hosted_platform ? "danger" : undefined,
            },
            {
              label: "Threat lists",
              icon: <IconShieldSearch stroke={1.75} aria-hidden="true" />,
              value: listed ? `${listed} listed` : `${checked} clear`,
              tone: listed ? "danger" : undefined,
            },
            { label: "Evidence", value: result.evidence[0].toUpperCase() + result.evidence.slice(1), icon: <IconScale stroke={1.75} aria-hidden="true" /> },
            { label: "Time", value: `${(result.elapsed_ms / 1000).toFixed(1)}s`, icon: <IconClock stroke={1.75} aria-hidden="true" /> },
          ]}
        />
      </Card>

      {/* Toolbar: how much detail on the left, what to do with it on the right. */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Segmented
          label="Result detail level"
          controls="view-panel"
          value={view}
          onChange={setView}
          segments={[
            {
              id: "overview",
              icon: <IconListDetails stroke={1.75} aria-hidden="true" />,
              label: (
                <>
                  Overview <span className="font-mono text-[11px] text-faint">{signals}</span>
                </>
              ),
            },
            { id: "advanced", icon: <IconAdjustmentsHorizontal stroke={1.75} aria-hidden="true" />, label: "Advanced details" },
          ]}
        />
        <div className="flex flex-wrap gap-1.5">
          <a href={reportUrl(result.input)} target="_blank" rel="noopener" className={pillClass("ghost", "sm")}>
            <IconFileText stroke={1.75} aria-hidden="true" />
            Report
          </a>
          <a href={reportUrl(result.input, true)} className={pillClass("ghost", "sm")} aria-label="Download PDF report">
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
            <IconSwap
              show={copied ? "b" : "a"}
              a={<IconShare className="size-4" stroke={1.75} />}
              b={<IconCheck className="size-4 text-ok" stroke={1.75} />}
            />
            <span aria-live="polite">{copied ? "Copied" : "Share"}</span>
          </button>
          <button type="button" onClick={onRescan} className={pillClass("ghost", "sm")}>
            <IconRefresh stroke={1.75} aria-hidden="true" />
            Rescan
          </button>
        </div>
      </div>

      <div role="tabpanel" id="view-panel" aria-labelledby={`view-panel-tab-${view}`} key={view} className="fade-in">
        {view === "overview" ? (
          <Card className="overflow-hidden">
            <CardHeader
              icon={<IconListDetails stroke={1.75} aria-hidden="true" />}
              title="Why this score"
              aside={<Badge>{signals} signals</Badge>}
            />
            <div className="p-2">
              <Findings findings={result.findings} />
            </div>
          </Card>
        ) : (
          <div className="grid gap-4 lg:grid-cols-2 lg:items-start">
            <div className="space-y-4">
              <Card className="overflow-hidden">
                <CardHeader
                  icon={<IconBrowser stroke={1.75} aria-hidden="true" />}
                  title="Page"
                  aside={!p ? <Badge>Skipped</Badge> : p.fetched ? <Badge tone="ok">HTTP {p.status}</Badge> : <Badge tone="warn">Offline</Badge>}
                />
                <PagePanel result={result} />
              </Card>
              <Card className="overflow-hidden">
                <CardHeader icon={<IconLink stroke={1.75} aria-hidden="true" />} title="Link text model" />
                <UrlPanel result={result} />
              </Card>
            </div>
            <div className="space-y-4">
              <Card className="overflow-hidden">
                <CardHeader
                  icon={<IconWorld stroke={1.75} aria-hidden="true" />}
                  title="Domain"
                  aside={<span className="truncate font-mono">{result.domain.domain}</span>}
                />
                <DomainPanel result={result} />
              </Card>
              <Card className="overflow-hidden">
                <CardHeader
                  icon={<IconShieldSearch stroke={1.75} aria-hidden="true" />}
                  title="Threat lists"
                  aside={listed ? <Badge tone="danger">{listed} listed</Badge> : <Badge>{checked} checked</Badge>}
                />
                <IntelPanel result={result} />
              </Card>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

/** The scanned link as text, never as a live link: a globe for the link
 * itself (no favicon, which would make the browser contact the site), the
 * landing page if it redirected, and a copy button. */
function LinkBox({ url, finalUrl }: { url: string; finalUrl: string | null }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="mt-auto flex items-start gap-2.5 rounded-[10px] border border-line bg-surface-2 py-2 pr-1.5 pl-3">
      <IconWorld className="mt-[3px] size-4 shrink-0 text-muted" stroke={1.75} aria-hidden="true" />
      <div className="min-w-0 flex-1 space-y-1 py-0.5 font-mono text-xs leading-5 text-fg-2">
        <p className="break-all">{url}</p>
        {finalUrl && (
          <p className="flex items-start gap-1.5 break-all text-muted">
            <IconCornerDownRight className="mt-[3px] size-3.5 shrink-0" stroke={1.75} aria-hidden="true" />
            <span>
              <span className="sr-only">Lands on </span>
              {finalUrl}
            </span>
          </p>
        )}
      </div>
      <button
        type="button"
        aria-label={copied ? "Link copied" : "Copy link"}
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(url);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          } catch {
            /* clipboard blocked */
          }
        }}
        className="grid size-7 shrink-0 place-items-center rounded-lg text-muted transition-[color,background-color,transform] duration-150 hover:bg-surface hover:text-fg active:scale-[0.92]"
      >
        <IconSwap
          show={copied ? "b" : "a"}
          a={<IconCopy className="size-4" stroke={1.75} />}
          b={<IconCheck className="size-4 text-ok" stroke={1.75} />}
        />
      </button>
    </div>
  );
}

/** The findings that pushed hardest toward the verdict, as chips under the
 * reason: warning signs for a risky link, reassurances for a safe one. A
 * safe link with a couple of minor flags should lead with why it's safe. */
function TopReasons({ findings, verdict }: { findings: Finding[]; verdict: Verdict }) {
  const towardRisk = verdict !== "safe";
  const list = findings
    .filter((f) => (towardRisk ? f.weight > 0 : f.weight < 0))
    .sort((x, y) => Math.abs(y.weight) - Math.abs(x.weight))
    .slice(0, 3);
  if (!list.length) return null;
  return (
    <ul className="flex flex-wrap gap-1.5" aria-label="Main reasons">
      {list.map((f) => {
        const I = CATEGORY_ICON[f.category];
        return (
          <li key={f.id + f.title}>
            <Badge tone={f.weight > 0 ? (f.weight >= 1.5 ? "danger" : "warn") : "ok"} className="py-1 pr-2.5 text-[12.5px]">
              <I stroke={1.75} aria-hidden="true" />
              {f.title}
            </Badge>
          </li>
        );
      })}
    </ul>
  );
}

function RiskMeter({ risk, tone }: { risk: number; tone: Tone }) {
  // Segmented track with the 40 and 70 verdict boundaries marked.
  return (
    <div aria-hidden="true">
      <div className="relative h-1.5 overflow-hidden rounded-full bg-fg/[0.08]">
        <div
          className={cx("grow-x absolute inset-y-0 left-0 rounded-full", TONE_BG[tone])}
          style={{ width: `${Math.max(2, risk)}%` }}
        />
        <span className="absolute inset-y-0 left-[40%] w-0.5 bg-surface" />
        <span className="absolute inset-y-0 left-[70%] w-0.5 bg-surface" />
      </div>
      <div className="relative mt-1.5 h-3 font-mono text-[10px] text-muted">
        <span className="absolute left-0">0</span>
        <span className="absolute left-[40%] -translate-x-1/2">40</span>
        <span className="absolute left-[70%] -translate-x-1/2">70</span>
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
              className="group grid w-full grid-cols-[auto_1fr_auto_auto] items-start gap-x-3 rounded-xl px-2 py-2.5 text-left transition-colors duration-200 hover:bg-surface-2 aria-expanded:bg-surface-2"
            >
              {(() => {
                const I = CATEGORY_ICON[f.category];
                return <I className={cx("mt-0.5 size-4", TONE_TEXT[SEVERITY[f.severity].tone])} stroke={1.75} aria-label={f.category} />;
              })()}
              <span className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 text-[14px] leading-5 font-medium text-fg">
                <span className="min-w-0">{f.title}</span>
                <Badge tone={SEVERITY[f.severity].tone} className="text-[11px] max-sm:hidden">
                  {SEVERITY[f.severity].label}
                </Badge>
              </span>
              <WeightBar weight={f.weight} max={max} />
              <IconChevronDown
                className={cx("mt-0.5 size-4 text-faint transition-[transform,color] duration-200 group-hover:text-fg", expanded && "rotate-180")}
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
                "grid transition-[grid-template-rows,opacity] duration-250 ease-[var(--ease-out)]",
                expanded ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0",
              )}
            >
              <p className="min-h-0 overflow-hidden px-2 pl-[36px] text-[13px] leading-relaxed text-muted">
                <span className="block">{f.detail}</span>
                {f.weight !== 0 && (
                  <span className="mt-1.5 block pb-2.5 font-mono text-[11.5px] text-faint">
                    {f.weight > 0 ? "Raises" : "Lowers"} risk, weight {f.weight > 0 ? "+" : ""}
                    {f.weight.toFixed(1)}
                  </span>
                )}
                {f.weight === 0 && <span className="block pb-2.5" />}
              </p>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="px-4 py-6 text-center text-[13px] text-muted">{children}</p>;
}

const IMPACT = [
  { min: 3, word: "Major" },
  { min: 1.5, word: "Strong" },
  { min: 0.6, word: "Moderate" },
  { min: 0, word: "Minor" },
];

/** How much a finding moved the score, in words and a bar. The raw
 * log-odds weight is in the expanded detail, not up front. */
function WeightBar({ weight, max }: { weight: number; max: number }) {
  if (!weight) return <span className="w-24 pt-0.5 text-right text-xs text-faint">No effect</span>;
  const raises = weight > 0;
  const word = IMPACT.find((l) => Math.abs(weight) >= l.min)!.word;
  const pct = Math.max(10, (Math.abs(weight) / max) * 100);
  const Trend = raises ? IconTrendingUp : IconTrendingDown;
  return (
    <div className="flex w-24 flex-col items-end gap-1.5 pt-0.5">
      <span className={cx("inline-flex items-center gap-1 text-xs font-medium", raises ? "text-danger" : "text-ok")}>
        <Trend className="size-3.5" stroke={2} aria-hidden="true" />
        {word}
        <span className="sr-only">{raises ? " increase in risk" : " decrease in risk"}</span>
      </span>
      <span className="h-1 w-full overflow-hidden rounded-full bg-surface-3" aria-hidden="true">
        <span className={cx("block h-1 rounded-full", raises ? "bg-danger" : "ml-auto bg-ok")} style={{ width: `${pct}%` }} />
      </span>
    </div>
  );
}

function Rows({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className="divide-y divide-line">
      {rows.map(([k, val]) => (
        <div key={k} className="grid grid-cols-[minmax(0,38%)_minmax(0,1fr)] gap-3 px-4 py-2.5">
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
          <li key={i.source} className="flex flex-wrap items-center gap-x-3 gap-y-1 px-4 py-2.5">
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
    <div className="space-y-3 px-4 py-3">
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
            <span className="block truncate text-fg-2 first-letter:uppercase" title={s.label}>
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
