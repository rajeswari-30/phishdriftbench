"use client";

import {
  IconAlertCircle,
  IconAlertTriangle,
  IconClockPause,
  IconRefresh,
  IconServerOff,
  IconWifiOff,
  IconAlphabetLatin,
  IconArrowRight,
  IconCodeOff,
  IconEyeOff,
  IconFishHook,
  IconHistory,
  IconLink,
  IconLinkPlus,
  IconLoader2,
  IconLock,
  IconShieldCheck,
  IconShieldX,
  IconTrash,
  IconWorldCheck,
} from "@tabler/icons-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { scan, ScanError, type ScanErrorKind } from "@/lib/api";
import { validateLink } from "@/lib/validate";
import { forget, loadRecent, remember, type RecentScan } from "@/lib/history";
import type { ScanResult } from "@/lib/types";
import { ResultView } from "./result-view";
import { ScanProgress } from "./scan-progress";
import { Badge, Card, CardHeader, cx, pillClass } from "./ui";

const EXAMPLES = [
  { label: "Phishing link", url: "http://paypal-secure-login-verify.account-update.xyz/signin", Icon: IconFishHook },
  { label: "Real site", url: "https://github.com/login", Icon: IconWorldCheck },
  { label: "Short link", url: "https://bit.ly/3xK9zQ2", Icon: IconLinkPlus },
  { label: "Look-alike", url: "https://xn--pple-43d.com", Icon: IconAlphabetLatin },
];

const VERDICT_ICON = {
  phishing: { Icon: IconShieldX, cls: "text-danger" },
  suspicious: { Icon: IconAlertTriangle, cls: "text-warn" },
  safe: { Icon: IconShieldCheck, cls: "text-ok" },
} as const;

type State =
  | { kind: "idle" }
  | { kind: "scanning"; url: string; deep: boolean }
  | { kind: "done"; result: ScanResult }
  | { kind: "error"; error: ScanError; url: string };

export function Scanner() {
  const [value, setValue] = useState("");
  const [deep, setDeep] = useState(true);
  const [state, setState] = useState<State>({ kind: "idle" });
  const [fieldError, setFieldError] = useState<string | null>(null);
  const [announce, setAnnounce] = useState("");
  const [recent, setRecent] = useState<RecentScan[]>([]);
  const inflight = useRef<AbortController | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setRecent(loadRecent());
  }, []);

  // ⌘K / Ctrl+K focuses the link field from anywhere on the page.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        input.current?.focus();
        input.current?.select();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const run = useCallback(
    async (url: string) => {
      setValue(url);
      const checked = validateLink(url);
      if (!checked.ok) {
        // Input problems belong under the field, not in a result card.
        setFieldError(checked.error);
        setAnnounce("");
        input.current?.focus();
        return;
      }
      const target = checked.value;
      setFieldError(null);
      inflight.current?.abort();
      const ctrl = new AbortController();
      inflight.current = ctrl;
      setState({ kind: "scanning", url: target, deep });
      setAnnounce("Scanning");
      requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
      try {
        const result = await scan(target, deep, ctrl.signal);
        setState({ kind: "done", result });
        setRecent(remember(result));
        setAnnounce(`Scan complete. ${result.verdict}, risk ${result.risk} out of 100. ${result.summary}`);
        window.history.replaceState(null, "", `/?${new URLSearchParams({ url: target })}`);
        requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        const err = e instanceof ScanError ? e : new ScanError("unexpected", "Something went wrong. Try again.");
        if (err.kind === "input") {
          setState({ kind: "idle" });
          setFieldError(err.message);
          setAnnounce("");
          input.current?.focus();
          return;
        }
        setState({ kind: "error", error: err, url: target });
        setAnnounce("");
      }
    },
    [deep],
  );

  // A shared link (/?url=...) scans on arrival.
  useEffect(() => {
    const shared = new URLSearchParams(window.location.search).get("url");
    if (shared) run(shared);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const busy = state.kind === "scanning";

  return (
    <div>
      <Card
        className={cx(
          "overflow-hidden transition-shadow",
          fieldError
            ? "border-danger/50 ring-4 ring-danger/10"
            : "has-[input:focus-visible]:border-line-2 has-[input:focus-visible]:ring-4 has-[input:focus-visible]:ring-fg/5",
        )}
      >
        <form
          onSubmit={(e) => {
            e.preventDefault();
            run(value);
          }}
          noValidate
          autoComplete="off"
          className="flex items-center gap-2 p-2 pl-4"
        >
          <IconLink className={cx("size-[18px] shrink-0", fieldError ? "text-danger" : "text-muted")} stroke={1.75} aria-hidden="true" />
          <label htmlFor="url" className="sr-only">
            Link to check
          </label>
          <input
            ref={input}
            id="url"
            type="text"
            inputMode="url"
            enterKeyHint="go"
            spellCheck={false}
            autoFocus
            placeholder="Paste a link"
            value={value}
            aria-invalid={fieldError ? true : undefined}
            aria-describedby={fieldError ? "url-error" : "url-hint"}
            onChange={(e) => {
              setValue(e.target.value);
              if (fieldError) setFieldError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                setValue("");
                setFieldError(null);
              }
            }}
            className="h-10 min-w-0 flex-1 bg-transparent font-mono text-[14px] text-fg outline-none focus-visible:outline-none placeholder:font-sans placeholder:text-muted"
          />
          <button type="submit" disabled={busy} aria-busy={busy} className={cx(pillClass("primary", "md"), "group")}>
            {busy ? <IconLoader2 className="animate-spin" stroke={2} aria-hidden="true" /> : null}
            {busy ? "Scanning" : "Scan"}
            {!busy && <IconArrowRight stroke={2} className="transition-transform duration-200 group-hover:translate-x-0.5" aria-hidden="true" />}
          </button>
        </form>
        <div className="flex flex-col gap-3 border-t border-line bg-surface-2/50 px-4 py-3 sm:flex-row sm:items-center">
          <Switch checked={deep} onChange={setDeep} label="Deep scan" />
          <div className="flex flex-wrap gap-1.5 sm:ml-auto sm:justify-end">
            {EXAMPLES.map((ex) => (
              <button key={ex.url} type="button" onClick={() => run(ex.url)} className={pillClass("ghost", "sm")}>
                <ex.Icon stroke={1.75} aria-hidden="true" />
                {ex.label}
              </button>
            ))}
          </div>
        </div>
      </Card>
      {fieldError && (
        <p id="url-error" role="alert" className="enter mt-2.5 flex items-center gap-1.5 px-1 text-[13px] text-danger">
          <IconAlertCircle className="size-4 shrink-0" stroke={1.75} aria-hidden="true" />
          {fieldError}
        </p>
      )}
      <p id="url-hint" className="sr-only">
        Paste any web link. Press Enter to scan.
      </p>
      <p className="sr-only" role="status" aria-live="polite">
        {announce}
      </p>
      <ul className="mt-3 flex flex-wrap justify-center gap-1.5" aria-label="Privacy">
        {(deep
          ? [
              { t: "Scripts never run", Icon: IconCodeOff },
              { t: "Your browser never opens it", Icon: IconEyeOff },
              { t: "No private links", Icon: IconLock },
            ]
          : [
              { t: "Nothing fetched", Icon: IconEyeOff },
              { t: "Link and lookups only", Icon: IconLink },
            ]
        ).map(({ t, Icon }) => (
          <li key={t}>
            <Badge>
              <Icon stroke={1.75} aria-hidden="true" />
              {t}
            </Badge>
          </li>
        ))}
      </ul>

      <div ref={resultRef} className="scroll-mt-20">
        {state.kind === "scanning" && <ScanProgress url={state.url} deep={state.deep} />}
        {state.kind === "error" && <ErrorCard error={state.error} onRetry={() => run(state.url)} />}
        {state.kind === "done" && <ResultView result={state.result} onRescan={() => run(state.result.input)} />}
      </div>

      {recent.length > 0 && state.kind !== "scanning" && (
        <Card className="enter mt-8 overflow-hidden">
          <CardHeader
            icon={<IconHistory stroke={1.75} aria-hidden="true" />}
            title="Recent scans"
            aside={
              <>
                <span className="rounded-full bg-surface-2 px-2 py-0.5 font-mono text-[11px] tabular-nums text-muted">{recent.length}</span>
                <button
                  type="button"
                  onClick={() => {
                    forget();
                    setRecent([]);
                  }}
                  className={pillClass("danger", "xs")}
                >
                  <IconTrash stroke={1.75} aria-hidden="true" />
                  Clear
                </button>
              </>
            }
          />
          <ul className="py-1">
            {recent.map((r) => (
              <li key={r.url}>
                <button
                  type="button"
                  onClick={() => run(r.url)}
                  className="group flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors duration-200 hover:bg-surface-2"
                >
                  {(() => {
                    const { Icon, cls } = VERDICT_ICON[r.verdict];
                    return <Icon className={cx("size-4 shrink-0", cls)} stroke={1.75} aria-label={r.verdict} />;
                  })()}
                  <span className="min-w-0 flex-1 truncate font-mono text-[13px] text-fg-2 transition-colors group-hover:text-fg">{r.url}</span>
                  <span className="font-mono text-xs tabular-nums text-muted">{r.risk}</span>
                  <IconArrowRight
                    className="size-4 shrink-0 -translate-x-1 text-faint opacity-0 transition-all duration-200 group-hover:translate-x-0 group-hover:opacity-100"
                    stroke={1.75}
                    aria-hidden="true"
                  />
                </button>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}

function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="group inline-flex shrink-0 items-center gap-2.5 text-[13px] text-fg-2"
    >
      <span
        className={cx(
          "relative h-5 w-9 rounded-full border transition-colors duration-200",
          checked ? "border-transparent bg-accent" : "border-line-2 bg-surface-3",
        )}
      >
        <span
          className={cx(
            "absolute top-1/2 left-0.5 size-3.5 -translate-y-1/2 rounded-full bg-white shadow transition-transform duration-300 ease-[var(--ease-spring)]",
            checked && "translate-x-4",
          )}
        />
      </span>
      {label}
    </button>
  );
}


const ERROR_LOOK: Record<Exclude<ScanErrorKind, "input">, { title: string; Icon: typeof IconWifiOff }> = {
  offline: { title: "You're offline", Icon: IconWifiOff },
  network: { title: "Can't reach the scanner", Icon: IconWifiOff },
  rate: { title: "Slow down a little", Icon: IconClockPause },
  server: { title: "Scanner error", Icon: IconServerOff },
  unexpected: { title: "Something went wrong", Icon: IconAlertCircle },
};

function ErrorCard({ error, onRetry }: { error: ScanError; onRetry: () => void }) {
  const look = ERROR_LOOK[error.kind as Exclude<ScanErrorKind, "input">] ?? ERROR_LOOK.unexpected;
  const [wait, setWait] = useState(error.kind === "rate" ? Math.min(error.retryAfter ?? 30, 60) : 0);
  useEffect(() => {
    if (wait <= 0) return;
    const t = setTimeout(() => setWait((w) => w - 1), 1000);
    return () => clearTimeout(t);
  }, [wait]);
  return (
    <Card role="alert" className="enter mt-6 flex flex-col items-start gap-4 p-5 sm:flex-row sm:items-center">
      <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-danger/10 text-danger">
        <look.Icon className="size-5" stroke={1.75} aria-hidden="true" />
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-[15px] font-medium text-fg">{look.title}</p>
        <p className="mt-0.5 text-[13.5px] text-muted">{error.message}</p>
      </div>
      <button type="button" onClick={onRetry} disabled={wait > 0} className={pillClass("ghost", "sm")}>
        <IconRefresh stroke={1.75} aria-hidden="true" />
        {wait > 0 ? `Try again in ${wait}s` : "Try again"}
      </button>
    </Card>
  );
}
