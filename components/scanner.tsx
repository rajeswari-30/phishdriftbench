"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { scan, ScanError } from "@/lib/api";
import { forget, loadRecent, remember, type RecentScan } from "@/lib/history";
import type { ScanResult } from "@/lib/types";
import { ResultView } from "./result-view";
import { ScanProgress } from "./scan-progress";
import { Card, CardHeader, cx, pillClass, TONE_BG } from "./ui";

const EXAMPLES = [
  { label: "Phishing link", url: "http://paypal-secure-login-verify.account-update.xyz/signin" },
  { label: "Real site", url: "https://github.com/login" },
  { label: "Short link", url: "https://bit.ly/3xK9zQ2" },
  { label: "Look-alike", url: "https://xn--pple-43d.com" },
];

const VERDICT_TONE = { phishing: "danger", suspicious: "warn", safe: "ok" } as const;

type State =
  | { kind: "idle" }
  | { kind: "scanning"; url: string; deep: boolean }
  | { kind: "done"; result: ScanResult }
  | { kind: "error"; message: string };

export function Scanner() {
  const [value, setValue] = useState("");
  const [deep, setDeep] = useState(true);
  const [state, setState] = useState<State>({ kind: "idle" });
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
      const target = url.trim();
      if (!target) return;
      setValue(target);
      inflight.current?.abort();
      const ctrl = new AbortController();
      inflight.current = ctrl;
      setState({ kind: "scanning", url: target, deep });
      requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
      try {
        const result = await scan(target, deep, ctrl.signal);
        setState({ kind: "done", result });
        setRecent(remember(result));
        window.history.replaceState(null, "", `/?${new URLSearchParams({ url: target })}`);
        requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        setState({ kind: "error", message: e instanceof ScanError ? e.message : "Something went wrong. Try again." });
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
      <Card className="overflow-hidden transition-shadow has-[input:focus-visible]:border-line-2 has-[input:focus-visible]:ring-4 has-[input:focus-visible]:ring-accent/15">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            run(value);
          }}
          autoComplete="off"
          className="flex items-center gap-2 p-2 pl-4"
        >
          <svg viewBox="0 0 24 24" className="size-[18px] shrink-0 text-muted" aria-hidden="true">
            <path
              d="M10 14a4.5 4.5 0 006.4 0l3-3a4.5 4.5 0 00-6.4-6.4l-1 1M14 10a4.5 4.5 0 00-6.4 0l-3 3a4.5 4.5 0 006.4 6.4l1-1"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
            />
          </svg>
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
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => e.key === "Escape" && setValue("")}
            className="h-10 min-w-0 flex-1 bg-transparent font-mono text-[14px] text-fg outline-none focus-visible:outline-none placeholder:font-sans placeholder:text-faint"
          />
          <button type="submit" disabled={busy || !value.trim()} className={pillClass("primary", "md")}>
            {busy ? "Scanning" : "Scan"}
          </button>
        </form>
        <div className="flex flex-col gap-3 border-t border-line bg-surface-2/50 px-4 py-3 sm:flex-row sm:items-center">
          <Switch checked={deep} onChange={setDeep} label="Deep scan" />
          <div className="flex flex-wrap gap-1.5 sm:ml-auto sm:justify-end">
            {EXAMPLES.map((ex) => (
              <button key={ex.url} type="button" onClick={() => run(ex.url)} className={pillClass("ghost", "sm")}>
                {ex.label}
              </button>
            ))}
          </div>
        </div>
      </Card>
      <ul className="mt-3 flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs text-faint" aria-label="Privacy">
        {(deep ? ["Scripts never run", "Your browser never opens it", "No private links"] : ["Nothing fetched", "Link and lookups only"]).map(
          (t) => (
            <li key={t} className="flex items-center gap-1.5">
              <span className="size-1 rounded-full bg-faint" aria-hidden="true" />
              {t}
            </li>
          ),
        )}
      </ul>

      <div ref={resultRef} className="scroll-mt-20">
        {state.kind === "scanning" && <ScanProgress url={state.url} deep={state.deep} />}
        {state.kind === "error" && (
          <div role="alert" className="enter mt-6 rounded-[18px] border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
            {state.message}
          </div>
        )}
        {state.kind === "done" && <ResultView result={state.result} onRescan={() => run(state.result.input)} />}
      </div>

      {recent.length > 0 && state.kind !== "scanning" && (
        <Card className="enter mt-8 overflow-hidden">
          <CardHeader
            title="Recent scans"
            aside={
              <>
                <span className="tabular-nums">{recent.length}</span>
                <button
                  type="button"
                  onClick={() => {
                    forget();
                    setRecent([]);
                  }}
                  className="rounded-full px-2 py-0.5 transition-colors hover:bg-surface-3 hover:text-fg"
                >
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
                  className="flex w-full items-center gap-3 px-4 py-2 text-left transition-colors hover:bg-surface-2"
                >
                  <span className={cx("size-1.5 shrink-0 rounded-full", TONE_BG[VERDICT_TONE[r.verdict]])} />
                  <span className="min-w-0 flex-1 truncate font-mono text-[13px] text-fg-2">{r.url}</span>
                  <span className="font-mono text-xs tabular-nums text-muted">{r.risk}</span>
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
