"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { scan, ScanError } from "@/lib/api";
import { forget, loadRecent, remember, type RecentScan } from "@/lib/history";
import type { ScanResult } from "@/lib/types";
import { ResultView } from "./result-view";
import { ScanProgress } from "./scan-progress";

const EXAMPLES = [
  { label: "a phishing link", url: "http://paypal-secure-login-verify.account-update.xyz/signin" },
  { label: "a real site", url: "https://github.com/login" },
  { label: "a short link", url: "https://bit.ly/3xK9zQ2" },
  { label: "a look-alike", url: "https://xn--pple-43d.com" },
];

const DOT: Record<string, string> = { phishing: "bg-phish", suspicious: "bg-unsure", safe: "bg-legit" };

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
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => setRecent(loadRecent()), []);

  const run = useCallback(
    async (url: string) => {
      const target = url.trim();
      if (!target) return;
      setValue(target);
      inflight.current?.abort();
      const ctrl = new AbortController();
      inflight.current = ctrl;
      setState({ kind: "scanning", url: target, deep });
      try {
        const result = await scan(target, deep, ctrl.signal);
        setState({ kind: "done", result });
        setRecent(remember(result));
        const params = new URLSearchParams({ url: target });
        window.history.replaceState(null, "", `/?${params}`);
        requestAnimationFrame(() =>
          resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }),
        );
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        setState({
          kind: "error",
          message: e instanceof ScanError ? e.message : "Something went wrong. Try again.",
        });
      }
    },
    [deep],
  );

  // A shared link (/?url=...) scans on arrival.
  useEffect(() => {
    const shared = new URLSearchParams(window.location.search).get("url");
    if (shared) run(shared);
    // run once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const busy = state.kind === "scanning";

  return (
    <div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          run(value);
        }}
        className="flex flex-col gap-2.5 sm:flex-row"
        autoComplete="off"
      >
        <label htmlFor="url" className="sr-only">
          Link to check
        </label>
        <input
          id="url"
          type="text"
          inputMode="url"
          spellCheck={false}
          autoFocus
          placeholder="Paste a link, e.g. https://example.com/login"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          className="h-13 min-w-0 flex-1 rounded-xl border border-rule-2 bg-card px-4 font-mono text-[15px] text-ink shadow-sm outline-none transition placeholder:text-faint focus:border-ink focus:ring-4 focus:ring-ink/5"
        />
        <button
          type="submit"
          disabled={busy || !value.trim()}
          className="h-13 rounded-xl bg-ink px-6 text-[15px] font-medium text-paper transition active:scale-[0.97] disabled:opacity-50"
        >
          {busy ? "Scanning…" : "Scan link"}
        </button>
      </form>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-muted">
        <label className="inline-flex cursor-pointer select-none items-center gap-2">
          <input
            type="checkbox"
            checked={deep}
            onChange={(e) => setDeep(e.target.checked)}
            className="size-4 accent-[var(--ink)]"
          />
          Visit the page safely
        </label>
        <span className="hidden text-faint sm:inline">·</span>
        <span>
          Try{" "}
          {EXAMPLES.map((ex, i) => (
            <span key={ex.url}>
              <button
                type="button"
                onClick={() => run(ex.url)}
                className="text-ink-2 underline decoration-rule-2 underline-offset-4 transition-colors hover:decoration-ink"
              >
                {ex.label}
              </button>
              {i < EXAMPLES.length - 1 ? <span className="mx-1.5 text-faint">·</span> : null}
            </span>
          ))}
        </span>
      </div>
      <p className="mt-3 text-[13px] leading-relaxed text-faint">
        {deep
          ? "The scanner reads the page's HTML from our server and never runs its scripts. Your browser never opens the link."
          : "Link-only mode: nothing is fetched, only the address and public lookups are used."}{" "}
        Don&apos;t paste links that contain private tokens.
      </p>

      <div ref={resultRef} className="scroll-mt-20">
        {state.kind === "scanning" && <ScanProgress url={state.url} deep={state.deep} />}
        {state.kind === "error" && (
          <div role="alert" className="rise mt-10 rounded-xl border border-phish/30 bg-phish-soft px-4 py-3 text-phish-ink">
            {state.message}
          </div>
        )}
        {state.kind === "done" && <ResultView result={state.result} onRescan={() => run(state.result.input)} />}
      </div>

      {recent.length > 0 && state.kind !== "scanning" && (
        <section className="mt-14">
          <div className="mb-3 flex items-baseline justify-between">
            <h2 className="text-sm font-medium text-muted">Recent scans on this device</h2>
            <button
              type="button"
              onClick={() => {
                forget();
                setRecent([]);
              }}
              className="text-xs text-faint hover:text-ink"
            >
              Clear
            </button>
          </div>
          <ul className="divide-y divide-rule rounded-xl border border-rule bg-card">
            {recent.map((r) => (
              <li key={r.url}>
                <button
                  type="button"
                  onClick={() => run(r.url)}
                  className="flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors hover:bg-paper-2"
                >
                  <span className={`size-2 shrink-0 rounded-full ${DOT[r.verdict]}`} />
                  <span className="min-w-0 flex-1 truncate font-mono text-[13px] text-ink-2">{r.url}</span>
                  <span className="font-mono text-xs tabular-nums text-muted">{r.risk}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
