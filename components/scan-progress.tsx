"use client";

import { useEffect, useState } from "react";

// The API answers in one response, so these steps are paced to its typical
// timing rather than streamed. They describe what is genuinely happening.
const DEEP = [
  "Reading the link",
  "Checking threat lists",
  "Looking up the domain",
  "Visiting the page safely",
  "Weighing the evidence",
];
const QUICK = ["Reading the link", "Checking threat lists", "Looking up the domain", "Weighing the evidence"];

export function ScanProgress({ url, deep }: { url: string; deep: boolean }) {
  const steps = deep ? DEEP : QUICK;
  const [at, setAt] = useState(0);
  useEffect(() => {
    setAt(0);
    const t = setInterval(() => setAt((i) => Math.min(i + 1, steps.length - 1)), 650);
    return () => clearInterval(t);
  }, [url, steps.length]);

  return (
    <div className="rise mt-10 rounded-2xl border border-rule bg-card p-6" aria-live="polite">
      <p className="truncate font-mono text-[13px] text-muted">{url}</p>
      <ol className="mt-4 space-y-2.5">
        {steps.map((s, i) => (
          <li key={s} className={`flex items-center gap-3 text-[15px] ${i > at ? "text-faint" : "text-ink"}`}>
            {i < at ? (
              <svg viewBox="0 0 16 16" className="size-4 text-legit" aria-hidden="true">
                <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            ) : i === at ? (
              <span className="size-4 animate-spin rounded-full border-2 border-ink border-r-transparent" aria-hidden="true" />
            ) : (
              <span className="size-4 rounded-full border-2 border-rule-2" aria-hidden="true" />
            )}
            {s}
          </li>
        ))}
      </ol>
    </div>
  );
}
