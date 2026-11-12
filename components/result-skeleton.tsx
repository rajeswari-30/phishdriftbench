"use client";

import { useEffect, useState } from "react";
import { Card, Skel, StatusIcon } from "./ui";

// The API answers in one response, so the status line is paced to its
// typical timing rather than streamed; each step names something that
// genuinely runs during a scan.
const DEEP = ["Reading the link", "Checking threat lists", "Looking up the domain", "Visiting the page safely", "Weighing the evidence"];
const QUICK = ["Reading the link", "Checking threat lists", "Looking up the domain", "Weighing the evidence"];

/** Placeholder for ResultView, block for block: same containers, same
 * paddings, same grid, so the real result replaces it without anything on
 * the page moving. */
export function ResultSkeleton({ url, deep }: { url: string; deep: boolean }) {
  const steps = deep ? DEEP : QUICK;
  const [at, setAt] = useState(0);
  useEffect(() => {
    setAt(0);
    const t = setInterval(() => setAt((i) => Math.min(i + 1, steps.length - 1)), 650);
    return () => clearInterval(t);
  }, [url, steps.length]);

  return (
    <section className="mt-8 space-y-4" aria-busy="true" aria-label="Scanning">
      <p className="sr-only" role="status" aria-live="polite">
        {steps[at]}
      </p>

      {/* Verdict card */}
      <Card className="enter overflow-hidden">
        <div className="grid lg:grid-cols-[300px_minmax(0,1fr)]">
          <div className="flex flex-col justify-center gap-4 border-b border-line bg-surface-2/60 p-5 sm:p-6 lg:border-r lg:border-b-0">
            {/* Where the verdict will be: what's running right now. */}
            <div className="flex h-[22px] items-center gap-2 text-[15px] font-medium">
              <StatusIcon state="active" />
              <span key={steps[at]} className="enter shimmer">
                {steps[at]}
              </span>
            </div>
            <div className="flex items-end gap-2">
              <Skel className="h-[60px] w-28 rounded-lg" />
              <Skel className="mb-1 h-4 w-16" />
            </div>
            <div>
              <Skel className="h-1.5 w-full rounded-full" />
              <div className="mt-1.5 flex justify-between">
                {[0, 1, 2, 3].map((i) => (
                  <Skel key={i} className="h-2.5 w-4" />
                ))}
              </div>
            </div>
          </div>
          <div className="flex min-w-0 flex-col gap-4 p-5 sm:p-6">
            <div className="space-y-2">
              <Skel className="h-5 w-11/12" />
              <Skel className="h-5 w-3/5" />
            </div>
            <div className="flex flex-wrap gap-1.5">
              <Skel className="h-[26px] w-36 rounded-full" />
              <Skel className="h-[26px] w-52 rounded-full" />
              <Skel className="h-[26px] w-32 rounded-full" />
            </div>
            <div className="mt-auto flex items-center gap-2.5 rounded-[10px] border border-line bg-surface-2 py-2 pr-1.5 pl-3">
              <Skel className="size-4 rounded-full" />
              <p className="min-w-0 flex-1 truncate py-0.5 font-mono text-xs leading-5 text-muted">{url}</p>
              <span className="size-7" />
            </div>
          </div>
        </div>
        <div className="grid grid-cols-2 border-t border-line sm:grid-cols-3 lg:grid-cols-5">
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="space-y-2 bg-surface px-4 py-3 shadow-[0_0_0_0.5px_var(--line)]">
              <Skel className="h-3 w-20" />
              <Skel className="h-[18px] w-16" />
            </div>
          ))}
        </div>
      </Card>

      {/* Toolbar */}
      <div className="enter flex flex-wrap items-center justify-between gap-3" style={{ animationDelay: "50ms" }}>
        <div className="inline-flex gap-1 rounded-full border border-line bg-surface-2 p-1">
          <Skel className="h-[30px] w-28 rounded-full" />
          <Skel className="h-[30px] w-36 rounded-full" />
        </div>
        <div className="flex flex-wrap gap-1.5">
          {["w-[84px]", "w-[66px]", "w-[78px]", "w-[86px]"].map((w) => (
            <Skel key={w} className={`h-8 rounded-[10px] ${w}`} />
          ))}
        </div>
      </div>

      {/* Findings */}
      <Card className="enter overflow-hidden" style={{ animationDelay: "100ms" }}>
        <div className="flex items-center gap-2.5 border-b border-line px-4 py-3">
          <Skel className="size-4" />
          <Skel className="h-4 w-28" />
          <Skel className="ml-auto h-5 w-16 rounded-full" />
        </div>
        <ul className="p-2">
          {["w-48", "w-72", "w-40", "w-56", "w-44"].map((w) => (
            <li key={w} className="grid grid-cols-[auto_1fr_auto_auto] items-start gap-x-3 px-2 py-2.5">
              <Skel className="mt-0.5 size-4" />
              <span className="flex items-center gap-2">
                <Skel className={`h-4 ${w} max-w-full`} />
                <Skel className="h-[18px] w-14 rounded-full max-sm:hidden" />
              </span>
              <span className="flex w-24 flex-col items-end gap-1.5 pt-0.5">
                <Skel className="h-3 w-14" />
                <Skel className="h-1 w-full rounded-full" />
              </span>
              <Skel className="mt-0.5 size-4" />
            </li>
          ))}
        </ul>
      </Card>
    </section>
  );
}
