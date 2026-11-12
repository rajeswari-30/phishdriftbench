"use client";

import { useEffect, useState } from "react";
import { Card, CardHeader, StatusIcon } from "./ui";

// The API answers in one response, so the steps are paced to its typical
// timing rather than streamed. Each describes something genuinely running.
const DEEP = ["Reading the link", "Checking threat lists", "Looking up the domain", "Visiting the page safely", "Weighing the evidence"];
const QUICK = ["Reading the link", "Checking threat lists", "Looking up the domain", "Weighing the evidence"];

export function ScanProgress({ url, deep }: { url: string; deep: boolean }) {
  const steps = deep ? DEEP : QUICK;
  const [at, setAt] = useState(0);
  useEffect(() => {
    setAt(0);
    const t = setInterval(() => setAt((i) => Math.min(i + 1, steps.length - 1)), 600);
    return () => clearInterval(t);
  }, [url, steps.length]);

  return (
    <Card className="enter mx-auto mt-6 max-w-3xl overflow-hidden" aria-live="polite">
      <CardHeader
        icon={<StatusIcon state="active" />}
        title="Scanning"
        aside={
          <span className="font-mono tabular-nums">
            {at}/{steps.length}
          </span>
        }
      />
      <p className="truncate border-b border-line px-4 py-2 font-mono text-xs text-muted">{url}</p>
      <ol className="space-y-0.5 p-2">
        {steps.map((s, i) => (
          <li key={s} className="flex items-center gap-3 rounded-lg px-2 py-1.5 text-[13.5px]">
            <StatusIcon state={i < at ? "done" : i === at ? "active" : "pending"} />
            <span className={i < at ? "text-muted line-through decoration-faint/60" : i === at ? "shimmer" : "text-muted"}>
              {s}
            </span>
          </li>
        ))}
      </ol>
    </Card>
  );
}
