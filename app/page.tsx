import { Suspense } from "react";
import { Scanner } from "@/components/scanner";
import { Badge } from "@/components/ui";

const CHECKS = ["Threat lists", "Domain age", "Page scan", "Link text", "AI review"];

export default function Home() {
  return (
    <div className="mx-auto max-w-3xl px-4 pt-10 pb-24 sm:px-6 sm:pt-16">
      <div className="enter mb-8 text-center">
        <h1 className="text-4xl font-semibold tracking-[-0.04em] text-balance sm:text-6xl">Check any link.</h1>
        <ul className="mt-5 flex flex-wrap justify-center gap-1.5" aria-label="What a scan checks">
          {CHECKS.map((c) => (
            <li key={c}>
              <Badge>{c}</Badge>
            </li>
          ))}
        </ul>
      </div>
      <Suspense>
        <Scanner />
      </Suspense>
    </div>
  );
}
