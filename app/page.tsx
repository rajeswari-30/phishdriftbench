import { Suspense } from "react";
import { Scanner } from "@/components/scanner";

export default function Home() {
  return (
    <div className="mx-auto max-w-3xl px-4 pb-24 pt-16 sm:px-6 sm:pt-24">
      <p className="mb-5 font-mono text-[12.5px] text-muted">Phishing link scanner</p>
      <h1 className="text-balance text-4xl font-semibold leading-[1.05] tracking-[-0.035em] sm:text-6xl">
        Check a link before you open it.
      </h1>
      <p className="mb-9 mt-5 max-w-xl text-pretty text-lg leading-relaxed text-muted">
        PhishLens checks threat lists, how old the domain is, and what the page really asks for, then explains every
        point of its score.
      </p>
      <Suspense>
        <Scanner />
      </Suspense>
    </div>
  );
}
