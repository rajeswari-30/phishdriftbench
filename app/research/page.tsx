import type { Metadata } from "next";

export const metadata: Metadata = { title: "Research" };

const REPO = "https://github.com/rajeswari-30/phishdriftbench";

const AXES = [
  ["Time", "−0.4 AUC pts", "Trained on older URLs, tested 12 months later. Barely moved: a null result, reported as one."],
  ["Source", "−23.3 AUC pts", "Train on one dataset, test on another. The worst baseline fell from 0.997 to 0.764."],
  ["Evasion", "−5.4 recall pts", "Simple tricks backfired; AI-generated phishing degraded every model tested."],
  ["Rarity", "5.9% precision", "At realistic phishing rates, most alerts from a 99%-AUC detector are false."],
];

export default function Research() {
  return (
    <div className="mx-auto max-w-3xl px-4 pb-24 pt-16 sm:px-6 sm:pt-20">
      <p className="mb-5 font-mono text-[12.5px] text-muted">PhishDriftBench</p>
      <h1 className="text-balance text-4xl font-semibold tracking-[-0.03em] sm:text-5xl">
        Anyone can build a detector. Knowing whether to trust it is harder.
      </h1>
      <p className="mt-5 max-w-xl text-lg leading-relaxed text-muted">
        PhishLens grew out of PhishDriftBench, a four-axis protocol for measuring what phishing-URL detectors do once
        they leave the test set, applied to five reimplemented baselines and a proposed mitigation, DASH.
      </p>

      <section className="mt-12 grid gap-6 sm:grid-cols-3">
        {[
          ["97.4%", "accuracy on 20,000 held-out real URLs"],
          ["5.9%", "precision once phishing is as rare as in live traffic"],
          ["18 µs", "to score one URL, with a 0.71 MB model"],
        ].map(([n, label]) => (
          <div key={n} className="border-t border-ink pt-4">
            <div className="text-4xl font-semibold tabular-nums tracking-tight">{n}</div>
            <p className="mt-2 text-sm leading-relaxed text-muted">{label}</p>
          </div>
        ))}
      </section>

      <section className="mt-16">
        <h2 className="text-2xl font-semibold tracking-tight">Tested four ways</h2>
        <p className="mt-2 leading-relaxed text-muted">
          Across eight recent detection papers, temporal testing, prevalence-corrected reporting and evasion testing were
          each performed by none of them.
        </p>
        <dl className="mt-6 divide-y divide-rule border-y border-rule">
          {AXES.map(([axis, figure, text]) => (
            <div key={axis} className="grid gap-1 py-5 sm:grid-cols-[120px_160px_1fr] sm:gap-4">
              <dt className="font-semibold">{axis}</dt>
              <dd className="font-mono text-sm text-ink-2">{figure}</dd>
              <dd className="text-sm leading-relaxed text-muted">{text}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="mt-16">
        <h2 className="text-2xl font-semibold tracking-tight">Why the scanner looks past the URL</h2>
        <p className="mt-3 leading-relaxed text-ink-2">
          The paper&apos;s provenance audit found that lookup features (WHOIS, DNS, certificates) added nothing to a model
          trained on archived URLs, because those lookups were made years after the URLs were collected. Made at the
          moment a link is scanned, they describe the present instead of leaking the future, which is why PhishLens uses
          them live while the paper&apos;s experiments stay URL-only.
        </p>
        <p className="mt-3 leading-relaxed text-ink-2">
          The same audit is why the URL model alone cannot be the answer: at a realistic 1-in-10,000 prevalence its
          precision is a fraction of a percent. The page and the domain carry the evidence the string does not.
        </p>
      </section>

      <div className="mt-14 flex flex-wrap gap-3">
        <a href={REPO} target="_blank" rel="noopener noreferrer" className="inline-flex h-10 items-center rounded-lg bg-ink px-4 text-sm font-medium text-paper">
          Source and data ↗
        </a>
        <a
          href={`${REPO}/blob/main/RESEARCH_GAP_AND_IMPROVEMENTS.md`}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex h-10 items-center rounded-lg border border-rule-2 px-4 text-sm font-medium"
        >
          Full results ↗
        </a>
      </div>
    </div>
  );
}
