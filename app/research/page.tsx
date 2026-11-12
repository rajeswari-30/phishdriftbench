import { IconChartBar, IconFlask, IconPercentage, IconTargetArrow, IconTimeline } from "@tabler/icons-react";
import type { Metadata } from "next";
import { DocsPage } from "@/components/docs-page";
import { Card, cx, TONE_TEXT, type Tone } from "@/components/ui";

export const metadata: Metadata = {
  title: "Research",
  description: "The research behind PhishLens: how phishing detectors behave outside their test sets, and what that changed.",
  alternates: { types: { "text/markdown": "/md/research" } },
};

const HEADLINES: { figure: string; label: string; tone: Tone; Icon: typeof IconChartBar }[] = [
  { figure: "99%", label: "Accuracy on balanced tests", tone: "ok", Icon: IconChartBar },
  { figure: "5.9%", label: "Precision at real rarity (B2)", tone: "danger", Icon: IconPercentage },
  { figure: "−23.3", label: "AUC pts on new data (B3)", tone: "danger", Icon: IconTargetArrow },
  { figure: "0 / 10", label: "Papers tested over time", tone: "warn", Icon: IconTimeline },
];

export default function Research() {
  return (
    <DocsPage
      page="research"
      eyebrow={
        <>
          <IconFlask stroke={1.75} aria-hidden="true" />
          Research
        </>
      }
      aside={
        <div className="mb-8 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {HEADLINES.map(({ figure, label, tone, Icon }, i) => (
            <Card key={label} className="enter p-4" style={{ animationDelay: `${i * 50}ms` }}>
              <Icon className="size-4 text-muted" stroke={1.75} aria-hidden="true" />
              <p className={cx("mt-3 text-2xl font-semibold tracking-tight tabular-nums", TONE_TEXT[tone])}>{figure}</p>
              <p className="mt-1 text-xs leading-snug text-muted">{label}</p>
            </Card>
          ))}
        </div>
      }
    />
  );
}
