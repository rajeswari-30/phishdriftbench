import type { Metadata } from "next";
import { Badge, Card, CardHeader, cx, Metrics, pillClass, TONE_TEXT, type Tone } from "@/components/ui";

export const metadata: Metadata = { title: "Research" };

const REPO = "https://github.com/rajeswari-30/phishdriftbench";

const AXES: { axis: string; figure: string; label: string; tone: Tone }[] = [
  { axis: "Time", figure: "−0.4", label: "AUC pts after 12 months", tone: "ok" },
  { axis: "Source", figure: "−23.3", label: "AUC pts across datasets", tone: "danger" },
  { axis: "Evasion", figure: "−5.4", label: "Recall pts under attack", tone: "warn" },
  { axis: "Rarity", figure: "5.9%", label: "Precision at 1 in 10⁴", tone: "danger" },
];

export default function Research() {
  return (
    <div className="mx-auto max-w-3xl space-y-5 px-4 pt-10 pb-24 sm:px-6 sm:pt-16">
      <div className="enter text-center">
        <Badge className="mb-4">PhishDriftBench</Badge>
        <h1 className="text-3xl font-semibold tracking-[-0.035em] text-balance sm:text-5xl">Built on research</h1>
      </div>

      <Card className="enter overflow-hidden">
        <CardHeader title="URL model" aside={<Badge>Held-out set</Badge>} />
        <Metrics
          items={[
            { label: "Accuracy", value: "97.4%" },
            { label: "Test URLs", value: "20,000" },
            { label: "Real-world precision", value: "5.9%", tone: "danger" },
            { label: "Per URL", value: "18 µs" },
            { label: "Model size", value: "0.71 MB" },
            { label: "Baselines", value: "5" },
          ]}
        />
      </Card>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {AXES.map((a, i) => (
          <Card key={a.axis} className="enter p-4" style={{ animationDelay: `${i * 50}ms` }}>
            <p className="text-xs text-muted">{a.axis}</p>
            <p className={cx("mt-2 text-3xl font-semibold tracking-tight tabular-nums", TONE_TEXT[a.tone])}>{a.figure}</p>
            <p className="mt-1 text-xs leading-snug text-muted">{a.label}</p>
          </Card>
        ))}
      </div>

      <Card className="enter overflow-hidden">
        <CardHeader title="Why look past the URL" />
        <ul className="divide-y divide-line">
          {[
            ["Archived lookups", "Leak the future", "danger"],
            ["Live lookups", "Describe the present", "ok"],
            ["URL alone", "<1% precision at real rates", "warn"],
          ].map(([k, v, tone]) => (
            <li key={k} className="flex items-center gap-3 px-4 py-3">
              <span className="text-[13.5px] text-fg">{k}</span>
              <Badge tone={tone as Tone} className="ml-auto">
                {v}
              </Badge>
            </li>
          ))}
        </ul>
      </Card>

      <div className="flex flex-wrap justify-center gap-2">
        <a href={REPO} target="_blank" rel="noopener noreferrer" className={pillClass("primary", "md")}>
          Source ↗
        </a>
        <a href={`${REPO}/blob/main/RESEARCH_GAP_AND_IMPROVEMENTS.md`} target="_blank" rel="noopener noreferrer" className={pillClass("ghost", "md")}>
          Results ↗
        </a>
      </div>
    </div>
  );
}
