import { Suspense } from "react";
import { Scanner } from "@/components/scanner";
import {
  IconCalendarTime,
  IconFileSearch,
  IconLink,
  IconShieldExclamation,
  IconSparkles,
} from "@tabler/icons-react";
import { Badge } from "@/components/ui";

const CHECKS = [
  { label: "Threat lists", Icon: IconShieldExclamation },
  { label: "Domain age", Icon: IconCalendarTime },
  { label: "Page scan", Icon: IconFileSearch },
  { label: "Link text", Icon: IconLink },
  { label: "AI review", Icon: IconSparkles },
];

export default function Home() {
  return (
    <div className="mx-auto max-w-6xl px-4 pt-10 pb-24 sm:px-6 sm:pt-16">
      <div className="stagger mx-auto mb-8 max-w-3xl text-center">
        <h1 className="text-4xl font-semibold tracking-[-0.04em] text-balance sm:text-6xl">Check any link.</h1>
        <ul className="mt-5 flex flex-wrap justify-center gap-1.5" aria-label="What a scan checks">
          {CHECKS.map(({ label, Icon }) => (
            <li key={label}>
              <Badge className="py-1 pr-2.5">
                <Icon className="text-accent" stroke={1.75} aria-hidden="true" />
                {label}
              </Badge>
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
