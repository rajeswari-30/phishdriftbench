import { Suspense } from "react";
import { Scanner } from "@/components/scanner";
import {
  IconCalendarTime,
  IconFileSearch,
  IconLink,
  IconShieldExclamation,
  IconWand,
} from "@tabler/icons-react";
import { Badge } from "@/components/ui";

const CHECKS = [
  { label: "Threat lists", Icon: IconShieldExclamation },
  { label: "Domain age", Icon: IconCalendarTime },
  { label: "Page scan", Icon: IconFileSearch },
  { label: "Link text", Icon: IconLink },
  { label: "AI review", Icon: IconWand },
];

export default function Home() {
  return (
    <div className="mx-auto max-w-6xl px-4 pt-10 pb-24 sm:px-6 sm:pt-16">
      <div className="stagger mx-auto mb-8 max-w-3xl text-center">
        <h1 className="text-4xl font-semibold tracking-[-0.04em] text-balance sm:text-6xl">Check any link.</h1>
        <ul className="mt-5 no-scrollbar -mx-4 flex justify-[safe_center] gap-1.5 overflow-x-auto px-4 [mask-image:linear-gradient(to_right,transparent,black_16px,black_calc(100%-16px),transparent)] sm:mx-0 sm:flex-wrap sm:justify-center sm:overflow-visible sm:px-0 sm:[mask-image:none]" aria-label="What a scan checks">
          {CHECKS.map(({ label, Icon }) => (
            <li key={label} className="shrink-0">
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
