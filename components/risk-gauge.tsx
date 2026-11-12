import type { Verdict } from "@/lib/types";

const TONE: Record<Verdict, string> = {
  phishing: "text-phish",
  suspicious: "text-unsure",
  safe: "text-legit",
};

/** 0-100 risk as a ring. The number is the content; the ring is a glance. */
export function RiskGauge({ risk, verdict, size = 132 }: { risk: number; verdict: Verdict; size?: number }) {
  const stroke = 10;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const filled = (Math.max(1, Math.min(100, risk)) / 100) * c;
  return (
    <div className={`relative shrink-0 ${TONE[verdict]}`} style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90" aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--rule)" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="currentColor"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${filled} ${c}`}
          className="transition-[stroke-dasharray] duration-700 ease-[var(--ease-out-strong)]"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-semibold tabular-nums tracking-tight text-ink">{risk}</span>
        <span className="text-xs text-muted">risk / 100</span>
      </div>
    </div>
  );
}
