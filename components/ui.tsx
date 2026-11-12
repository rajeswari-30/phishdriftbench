import type { ComponentProps, ReactNode } from "react";

export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

/** Raised surface: one hairline border, a soft shadow, ~14px corners. */
export function Card({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      className={cx("rounded-[18px] border border-line bg-surface shadow-card", className)}
      {...props}
    />
  );
}

export function CardHeader({ title, aside, icon }: { title: ReactNode; aside?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex items-center gap-2.5 border-b border-line px-4 py-3">
      {icon}
      <h3 className="text-sm font-semibold tracking-tight text-fg">{title}</h3>
      {aside && <div className="ml-auto flex items-center gap-2 text-xs text-muted">{aside}</div>}
    </div>
  );
}

const PILL = {
  primary: "bg-pill text-pill-fg hover:opacity-90",
  ghost: "border border-line-2 bg-surface-2 text-fg hover:bg-surface-3",
};

export function pillClass(variant: keyof typeof PILL = "primary", size: "sm" | "md" | "lg" = "md") {
  const sizes = { sm: "h-8 px-3 text-[13px]", md: "h-9 px-4 text-sm", lg: "h-11 px-5 text-[15px]" };
  return cx(
    "inline-flex shrink-0 select-none items-center justify-center gap-1.5 whitespace-nowrap rounded-full font-medium",
    "transition-[transform,opacity,background-color] duration-150 ease-[var(--ease-out)] active:scale-[0.96]",
    "disabled:pointer-events-none disabled:opacity-40",
    sizes[size],
    PILL[variant],
  );
}

export function Badge({ children, tone = "neutral", className }: { children: ReactNode; tone?: Tone; className?: string }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs leading-4",
        TONE_BADGE[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export type Tone = "neutral" | "danger" | "warn" | "ok" | "accent";
const TONE_BADGE: Record<Tone, string> = {
  neutral: "border-line-2 text-muted",
  danger: "border-danger/30 bg-danger/10 text-danger",
  warn: "border-warn/30 bg-warn/10 text-warn",
  ok: "border-ok/30 bg-ok/10 text-ok",
  accent: "border-accent/30 bg-accent/10 text-accent",
};
export const TONE_TEXT: Record<Tone, string> = {
  neutral: "text-muted",
  danger: "text-danger",
  warn: "text-warn",
  ok: "text-ok",
  accent: "text-accent",
};
export const TONE_BG: Record<Tone, string> = {
  neutral: "bg-faint",
  danger: "bg-danger",
  warn: "bg-warn",
  ok: "bg-ok",
  accent: "bg-accent",
};

/** Task-list style status circle: pending (dashed), active (spinning arc), done (check pops in). */
export function StatusIcon({ state, tone = "ok" }: { state: "pending" | "active" | "done"; tone?: Tone }) {
  if (state === "done") {
    return (
      <span className={cx("pop grid size-4 place-items-center rounded-full", TONE_BG[tone])} aria-hidden="true">
        <svg viewBox="0 0 16 16" className="size-3 text-bg">
          <path d="M4 8.5l2.5 2.5L12 5.5" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
    );
  }
  if (state === "active") {
    return (
      <span className="relative size-4" aria-hidden="true">
        <span className="absolute inset-0 rounded-full border-[1.5px] border-line-2" />
        <span className="absolute inset-0 animate-spin rounded-full border-[1.5px] border-transparent border-t-fg" />
      </span>
    );
  }
  return <span className="size-4 rounded-full border-[1.5px] border-dashed border-faint/70" aria-hidden="true" />;
}

/** A strip of labelled figures separated by hairlines, wrapping on small screens. */
export function Metrics({ items }: { items: { label: string; value: ReactNode; tone?: Tone }[] }) {
  return (
    // A 1px gap over a line-coloured background draws every hairline, at
    // every column count, without per-breakpoint border rules.
    <dl className="grid grid-cols-2 gap-px border-b border-line bg-line sm:grid-cols-3 lg:grid-cols-6">
      {items.map((m) => (
        <div key={m.label} className="bg-surface px-4 py-3">
          <dt className="text-xs text-muted">{m.label}</dt>
          <dd className={cx("mt-1 truncate text-[15px] font-medium tabular-nums", m.tone ? TONE_TEXT[m.tone] : "text-fg")}>
            {m.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** Numbers that roll in digit by digit. */
export function RollingNumber({ value, className }: { value: number | string; className?: string }) {
  return (
    <span className={className} aria-label={String(value)}>
      {String(value)
        .split("")
        .map((ch, i) => (
          <span key={`${value}-${i}`} className="digit" style={{ animationDelay: `${i * 70}ms` }} aria-hidden="true">
            {ch}
          </span>
        ))}
    </span>
  );
}
