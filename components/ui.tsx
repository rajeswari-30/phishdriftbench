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
    <div className="flex items-center gap-2.5 border-b border-line px-4 py-3 [&>svg]:size-4 [&>svg]:shrink-0 [&>svg]:text-muted">
      {icon}
      <h2 className="text-sm font-semibold tracking-tight text-fg">{title}</h2>
      {aside && <div className="ml-auto flex items-center gap-2 text-xs text-muted">{aside}</div>}
    </div>
  );
}

// Ghost buttons are flat white keys: a hairline ring plus a 1px bottom
// edge, no drop shadow. The primary action is a dark pill with a faint top
// highlight. Both are drawn with box-shadow rings, not borders, so a hover
// can change the ring without shifting layout.
const RING = "shadow-[0_0_0_1px_oklch(0%_0_0/0.09),0_1px_0_oklch(0%_0_0/0.06)]";
const PILL = {
  primary:
    "rounded-full bg-[linear-gradient(to_bottom,oklch(30%_0_0),oklch(16%_0_0))] text-pill-fg shadow-[inset_0_1px_0_oklch(100%_0_0/0.16),0_0_0_1px_oklch(0%_0_0/0.9)] hover:bg-[linear-gradient(to_bottom,oklch(36%_0_0),oklch(22%_0_0))]",
  ghost: `rounded-[10px] bg-surface text-fg ${RING} hover:bg-surface-2 hover:shadow-[0_0_0_1px_oklch(0%_0_0/0.16),0_1px_0_oklch(0%_0_0/0.08)] [&>svg]:text-muted hover:[&>svg]:text-fg`,
  danger: `rounded-[10px] bg-surface text-fg-2 ${RING} hover:bg-danger/5 hover:text-danger hover:shadow-[0_0_0_1px_color-mix(in_oklch,var(--danger)_40%,transparent)] [&>svg]:text-muted hover:[&>svg]:text-danger`,
};

export function pillClass(variant: keyof typeof PILL = "primary", size: "xs" | "sm" | "md" | "lg" = "md") {
  const sizes = {
    xs: "h-7 px-2.5 text-xs [&>svg]:size-3.5",
    sm: "h-8 px-2.5 text-[13.5px] [&>svg]:size-4",
    md: "h-9 px-4 text-sm [&>svg]:size-4",
    lg: "h-11 px-5 text-[15px] [&>svg]:size-[18px]",
  };
  return cx(
    "inline-flex shrink-0 select-none items-center justify-center gap-1.5 whitespace-nowrap font-normal [&>svg]:shrink-0 [&>svg]:transition-colors",
    "transition-[transform,background-color,box-shadow,color] duration-200 ease-[var(--ease-out)] active:scale-[0.96]",
    "focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-accent/20",
    "disabled:pointer-events-none disabled:opacity-40",
    sizes[size],
    PILL[variant],
  );
}

export function Badge({ children, tone = "neutral", className }: { children: ReactNode; tone?: Tone; className?: string }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs leading-4 [&>svg]:size-3.5 [&>svg]:shrink-0",
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
export function Metrics({ items }: { items: { label: string; value: ReactNode; tone?: Tone; icon?: ReactNode }[] }) {
  return (
    // A 1px gap over a line-coloured background draws every hairline, at
    // every column count, without per-breakpoint border rules.
    <dl className="grid grid-cols-2 gap-px border-b border-line bg-line sm:grid-cols-3 lg:grid-cols-6">
      {items.map((m) => (
        <div key={m.label} className="bg-surface px-4 py-3">
          <dt className="flex items-center gap-1.5 text-xs text-muted [&>svg]:size-3.5 [&>svg]:shrink-0">
            {m.icon}
            {m.label}
          </dt>
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
