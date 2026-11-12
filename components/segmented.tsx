"use client";

import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { cx } from "./ui";

export interface Segment<T extends string> {
  id: T;
  label: ReactNode;
  icon?: ReactNode;
}

/** Tabs drawn as a segmented control. The highlight slides between options
 * with a transform (compositor-only) instead of each option repainting its
 * own background, and arrow keys move between them (WAI-ARIA tabs). */
export function Segmented<T extends string>({
  segments,
  value,
  onChange,
  label,
  controls,
}: {
  segments: Segment<T>[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  controls: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [pill, setPill] = useState<{ x: number; w: number } | null>(null);

  useLayoutEffect(() => {
    const el = ref.current?.querySelector<HTMLElement>(`[data-seg="${value}"]`);
    if (el) setPill({ x: el.offsetLeft, w: el.offsetWidth });
  }, [value, segments.length]);

  return (
    <div
      ref={ref}
      role="tablist"
      aria-label={label}
      onKeyDown={(e) => {
        const i = segments.findIndex((s) => s.id === value);
        const next =
          e.key === "ArrowRight" ? (i + 1) % segments.length
          : e.key === "ArrowLeft" ? (i - 1 + segments.length) % segments.length
          : e.key === "Home" ? 0
          : e.key === "End" ? segments.length - 1
          : -1;
        if (next < 0) return;
        e.preventDefault();
        onChange(segments[next].id);
        ref.current?.querySelector<HTMLElement>(`[data-seg="${segments[next].id}"]`)?.focus();
      }}
      className="relative inline-flex rounded-full border border-line bg-surface-2 p-1"
    >
      {pill && (
        <span
          aria-hidden="true"
          className="absolute top-1 bottom-1 left-0 rounded-full bg-surface shadow-[0_0_0_1px_oklch(0%_0_0/0.08)] transition-[transform,width] duration-250 ease-[var(--ease-out)]"
          style={{ transform: `translateX(${pill.x}px)`, width: pill.w }}
        />
      )}
      {segments.map((s) => (
        <button
          key={s.id}
          data-seg={s.id}
          id={`${controls}-tab-${s.id}`}
          role="tab"
          type="button"
          aria-selected={value === s.id}
          aria-controls={controls}
          tabIndex={value === s.id ? 0 : -1}
          onClick={() => onChange(s.id)}
          className={cx(
            "relative inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-[13px] whitespace-nowrap transition-colors duration-200 [&>svg]:size-4",
            value === s.id ? "text-fg" : "text-muted hover:text-fg",
          )}
        >
          {s.icon}
          {s.label}
        </button>
      ))}
    </div>
  );
}
