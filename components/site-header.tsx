"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Logo } from "./logo";
import { cx, pillClass } from "./ui";

const LINKS = [
  { href: "/", label: "Scan" },
  { href: "/how-it-works", label: "How it works" },
  { href: "/research", label: "Research" },
];

/** Floating glass pill. A highlight slides between links on desktop; on
 * small screens the links fold into a menu that closes on navigation,
 * Escape or a tap outside. */
export function SiteHeader({ github }: { github: string }) {
  const path = usePathname();
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const [hover, setHover] = useState<string | null>(null);
  const [pill, setPill] = useState<{ left: number; width: number } | null>(null);
  const navRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);
  const active = hover ?? LINKS.find((l) => l.href === path)?.href ?? null;

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => setOpen(false), [path]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    const onDown = (e: PointerEvent) => {
      if (!headerRef.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown);
    };
  }, [open]);

  // Measure the active (or hovered) link so the highlight can slide to it.
  useLayoutEffect(() => {
    const nav = navRef.current;
    const el = active ? nav?.querySelector<HTMLElement>(`[data-href="${active}"]`) : null;
    setPill(el && nav ? { left: el.offsetLeft, width: el.offsetWidth } : null);
  }, [active]);

  return (
    <div className="sticky top-0 z-50 px-3 pt-3 sm:px-4">
      <header
        ref={headerRef}
        className={cx(
          "mx-auto max-w-3xl rounded-[22px] border transition-[background-color,box-shadow,border-color] duration-300",
          scrolled || open
            ? "border-line-2 bg-surface/80 shadow-[0_8px_30px_-12px_oklch(0%_0_0/0.18)] backdrop-blur-xl backdrop-saturate-150"
            : "border-transparent bg-transparent",
        )}
      >
        <div className="flex h-12 items-center gap-2 pr-1.5 pl-2">
          <Link href="/" className="flex items-center gap-2 rounded-full py-1 pr-2 pl-1 text-[15px] font-semibold tracking-tight">
            <span className="grid size-7 place-items-center rounded-full bg-fg text-bg">
              <Logo className="size-4" />
            </span>
            PhishLens
          </Link>

          <div
            ref={navRef}
            onMouseLeave={() => setHover(null)}
            className="relative mx-auto hidden items-center rounded-full border border-line bg-surface-2/70 p-0.5 md:flex"
          >
            {pill && (
              <span
                aria-hidden="true"
                className="absolute top-0.5 bottom-0.5 rounded-full bg-surface shadow-[0_1px_2px_oklch(0%_0_0/0.08)] transition-all duration-300 ease-[var(--ease-out)]"
                style={{ left: pill.left, width: pill.width }}
              />
            )}
            {LINKS.map((l) => (
              <Link
                key={l.href}
                href={l.href}
                data-href={l.href}
                onMouseEnter={() => setHover(l.href)}
                onFocus={() => setHover(l.href)}
                onBlur={() => setHover(null)}
                aria-current={path === l.href ? "page" : undefined}
                className={cx(
                  "relative rounded-full px-3.5 py-1.5 text-[13px] whitespace-nowrap transition-colors",
                  active === l.href ? "text-fg" : "text-muted hover:text-fg",
                )}
              >
                {l.label}
              </Link>
            ))}
          </div>

          <div className="ml-auto flex items-center gap-1.5 md:ml-0">
            <a href={github} target="_blank" rel="noopener noreferrer" className={cx(pillClass("primary", "sm"), "max-sm:hidden")}>
              <GithubMark />
              GitHub
            </a>
            <button
              type="button"
              aria-expanded={open}
              aria-controls="mobile-nav"
              aria-label={open ? "Close menu" : "Open menu"}
              onClick={() => setOpen((v) => !v)}
              className="grid size-9 place-items-center rounded-full border border-line-2 bg-surface transition active:scale-95 md:hidden"
            >
              <span className="relative block h-2.5 w-4" aria-hidden="true">
                <span className={cx("absolute left-0 h-[1.5px] w-4 rounded bg-fg transition-transform duration-300", open ? "top-1 rotate-45" : "top-0")} />
                <span className={cx("absolute left-0 h-[1.5px] w-4 rounded bg-fg transition-transform duration-300", open ? "top-1 -rotate-45" : "top-2")} />
              </span>
            </button>
          </div>
        </div>

        <nav
          id="mobile-nav"
          aria-label="Main"
          className={cx(
            "grid overflow-hidden transition-[grid-template-rows,opacity] duration-300 ease-[var(--ease-out)] md:hidden",
            open ? "grid-rows-[1fr] opacity-100" : "pointer-events-none grid-rows-[0fr] opacity-0",
          )}
        >
          <div className="min-h-0">
            <ul className="space-y-0.5 border-t border-line p-2">
              {LINKS.map((l) => (
                <li key={l.href}>
                  <Link
                    href={l.href}
                    tabIndex={open ? 0 : -1}
                    aria-current={path === l.href ? "page" : undefined}
                    className="flex items-center justify-between rounded-2xl px-3 py-2.5 text-[15px] text-fg-2 transition-colors hover:bg-surface-2 aria-[current=page]:bg-surface-2 aria-[current=page]:text-fg"
                  >
                    {l.label}
                    <span className="text-faint" aria-hidden="true">→</span>
                  </Link>
                </li>
              ))}
              <li className="sm:hidden">
                <a
                  href={github}
                  target="_blank"
                  rel="noopener noreferrer"
                  tabIndex={open ? 0 : -1}
                  className="flex items-center justify-between rounded-2xl px-3 py-2.5 text-[15px] text-fg-2 transition-colors hover:bg-surface-2"
                >
                  GitHub
                  <span className="text-faint" aria-hidden="true">↗</span>
                </a>
              </li>
            </ul>
          </div>
        </nav>
      </header>
    </div>
  );
}

function GithubMark() {
  return (
    <svg viewBox="0 0 16 16" className="size-3.5" aria-hidden="true" fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 005.47 7.59c.4.07.55-.17.55-.38v-1.33c-2.23.48-2.7-1.07-2.7-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.05-.49.05-.49.8.06 1.23.83 1.23.83.72 1.23 1.88.87 2.34.67.07-.52.28-.87.5-1.07-1.78-.2-3.65-.89-3.65-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 014 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}
