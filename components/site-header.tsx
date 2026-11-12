"use client";

import { IconArrowUpRight, IconBrandGithub, IconChevronRight } from "@tabler/icons-react";
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

  // Scrolling down tucks the header away; any scroll up brings it back.
  // A small dead zone keeps trackpad jitter from making it flicker.
  const [hidden, setHidden] = useState(false);
  useEffect(() => {
    let last = window.scrollY;
    const onScroll = () => {
      const y = window.scrollY;
      setScrolled(y > 8);
      if (Math.abs(y - last) < 6) return;
      setHidden(y > last && y > 120);
      last = y;
    };
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
    <div
      className={cx(
        "sticky top-0 z-50 px-3 pt-3 transition-transform duration-300 ease-[var(--ease-out)] sm:px-4",
        hidden && !open && "-translate-y-[calc(100%+12px)]",
      )}
      onFocusCapture={() => setHidden(false)}
    >
      <header
        ref={headerRef}
        className={cx(
          "mx-auto max-w-3xl rounded-[22px] border transition-[background-color,box-shadow,border-color] duration-300",
          scrolled || open
            ? "border-line-2 bg-surface/80 backdrop-blur-xl backdrop-saturate-150"
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
                className="absolute top-0.5 bottom-0.5 rounded-full bg-surface shadow-[0_0_0_1px_oklch(0%_0_0/0.08)] transition-all duration-300 ease-[var(--ease-out)]"
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
              <IconBrandGithub stroke={1.75} aria-hidden="true" />
              GitHub
            </a>
            <button
              type="button"
              aria-expanded={open}
              aria-controls="mobile-nav"
              aria-label={open ? "Close menu" : "Open menu"}
              onClick={() => setOpen((v) => !v)}
              className="grid size-9 place-items-center rounded-[10px] bg-surface shadow-[0_0_0_1px_oklch(0%_0_0/0.09),0_1px_0_oklch(0%_0_0/0.06)] transition hover:bg-surface-2 active:scale-95 md:hidden"
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
                    <IconChevronRight className="size-4 text-faint" stroke={1.75} aria-hidden="true" />
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
                  <IconArrowUpRight className="size-4 text-faint" stroke={1.75} aria-hidden="true" />
                </a>
              </li>
            </ul>
          </div>
        </nav>
      </header>
    </div>
  );
}
