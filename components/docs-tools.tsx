"use client";

import {
  IconBrandOpenai,
  IconCheck,
  IconChevronDown,
  IconCopy,
  IconExternalLink,
  IconListTree,
  IconMarkdown,
  IconRobot,
  IconSparkles,
} from "@tabler/icons-react";
import { useEffect, useRef, useState } from "react";
import type { TocItem } from "@/lib/docs";
import { cx } from "./ui";

/** "Copy page" split button: one click copies the docs as Markdown; the
 * menu opens the same text for an agent or in an AI assistant. */
export function CopyPage({ markdown, page }: { markdown: string; page: string }) {
  const [copied, setCopied] = useState(false);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (e: Event) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !ref.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("pointerdown", close);
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("pointerdown", close);
      window.removeEventListener("keydown", close);
    };
  }, [open]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(markdown);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      window.open(`/md/${page}`, "_blank", "noopener");
    }
  };

  const ask = (base: string) => {
    const prompt = `Read ${window.location.origin}/md/${page} so I can ask you questions about PhishLens.`;
    return `${base}${encodeURIComponent(prompt)}`;
  };

  const item =
    "flex w-full items-center gap-2.5 rounded-xl px-2.5 py-2 text-left text-[13px] text-fg-2 transition-colors hover:bg-surface-2 hover:text-fg [&>svg]:size-4 [&>svg]:shrink-0 [&>svg]:text-muted";

  return (
    <div ref={ref} className="relative inline-flex shrink-0">
      <button
        type="button"
        onClick={copy}
        className="inline-flex h-8 items-center gap-1.5 rounded-l-[10px] bg-surface pr-2.5 pl-3 text-[13.5px] shadow-[0_0_0_1px_oklch(0%_0_0/0.09),0_1px_0_oklch(0%_0_0/0.06)] transition-colors hover:bg-surface-2"
      >
        {copied ? <IconCheck className="size-4 text-ok" stroke={1.75} aria-hidden="true" /> : <IconCopy className="size-4" stroke={1.75} aria-hidden="true" />}
        {copied ? "Copied" : "Copy page"}
      </button>
      <button
        type="button"
        aria-label="More options"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="ml-px inline-flex h-8 items-center rounded-r-[10px] bg-surface pr-2 pl-1.5 shadow-[0_0_0_1px_oklch(0%_0_0/0.09),0_1px_0_oklch(0%_0_0/0.06)] transition-colors hover:bg-surface-2"
      >
        <IconChevronDown className={cx("size-4 transition-transform", open && "rotate-180")} stroke={1.75} aria-hidden="true" />
      </button>
      {open && (
        <div
          role="menu"
          className="enter absolute top-10 right-0 z-30 w-64 rounded-2xl border border-line-2 bg-surface p-1.5 shadow-[0_12px_32px_-12px_oklch(0%_0_0/0.18)]"
        >
          <button type="button" role="menuitem" className={item} onClick={() => (copy(), setOpen(false))}>
            <IconMarkdown stroke={1.75} aria-hidden="true" />
            <span className="flex-1">
              Copy as Markdown
              <span className="block text-xs text-faint">Paste into any agent</span>
            </span>
          </button>
          <a role="menuitem" className={item} href={`/md/${page}`} target="_blank" rel="noopener">
            <IconExternalLink stroke={1.75} aria-hidden="true" />
            View as Markdown
          </a>
          <a role="menuitem" className={item} href="/llms.txt" target="_blank" rel="noopener">
            <IconRobot stroke={1.75} aria-hidden="true" />
            llms.txt
          </a>
          <div className="my-1 h-px bg-line" />
          <a role="menuitem" className={item} href={ask("https://claude.ai/new?q=")} target="_blank" rel="noopener noreferrer">
            <IconSparkles stroke={1.75} aria-hidden="true" />
            Open in Claude
          </a>
          <a role="menuitem" className={item} href={ask("https://chatgpt.com/?q=")} target="_blank" rel="noopener noreferrer">
            <IconBrandOpenai stroke={1.75} aria-hidden="true" />
            Open in ChatGPT
          </a>
        </div>
      )}
    </div>
  );
}

/** Table of contents that highlights the section being read. */
export function Toc({ items, mobile = false }: { items: TocItem[]; mobile?: boolean }) {
  const [active, setActive] = useState(items[0]?.id);

  useEffect(() => {
    const heads = items.map((i) => document.getElementById(i.id)).filter(Boolean) as HTMLElement[];
    const obs = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { rootMargin: "-80px 0px -70% 0px" },
    );
    heads.forEach((h) => obs.observe(h));
    return () => obs.disconnect();
  }, [items]);

  const list = (
    <ul className="space-y-0.5">
      {items.map((i) => (
        <li key={i.id}>
          <a
            href={`#${i.id}`}
            aria-current={active === i.id ? "location" : undefined}
            className={cx(
              "block rounded-lg py-1 text-[13px] leading-snug transition-colors",
              i.depth === 3 ? "pl-5" : "pl-2",
              active === i.id ? "bg-accent-soft font-medium text-accent-2" : "text-muted hover:text-fg",
            )}
          >
            {i.text}
          </a>
        </li>
      ))}
    </ul>
  );

  if (!mobile) return list;
  return (
    <details className="group rounded-2xl border border-line bg-surface px-3 py-2 lg:hidden">
      <summary className="flex cursor-pointer list-none items-center gap-2 text-[13px] font-medium [&::-webkit-details-marker]:hidden">
        <IconListTree className="size-4 text-muted" stroke={1.75} aria-hidden="true" />
        On this page
        <IconChevronDown className="ml-auto size-4 text-muted transition-transform group-open:rotate-180" stroke={1.75} aria-hidden="true" />
      </summary>
      <div className="mt-2 border-t border-line pt-2">{list}</div>
    </details>
  );
}
