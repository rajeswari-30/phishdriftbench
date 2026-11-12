"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Scan" },
  { href: "/how-it-works", label: "How it works" },
  { href: "/research", label: "Research" },
];

export function NavLinks({ github }: { github: string }) {
  const path = usePathname();
  return (
    <nav className="ml-auto flex items-center gap-5 text-sm">
      {LINKS.map((l) => (
        <Link
          key={l.href}
          href={l.href}
          aria-current={path === l.href ? "page" : undefined}
          className="whitespace-nowrap text-muted transition-colors hover:text-ink aria-[current=page]:text-ink"
        >
          {l.label}
        </Link>
      ))}
      <a
        href={github}
        target="_blank"
        rel="noopener noreferrer"
        className="hidden text-ink sm:inline"
      >
        GitHub ↗
      </a>
    </nav>
  );
}
