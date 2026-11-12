import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import { Logo } from "@/components/logo";
import { NavLinks } from "@/components/nav-links";
import "./globals.css";

const sans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const mono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "PhishLens · Check a link before you open it", template: "%s · PhishLens" },
  description:
    "PhishLens scans a link for phishing: threat lists, domain age, a safe read of the page, and an explainable risk score.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fafaf9" },
    { media: "(prefers-color-scheme: dark)", color: "#0e0e0d" },
  ],
};

const GITHUB = "https://github.com/rajeswari-30/phishdriftbench";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${sans.variable} ${mono.variable} min-h-dvh flex flex-col`}>
        <header className="sticky top-0 z-40 border-b border-rule bg-paper/85 backdrop-blur-md backdrop-saturate-150">
          <div className="mx-auto flex h-14 max-w-6xl items-center gap-6 px-4 sm:px-6">
            <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
              <Logo className="size-5" />
              PhishLens
            </Link>
            <NavLinks github={GITHUB} />
          </div>
        </header>
        <main className="flex-1">{children}</main>
        <footer className="border-t border-rule">
          <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-8 text-sm text-muted sm:flex-row sm:items-center sm:px-6">
            <span>
              <b className="font-semibold text-ink">PhishLens</b> · built on the PhishDriftBench research
              project.
            </span>
            <div className="flex gap-5 sm:ml-auto">
              <Link href="/how-it-works" className="hover:text-ink">How it works</Link>
              <Link href="/research" className="hover:text-ink">Research</Link>
              <a href={GITHUB} target="_blank" rel="noopener noreferrer" className="hover:text-ink">
                Source
              </a>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
