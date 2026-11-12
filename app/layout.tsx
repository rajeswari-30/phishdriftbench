import type { Metadata, Viewport } from "next";
import { Geist_Mono, Inter } from "next/font/google";
import Link from "next/link";
import { GithubMark } from "@/components/github-mark";
import { SiteHeader } from "@/components/site-header";
import "./globals.css";

const sans = Inter({ variable: "--font-inter", subsets: ["latin"] });
const mono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "PhishLens: check a link before you open it", template: "%s | PhishLens" },
  description:
    "PhishLens scans a link for phishing: threat lists, domain age, a safe read of the page, and an explainable risk score.",
};

export const viewport: Viewport = {
  themeColor: "#fafafa",
  colorScheme: "light",
};

const GITHUB = "https://github.com/rajeswari-30/phishdriftbench";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${sans.variable} ${mono.variable} flex min-h-dvh flex-col`}>
        <a
          href="#main"
          className="sr-only z-[60] rounded-[10px] bg-fg px-3 py-2 text-sm text-bg focus:not-sr-only focus:fixed focus:top-3 focus:left-3"
        >
          Skip to content
        </a>
        <SiteHeader github={GITHUB} />
        <main id="main" tabIndex={-1} className="flex-1 focus:outline-none">
          {children}
        </main>
        <footer className="border-t border-line">
          <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-6 text-[13px] text-muted sm:flex-row sm:items-center sm:px-6">
            <span>
              Developed by{" "}
              <a
                href="https://github.com/rajeswari-30"
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-fg underline decoration-line-2 underline-offset-4 transition-colors hover:decoration-fg"
              >
                Rajarajeswari V
              </a>
            </span>
            <div className="flex gap-4 sm:ml-auto">
              <Link href="/how-it-works" className="transition-colors hover:text-fg">How it works</Link>
              <Link href="/research" className="transition-colors hover:text-fg">Research</Link>
              <a href={GITHUB} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1.5 transition-colors hover:text-fg">
                <GithubMark className="size-3.5" />
                Source
              </a>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
