import type { Metadata, Viewport } from "next";
import { Geist_Mono, Inter } from "next/font/google";
import Link from "next/link";
import { SiteHeader } from "@/components/site-header";
import "./globals.css";

const sans = Inter({ variable: "--font-inter", subsets: ["latin"] });
const mono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "PhishLens · Check a link before you open it", template: "%s · PhishLens" },
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
        <SiteHeader github={GITHUB} />
        <main className="flex-1">{children}</main>
        <footer className="border-t border-line">
          <div className="mx-auto flex max-w-3xl flex-col gap-3 px-4 py-6 text-[13px] text-muted sm:flex-row sm:items-center sm:px-6">
            <span>PhishLens · PhishDriftBench</span>
            <div className="flex gap-4 sm:ml-auto">
              <Link href="/how-it-works" className="transition-colors hover:text-fg">How it works</Link>
              <Link href="/research" className="transition-colors hover:text-fg">Research</Link>
              <a href={GITHUB} target="_blank" rel="noopener noreferrer" className="transition-colors hover:text-fg">
                Source
              </a>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
