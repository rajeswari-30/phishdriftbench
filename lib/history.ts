import type { ScanResult, Verdict } from "./types";

// Recent scans live only in this browser. Storage can be unavailable
// (private windows, blocked site data), so every access is guarded and the
// page works without it.
const KEY = "phishlens:recent";
const MAX = 8;

export interface RecentScan {
  url: string;
  risk: number;
  verdict: Verdict;
  at: string;
}

export function loadRecent(): RecentScan[] {
  try {
    const raw = window.localStorage.getItem(KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.slice(0, MAX) : [];
  } catch {
    return [];
  }
}

export function remember(result: ScanResult): RecentScan[] {
  const entry: RecentScan = {
    url: result.url,
    risk: result.risk,
    verdict: result.verdict,
    at: result.scanned_at,
  };
  const next = [entry, ...loadRecent().filter((r) => r.url !== entry.url)].slice(0, MAX);
  try {
    window.localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* storage unavailable: keep the in-memory list */
  }
  return next;
}

export function forget(): void {
  try {
    window.localStorage.removeItem(KEY);
  } catch {
    /* nothing to clear */
  }
}
