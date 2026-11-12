import type { Health, ScanResult } from "./types";

export class ScanError extends Error {}

async function detail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) return "That doesn't look like a web address.";
  } catch {
    /* not JSON */
  }
  if (res.status === 429) return "Too many scans. Wait a minute and try again.";
  if (res.status >= 500) return "The scanner hit an error. Try again in a moment.";
  return `Request failed (HTTP ${res.status}).`;
}

export async function scan(url: string, deep: boolean, signal?: AbortSignal): Promise<ScanResult> {
  let res: Response;
  try {
    res = await fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, deep }),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ScanError("Couldn't reach the scanner. Check your connection and try again.");
  }
  if (!res.ok) throw new ScanError(await detail(res));
  return res.json();
}

export async function health(): Promise<Health | null> {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    return res.ok ? res.json() : null;
  } catch {
    return null;
  }
}

export function reportUrl(url: string, download = false): string {
  return `/api/report?url=${encodeURIComponent(url)}${download ? "&download=true" : ""}`;
}
