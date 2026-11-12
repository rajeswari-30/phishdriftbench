import type { Health, ScanResult } from "./types";
import { scanResultShape } from "./validate";

/** Why a scan failed, so the UI can say the right thing in the right place:
 * an input problem belongs under the field, everything else in a card with
 * a way to retry. */
export type ScanErrorKind = "input" | "network" | "offline" | "rate" | "server" | "unexpected";

export class ScanError extends Error {
  constructor(
    public kind: ScanErrorKind,
    message: string,
    public retryAfter?: number,
  ) {
    super(message);
  }
}

async function detail(res: Response): Promise<string | null> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    /* not JSON */
  }
  return null;
}

export async function scan(url: string, deep: boolean, signal?: AbortSignal): Promise<ScanResult> {
  if (typeof navigator !== "undefined" && navigator.onLine === false) {
    throw new ScanError("offline", "You're offline. Reconnect and try again.");
  }
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
    throw new ScanError("network", "Couldn't reach the scanner. Check your connection and try again.");
  }

  if (res.status === 400 || res.status === 422) {
    throw new ScanError("input", (await detail(res)) ?? "That doesn't look like a web address.");
  }
  if (res.status === 429) {
    const after = Number(res.headers.get("retry-after")) || 60;
    throw new ScanError("rate", "Too many scans in a short time. Wait a moment, then try again.", after);
  }
  if (!res.ok) {
    throw new ScanError("server", "The scanner ran into a problem on its side. Try again in a moment.");
  }

  let body: unknown;
  try {
    body = await res.json();
  } catch {
    throw new ScanError("unexpected", "The scanner sent back something unreadable. Try again.");
  }
  const parsed = scanResultShape.safeParse(body);
  if (!parsed.success) {
    throw new ScanError("unexpected", "The scanner's answer was incomplete. Try again.");
  }
  return body as ScanResult;
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
