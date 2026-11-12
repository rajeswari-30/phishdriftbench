import { z } from "zod";

// Client-side checks mirror the server's input repair (src/phishdriftbench/
// demo/trust.py): they catch what can never be a web link before a request
// is made, and leave everything subtler (private addresses, dead domains)
// to the server, which answers with its own message.

const OPAQUE_SCHEME = /^(mailto|javascript|data|tel|file|about|chrome|ftp|ssh|sms|blob):/i;

function repair(raw: string): string {
  let s = raw.trim().replace(/^[<"'“]+|[>"'”]+$/g, "");
  s = s.replace(/^hxxp(s?)(?:\[:\]|:)\/\//i, "http$1://").replace(/\[\.\]|\(\.\)|\{\.\}/g, ".");
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(s)) s = "https://" + s.replace(/^\/+/, "");
  return s;
}

export const linkInput = z
  .string()
  .trim()
  .min(1, "Paste a link to scan.")
  .max(4096, "That link is too long. Links can be up to 4,096 characters.")
  .refine((s) => !OPAQUE_SCHEME.test(s), {
    message: "Only web links (http or https) can be scanned.",
  })
  .refine(
    (s) => {
      try {
        const u = new URL(repair(s));
        if (u.protocol !== "http:" && u.protocol !== "https:") return false;
        return u.hostname.includes(".") || u.hostname === "localhost";
      } catch {
        return false;
      }
    },
    { message: "That doesn't look like a web address. Try something like example.com/login." },
  );

export function validateLink(raw: string): { ok: true; value: string } | { ok: false; error: string } {
  const r = linkInput.safeParse(raw);
  return r.success ? { ok: true, value: r.data } : { ok: false, error: r.error.issues[0]?.message ?? "Invalid link." };
}

// Only the fields the UI cannot render without. Everything else is optional
// in practice, and a missing extra must not take the whole result down.
const finding = z.object({
  id: z.string(),
  category: z.enum(["url", "domain", "page", "intel", "ai"]),
  title: z.string(),
  detail: z.string(),
  weight: z.number(),
  severity: z.enum(["critical", "high", "medium", "low", "info", "good"]),
});

export const scanResultShape = z.looseObject({
  url: z.string(),
  final_url: z.string(),
  risk: z.number().min(0).max(100),
  verdict: z.enum(["phishing", "suspicious", "safe"]),
  evidence: z.enum(["strong", "moderate", "limited"]),
  summary: z.string(),
  findings: z.array(finding),
  domain: z.looseObject({ domain: z.string() }),
  intel: z.array(z.looseObject({ source: z.string(), status: z.string() })),
  url_model: z.looseObject({ signals: z.array(z.looseObject({ name: z.string() })) }),
});
