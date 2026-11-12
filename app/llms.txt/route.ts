import { siteOrigin } from "@/lib/docs";

// llms.txt (https://llmstxt.org): a short Markdown index an agent can read
// to learn what this site is and where the full docs live.
export const dynamic = "force-static";

export function GET() {
  const o = siteOrigin();
  const body = `# PhishLens

> PhishLens checks whether a link is phishing. It combines threat lists, domain age, a safe read of the page (HTML only, scripts never run) and a lexical URL model into one explainable 0-100 risk score.

To scan a link, POST {"url": "..."} to ${o}/api/scan. Report the verdict (phishing, suspicious or safe), the risk score, and the titles of findings with a positive weight. A "safe" verdict means no warning signs were found, not a guarantee.

## Docs

- [How it works](${o}/md/how-it-works): how each check works, the scoring formula, accuracy, safety and the API
- [Research](${o}/md/research): the benchmark and findings the URL model and the scanner's design come from
- [Everything](${o}/llms-full.txt): all of the above as one Markdown file

## API

- [OpenAPI schema](${o}/api/openapi.json): machine-readable API description
- [Scan endpoint](${o}/api/docs#/default/scan_api_scan_post): POST /api/scan, full scan
- [Health](${o}/api/health): which sources are configured

## Optional

- [Source code](https://github.com/rajeswari-30/phishdriftbench)
`;
  return new Response(body, { headers: { "Content-Type": "text/plain; charset=utf-8" } });
}
