import "server-only";
import { readFileSync } from "node:fs";
import path from "node:path";
import { Marked } from "marked";

// Each docs page is one Markdown file in content/. That file is the single
// source for the rendered page, its "Copy page" text, /md/<page> and
// /llms-full.txt, so none of them can disagree.
export const DOC_PAGES = ["how-it-works", "research"] as const;
export type DocPage = (typeof DOC_PAGES)[number];

export interface TocItem {
  id: string;
  text: string;
  depth: 2 | 3;
}

export function docsMarkdown(page: DocPage): string {
  return readFileSync(path.join(process.cwd(), "content", `${page}.md`), "utf8");
}

// marked escapes text for HTML; the contents list renders it as plain text.
function decode(text: string): string {
  return text
    .replace(/&#39;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");
}

function slug(text: string): string {
  return text
    .toLowerCase()
    .replace(/<[^>]+>/g, "")
    .replace(/&(?:[a-z]+|#\d+);/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

/** Rendered HTML (without the H1, which the page lays out itself), the
 * title, the lead paragraph and the table of contents. The Markdown is our
 * own file, not user input. */
export function renderDocs(page: DocPage): { title: string; html: string; toc: TocItem[] } {
  const md = docsMarkdown(page);
  const title = md.match(/^# (.+)$/m)?.[1] ?? "Docs";
  const toc: TocItem[] = [];
  const marked = new Marked({
    gfm: true,
    renderer: {
      heading({ tokens, depth }) {
        const text = this.parser.parseInline(tokens);
        if (depth === 1) return "";
        const id = slug(text);
        if (depth === 2 || depth === 3) toc.push({ id, text: decode(text.replace(/<[^>]+>/g, "")), depth });
        return `<h${depth} id="${id}"><a href="#${id}" class="anchor" aria-hidden="true">#</a>${text}</h${depth}>`;
      },
      table(token) {
        // Wrapped so a wide table scrolls inside the column, not the page.
        const header = token.header.map((c) => `<th>${this.parser.parseInline(c.tokens)}</th>`).join("");
        const rows = token.rows
          .map((r) => `<tr>${r.map((c) => `<td>${this.parser.parseInline(c.tokens)}</td>`).join("")}</tr>`)
          .join("");
        return `<div class="table-wrap"><table><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table></div>`;
      },
      blockquote({ tokens }) {
        return `<aside class="callout">${this.parser.parse(tokens)}</aside>`;
      },
    },
  });
  const html = marked.parse(md, { async: false }) as string;
  return { title, html, toc };
}

/** Absolute site origin when known at build time (Vercel), else "". */
export function siteOrigin(): string {
  const host = process.env.VERCEL_PROJECT_PRODUCTION_URL ?? process.env.VERCEL_URL;
  return host ? `https://${host}` : "";
}
