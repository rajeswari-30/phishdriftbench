import { DOC_PAGES, docsMarkdown } from "@/lib/docs";

// Every docs page as one Markdown file, for agents that want it all at once.
export const dynamic = "force-static";

export function GET() {
  const body = DOC_PAGES.map((page) => docsMarkdown(page)).join("\n\n---\n\n");
  return new Response(body, { headers: { "Content-Type": "text/plain; charset=utf-8" } });
}
