import { DOC_PAGES, docsMarkdown, type DocPage } from "@/lib/docs";

// /md/how-it-works and /md/research: each docs page as raw Markdown.
export const dynamic = "force-static";
export const dynamicParams = false;

export function generateStaticParams() {
  return DOC_PAGES.map((page) => ({ page }));
}

export async function GET(_req: Request, { params }: { params: Promise<{ page: string }> }) {
  const { page } = await params;
  return new Response(docsMarkdown(page as DocPage), {
    headers: { "Content-Type": "text/markdown; charset=utf-8" },
  });
}
