import { IconListTree } from "@tabler/icons-react";
import type { ReactNode } from "react";
import { docsMarkdown, renderDocs, type DocPage } from "@/lib/docs";
import { CopyPage, Toc } from "./docs-tools";
import { Badge } from "./ui";

/** Docs layout: sticky contents on the left, prose on the right, and a
 * "Copy page" menu that hands the page's Markdown to an agent. */
export function DocsPage({ page, eyebrow, aside }: { page: DocPage; eyebrow: ReactNode; aside?: ReactNode }) {
  const { title, html, toc } = renderDocs(page);
  return (
    <div className="mx-auto grid max-w-6xl gap-10 px-4 pt-10 pb-24 sm:px-6 sm:pt-14 lg:grid-cols-[200px_minmax(0,1fr)]">
      <aside className="hidden lg:block">
        <nav aria-label="On this page" className="sticky top-24">
          <p className="mb-2 flex items-center gap-2 pl-2 text-xs font-medium text-muted">
            <IconListTree className="size-3.5" stroke={1.75} aria-hidden="true" />
            On this page
          </p>
          <Toc items={toc} />
        </nav>
      </aside>

      <div className="min-w-0 max-w-3xl">
        <header className="enter relative z-20 mb-8">
          <Badge className="mb-4">{eyebrow}</Badge>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <h1 className="text-3xl font-semibold tracking-[-0.035em] text-balance sm:text-4xl">{title}</h1>
            <CopyPage markdown={docsMarkdown(page)} page={page} />
          </div>
        </header>
        {aside}
        <div className="mb-8">
          <Toc items={toc} mobile />
        </div>
        <article className="docs enter" dangerouslySetInnerHTML={{ __html: html }} />
      </div>
    </div>
  );
}
