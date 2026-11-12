import { IconBook2 } from "@tabler/icons-react";
import type { Metadata } from "next";
import { DocsPage } from "@/components/docs-page";

export const metadata: Metadata = {
  title: "How it works",
  description: "How PhishLens scans a link: threat lists, the page scan, domain facts, the URL model, scoring and the API.",
  alternates: { types: { "text/markdown": "/md/how-it-works" } },
};

export default function HowItWorks() {
  return (
    <DocsPage
      page="how-it-works"
      eyebrow={
        <>
          <IconBook2 stroke={1.75} aria-hidden="true" />
          Docs
        </>
      }
    />
  );
}
