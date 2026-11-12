import { IconBook2, IconFileUnknown, IconScan } from "@tabler/icons-react";
import type { Metadata } from "next";
import Link from "next/link";
import { StatusPage } from "@/components/status-page";
import { pillClass } from "@/components/ui";

export const metadata: Metadata = { title: "Page not found" };

export default function NotFound() {
  return (
    <StatusPage
      icon={<IconFileUnknown stroke={1.5} aria-hidden="true" />}
      code="404"
      title="This page doesn't exist"
      body="The link may be mistyped, or the page may have moved."
    >
      <Link href="/" className={pillClass("primary", "md")}>
        <IconScan stroke={1.75} aria-hidden="true" />
        Scan a link
      </Link>
      <Link href="/how-it-works" className={pillClass("ghost", "md")}>
        <IconBook2 stroke={1.75} aria-hidden="true" />
        Read the docs
      </Link>
    </StatusPage>
  );
}
