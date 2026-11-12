"use client";

import { IconBug, IconHome, IconRefresh } from "@tabler/icons-react";
import Link from "next/link";
import { useEffect } from "react";
import { StatusPage } from "@/components/status-page";
import { pillClass } from "@/components/ui";

// Catches any render error below the root layout, so a crash shows a way
// back instead of a blank page.
export default function ErrorPage({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <StatusPage
      icon={<IconBug stroke={1.5} aria-hidden="true" />}
      code={error.digest ? `Error ${error.digest}` : "Error"}
      title="Something broke on this page"
      body="It's not your link. Try again, and if it keeps happening, go back to the scanner."
    >
      <button type="button" onClick={reset} className={pillClass("primary", "md")}>
        <IconRefresh stroke={1.75} aria-hidden="true" />
        Try again
      </button>
      <Link href="/" className={pillClass("ghost", "md")}>
        <IconHome stroke={1.75} aria-hidden="true" />
        Home
      </Link>
    </StatusPage>
  );
}
