import type { ReactNode } from "react";

/** Centred full-page state (404, crash): an icon tile, a big code, one line
 * of explanation and the ways out. */
export function StatusPage({
  icon,
  code,
  title,
  body,
  children,
}: {
  icon: ReactNode;
  code: string;
  title: string;
  body: string;
  children: ReactNode;
}) {
  return (
    <div className="mx-auto flex min-h-[70dvh] max-w-md flex-col items-center justify-center px-4 py-16 text-center">
      <span className="enter grid size-14 place-items-center rounded-2xl border border-line-2 bg-surface text-muted [&>svg]:size-7">
        {icon}
      </span>
      <p className="enter mt-6 font-mono text-sm text-muted" style={{ animationDelay: "40ms" }}>
        {code}
      </p>
      <h1 className="enter mt-2 text-3xl font-semibold tracking-[-0.03em] sm:text-4xl" style={{ animationDelay: "80ms" }}>
        {title}
      </h1>
      <p className="enter mt-3 text-[15px] leading-relaxed text-muted" style={{ animationDelay: "120ms" }}>
        {body}
      </p>
      <div className="enter mt-8 flex flex-wrap justify-center gap-2" style={{ animationDelay: "160ms" }}>
        {children}
      </div>
    </div>
  );
}
