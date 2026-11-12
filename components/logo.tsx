export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" fill="none" aria-hidden="true" className={className}>
      <circle cx="14" cy="14" r="9.25" stroke="currentColor" strokeWidth="2.4" />
      <path d="M20.9 20.9 L27.5 27.5" stroke="currentColor" strokeWidth="2.9" strokeLinecap="round" />
      <path
        d="M10.1 14.35 L12.85 17.1 L18 11.1"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
