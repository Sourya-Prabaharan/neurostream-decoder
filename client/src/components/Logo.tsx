export function Logo({ className = "h-8 w-8" }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 32 32"
      fill="none"
      aria-label="NeuroStream Decoder logo"
      role="img"
    >
      <rect x="1" y="1" width="30" height="30" rx="7" stroke="currentColor" strokeWidth="1.2" opacity="0.35" />
      <path
        d="M3 18 L8 18 L10 10 L14 24 L18 14 L21 20 L29 20"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      <circle cx="10" cy="10" r="1.6" fill="currentColor" />
      <circle cx="18" cy="14" r="1.6" fill="currentColor" />
    </svg>
  );
}
