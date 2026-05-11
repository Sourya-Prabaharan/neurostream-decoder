import { useMemo } from "react";
import type { CursorPoint } from "@/lib/types";

interface Props {
  points: CursorPoint[];
  current?: CursorPoint;
  size?: number;
  testId?: string;
}

/** Decoded 2-D cursor trajectory. */
export function CursorTrail({ points, current, size = 220, testId = "chart-cursor" }: Props) {
  const inset = 14;
  const draw = useMemo(() => {
    if (!points.length) return "";
    return points
      .map((p, i) => {
        const x = inset + ((p.x + 1) / 2) * (size - inset * 2);
        const y = inset + ((1 - (p.y + 1) / 2)) * (size - inset * 2);
        return (i === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1);
      })
      .join(" ");
  }, [points, size]);

  if (!points.length) {
    return (
      <div
        className="flex items-center justify-center text-xs text-muted-foreground border border-dashed border-border rounded-md"
        style={{ width: size, height: size }}
        data-testid={`${testId}-empty`}
      >
        No cursor data
      </div>
    );
  }

  const cx = current
    ? inset + ((current.x + 1) / 2) * (size - inset * 2)
    : null;
  const cy = current
    ? inset + ((1 - (current.y + 1) / 2)) * (size - inset * 2)
    : null;

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="w-full h-auto" data-testid={testId}>
      <rect
        x="1"
        y="1"
        width={size - 2}
        height={size - 2}
        rx="6"
        fill="hsl(var(--card))"
        stroke="hsl(var(--border))"
      />
      {/* crosshair */}
      <line x1={size / 2} y1={inset} x2={size / 2} y2={size - inset} stroke="hsl(var(--border))" strokeDasharray="2 3" />
      <line x1={inset} y1={size / 2} x2={size - inset} y2={size / 2} stroke="hsl(var(--border))" strokeDasharray="2 3" />
      <path d={draw} fill="none" stroke="hsl(var(--chart-2))" strokeWidth="1.4" strokeOpacity="0.85" strokeLinecap="round" strokeLinejoin="round" />
      {cx !== null && cy !== null && (
        <>
          <circle cx={cx} cy={cy} r="9" fill="hsl(var(--primary) / 0.18)" />
          <circle cx={cx} cy={cy} r="3.2" fill="hsl(var(--primary))" />
        </>
      )}
    </svg>
  );
}
