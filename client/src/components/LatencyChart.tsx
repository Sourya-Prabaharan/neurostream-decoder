import { useMemo } from "react";

interface Props {
  latencies: number[];
  testId?: string;
  height?: number;
}

/** Compact sparkline of per-window latency with p50/p95 reference lines. */
export function LatencyChart({ latencies, testId = "chart-latency", height = 120 }: Props) {
  const width = 480;
  const padding = { top: 8, right: 8, bottom: 16, left: 28 };

  const { path, p50, p95, max } = useMemo(() => {
    if (!latencies.length) return { path: "", p50: 0, p95: 0, max: 1 };
    const sorted = [...latencies].sort((a, b) => a - b);
    const pick = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
    const max = Math.max(...latencies, 1);
    const path = latencies
      .map((v, i) => {
        const x = padding.left + (i / Math.max(latencies.length - 1, 1)) * (width - padding.left - padding.right);
        const y = padding.top + (1 - v / max) * (height - padding.top - padding.bottom);
        return (i === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1);
      })
      .join(" ");
    return { path, p50: pick(50), p95: pick(95), max };
  }, [latencies, height]);

  const lineY = (v: number) =>
    padding.top + (1 - v / max) * (height - padding.top - padding.bottom);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto" data-testid={testId}>
      <rect x="0" y="0" width={width} height={height} fill="transparent" />
      <line x1={padding.left} x2={width - padding.right} y1={lineY(p50)} y2={lineY(p50)} stroke="hsl(var(--chart-2))" strokeDasharray="3 4" strokeOpacity="0.55" />
      <line x1={padding.left} x2={width - padding.right} y1={lineY(p95)} y2={lineY(p95)} stroke="hsl(var(--chart-4))" strokeDasharray="3 4" strokeOpacity="0.55" />
      <path d={path} fill="none" stroke="hsl(var(--chart-1))" strokeWidth="1.3" />
      <text x={padding.left} y={lineY(p50) - 4} fontSize="9" fontFamily="var(--font-mono)" fill="hsl(var(--chart-2))">
        p50 {p50.toFixed(2)}ms
      </text>
      <text x={padding.left} y={lineY(p95) - 4} fontSize="9" fontFamily="var(--font-mono)" fill="hsl(var(--chart-4))">
        p95 {p95.toFixed(2)}ms
      </text>
      <text x={width - padding.right} y={height - 4} fontSize="9" textAnchor="end" fontFamily="var(--font-mono)" fill="hsl(var(--muted-foreground))">
        {latencies.length} windows
      </text>
    </svg>
  );
}
