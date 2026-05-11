import { useMemo } from "react";

interface OscilloscopeProps {
  time: number[];
  channels: number[][];
  spikeRaster?: number[][];
  height?: number;
  testId?: string;
  /** Optional time-window highlighted in primary colour (e.g. decoder cursor) */
  highlight?: { start: number; end: number };
}

/** Multi-channel neural oscilloscope rendered as a single SVG. */
export function Oscilloscope({
  time,
  channels,
  spikeRaster,
  height = 360,
  testId = "chart-oscilloscope",
  highlight,
}: OscilloscopeProps) {
  const width = 880;
  const padding = { top: 8, right: 16, bottom: 22, left: 36 };

  const { paths, ticks, t0, t1, lanes } = useMemo(() => {
    if (!time.length || !channels.length) {
      return { paths: [] as string[], ticks: [] as { t: number; x: number }[], t0: 0, t1: 1, lanes: [] as { y: number; h: number }[] };
    }
    const t0 = time[0];
    const t1 = time[time.length - 1];
    const xs = (t: number) =>
      padding.left + ((t - t0) / Math.max(t1 - t0, 1e-6)) * (width - padding.left - padding.right);
    const laneH = (height - padding.top - padding.bottom) / channels.length;
    const lanes = channels.map((_c, i) => ({
      y: padding.top + i * laneH + laneH / 2,
      h: laneH,
    }));

    // Normalize each channel independently to its own peak so the stack reads cleanly.
    const paths = channels.map((ch, idx) => {
      const peak = Math.max(...ch.map((v) => Math.abs(v))) || 1;
      const lane = lanes[idx];
      const amplitude = lane.h * 0.42;
      let d = "";
      for (let i = 0; i < ch.length; i++) {
        const x = xs(time[i]);
        const y = lane.y - (ch[i] / peak) * amplitude;
        d += (i === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1) + " ";
      }
      return d;
    });

    // 5 evenly spaced ticks
    const ticks = Array.from({ length: 5 }, (_, i) => {
      const t = t0 + ((t1 - t0) * i) / 4;
      return { t, x: xs(t) };
    });
    return { paths, ticks, t0, t1, lanes };
  }, [time, channels, height]);

  if (!time.length) {
    return (
      <div
        className="flex items-center justify-center text-xs text-muted-foreground border border-dashed border-border rounded-md"
        style={{ height }}
        data-testid={`${testId}-empty`}
      >
        Awaiting signal…
      </div>
    );
  }

  const xs = (t: number) =>
    padding.left + ((t - t0) / Math.max(t1 - t0, 1e-6)) * (width - padding.left - padding.right);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-auto"
      preserveAspectRatio="none"
      data-testid={testId}
    >
      <defs>
        <linearGradient id="osc-bg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="hsl(var(--card))" stopOpacity="0.0" />
          <stop offset="100%" stopColor="hsl(var(--background))" stopOpacity="0.0" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width={width} height={height} fill="url(#osc-bg)" />
      {/* grid */}
      {ticks.map((tk, i) => (
        <line
          key={i}
          x1={tk.x}
          x2={tk.x}
          y1={padding.top}
          y2={height - padding.bottom}
          stroke="hsl(var(--border))"
          strokeWidth={0.5}
          strokeDasharray="2 3"
        />
      ))}
      {highlight && (
        <rect
          x={xs(highlight.start)}
          y={padding.top}
          width={Math.max(2, xs(highlight.end) - xs(highlight.start))}
          height={height - padding.top - padding.bottom}
          fill="hsl(var(--primary) / 0.07)"
          stroke="hsl(var(--primary) / 0.35)"
          strokeWidth={0.5}
        />
      )}
      {/* channel separators */}
      {lanes.map((l, i) => (
        <line
          key={i}
          x1={padding.left}
          x2={width - padding.right}
          y1={l.y}
          y2={l.y}
          stroke="hsl(var(--border))"
          strokeOpacity={0.25}
          strokeWidth={0.5}
        />
      ))}
      {/* spike raster ticks */}
      {spikeRaster?.map((times, i) =>
        times
          .filter((t) => t >= t0 && t <= t1)
          .map((t, j) => {
            const lane = lanes[i % lanes.length];
            const x = xs(t);
            return (
              <line
                key={`sp-${i}-${j}`}
                x1={x}
                x2={x}
                y1={lane.y - lane.h * 0.32}
                y2={lane.y + lane.h * 0.32}
                stroke="hsl(var(--chart-3))"
                strokeOpacity={0.65}
                strokeWidth={0.7}
              />
            );
          }),
      )}
      {/* signal traces */}
      {paths.map((d, i) => (
        <path
          key={i}
          d={d}
          fill="none"
          stroke="hsl(var(--chart-1))"
          strokeOpacity={0.95}
          strokeWidth={1.1}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      ))}
      {/* x-axis */}
      <line
        x1={padding.left}
        x2={width - padding.right}
        y1={height - padding.bottom}
        y2={height - padding.bottom}
        stroke="hsl(var(--border))"
      />
      {ticks.map((tk, i) => (
        <text
          key={`tx-${i}`}
          x={tk.x}
          y={height - 6}
          textAnchor="middle"
          fontSize="10"
          fill="hsl(var(--muted-foreground))"
          fontFamily="var(--font-mono)"
        >
          {tk.t.toFixed(2)}s
        </text>
      ))}
      {/* per-lane channel labels — one for every displayed trace */}
      {lanes.map((lane, i) => (
        <text
          key={`lbl-${i}`}
          x={padding.left - 6}
          y={lane.y + 3}
          textAnchor="end"
          fontSize="10"
          fill="hsl(var(--muted-foreground))"
          fontFamily="var(--font-mono)"
          data-testid={`text-osc-channel-${i}`}
        >
          ch{i}
        </text>
      ))}
    </svg>
  );
}
