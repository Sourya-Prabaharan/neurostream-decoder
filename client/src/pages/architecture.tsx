import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

// EEG-specific pipeline. Each stage lists what it owns and what it hands off.
interface Stage {
  id: string;
  name: string;
  short: string;
  plain: string;
  io: { in: string; out: string };
  ref?: string;
}

const STAGES: Stage[] = [
  {
    id: "source",
    name: "EEG source",
    short: "PhysioNet / synthetic",
    plain:
      "Real EEG read from an EDF file at 160 Hz (PhysioNet eegmmidb v1.0.0), or a synthetic generator with the same shape.",
    io: { in: "64-ch electrode array", out: "raw samples (T, 64)" },
    ref: "engine/scripts/real_eeg_smoke.py",
  },
  {
    id: "transport",
    name: "Redis Stream / asyncio.Queue",
    short: "transport",
    plain:
      "Producer packs samples into fixed-size packets and pushes them onto the stream. Redis Streams in production, in-process queue when Redis isn't reachable.",
    io: { in: "EEGPacket(T=16, C=64)", out: "stream messages" },
    ref: "engine/neurostream/streaming_api/stream.py",
  },
  {
    id: "ring",
    name: "RingBuffer",
    short: "per-stream ring",
    plain:
      "Bounded circular buffer (~50 s @ 160 Hz). If the producer outruns the consumer, oldest samples are overwritten and counted as overruns — no unbounded memory growth.",
    io: { in: "stream messages", out: "samples in RAM" },
    ref: "engine/neurostream/streaming_api/ring_buffer.py",
  },
  {
    id: "window",
    name: "Sliding window",
    short: "1.0 s · 250 ms hop",
    plain:
      "Reads the latest 160 samples (1.0 s) without advancing the read cursor, so the next window overlaps by 75 %.",
    io: { in: "ring buffer", out: "window (160, 64)" },
  },
  {
    id: "preproc",
    name: "Preprocessing",
    short: "bandpass + z-score + rFFT",
    plain:
      "IIR bandpass (1–45 Hz) with state carried across windows, per-channel z-score, rFFT band powers (δ θ α β γ).",
    io: { in: "window (160, 64)", out: "features (64, 5)" },
    ref: "engine/neurostream/streaming_api/preprocessing.py",
  },
  {
    id: "cnn",
    name: "CNN inference",
    short: "EEGNetSmall · ~7.9 k params",
    plain:
      "Two 1-D conv blocks + linear head. eval mode, no_grad, batch=1 on CPU. ~0.6 ms p50 inference.",
    io: { in: "features (64, 5)", out: "logits (3,)" },
    ref: "engine/neurostream/streaming_api/classifier.py",
  },
  {
    id: "out",
    name: "Dashboard / metrics",
    short: "live decode + telemetry",
    plain:
      "Class + confidence go to the dashboard. Per-stage latency + queue stats are aggregated by MetricsAggregator and exposed at /metrics.",
    io: { in: "predictions + timings", out: "websocket / http json" },
    ref: "client/src/pages/eeg.tsx",
  },
];

export default function ArchitecturePage() {
  return (
    <div className="space-y-6 max-w-[1100px] mx-auto">
      <div>
        <h1 className="text-xl font-semibold tracking-tight" data-testid="text-architecture-title">
          Pipeline
        </h1>
        <p className="text-sm text-muted-foreground">
          One pass from electrode samples to a class label. Each stage carries state across
          windows so behaviour matches the offline batch run.
        </p>
      </div>

      {/* Flow diagram */}
      <Card className="border-card-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Data flow</CardTitle>
          <CardDescription>
            EEG Stream → Redis Stream buffer → Ring buffer / windowing → Preprocessing → CNN
            inference → Dashboard.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <FlowDiagram />
          <ol className="mt-6 space-y-3" data-testid="list-architecture-stages">
            {STAGES.map((s, i) => (
              <li
                key={s.id}
                className="flex items-start gap-4 rounded-md border border-card-border p-4 bg-card"
                data-testid={`stage-${i}`}
              >
                <div className="font-mono text-[11px] text-muted-foreground pt-0.5 w-6">
                  {String(i + 1).padStart(2, "0")}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-medium text-sm">{s.name}</span>
                    <Badge variant="outline" className="font-mono text-[10px]">
                      {s.short}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{s.plain}</p>
                  <div className="mt-2 flex items-center gap-2 text-[11px] font-mono text-muted-foreground flex-wrap">
                    <span className="rounded bg-muted/40 px-1.5 py-0.5">{s.io.in}</span>
                    <span className="text-muted-foreground/60">→</span>
                    <span className="rounded bg-muted/40 px-1.5 py-0.5">{s.io.out}</span>
                    {s.ref && (
                      <code className="ml-auto text-[10px] text-muted-foreground/80">{s.ref}</code>
                    )}
                  </div>
                </div>
              </li>
            ))}
          </ol>
        </CardContent>
      </Card>

      {/* Side notes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="border-card-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Hardware modes</CardTitle>
            <CardDescription>What runs where.</CardDescription>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground space-y-2 leading-relaxed">
            <p>
              <code className="font-mono text-foreground">torch.device('cuda')</code> is selected
              automatically if a GPU is visible. With no GPU, the entire pipeline runs on CPU and
              still hits 2–3 ms p50 end-to-end on a developer laptop.
            </p>
            <p>
              Mixed-precision <code className="font-mono">autocast</code> is enabled on CUDA only.
              The small CNN doesn't benefit much from fp16 on CPU.
            </p>
          </CardContent>
        </Card>
        <Card className="border-card-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Datasets</CardTitle>
            <CardDescription>What goes in.</CardDescription>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            <ul className="list-disc pl-4 space-y-1">
              <li>
                <span className="text-foreground font-medium">PhysioNet eegmmidb v1.0.0</span> —
                64-ch EEG @ 160 Hz, motor-imagery annotations T0 / T1 / T2.
              </li>
              <li>BCI2000 BNCI Horizon 2020 — EEG BCI competition data (planned).</li>
              <li>Synthetic generator — same shape, deterministic seed, for benchmarks.</li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Simple horizontal flow diagram drawn in SVG. No animations, no gradients —
// reads like a textbook diagram, not a marketing graphic.
// ---------------------------------------------------------------------------
function FlowDiagram() {
  const boxes = [
    { label: "EEG\nstream", sub: "160 Hz" },
    { label: "Redis\nStream", sub: "XADD/XREAD" },
    { label: "Ring buffer\n+ window", sub: "1.0 s / 250 ms" },
    { label: "Preprocess", sub: "bandpass · z-score · rFFT" },
    { label: "CNN\ninference", sub: "EEGNetSmall" },
    { label: "Dashboard", sub: "class · latency · throughput" },
  ];
  const w = 880;
  const h = 110;
  const boxW = 120;
  const boxH = 56;
  const gap = (w - boxes.length * boxW) / (boxes.length + 1);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" data-testid="diagram-flow">
      {boxes.map((b, i) => {
        const x = gap + i * (boxW + gap);
        const y = (h - boxH) / 2;
        return (
          <g key={i} data-testid={`flow-box-${i}`}>
            <rect
              x={x}
              y={y}
              width={boxW}
              height={boxH}
              rx={4}
              fill="hsl(var(--card))"
              stroke="hsl(var(--border))"
              strokeWidth={1}
            />
            {b.label.split("\n").map((line, j) => (
              <text
                key={j}
                x={x + boxW / 2}
                y={y + 20 + j * 12}
                textAnchor="middle"
                fontSize="11"
                fontFamily="var(--font-sans)"
                fill="hsl(var(--foreground))"
              >
                {line}
              </text>
            ))}
            <text
              x={x + boxW / 2}
              y={y + boxH - 6}
              textAnchor="middle"
              fontSize="9"
              fontFamily="var(--font-mono)"
              fill="hsl(var(--muted-foreground))"
            >
              {b.sub}
            </text>
            {i < boxes.length - 1 && (
              <g>
                <line
                  x1={x + boxW}
                  y1={h / 2}
                  x2={x + boxW + gap}
                  y2={h / 2}
                  stroke="hsl(var(--muted-foreground))"
                  strokeWidth={1}
                />
                <polygon
                  points={`${x + boxW + gap - 6},${h / 2 - 3} ${x + boxW + gap},${h / 2} ${x + boxW + gap - 6},${h / 2 + 3}`}
                  fill="hsl(var(--muted-foreground))"
                />
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}
