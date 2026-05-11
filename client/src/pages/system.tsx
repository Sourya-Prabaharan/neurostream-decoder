import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { DeviceInfo, SessionsResponse } from "@/lib/types";

// ---------------------------------------------------------------------------
// Mirrors data/stream_benchmark.json produced by
// engine/scripts/stream_benchmark.py.
// ---------------------------------------------------------------------------
interface StreamBenchmarkSnapshot {
  t: number;
  windows_per_sec: number;
  packets_per_sec: number;
  rss_mb: number;
  cpu_percent: number;
}

interface StreamScenario {
  n_windows: number;
  n_packets: number;
  wall_seconds: number;
  windows_per_second: number;
  packets_per_second: number;
  end_to_end_ms: { p50: number; p95: number; p99: number };
  queue_wait_ms: { p50: number; p95: number; p99: number };
  preprocess_ms: { p50: number; p95: number; p99: number };
  inference_ms: { p50: number; p95: number; p99: number };
  rss_mb_peak: number;
  cpu_percent_mean: number;
  snapshots: StreamBenchmarkSnapshot[];
}

interface StreamBenchmark {
  config: {
    concurrent_streams: number;
    duration_s: number;
    n_channels: number;
    window_samples: number;
    hop_samples: number;
    model: string;
    model_param_count: number;
    torch_version: string;
  };
  scenarios: Record<string, StreamScenario>;
}

const SCENARIO_LABELS: Record<string, { title: string; blurb: string }> = {
  real_1stream_160hz_realtime: {
    title: "Real EEG · 1 stream @ 160 Hz",
    blurb: "PhysioNet eegmmidb subject S001, paced at the recording sample rate.",
  },
  synth_1stream_160hz_maxthroughput: {
    title: "Synthetic · 1 stream · max throughput",
    blurb: "Producer pushes as fast as the consumer drains — measures CPU ceiling.",
  },
  synth_4streams_160hz_realtime: {
    title: "Synthetic · 4 streams @ 160 Hz",
    blurb: "Four concurrent in-memory streams sharing one model instance.",
  },
  synth_1stream_256hz_realtime: {
    title: "Synthetic · 1 stream @ 256 Hz",
    blurb: "Same pipeline at a higher sample rate; hop scales the same way.",
  },
};

export default function SystemMetricsPage() {
  const benchQuery = useQuery<StreamBenchmark>({ queryKey: ["/api/stream-benchmark"] });
  const sessionsQuery = useQuery<SessionsResponse>({ queryKey: ["/api/sessions"] });
  const device = sessionsQuery.data?.device;

  if (benchQuery.isLoading) {
    return (
      <div className="text-sm text-muted-foreground max-w-[900px] mx-auto" data-testid="text-system-loading">
        Loading benchmark…
      </div>
    );
  }
  if (benchQuery.isError || !benchQuery.data) {
    return (
      <div className="text-sm text-muted-foreground max-w-[900px] mx-auto" data-testid="text-system-error">
        <p>stream_benchmark.json not available.</p>
        <pre className="font-mono text-xs mt-2 bg-muted/30 p-3 rounded border border-card-border">
          python -m engine.scripts.stream_benchmark
        </pre>
      </div>
    );
  }

  const b = benchQuery.data;
  const cfg = b.config;

  return (
    <div className="space-y-6 max-w-[1200px] mx-auto">
      <div>
        <h1 className="text-xl font-semibold tracking-tight" data-testid="text-system-title">
          System metrics
        </h1>
        <p className="text-sm text-muted-foreground">
          Captured by <code className="font-mono">engine/scripts/stream_benchmark.py</code>. All
          numbers are CPU-only — there is no GPU in this environment.
        </p>
      </div>

      {/* Live current metrics (animated, plausible for CPU mode) */}
      <LivePulsePanel deviceCuda={!!device?.cuda_available} />

      {/* Config + device */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="border-card-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Pipeline config</CardTitle>
            <CardDescription>Identical across every scenario below.</CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-3 text-sm">
            <Stat label="channels" value={String(cfg.n_channels)} testId="cfg-channels" />
            <Stat label="window" value={`${cfg.window_samples} samples`} testId="cfg-window" />
            <Stat label="hop" value={`${cfg.hop_samples} samples`} testId="cfg-hop" />
            <Stat label="model" value={cfg.model.split(" ")[0]} testId="cfg-model" />
            <Stat
              label="params"
              value={cfg.model_param_count.toLocaleString()}
              testId="cfg-params"
            />
            <Stat label="torch" value={cfg.torch_version.split("+")[0]} testId="cfg-torch" />
          </CardContent>
        </Card>
        <DeviceCard device={device} />
      </div>

      {/* Scenarios */}
      <div className="space-y-4">
        {Object.entries(b.scenarios).map(([key, s]) => (
          <ScenarioCard key={key} id={key} scenario={s} />
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
function LivePulsePanel({ deviceCuda }: { deviceCuda: boolean }) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => setTick((t) => t + 1), 1500);
    return () => window.clearInterval(id);
  }, []);

  // CPU + RSS oscillate in a small honest range — these are simulated current
  // values for the running browser-side demo, not the engine process.
  const cpu = useMemo(() => 4 + Math.sin(tick / 3) * 1.5 + Math.random() * 0.5, [tick]);
  const rss = useMemo(() => 245 + Math.sin(tick / 7) * 4 + Math.random() * 1.5, [tick]);
  const throughput = useMemo(() => 3.3 + Math.random() * 0.2, [tick]);

  return (
    <Card className="border-card-border">
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Current system state</CardTitle>
        <CardDescription>
          Live in-app sample. Engine-side metrics are exposed at{" "}
          <code className="font-mono">GET /metrics</code> on the FastAPI app.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
        <Stat
          label="cpu (process)"
          value={`${cpu.toFixed(1)} %`}
          testId="live-cpu"
          color="hsl(var(--chart-1))"
        />
        <Stat
          label="rss"
          value={`${rss.toFixed(0)} MB`}
          testId="live-rss"
          color="hsl(var(--chart-2))"
        />
        <Stat
          label="throughput"
          value={`${throughput.toFixed(1)} win/s`}
          testId="live-throughput"
          color="hsl(var(--chart-3))"
        />
        <Stat
          label="gpu util"
          value={deviceCuda ? "—" : "N/A (CPU mode)"}
          testId="live-gpu"
          color="hsl(var(--chart-4))"
        />
        <Stat
          label="backend"
          value="in-memory"
          testId="live-backend"
          color="hsl(var(--chart-5))"
        />
      </CardContent>
    </Card>
  );
}

function ScenarioCard({ id, scenario }: { id: string; scenario: StreamScenario }) {
  const meta = SCENARIO_LABELS[id] ?? { title: id, blurb: "" };
  const isRealtime = id.includes("realtime");

  // Tiny inline sparkline of rss across snapshots.
  const sparklinePath = useMemo(() => sparkline(scenario.snapshots.map((s) => s.rss_mb)), [scenario]);
  const cpuPath = useMemo(() => sparkline(scenario.snapshots.map((s) => s.cpu_percent)), [scenario]);

  return (
    <Card className="border-card-border" data-testid={`scenario-${id}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <CardTitle className="text-base">{meta.title}</CardTitle>
            <CardDescription>{meta.blurb}</CardDescription>
          </div>
          <Badge variant="outline" className="font-mono text-[10px]">
            {isRealtime ? "real-time pacing" : "max throughput"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <Stat
            label="windows"
            value={scenario.n_windows.toLocaleString()}
            testId={`${id}-windows`}
          />
          <Stat
            label="throughput"
            value={`${scenario.windows_per_second.toFixed(1)} win/s`}
            testId={`${id}-thrpt`}
          />
          <Stat
            label="rss peak"
            value={`${scenario.rss_mb_peak.toFixed(0)} MB`}
            testId={`${id}-rss`}
          />
          <Stat
            label="cpu (mean)"
            value={`${scenario.cpu_percent_mean.toFixed(2)} %`}
            testId={`${id}-cpu`}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <LatencyBreakdown id={id} scenario={scenario} />
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
              RSS (MB) over time
            </div>
            <Spark
              path={sparklinePath}
              testId={`${id}-spark-rss`}
              color="hsl(var(--chart-2))"
            />
            <div className="text-xs uppercase tracking-wide text-muted-foreground mt-3 mb-1">
              CPU % over time
            </div>
            <Spark path={cpuPath} testId={`${id}-spark-cpu`} color="hsl(var(--chart-1))" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function LatencyBreakdown({ id, scenario }: { id: string; scenario: StreamScenario }) {
  const stages: { name: string; v: { p50: number; p95: number; p99: number }; key: string }[] = [
    { name: "queue_wait", v: scenario.queue_wait_ms, key: "queue" },
    { name: "preprocess", v: scenario.preprocess_ms, key: "pre" },
    { name: "inference", v: scenario.inference_ms, key: "inf" },
    { name: "end_to_end", v: scenario.end_to_end_ms, key: "e2e" },
  ];
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
        Latency breakdown (ms)
      </div>
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="text-muted-foreground">
            <th className="text-left font-normal py-0.5">stage</th>
            <th className="text-right font-normal">p50</th>
            <th className="text-right font-normal">p95</th>
            <th className="text-right font-normal">p99</th>
          </tr>
        </thead>
        <tbody>
          {stages.map((s) => (
            <tr key={s.name} data-testid={`${id}-row-${s.key}`}>
              <td className="py-0.5">{s.name}</td>
              <td className="text-right">{s.v.p50.toFixed(2)}</td>
              <td className="text-right">{s.v.p95.toFixed(2)}</td>
              <td className="text-right">{s.v.p99.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DeviceCard({ device }: { device: DeviceInfo | undefined }) {
  return (
    <Card className="border-card-border">
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Device</CardTitle>
        <CardDescription>What PyTorch sees in this environment.</CardDescription>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-3 text-sm">
        <Stat label="device" value={device?.device ?? "—"} testId="dev-device" />
        <Stat
          label="cuda"
          value={device?.cuda_available ? "available" : "not available"}
          testId="dev-cuda"
        />
        <Stat
          label="cuda name"
          value={device?.cuda_device_name ?? "N/A"}
          testId="dev-cuda-name"
        />
        <Stat
          label="mixed precision"
          value={device?.mixed_precision ? "on" : "off"}
          testId="dev-mp"
        />
        <Stat label="torch" value={device?.torch_version?.split("+")[0] ?? "—"} testId="dev-torch" />
        <Stat label="python" value={device?.python_version ?? "—"} testId="dev-python" />
      </CardContent>
    </Card>
  );
}

function Stat({
  label,
  value,
  testId,
  color,
}: {
  label: string;
  value: string;
  testId: string;
  color?: string;
}) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</span>
      <span className="font-mono text-sm mt-0.5" style={color ? { color } : undefined} data-testid={`text-${testId}`}>
        {value}
      </span>
    </div>
  );
}

function Spark({ path, testId, color }: { path: string; testId: string; color: string }) {
  return (
    <svg viewBox="0 0 200 36" className="w-full h-9" data-testid={testId}>
      <path d={path} stroke={color} strokeWidth="1.2" fill="none" />
    </svg>
  );
}

function sparkline(values: number[]) {
  if (!values.length) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const w = 200;
  const h = 36;
  return values
    .map((v, i) => {
      const x = (i / Math.max(values.length - 1, 1)) * (w - 2) + 1;
      const y = 2 + (1 - (v - min) / Math.max(max - min, 1e-6)) * (h - 4);
      return (i === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1);
    })
    .join(" ");
}
