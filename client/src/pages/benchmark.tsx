import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { DeviceInfo } from "@/lib/types";

interface BenchmarkResponse {
  device_report: DeviceInfo;
  training: { epochs: number; loss_history: number[]; wall_seconds: number; model_kind: string };
  streaming: {
    n_windows: number;
    wall_seconds: number;
    windows_per_second: number;
    p50_latency_ms: number;
    p95_latency_ms: number;
    p99_latency_ms: number;
    accuracy_proxy: number;
  };
  task: string;
  label_names: string[];
}

export default function BenchmarkPage() {
  const { data, isLoading } = useQuery<BenchmarkResponse>({ queryKey: ["/api/benchmark"] });
  if (isLoading || !data) {
    return (
      <div className="text-sm text-muted-foreground max-w-[900px] mx-auto" data-testid="text-benchmark-loading">
        Running benchmark…
      </div>
    );
  }
  const s = data.streaming;
  const t = data.training;

  return (
    <div className="space-y-6 max-w-[1100px] mx-auto">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Benchmark</h1>
        <p className="text-sm text-muted-foreground">
          Headless end-to-end run captured by <code className="font-mono">engine/scripts/generate_demo.py</code>.
          Re-run any time to refresh.
        </p>
      </div>

      <Card className="border-card-border">
        <CardHeader>
          <CardTitle className="text-base">Hardware</CardTitle>
          <CardDescription>Device PyTorch saw when the demo was generated.</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <DevStat label="device" value={data.device_report.device} testId="bench-device" />
          <DevStat label="cuda" value={data.device_report.cuda_available ? "available" : "none"} testId="bench-cuda" />
          <DevStat label="torch" value={data.device_report.torch_version.split("+")[0]} testId="bench-torch" />
          <DevStat label="mixed precision" value={data.device_report.mixed_precision ? "on" : "off"} testId="bench-mp" />
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="border-card-border">
          <CardHeader>
            <CardTitle className="text-base">Streaming inference</CardTitle>
            <CardDescription>
              Task: <Badge variant="outline" className="font-mono ml-1">{data.task}</Badge>
            </CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-4 text-sm">
            <DevStat label="windows" value={s.n_windows.toLocaleString()} testId="bench-windows" />
            <DevStat label="windows/sec" value={s.windows_per_second.toFixed(1)} testId="bench-thrpt" />
            <DevStat label="p50 latency" value={`${s.p50_latency_ms.toFixed(2)} ms`} testId="bench-p50" />
            <DevStat label="p95 latency" value={`${s.p95_latency_ms.toFixed(2)} ms`} testId="bench-p95" />
            <DevStat label="p99 latency" value={`${s.p99_latency_ms.toFixed(2)} ms`} testId="bench-p99" />
            <DevStat label="accuracy proxy" value={`${(s.accuracy_proxy * 100).toFixed(1)}%`} testId="bench-acc" />
          </CardContent>
        </Card>
        <Card className="border-card-border">
          <CardHeader>
            <CardTitle className="text-base">Training</CardTitle>
            <CardDescription>Model: {t.model_kind.toUpperCase()}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="grid grid-cols-2 gap-4">
              <DevStat label="epochs" value={String(t.epochs)} testId="bench-epochs" />
              <DevStat label="wall time" value={`${t.wall_seconds.toFixed(2)} s`} testId="bench-train-time" />
            </div>
            <LossSpark losses={t.loss_history} />
          </CardContent>
        </Card>
      </div>

      <Card className="border-card-border">
        <CardHeader>
          <CardTitle className="text-base">Reproduce</CardTitle>
        </CardHeader>
        <CardContent className="text-sm space-y-2">
          <p className="text-muted-foreground">From the project root:</p>
          <pre className="rounded-md border border-card-border bg-muted/30 p-3 font-mono text-xs overflow-x-auto"><code data-testid="text-bench-cmd">{`pip install -r engine/requirements.txt
python -m engine.scripts.generate_demo
pytest engine/tests`}</code></pre>
          <p className="text-xs text-muted-foreground">
            Add <code className="font-mono">--task speech</code> or <code className="font-mono">--kind transformer</code> to{" "}
            <code className="font-mono">python -m engine.scripts.benchmark</code> for ad-hoc runs.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function DevStat({ label, value, testId }: { label: string; value: string; testId: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</span>
      <span className="font-mono text-sm mt-0.5" data-testid={`text-${testId}`}>{value}</span>
    </div>
  );
}

function LossSpark({ losses }: { losses: number[] }) {
  if (!losses.length) return null;
  const w = 280, h = 60;
  const max = Math.max(...losses), min = Math.min(...losses);
  const path = losses
    .map((v, i) => {
      const x = (i / Math.max(losses.length - 1, 1)) * (w - 8) + 4;
      const y = 6 + (1 - (v - min) / Math.max(max - min, 1e-6)) * (h - 14);
      return (i === 0 ? "M" : "L") + x + " " + y.toFixed(1);
    })
    .join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" data-testid="bench-loss-spark">
      <path d={path} stroke="hsl(var(--chart-1))" strokeWidth="1.5" fill="none" />
      <text x={4} y={h - 4} fontSize="9" fontFamily="var(--font-mono)" fill="hsl(var(--muted-foreground))">
        {losses[0].toFixed(3)} → {losses[losses.length - 1].toFixed(3)}
      </text>
    </svg>
  );
}
