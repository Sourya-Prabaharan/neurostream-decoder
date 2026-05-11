import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Play, Pause, RotateCcw, Activity, Brain, Gauge, Zap } from "lucide-react";
import { Oscilloscope } from "@/components/Oscilloscope";
import { CursorTrail } from "@/components/CursorTrail";
import { LatencyChart } from "@/components/LatencyChart";
import type {
  SessionsResponse,
  SessionDetail,
  Prediction,
  CursorPoint,
} from "@/lib/types";
import { cn } from "@/lib/utils";

const TRACE_SPAN = 3; // seconds of trace shown at any time
const MAX_BUFFER = 240; // recent windows kept for live charts

export default function Dashboard() {
  const sessionsQuery = useQuery<SessionsResponse>({ queryKey: ["/api/sessions"] });
  const sessions = sessionsQuery.data?.sessions ?? [];

  const [task, setTask] = useState<"movement" | "speech">("movement");
  const [model, setModel] = useState<"gru" | "transformer">("gru");
  const sessionId = useMemo(() => {
    const match = sessions.find((s) => s.task === task && s.model_kind === model);
    return match?.id ?? sessions.find((s) => s.task === task)?.id ?? sessions[0]?.id;
  }, [sessions, task, model]);

  const sessionQuery = useQuery<SessionDetail>({
    queryKey: ["/api/sessions", sessionId],
    enabled: !!sessionId,
  });
  const session = sessionQuery.data;

  // Live playback state.
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [cursor, setCursor] = useState(0); // window index
  const [thresholdSigma, setThresholdSigma] = useState(4);

  useEffect(() => {
    setCursor(0);
  }, [sessionId]);

  // Drive playback off requestAnimationFrame so the trace feels live.
  const lastTickRef = useRef<number | null>(null);
  useEffect(() => {
    if (!running || !session) return;
    let raf = 0;
    const totalWindows = session.predictions.length;
    const windowsPerSec = 1000 / session.config.feature_hop_ms; // ~40 Hz

    const step = (now: number) => {
      if (lastTickRef.current == null) lastTickRef.current = now;
      const dt = (now - lastTickRef.current) / 1000;
      lastTickRef.current = now;
      setCursor((c) => {
        const next = c + dt * windowsPerSec * speed;
        if (next >= totalWindows) return 0; // loop
        return next;
      });
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => {
      cancelAnimationFrame(raf);
      lastTickRef.current = null;
    };
  }, [running, session, speed]);

  // Build the live buffers from the session detail.
  const liveData = useMemo(() => {
    if (!session) {
      return {
        preds: [] as Prediction[],
        latencies: [] as number[],
        cursorPath: [] as CursorPoint[],
        currentPred: undefined as Prediction | undefined,
        windowTime: 0,
      };
    }
    const idx = Math.min(session.predictions.length - 1, Math.floor(cursor));
    const start = Math.max(0, idx - MAX_BUFFER);
    const preds = session.predictions.slice(start, idx + 1);
    const latencies = preds.map((p) => p.latency_ms);
    const tEnd = session.predictions[idx]?.t ?? 0;
    const tStart = Math.max(0, tEnd - TRACE_SPAN);
    const cursorPath = session.cursor.filter((p) => p.t <= tEnd && p.t >= tStart - 0.05);
    const currentPred = session.predictions[idx];
    return { preds, latencies, cursorPath, currentPred, windowTime: tEnd };
  }, [session, cursor]);

  // Filter the signal-preview trace to the moving window.
  const traceData = useMemo(() => {
    if (!session) return { time: [] as number[], channels: [] as number[][], spike_raster: [] as number[][] };
    const tEnd = liveData.windowTime;
    const tStart = Math.max(0, tEnd - TRACE_SPAN);
    const time = session.signal_preview.time;
    let i0 = 0;
    let i1 = time.length;
    for (let i = 0; i < time.length; i++) {
      if (time[i] >= tStart) {
        i0 = i;
        break;
      }
    }
    for (let i = i0; i < time.length; i++) {
      if (time[i] > tEnd) {
        i1 = i;
        break;
      }
    }
    if (i1 <= i0) {
      i1 = Math.min(time.length, i0 + 2);
    }
    return {
      time: time.slice(i0, i1),
      channels: session.signal_preview.channels.map((ch) => ch.slice(i0, i1)),
      spike_raster: session.spike_raster.map((times) =>
        times.filter((t) => t >= tStart && t <= tEnd),
      ),
    };
  }, [session, liveData.windowTime]);

  // Class distribution for the visible window.
  const classCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const p of liveData.preds) counts.set(p.label, (counts.get(p.label) ?? 0) + 1);
    return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  }, [liveData.preds]);

  const lossHistory = session?.training.loss_history ?? [];

  const currentLabel = liveData.currentPred?.label ?? "—";
  const currentConf = liveData.currentPred?.confidence ?? 0;
  const currentLat = liveData.currentPred?.latency_ms ?? 0;

  const labelColours: Record<string, string> = {
    IDLE: "hsl(var(--muted-foreground))",
    LEFT: "hsl(var(--chart-1))",
    RIGHT: "hsl(var(--chart-2))",
    UP: "hsl(var(--chart-3))",
    DOWN: "hsl(var(--chart-4))",
    SELECT: "hsl(var(--chart-5))",
    STOP: "hsl(var(--destructive))",
    YES: "hsl(var(--chart-1))",
    NO: "hsl(var(--chart-5))",
  };

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      <PageHeader
        title="Live decode"
        subtitle={`Streaming inference @ ${session?.config.feature_hop_ms ?? 25}ms hop · ${session?.config.n_channels ?? 0}-ch synthetic recording`}
      />

      {/* Controls */}
      <Card className="border-card-border">
        <CardContent className="py-4 flex flex-wrap items-end gap-4">
          <div className="flex gap-1.5">
            <Button
              variant={running ? "secondary" : "default"}
              size="sm"
              onClick={() => setRunning((r) => !r)}
              data-testid="button-toggle-stream"
            >
              {running ? <Pause className="h-4 w-4 mr-1.5" /> : <Play className="h-4 w-4 mr-1.5" />}
              {running ? "Pause" : "Play"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCursor(0)}
              data-testid="button-reset-stream"
            >
              <RotateCcw className="h-4 w-4 mr-1.5" /> Reset
            </Button>
          </div>

          <div className="flex flex-col gap-1.5">
            <span className="text-xs text-muted-foreground">Task</span>
            <Tabs value={task} onValueChange={(v) => setTask(v as "movement" | "speech")} data-testid="tabs-task">
              <TabsList>
                <TabsTrigger value="movement" data-testid="tab-task-movement">
                  Movement
                </TabsTrigger>
                <TabsTrigger value="speech" data-testid="tab-task-speech">
                  Speech tokens
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>

          <div className="flex flex-col gap-1.5">
            <span className="text-xs text-muted-foreground">Decoder</span>
            <Select value={model} onValueChange={(v) => setModel(v as "gru" | "transformer")}>
              <SelectTrigger className="w-[160px]" data-testid="select-model">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="gru" data-testid="option-model-gru">
                  GRU (2-layer)
                </SelectItem>
                <SelectItem value="transformer" data-testid="option-model-transformer">
                  Tiny Transformer
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-col gap-1.5 min-w-[180px]">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Playback × {speed.toFixed(1)}</span>
            </div>
            <Slider
              value={[speed]}
              min={0.25}
              max={4}
              step={0.25}
              onValueChange={(v) => setSpeed(v[0])}
              data-testid="slider-speed"
            />
          </div>

          <div className="flex flex-col gap-1.5 min-w-[180px]">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Spike threshold</span>
              <span className="font-mono">{thresholdSigma.toFixed(1)} σ</span>
            </div>
            <Slider
              value={[thresholdSigma]}
              min={2}
              max={7}
              step={0.5}
              onValueChange={(v) => setThresholdSigma(v[0])}
              data-testid="slider-threshold"
            />
          </div>
        </CardContent>
      </Card>

      {/* Top KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard
          icon={<Brain className="h-4 w-4" />}
          label="Decoded output"
          value={currentLabel}
          valueClass="text-lg font-semibold tracking-tight"
          testId="kpi-decoded-label"
          accent={labelColours[currentLabel] ?? "hsl(var(--primary))"}
        />
        <KpiCard
          icon={<Activity className="h-4 w-4" />}
          label="Confidence"
          value={`${(currentConf * 100).toFixed(1)}%`}
          testId="kpi-confidence"
          accent="hsl(var(--chart-2))"
        />
        <KpiCard
          icon={<Gauge className="h-4 w-4" />}
          label="Latency / window"
          value={`${currentLat.toFixed(2)} ms`}
          testId="kpi-latency"
          accent="hsl(var(--chart-3))"
        />
        <KpiCard
          icon={<Zap className="h-4 w-4" />}
          label="Spikes / window"
          value={liveData.currentPred?.spike_count?.toString() ?? "0"}
          testId="kpi-spikes"
          accent="hsl(var(--chart-4))"
        />
      </div>

      {/* Trace + cursor */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="lg:col-span-2 border-card-border">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">Filtered neural trace</CardTitle>
                <CardDescription>
                  6 of {session?.config.n_channels ?? 0} channels, 5–200 Hz biquad bandpass · spike events overlaid
                </CardDescription>
              </div>
              <Badge variant="outline" className="font-mono text-xs">
                t = {liveData.windowTime.toFixed(2)}s
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <Oscilloscope
              time={traceData.time}
              channels={traceData.channels}
              spikeRaster={traceData.spike_raster}
              height={320}
            />
          </CardContent>
        </Card>

        <Card className="border-card-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Decoded cursor / command</CardTitle>
            <CardDescription>2-D output trajectory · ground-truth label colour</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col items-center gap-3">
            <CursorTrail
              points={liveData.cursorPath}
              current={liveData.cursorPath[liveData.cursorPath.length - 1]}
            />
            <ClassChips counts={classCounts} testId="chips-classdist" />
          </CardContent>
        </Card>
      </div>

      {/* Telemetry + training */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="lg:col-span-2 border-card-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Per-window latency</CardTitle>
            <CardDescription>
              Rolling buffer of the last {liveData.latencies.length} streaming inferences.
              p50/p95 reference lines update live.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <LatencyChart latencies={liveData.latencies} />
          </CardContent>
        </Card>
        <Card className="border-card-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Training loss</CardTitle>
            <CardDescription>
              {session?.training.epochs ?? 0} epochs · {session?.training.wall_seconds?.toFixed(2) ?? "—"}s
            </CardDescription>
          </CardHeader>
          <CardContent>
            <TrainingLoss losses={lossHistory} />
          </CardContent>
        </Card>
      </div>

      <SessionStats session={session} sessions={sessions} sessionId={sessionId} />
    </div>
  );
}

function PageHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="flex flex-col gap-1">
      <h1 className="text-xl font-semibold tracking-tight" data-testid="text-page-title">
        {title}
      </h1>
      <p className="text-sm text-muted-foreground" data-testid="text-page-subtitle">
        {subtitle}
      </p>
    </div>
  );
}

function KpiCard({
  icon,
  label,
  value,
  valueClass,
  testId,
  accent,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  valueClass?: string;
  testId: string;
  accent: string;
}) {
  return (
    <Card className="border-card-border overflow-hidden relative" data-testid={`card-${testId}`}>
      <div
        className="absolute inset-y-0 left-0 w-[3px]"
        style={{ background: accent }}
        aria-hidden
      />
      <CardContent className="py-4 pl-5">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground uppercase tracking-wide">
          {icon} {label}
        </div>
        <div className={cn("mt-1.5 font-mono", valueClass ?? "text-base font-semibold")} data-testid={`text-${testId}`}>
          {value}
        </div>
      </CardContent>
    </Card>
  );
}

function ClassChips({ counts, testId }: { counts: [string, number][]; testId: string }) {
  const total = counts.reduce((a, [, c]) => a + c, 0) || 1;
  if (!counts.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5 justify-center" data-testid={testId}>
      {counts.map(([label, c]) => (
        <span
          key={label}
          data-testid={`chip-class-${label}`}
          className="px-2 py-0.5 rounded-full text-[10px] font-mono border border-border bg-muted/30 text-muted-foreground"
        >
          {label} · {Math.round((c / total) * 100)}%
        </span>
      ))}
    </div>
  );
}

function TrainingLoss({ losses }: { losses: number[] }) {
  if (!losses.length) return <div className="text-sm text-muted-foreground">No training history</div>;
  const w = 280;
  const h = 110;
  const max = Math.max(...losses);
  const min = Math.min(...losses);
  const path = losses
    .map((v, i) => {
      const x = (i / Math.max(losses.length - 1, 1)) * (w - 16) + 8;
      const y = 6 + (1 - (v - min) / Math.max(max - min, 1e-6)) * (h - 24);
      return (i === 0 ? "M" : "L") + x + " " + y.toFixed(1);
    })
    .join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" data-testid="chart-training-loss">
      <path d={path} stroke="hsl(var(--chart-1))" strokeWidth="1.6" fill="none" />
      {losses.map((v, i) => {
        const x = (i / Math.max(losses.length - 1, 1)) * (w - 16) + 8;
        const y = 6 + (1 - (v - min) / Math.max(max - min, 1e-6)) * (h - 24);
        return <circle key={i} cx={x} cy={y} r="2.4" fill="hsl(var(--chart-1))" />;
      })}
      <text x={6} y={h - 4} fontSize="10" fill="hsl(var(--muted-foreground))" fontFamily="var(--font-mono)">
        loss {losses[0].toFixed(3)} → {losses[losses.length - 1].toFixed(3)}
      </text>
    </svg>
  );
}

function SessionStats({
  session,
  sessions,
  sessionId,
}: {
  session: SessionDetail | undefined;
  sessions: SessionsResponse["sessions"];
  sessionId: string | undefined;
}) {
  if (!session) {
    return (
      <Card className="border-card-border">
        <CardContent className="py-8 text-sm text-muted-foreground" data-testid="text-session-loading">
          Loading session metadata…
        </CardContent>
      </Card>
    );
  }
  const stats = session.streaming;
  return (
    <Card className="border-card-border">
      <CardHeader className="pb-2 flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-base">Streaming benchmark · {session.id}</CardTitle>
          <CardDescription>
            {sessions.length} demo sessions loaded · current = {sessionId}
          </CardDescription>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {session.label_names.map((l) => (
            <Badge key={l} variant="outline" className="font-mono text-[10px]">
              {l}
            </Badge>
          ))}
        </div>
      </CardHeader>
      <CardContent className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 text-sm">
        <Stat label="windows" value={stats.n_windows.toLocaleString()} testId="stat-windows" />
        <Stat label="throughput" value={`${stats.windows_per_second.toFixed(0)} win/s`} testId="stat-throughput" />
        <Stat label="p50 latency" value={`${stats.p50_latency_ms.toFixed(2)} ms`} testId="stat-p50" />
        <Stat label="p95 latency" value={`${stats.p95_latency_ms.toFixed(2)} ms`} testId="stat-p95" />
        <Stat label="p99 latency" value={`${stats.p99_latency_ms.toFixed(2)} ms`} testId="stat-p99" />
        <Stat label="mean confidence" value={`${(stats.mean_confidence * 100).toFixed(1)}%`} testId="stat-confidence" />
      </CardContent>
    </Card>
  );
}

function Stat({ label, value, testId }: { label: string; value: string; testId: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-xs uppercase tracking-wide text-muted-foreground">{label}</span>
      <span className="font-mono text-sm mt-0.5" data-testid={`text-${testId}`}>{value}</span>
    </div>
  );
}
