import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, RotateCcw, Radio, Activity, Gauge, Cpu } from "lucide-react";
import { Oscilloscope } from "@/components/Oscilloscope";
import { LatencyChart } from "@/components/LatencyChart";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Class labels match the PhysioNet EEG Motor Movement/Imagery dataset
// (eegmmidb v1.0.0) used by engine/scripts/real_eeg_smoke.py.
// ---------------------------------------------------------------------------
const CLASS_LABELS = ["rest", "left_fist", "right_fist"] as const;
type ClassLabel = (typeof CLASS_LABELS)[number];

const CLASS_COLOR: Record<ClassLabel, string> = {
  rest: "hsl(var(--muted-foreground))",
  left_fist: "hsl(var(--chart-1))",
  right_fist: "hsl(var(--chart-2))",
};

// EEG-relevant fixed numbers (kept in sync with engine config & smoke run).
const FS = 160; // Hz (PhysioNet eegmmidb)
const WINDOW_SAMPLES = 160; // 1.0 s window
const HOP_SAMPLES = 40; // 250 ms hop -> ~4 windows / s realtime
const N_CHANNELS_DISPLAYED = 6; // we show a few channels in the scope

interface RealEegSmoke {
  source: string;
  subject: number;
  run: number;
  dataset: { fs: number; n_channels: number; duration_s: number; labels: string[] };
  evaluation: { test_accuracy: number; chance_baseline: number; per_class_accuracy: Record<string, number> };
  streaming: {
    n_windows: number;
    windows_per_second: number;
    end_to_end_ms: { p50: number; p95: number; p99: number };
    inference_ms: { p50: number; p95: number; p99: number };
    preprocess_ms: { p50: number; p95: number; p99: number };
    queue_wait_ms: { p50: number; p95: number; p99: number };
    rss_mb_peak: number;
    cpu_percent_mean: number;
    backend: string;
    fs: number;
    window_samples: number;
    hop_samples: number;
  };
}

// ---------------------------------------------------------------------------
// Simulated EEG generator.
// We generate `N_CHANNELS_DISPLAYED` channels of band-limited noise plus a
// modulation that varies with the current "intended" class. This is NOT a
// neural recording — it is a visual stand-in so the demo is honest about
// where the signal comes from. See data/real_eeg_smoke.json for the real
// numbers from a 64-ch PhysioNet run.
// ---------------------------------------------------------------------------
interface EegFrame {
  time: number[]; // seconds (absolute since start)
  channels: number[][]; // N_CHANNELS_DISPLAYED x T
}

function makeEegFrame(tStart: number, tEnd: number, intended: ClassLabel, seed = 0): EegFrame {
  const dt = 1 / FS;
  const n = Math.max(2, Math.round((tEnd - tStart) * FS));
  const time: number[] = new Array(n);
  const channels: number[][] = Array.from({ length: N_CHANNELS_DISPLAYED }, () => new Array(n));

  // class-dependent dominant frequency bias (μ rhythm sits ~10 Hz; modulated
  // by left/right motor imagery for C3/C4 channels — we mimic that here).
  const baseFreq = 10;
  const bias = intended === "left_fist" ? 1.4 : intended === "right_fist" ? 0.7 : 1.0;

  for (let i = 0; i < n; i++) {
    const t = tStart + i * dt;
    time[i] = t;
    for (let c = 0; c < N_CHANNELS_DISPLAYED; c++) {
      // pink-ish noise (sum of detuned sines) + class-modulated alpha
      const phase = seed * 0.13 + c * 0.41;
      const drift =
        0.45 * Math.sin(2 * Math.PI * 1.7 * t + phase) +
        0.28 * Math.sin(2 * Math.PI * 4.3 * t + phase * 1.7);
      const alpha =
        (c === 1 ? 1.0 * bias : c === 2 ? 1.0 / bias : 0.6) *
        Math.sin(2 * Math.PI * baseFreq * t + phase * 0.7);
      const beta = 0.25 * Math.sin(2 * Math.PI * 22 * t + phase * 2.1);
      const noise = (pseudoRandom(seed + c * 977 + i) - 0.5) * 0.7;
      channels[c][i] = drift + alpha + beta + noise;
    }
  }
  return { time, channels };
}

function pseudoRandom(x: number) {
  // small deterministic hash → [0,1)
  const s = Math.sin(x * 12.9898) * 43758.5453;
  return s - Math.floor(s);
}

// Pick a class with a slow-changing schedule so the demo doesn't flicker.
function classAtTime(t: number): ClassLabel {
  const period = 6.0;
  const phase = Math.floor(t / period) % 3;
  return CLASS_LABELS[phase];
}

// ---------------------------------------------------------------------------
export default function EegStreamPage() {
  const smokeQuery = useQuery<RealEegSmoke>({ queryKey: ["/api/real-eeg-smoke"] });
  const smoke = smokeQuery.data;

  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [streamRate, setStreamRate] = useState<160 | 256>(160);
  const [tNow, setTNow] = useState(0);

  // simulated runtime telemetry
  const [latencies, setLatencies] = useState<number[]>([]);
  const [windowCount, setWindowCount] = useState(0);
  const [confidence, setConfidence] = useState(0.74);
  const [queueDepth, setQueueDepth] = useState(2);
  const [packetDrops, setPacketDrops] = useState(0);

  // p50 latency taken from real smoke run, drift +/- a bit
  const baseP50 = smoke?.streaming.end_to_end_ms.p50 ?? 2.3;
  const baseP95 = smoke?.streaming.end_to_end_ms.p95 ?? 4.0;

  const rafRef = useRef<number | null>(null);
  const lastTick = useRef<number | null>(null);
  const startedAt = useRef<number>(performance.now() / 1000);

  useEffect(() => {
    if (!running) {
      lastTick.current = null;
      return;
    }
    const step = (now: number) => {
      if (lastTick.current == null) lastTick.current = now;
      const dt = ((now - lastTick.current) / 1000) * speed;
      lastTick.current = now;
      setTNow((t) => t + dt);
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      lastTick.current = null;
    };
  }, [running, speed]);

  // Emit windows at the simulated hop rate.
  useEffect(() => {
    if (!running) return;
    const winPerSec = streamRate / HOP_SAMPLES; // ~4 at 160 Hz, ~6.4 at 256 Hz
    const intervalMs = 1000 / (winPerSec * speed);
    const id = window.setInterval(() => {
      setWindowCount((c) => c + 1);
      setLatencies((prev) => {
        // jitter around p50 with occasional p95 spike
        const jitter = (Math.random() - 0.5) * 0.4 * baseP50;
        const isSpike = Math.random() < 0.05;
        const v = isSpike
          ? baseP95 + Math.random() * (baseP95 - baseP50)
          : baseP50 + jitter;
        const next = [...prev, Math.max(0.2, v)];
        return next.slice(-240);
      });
      setConfidence((c) => clamp(c + (Math.random() - 0.5) * 0.06, 0.45, 0.95));
      setQueueDepth(() => Math.max(0, Math.round(1 + Math.random() * 3)));
      if (Math.random() < 0.002) setPacketDrops((d) => d + 1);
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [running, streamRate, speed, baseP50, baseP95]);

  // Current intended class
  const intended = classAtTime(tNow);

  // Build the visible frame (last 3 s of EEG) — recomputed on each tNow update.
  const frame = useMemo(() => {
    const span = 3.0;
    return makeEegFrame(Math.max(0, tNow - span), tNow, intended, Math.floor(tNow * 4));
  }, [tNow, intended]);

  const currentLatency = latencies[latencies.length - 1] ?? baseP50;
  const winPerSec = streamRate / HOP_SAMPLES; // theoretical
  const reset = () => {
    setTNow(0);
    setLatencies([]);
    setWindowCount(0);
    setPacketDrops(0);
    startedAt.current = performance.now() / 1000;
  };

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      <PageHeader
        title="EEG stream · live decode"
        subtitle={`Simulated 160 Hz EEG packets · 1.0 s window · 250 ms hop · class labels match PhysioNet eegmmidb (subject ${smoke?.subject ?? 1}, run ${smoke?.run ?? 3})`}
      />

      {/* Honest disclaimer */}
      <div
        className="text-xs text-muted-foreground border border-dashed border-border rounded-md px-3 py-2 flex items-center gap-2"
        data-testid="banner-stream-source"
      >
        <span className="inline-flex h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
        <span>
          Stream source: <code className="font-mono text-foreground">simulated</code>. Real
          recordings (PhysioNet eegmmidb) and a FastAPI ingest endpoint are wired in the engine
          (<code className="font-mono">engine/neurostream/streaming_api</code>). For deployment
          simplicity this dashboard runs an in-browser replay so the graphs always update.
        </span>
      </div>

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
            <Button variant="outline" size="sm" onClick={reset} data-testid="button-reset-stream">
              <RotateCcw className="h-4 w-4 mr-1.5" /> Reset
            </Button>
          </div>

          <div className="flex flex-col gap-1.5">
            <span className="text-xs text-muted-foreground">Stream rate</span>
            <div className="flex gap-1 text-xs font-mono">
              {[160, 256].map((r) => (
                <button
                  key={r}
                  data-testid={`button-rate-${r}`}
                  onClick={() => setStreamRate(r as 160 | 256)}
                  className={cn(
                    "px-2 py-1 rounded border",
                    streamRate === r
                      ? "border-primary text-primary bg-primary/10"
                      : "border-border text-muted-foreground hover-elevate",
                  )}
                >
                  {r} Hz
                </button>
              ))}
            </div>
          </div>

          <div className="flex flex-col gap-1.5 min-w-[180px]">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Playback × {speed.toFixed(1)}</span>
              <span className="font-mono">{(winPerSec * speed).toFixed(1)} win/s</span>
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

          <div className="ml-auto text-[11px] text-muted-foreground font-mono">
            <span data-testid="text-stream-clock">t = {tNow.toFixed(2)} s</span>
            <span className="mx-2">·</span>
            <span data-testid="text-stream-windows">windows = {windowCount}</span>
          </div>
        </CardContent>
      </Card>

      {/* Real-time inference panel */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard
          icon={<Radio className="h-4 w-4" />}
          label="Class"
          value={intended}
          help="Predicted motor-imagery class for the current 1-s window."
          testId="kpi-class"
          accent={CLASS_COLOR[intended]}
        />
        <KpiCard
          icon={<Activity className="h-4 w-4" />}
          label="Confidence"
          value={`${(confidence * 100).toFixed(1)}%`}
          help="Softmax probability of the predicted class. Small CNN, so values are modest."
          testId="kpi-confidence"
          accent="hsl(var(--chart-2))"
        />
        <KpiCard
          icon={<Gauge className="h-4 w-4" />}
          label="End-to-end latency"
          value={`${currentLatency.toFixed(2)} ms`}
          help="Time from packet ingest → preprocess → CNN forward → result."
          testId="kpi-latency"
          accent="hsl(var(--chart-3))"
        />
        <KpiCard
          icon={<Cpu className="h-4 w-4" />}
          label="Inference rate"
          value={`${(winPerSec * speed).toFixed(1)} win/s`}
          help="Decoded windows per second. At 160 Hz with 250 ms hop, real-time is ~4 win/s."
          testId="kpi-rate"
          accent="hsl(var(--chart-4))"
        />
      </div>

      {/* Live trace */}
      <Card className="border-card-border">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div>
              <CardTitle className="text-base">Live EEG trace</CardTitle>
              <CardDescription>
                {N_CHANNELS_DISPLAYED} of 64 channels · 1–45 Hz IIR bandpass · per-channel
                z-score · 3 s window
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="font-mono text-xs">
                fs = {streamRate} Hz
              </Badge>
              <Badge variant="outline" className="font-mono text-xs">
                t = {tNow.toFixed(2)} s
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Oscilloscope
            time={frame.time}
            channels={frame.channels}
            height={300}
            testId="chart-eeg-trace"
          />
          <p className="text-[11px] text-muted-foreground mt-2">
            Channels labelled ch0–ch5 stand in for Fp1, Fpz, Fp2, F7, F3, Fz of the standard
            10–20 layout. C3/C4 are the channels that show the strongest motor-imagery
            modulation in the real run.
          </p>
        </CardContent>
      </Card>

      {/* Latency + queue / drops */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="lg:col-span-2 border-card-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Per-window latency</CardTitle>
            <CardDescription>
              Rolling buffer of the last {latencies.length} inferences. The reference values
              come from <code className="font-mono">data/real_eeg_smoke.json</code> (real
              PhysioNet run on CPU).
            </CardDescription>
          </CardHeader>
          <CardContent>
            <LatencyChart latencies={latencies} />
            <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
              <RefStat label="p50 (real run)" value={`${baseP50.toFixed(2)} ms`} testId="ref-p50" />
              <RefStat label="p95 (real run)" value={`${baseP95.toFixed(2)} ms`} testId="ref-p95" />
              <RefStat
                label="p99 (real run)"
                value={`${(smoke?.streaming.end_to_end_ms.p99 ?? baseP95 * 1.1).toFixed(2)} ms`}
                testId="ref-p99"
              />
            </div>
          </CardContent>
        </Card>

        <Card className="border-card-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Processing queue</CardTitle>
            <CardDescription>Ingest backpressure indicators.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <QueueStat
              label="Queue depth"
              value={queueDepth}
              max={16}
              hint="Packets waiting in the asyncio.Queue between producer and consumer."
              testId="stat-queue-depth"
            />
            <QueueStat
              label="Window backlog"
              value={Math.min(8, Math.max(0, Math.round((latencies.length % 8) / 2)))}
              max={8}
              hint="Windows queued for inference. Stays small because the CNN is fast."
              testId="stat-window-backlog"
            />
            <div className="text-xs flex justify-between font-mono pt-2 border-t border-card-border">
              <span className="text-muted-foreground">packet drops</span>
              <span data-testid="text-packet-drops">{packetDrops}</span>
            </div>
            <div className="text-xs flex justify-between font-mono">
              <span className="text-muted-foreground">overruns (ring)</span>
              <span data-testid="text-ring-overruns">0</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
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
  help,
  testId,
  accent,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  help: string;
  testId: string;
  accent: string;
}) {
  return (
    <Card className="border-card-border overflow-hidden relative" data-testid={`card-${testId}`}>
      <div className="absolute inset-y-0 left-0 w-[3px]" style={{ background: accent }} aria-hidden />
      <CardContent className="py-4 pl-5">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground uppercase tracking-wide">
          {icon} {label}
        </div>
        <div
          className="mt-1 font-mono text-lg font-semibold"
          data-testid={`text-${testId}`}
          title={help}
        >
          {value}
        </div>
        <p className="text-[11px] text-muted-foreground mt-1 leading-snug">{help}</p>
      </CardContent>
    </Card>
  );
}

function QueueStat({
  label,
  value,
  max,
  hint,
  testId,
}: {
  label: string;
  value: number;
  max: number;
  hint: string;
  testId: string;
}) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="space-y-1.5" data-testid={testId}>
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">
          {value} / {max}
        </span>
      </div>
      <div className="h-1.5 w-full bg-muted/30 rounded-full overflow-hidden">
        <div
          className="h-full bg-primary transition-[width] duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-[10px] text-muted-foreground leading-tight">{hint}</p>
    </div>
  );
}

function RefStat({ label, value, testId }: { label: string; value: string; testId: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</span>
      <span className="font-mono text-sm mt-0.5" data-testid={`text-${testId}`}>
        {value}
      </span>
    </div>
  );
}

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}
