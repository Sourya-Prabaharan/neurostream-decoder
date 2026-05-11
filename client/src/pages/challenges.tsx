import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface Challenge {
  id: string;
  title: string;
  plain: string;
  technical: string[];
  status: "solved" | "ongoing" | "planned";
}

const CHALLENGES: Challenge[] = [
  {
    id: "async-sync",
    title: "Async stream synchronization",
    plain:
      "EEG packets arrive at fixed intervals (one every 100 ms for the 16-sample packet size). The decoder runs at its own pace. Keeping the two in step without dropping samples or doing extra work is the core problem.",
    technical: [
      "Producer pushes timestamped EEGPacket onto an asyncio.Queue / Redis Stream.",
      "Consumer awaits the next packet, writes into a per-stream RingBuffer, and triggers windowing.",
      "t_send is set on the producer side; the consumer subtracts it from t_recv to get queue_wait_ms (real wait, not wall-clock).",
      "Single model instance is shared across streams via a small lock — fine for batch=1 at this scale.",
    ],
    status: "solved",
  },
  {
    id: "latency",
    title: "Latency reduction",
    plain:
      "Each window must turn into a prediction faster than the next window arrives. With a 250 ms hop that means a 250 ms budget — we land at ~2.3 ms p50.",
    technical: [
      "IIR bandpass implemented as a Direct-Form-II biquad with state carried across windows (no re-warm-up).",
      "rFFT band powers use NumPy on contiguous float32 buffers; no per-call allocation.",
      "PyTorch forward in eval mode, no_grad, batch=1 — the small CNN keeps inference under 1 ms p50 on CPU.",
      "End-to-end p99 stays under 5 ms on a developer laptop CPU.",
    ],
    status: "solved",
  },
  {
    id: "buffering",
    title: "Buffering and window sizing",
    plain:
      "Windows shorter than ~0.5 s are too noisy for the CNN. Longer than ~2 s and the lag is felt by anyone using it. We picked 1.0 s with 250 ms hop, which is a common BCI default.",
    technical: [
      "Window = 160 samples (1.0 s @ 160 Hz). Hop = 40 samples (250 ms) → 75 % overlap.",
      "RingBuffer sized to ~50 s; drop-oldest on overrun with a counted overrun stat.",
      "Windows are read with `ring_buffer.latest(window_samples)` — the read cursor is NOT advanced, so the next hop overlaps naturally.",
    ],
    status: "solved",
  },
  {
    id: "tradeoff",
    title: "Latency vs throughput tradeoff",
    plain:
      "Going faster sometimes means processing fewer windows at once and using more CPU. Going slower means a smoother system but a noticeable lag.",
    technical: [
      "Real-time pacing: producer sleeps to match the recording sample rate. CPU stays ~1 %.",
      "Max-throughput pacing: producer pushes as fast as possible. CPU climbs but windows/sec hits ~450+ at fs=160 Hz.",
      "Both modes are captured side-by-side in data/stream_benchmark.json so the cost is explicit.",
    ],
    status: "solved",
  },
  {
    id: "redis",
    title: "Why Redis Streams instead of Kafka",
    plain:
      "Both can do this. Kafka is heavier to set up and is overkill for a single-process EEG decoder. Redis Streams gives us the same `XADD / XREAD` semantics with one container.",
    technical: [
      "RedisStream uses XADD for produce, XREAD with BLOCK and last-id tracking for consume — same pattern as Kafka topics, lighter operationally.",
      "get_stream() probes the configured REDIS_URL and falls back to an in-process asyncio.Queue. Backend used is reported in every benchmark.",
      "Kafka becomes interesting once we have many decoder workers consuming the same EEG feed — not the current scale.",
    ],
    status: "solved",
  },
  {
    id: "fastapi",
    title: "Why FastAPI",
    plain:
      "Async-first server. Pydantic gives us body validation for free. Easy to plug uvicorn workers in front when needed.",
    technical: [
      "Endpoints: GET /healthz, POST /ingest, GET /metrics, GET /windows.",
      "App starts an in-process consumer task on lifespan startup that owns the stream + ring buffer.",
      "Pydantic IngestBody validates (T, C) shape before it ever touches the ring buffer.",
    ],
    status: "solved",
  },
  {
    id: "cuda",
    title: "Future CUDA acceleration",
    plain:
      "The CNN is small enough that CPU is fine right now. With more streams or a bigger model, a single CUDA device would give us a lot of headroom.",
    technical: [
      "torch.device('cuda') auto-selected when available; mixed-precision autocast turned on only on CUDA.",
      "Hot path ops are stock PyTorch (bandpass via filtfilt → vectorised matmuls), so they hit cuBLAS/cuDNN with no rewrite.",
      "Native CUDA / Triton extensions are scaffolded in engine/neurostream/cuda_ext; integration plan in docs/cuda_notes.md.",
    ],
    status: "planned",
  },
];

const STATUS_LABEL: Record<Challenge["status"], { text: string; classes: string }> = {
  solved: { text: "shipped", classes: "border-primary/40 text-primary bg-primary/10" },
  ongoing: { text: "in progress", classes: "border-amber-500/40 text-amber-500 bg-amber-500/10" },
  planned: { text: "planned", classes: "border-border text-muted-foreground bg-muted/30" },
};

export default function ChallengesPage() {
  return (
    <div className="space-y-6 max-w-[1100px] mx-auto">
      <div>
        <h1 className="text-xl font-semibold tracking-tight" data-testid="text-challenges-title">
          Engineering challenges
        </h1>
        <p className="text-sm text-muted-foreground">
          The non-obvious problems behind a real-time EEG decoder, in plain language with the
          technical detail below each one.
        </p>
      </div>

      <div className="space-y-4">
        {CHALLENGES.map((c) => (
          <Card key={c.id} className="border-card-border" data-testid={`challenge-${c.id}`}>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <CardTitle className="text-base">{c.title}</CardTitle>
                <Badge
                  variant="outline"
                  className={`font-mono text-[10px] ${STATUS_LABEL[c.status].classes}`}
                >
                  {STATUS_LABEL[c.status].text}
                </Badge>
              </div>
              <CardDescription className="pt-1 text-sm leading-relaxed">
                {c.plain}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="text-xs text-muted-foreground space-y-1.5 font-mono leading-relaxed border-l border-card-border pl-4">
                {c.technical.map((t, i) => (
                  <li key={i} data-testid={`challenge-${c.id}-bullet-${i}`}>
                    {t}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
