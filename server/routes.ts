import type { Express } from "express";
import type { Server } from "node:http";
import { promises as fs, statSync } from "node:fs";
import path from "node:path";

// Resolve the demo data directory robustly in both dev (tsx ESM) and
// production (esbuild CJS bundle). Search common locations relative to the
// process cwd so the same code works whether started from the project root
// or from dist/.
function resolveDataDir(): string {
  const candidates = [
    path.resolve(process.cwd(), "data"),
    path.resolve(process.cwd(), "..", "data"),
  ];
  for (const c of candidates) {
    try {
      if (statSync(c).isDirectory()) return c;
    } catch {}
  }
  return candidates[0];
}

const DATA_DIR = resolveDataDir();

// Cache parsed JSON in memory.
const cache = new Map<string, any>();

async function loadJson(file: string) {
  if (cache.has(file)) return cache.get(file);
  const buf = await fs.readFile(path.join(DATA_DIR, file), "utf8");
  const parsed = JSON.parse(buf);
  cache.set(file, parsed);
  return parsed;
}

export async function registerRoutes(httpServer: Server, app: Express): Promise<Server> {
  // ---------------------------------------------------------------
  // Health / device telemetry
  // ---------------------------------------------------------------
  app.get("/api/health", async (_req, res) => {
    try {
      const sessions = await loadJson("sessions.json");
      res.json({
        ok: true,
        device: sessions.device,
        n_sessions: sessions.sessions.length,
        timestamp: Date.now(),
      });
    } catch (e: any) {
      res.status(500).json({ ok: false, error: e?.message ?? "unknown" });
    }
  });

  // List demo sessions.
  app.get("/api/sessions", async (_req, res) => {
    const data = await loadJson("sessions.json");
    res.json(data);
  });

  // Full session preview (signal, spikes, cursor, predictions).
  app.get("/api/sessions/:id", async (req, res) => {
    try {
      const data = await loadJson(`session_${req.params.id}.json`);
      res.json(data);
    } catch (e: any) {
      res.status(404).json({ error: "session not found", id: req.params.id });
    }
  });

  // Stream a slice of the predicted windows — simulates a live feed.
  // Query params:
  //   start  – window index to start from (default 0)
  //   limit  – max windows to return (default 60)
  app.get("/api/sessions/:id/stream", async (req, res) => {
    let data;
    try {
      data = await loadJson(`session_${req.params.id}.json`);
    } catch {
      return res.status(404).json({ error: "session not found" });
    }
    const start = Math.max(0, parseInt(String(req.query.start ?? "0"), 10) || 0);
    const limit = Math.min(500, Math.max(1, parseInt(String(req.query.limit ?? "60"), 10) || 60));
    const total = data.predictions.length;
    const end = Math.min(total, start + limit);
    const slice = data.predictions.slice(start, end);
    // Also send a matching slice of cursor + spike density for the same time window.
    const timeStart = slice[0]?.t ?? 0;
    const timeEnd = slice[slice.length - 1]?.t ?? 0;
    const cursor = data.cursor.filter(
      (p: any) => p.t >= timeStart - 0.05 && p.t <= timeEnd + 0.05,
    );
    res.json({
      id: data.id,
      task: data.task,
      label_names: data.label_names,
      start,
      end,
      total,
      done: end >= total,
      predictions: slice,
      cursor,
    });
  });

  // Signal trace window (down-sampled). Useful for the live oscilloscope view.
  app.get("/api/sessions/:id/trace", async (req, res) => {
    let data;
    try {
      data = await loadJson(`session_${req.params.id}.json`);
    } catch {
      return res.status(404).json({ error: "session not found" });
    }
    const tStart = parseFloat(String(req.query.start ?? "0")) || 0;
    const span = Math.max(0.5, Math.min(10, parseFloat(String(req.query.span ?? "3")) || 3));
    const tEnd = tStart + span;
    const time = data.signal_preview.time;
    const i0 = time.findIndex((t: number) => t >= tStart);
    const i1Raw = time.findIndex((t: number) => t >= tEnd);
    const i1 = i1Raw === -1 ? time.length : i1Raw;
    const idx0 = i0 === -1 ? 0 : i0;
    res.json({
      id: data.id,
      time: time.slice(idx0, i1),
      channels: data.signal_preview.channels.map((ch: number[]) => ch.slice(idx0, i1)),
      spike_raster: data.spike_raster.map((times: number[]) =>
        times.filter((t) => t >= tStart && t <= tEnd),
      ),
    });
  });

  // Benchmark summary.
  app.get("/api/benchmark", async (_req, res) => {
    const data = await loadJson("benchmark.json");
    res.json(data);
  });

  // Streaming-pipeline benchmark (real EEG + synthetic scenarios).
  app.get("/api/stream-benchmark", async (_req, res) => {
    try {
      const data = await loadJson("stream_benchmark.json");
      res.json(data);
    } catch {
      res.status(404).json({ error: "stream_benchmark.json not found — run engine.scripts.stream_benchmark" });
    }
  });

  // Real-EEG smoke test report.
  app.get("/api/real-eeg-smoke", async (_req, res) => {
    try {
      const data = await loadJson("real_eeg_smoke.json");
      res.json(data);
    } catch {
      res.status(404).json({ error: "real_eeg_smoke.json not found — run engine.scripts.real_eeg_smoke" });
    }
  });

  // Pipeline / architecture metadata.
  app.get("/api/architecture", async (_req, res) => {
    const data = await loadJson("architecture.json");
    res.json(data);
  });

  // Aggregated telemetry: latency percentiles + class distribution per session.
  app.get("/api/sessions/:id/telemetry", async (req, res) => {
    let data;
    try {
      data = await loadJson(`session_${req.params.id}.json`);
    } catch {
      return res.status(404).json({ error: "session not found" });
    }
    const preds: any[] = data.predictions;
    const counts: Record<string, number> = {};
    let confSum = 0;
    for (const p of preds) {
      counts[p.label] = (counts[p.label] ?? 0) + 1;
      confSum += p.confidence;
    }
    const dist = Object.entries(counts).map(([label, count]) => ({
      label,
      count,
      fraction: count / preds.length,
    }));
    res.json({
      id: data.id,
      label_distribution: dist,
      mean_confidence: confSum / Math.max(1, preds.length),
      streaming: data.streaming,
      training: data.training,
      config: data.config,
    });
  });

  return httpServer;
}
