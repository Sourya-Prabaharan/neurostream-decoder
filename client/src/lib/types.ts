// Shared types matching the Python-generated JSON shape.

export interface DeviceInfo {
  device: string;
  cuda_available: boolean;
  cuda_device_count: number;
  cuda_device_name: string | null;
  torch_version: string;
  python_version: string;
  mixed_precision: boolean;
}

export interface StreamingStats {
  n_windows: number;
  wall_seconds: number;
  windows_per_second: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  mean_confidence: number;
  accuracy_proxy: number;
}

export interface TrainingStats {
  epochs: number;
  loss_history: number[];
  wall_seconds: number;
}

export interface SessionSummary {
  id: string;
  task: "movement" | "speech";
  model_kind: "gru" | "transformer";
  duration_sec: number;
  n_channels: number;
  label_names: string[];
  streaming: StreamingStats;
}

export interface SessionsResponse {
  sessions: SessionSummary[];
  device: DeviceInfo;
}

export interface Prediction {
  t: number;
  class: number;
  label: string;
  confidence: number;
  latency_ms: number;
  spike_count: number;
}

export interface CursorPoint {
  t: number;
  x: number;
  y: number;
}

export interface SessionDetail {
  id: string;
  task: "movement" | "speech";
  model_kind: "gru" | "transformer";
  label_names: string[];
  config: {
    n_channels: number;
    sample_rate: number;
    duration_sec: number;
    feature_window_ms: number;
    feature_hop_ms: number;
    spike_threshold_sigma: number;
  };
  signal_preview: {
    time: number[];
    channels: number[][];
  };
  cursor: CursorPoint[];
  spike_raster: number[][];
  predictions: Prediction[];
  training: TrainingStats;
  streaming: StreamingStats;
}

export interface ArchitectureStage {
  name: string;
  detail: string;
  input: string;
  output: string;
}

export interface ArchitectureResponse {
  stages: ArchitectureStage[];
}

export interface StreamResponse {
  id: string;
  task: string;
  label_names: string[];
  start: number;
  end: number;
  total: number;
  done: boolean;
  predictions: Prediction[];
  cursor: CursorPoint[];
}

export interface TelemetryResponse {
  id: string;
  label_distribution: { label: string; count: number; fraction: number }[];
  mean_confidence: number;
  streaming: StreamingStats;
  training: TrainingStats;
  config: SessionDetail["config"];
}
