import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowRight, Database, Filter, Zap, Layers as LayersIcon, Brain, Activity } from "lucide-react";
import type { ArchitectureResponse } from "@/lib/types";

const STAGE_ICONS = [Database, Filter, Zap, LayersIcon, Brain, Activity];

export default function ArchitecturePage() {
  const { data, isLoading } = useQuery<ArchitectureResponse>({ queryKey: ["/api/architecture"] });

  return (
    <div className="space-y-6 max-w-[1100px] mx-auto">
      <div>
        <h1 className="text-xl font-semibold tracking-tight" data-testid="text-architecture-title">Pipeline</h1>
        <p className="text-sm text-muted-foreground">
          End-to-end neural decoding pipeline. Streaming filters maintain per-channel state so latency
          stays low and behaviour matches batch processing exactly.
        </p>
      </div>

      <Card className="border-card-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Data flow</CardTitle>
          <CardDescription>Each stage runs window-by-window during streaming inference.</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading || !data ? (
            <div className="text-sm text-muted-foreground" data-testid="text-architecture-loading">
              Loading pipeline metadata…
            </div>
          ) : (
            <div className="space-y-3" data-testid="list-architecture-stages">
              {data.stages.map((s, i) => {
                const Icon = STAGE_ICONS[i % STAGE_ICONS.length];
                return (
                  <div
                    key={s.name}
                    className="flex items-start gap-4 rounded-md border border-card-border p-4 bg-card"
                    data-testid={`stage-${i}`}
                  >
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
                      <Icon className="h-5 w-5" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-medium text-sm">{s.name}</span>
                        <Badge variant="outline" className="font-mono text-[10px]">
                          stage {i + 1}
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">{s.detail}</p>
                      <div className="mt-2 flex items-center gap-2 text-[11px] font-mono text-muted-foreground">
                        <span className="rounded bg-muted/40 px-1.5 py-0.5">{s.input}</span>
                        <ArrowRight className="h-3 w-3" />
                        <span className="rounded bg-muted/40 px-1.5 py-0.5">{s.output}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="border-card-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">GPU acceleration</CardTitle>
            <CardDescription>How the engine takes advantage of hardware when it’s present.</CardDescription>
          </CardHeader>
          <CardContent className="text-sm space-y-2 text-muted-foreground">
            <p>
              The decoder selects <code className="font-mono text-foreground">torch.device('cuda')</code> automatically
              when a GPU is visible to PyTorch and falls back to CPU otherwise. Mixed-precision <code className="font-mono">autocast</code> is
              enabled on CUDA only.
            </p>
            <p>
              Hot kernels (spike detection, feature reduction) are written as plain PyTorch ops so they hit cuBLAS/cuDNN paths.
              Native CUDA / Triton extensions live in <code className="font-mono">engine/neurostream/cuda_ext</code>;
              see <code className="font-mono">docs/cuda_notes.md</code> for the integration plan.
            </p>
          </CardContent>
        </Card>
        <Card className="border-card-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Public dataset hooks</CardTitle>
            <CardDescription>Replace synthetic generation with real neural recordings.</CardDescription>
          </CardHeader>
          <CardContent className="text-sm space-y-1.5 text-muted-foreground">
            <ul className="list-disc pl-4 space-y-1">
              <li>Allen Brain Observatory — Neuropixels visual cortex recordings</li>
              <li>OpenNeuro — open MEG / ECoG / iEEG datasets</li>
              <li>BNCI Horizon 2020 — EEG BCI competition data</li>
              <li>PhysioNet EEG Motor Imagery — 64-ch movement classification</li>
            </ul>
            <p className="pt-2 text-xs">Loaders that match the engine’s window-by-window streaming API are sketched in <code className="font-mono">docs/architecture.md</code>.</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
