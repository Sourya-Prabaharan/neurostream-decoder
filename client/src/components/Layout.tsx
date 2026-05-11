import { type ReactNode } from "react";
import { Link, useLocation } from "wouter";
import { Logo } from "./Logo";
import { useTheme } from "@/lib/theme";
import { Button } from "@/components/ui/button";
import { Activity, Cpu, Layers, Moon, Sun, BarChart3 } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import type { DeviceInfo, SessionsResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

interface NavItem {
  href: string;
  label: string;
  icon: typeof Activity;
}

const NAV: NavItem[] = [
  { href: "/", label: "Live decode", icon: Activity },
  { href: "/architecture", label: "Pipeline", icon: Layers },
  { href: "/benchmark", label: "Benchmark", icon: BarChart3 },
];

export function Layout({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  const { theme, setTheme } = useTheme();
  const { data } = useQuery<SessionsResponse>({ queryKey: ["/api/sessions"] });
  const device = data?.device;

  return (
    <div className="flex min-h-screen w-full bg-background text-foreground font-sans">
      <aside className="hidden md:flex md:w-60 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground">
        <div className="flex items-center gap-2 px-5 py-5 border-b border-sidebar-border">
          <div className="text-primary">
            <Logo className="h-7 w-7" />
          </div>
          <div className="leading-tight">
            <div className="font-semibold tracking-tight text-sm">NeuroStream</div>
            <div className="text-xs text-muted-foreground">Decoder</div>
          </div>
        </div>
        <nav className="px-3 py-4 flex-1 space-y-1" data-testid="nav-primary">
          {NAV.map((item) => {
            const Icon = item.icon;
            const isActive = location === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                data-testid={`link-nav-${item.label.toLowerCase().replace(/\s+/g, "-")}`}
                className={cn(
                  "flex items-center gap-2.5 px-3 py-2 rounded-md text-sm hover-elevate transition-colors",
                  isActive
                    ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                    : "text-sidebar-foreground/80",
                )}
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>
        <DeviceFooter device={device} />
      </aside>
      <main className="flex-1 flex flex-col min-w-0">
        <header className="flex items-center justify-between border-b border-border px-5 md:px-8 py-3 bg-background/70 backdrop-blur sticky top-0 z-10">
          <div className="md:hidden flex items-center gap-2">
            <span className="text-primary">
              <Logo className="h-6 w-6" />
            </span>
            <span className="font-semibold text-sm">NeuroStream</span>
          </div>
          <div className="hidden md:flex items-center gap-3 text-xs text-muted-foreground" data-testid="text-header-subtitle">
            <span className="inline-flex h-2 w-2 rounded-full bg-primary animate-pulse" />
            Real-time neural signal decoder · synthetic dataset
          </div>
          <div className="flex items-center gap-2">
            <DeviceBadge device={device} />
            <Button
              variant="ghost"
              size="icon"
              data-testid="button-theme-toggle"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          </div>
        </header>
        <div className="flex-1 px-5 md:px-8 py-6 md:py-8 min-w-0">{children}</div>
      </main>
    </div>
  );
}

function DeviceBadge({ device }: { device: DeviceInfo | undefined }) {
  const cuda = device?.cuda_available;
  return (
    <span
      data-testid="status-device"
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-mono",
        cuda
          ? "border-primary/40 bg-primary/10 text-primary"
          : "border-border bg-muted/30 text-muted-foreground",
      )}
    >
      <Cpu className="h-3 w-3" />
      {cuda ? device?.cuda_device_name?.split(" ").slice(-2).join(" ") ?? "CUDA" : "CPU"}
    </span>
  );
}

function DeviceFooter({ device }: { device: DeviceInfo | undefined }) {
  return (
    <div className="border-t border-sidebar-border p-4 text-xs space-y-1 font-mono text-muted-foreground" data-testid="panel-device-footer">
      <div className="flex justify-between"><span>device</span><span data-testid="text-device-kind">{device?.device ?? "—"}</span></div>
      <div className="flex justify-between"><span>cuda</span><span>{device?.cuda_available ? "yes" : "no"}</span></div>
      <div className="flex justify-between"><span>torch</span><span>{device?.torch_version?.split("+")[0] ?? "—"}</span></div>
      <div className="flex justify-between"><span>mixed-fp16</span><span>{device?.mixed_precision ? "enabled" : "off"}</span></div>
    </div>
  );
}
