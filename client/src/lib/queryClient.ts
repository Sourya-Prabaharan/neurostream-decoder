import { QueryClient, QueryFunction } from "@tanstack/react-query";

const API_BASE = "__PORT_5000__".startsWith("__") ? "" : "__PORT_5000__";

// Static JSON fallbacks for endpoints that ship a checked-in artifact under
// client/public/data. Used when no Express backend is reachable (e.g. when
// the dashboard is served as a static site from GitHub Pages).
// Map: API path → JSON file name relative to <base>/data/.
const STATIC_FALLBACKS: Record<string, string> = {
  "/api/sessions": "sessions.json",
  "/api/benchmark": "benchmark.json",
  "/api/stream-benchmark": "stream_benchmark.json",
  "/api/real-eeg-smoke": "real_eeg_smoke.json",
  "/api/architecture": "architecture.json",
  "/api/health": "health.json",
};

// Vite injects BASE_URL from vite.config.ts (base: "./"). Anchor to the
// document base so the same code works at the repo root and under a
// GitHub Pages subpath. Falls back to "./" if BASE_URL is unset.
function staticDataUrl(file: string): string {
  const base = (import.meta as any).env?.BASE_URL ?? "./";
  const prefix = base.endsWith("/") ? base : `${base}/`;
  return `${prefix}data/${file}`;
}

async function throwIfResNotOk(res: Response) {
  if (!res.ok) {
    const text = (await res.text()) || res.statusText;
    throw new Error(`${res.status}: ${text}`);
  }
}

export async function apiRequest(
  method: string,
  url: string,
  data?: unknown | undefined,
): Promise<Response> {
  const res = await fetch(`${API_BASE}${url}`, {
    method,
    headers: data ? { "Content-Type": "application/json" } : {},
    body: data ? JSON.stringify(data) : undefined,
  });

  await throwIfResNotOk(res);
  return res;
}

async function fetchStaticFallback(apiPath: string): Promise<any | undefined> {
  const file = STATIC_FALLBACKS[apiPath];
  if (!file) return undefined;
  const res = await fetch(staticDataUrl(file));
  if (!res.ok) return undefined;
  return res.json();
}

type UnauthorizedBehavior = "returnNull" | "throw";
export const getQueryFn: <T>(options: {
  on401: UnauthorizedBehavior;
}) => QueryFunction<T> =
  ({ on401: unauthorizedBehavior }) =>
  async ({ queryKey }) => {
    const apiPath = queryKey.join("/");
    try {
      const res = await fetch(`${API_BASE}${apiPath}`);
      if (unauthorizedBehavior === "returnNull" && res.status === 401) {
        return null;
      }
      if (!res.ok) {
        const fallback = await fetchStaticFallback(apiPath);
        if (fallback !== undefined) return fallback;
        await throwIfResNotOk(res);
      }
      return await res.json();
    } catch (err) {
      const fallback = await fetchStaticFallback(apiPath);
      if (fallback !== undefined) return fallback;
      throw err;
    }
  };

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: getQueryFn({ on401: "throw" }),
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
      retry: false,
    },
    mutations: {
      retry: false,
    },
  },
});
