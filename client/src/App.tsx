import { Switch, Route, Router } from "wouter";
import { useHashLocation } from "wouter/use-hash-location";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/not-found";
import EegStream from "@/pages/eeg";
import SystemMetrics from "@/pages/system";
import ChallengesPage from "@/pages/challenges";
import ArchitecturePage from "@/pages/architecture";
import BenchmarkPage from "@/pages/benchmark";
import { Layout } from "@/components/Layout";
import { ThemeProvider } from "@/lib/theme";

function AppRouter() {
  return (
    <Layout>
      <Switch>
        <Route path="/" component={EegStream} />
        <Route path="/system" component={SystemMetrics} />
        <Route path="/architecture" component={ArchitecturePage} />
        <Route path="/challenges" component={ChallengesPage} />
        <Route path="/benchmark" component={BenchmarkPage} />
        <Route component={NotFound} />
      </Switch>
    </Layout>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <TooltipProvider>
          <Toaster />
          <Router hook={useHashLocation}>
            <AppRouter />
          </Router>
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
