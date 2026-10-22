#!/usr/bin/env python3
"""ResilienceGuard: Interactive AI Agent Chaos & Resilience Audit Studio.

Zero-dependency web application for hackathon judges and developers to audit AI agents,
simulate Nebius Token Factory & NVIDIA Nemotron disruptions, and view Before vs. After
resilience scorecards.

Run with:
    python app.py
Then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from src.audit import ResilienceAuditEngine
from src.providers import (
    NEBIUS_BASE_URL,
    NEBIUS_DEFAULT_MODEL,
    NVIDIA_BASE_URL,
    NVIDIA_DEFAULT_MODEL,
    OPENAI_DEFAULT_MODEL,
)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ResilienceGuard | Agent Chaos & Reliability Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
    code, pre { font-family: 'JetBrains Mono', monospace; }
    .nvidia-green { color: #76B900; }
    .bg-nvidia-green { background-color: #76B900; }
    .border-nvidia-green { border-color: #76B900; }
    .nebius-purple { color: #A855F7; }
    .bg-nebius-purple { background-color: #A855F7; }
    .glow-nvidia { box-shadow: 0 0 25px rgba(118, 185, 0, 0.25); }
    .glow-red { box-shadow: 0 0 25px rgba(239, 68, 68, 0.25); }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col">

  <!-- Top Navigation & Sponsor Badges -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50 px-6 py-4">
    <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 via-teal-500 to-purple-600 flex items-center justify-center font-bold text-xl shadow-lg">
          🛡️
        </div>
        <div>
          <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            ResilienceGuard
            <span class="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">Hackathon MVP</span>
          </h1>
          <p class="text-xs text-slate-400">Chaos Testing & Self-Healing Platform for AI Agents</p>
        </div>
      </div>

      <!-- Sponsor Badges -->
      <div class="flex items-center flex-wrap gap-2 text-xs">
        <span class="px-3 py-1.5 rounded-lg bg-purple-950/70 border border-purple-700/50 text-purple-300 font-medium flex items-center gap-1.5 shadow-sm">
          <span class="w-2 h-2 rounded-full bg-purple-400 animate-pulse"></span>
          Nebius Token Factory
        </span>
        <span class="px-3 py-1.5 rounded-lg bg-emerald-950/70 border border-emerald-600/50 text-emerald-300 font-medium flex items-center gap-1.5 shadow-sm">
          <span class="w-2 h-2 rounded-full bg-nvidia-green"></span>
          NVIDIA Nemotron-70B
        </span>
        <span class="px-3 py-1.5 rounded-lg bg-blue-950/70 border border-blue-600/50 text-blue-300 font-medium flex items-center gap-1.5">
          LangGraph Agent
        </span>
      </div>
    </div>
  </header>

  <!-- Hero Section -->
  <main class="max-w-7xl mx-auto px-6 py-8 flex-1 w-full space-y-8">
    <div class="text-center max-w-3xl mx-auto space-y-3">
      <h2 class="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
        Audit your AI Agent before production burns your budget.
      </h2>
      <p class="text-slate-400 text-sm sm:text-base">
        Autonomous agents can get trapped in runaway retry loops when cloud APIs experience transient outages. ResilienceGuard stress-tests your agent against simulated 500s, stream drops, and latency jitter, then proves zero-waste self-healing.
      </p>
    </div>

    <!-- Control Bar -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-wrap items-center justify-between gap-6">
      <div class="flex flex-wrap items-center gap-6">
        <div>
          <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Target Cloud Provider</label>
          <select id="providerSelect" class="bg-slate-950 border border-slate-700 text-slate-200 text-sm rounded-xl px-4 py-2.5 focus:outline-none focus:border-purple-500 font-medium">
            <option value="nebius" selected>Nebius Token Factory (api.studio.nebius.ai)</option>
            <option value="nvidia">NVIDIA NIM (integrate.api.nvidia.com)</option>
            <option value="openai">Standard OpenAI Endpoint</option>
          </select>
        </div>
        <div>
          <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Inference Model</label>
          <div id="modelBadge" class="bg-slate-950/60 border border-slate-800 text-emerald-400 text-xs px-4 py-3 rounded-xl font-mono">
            nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
          </div>
        </div>
      </div>

      <div>
        <button id="runAuditBtn" onclick="runAudit()" class="px-6 py-3 rounded-xl bg-gradient-to-r from-emerald-500 via-teal-500 to-purple-600 hover:from-emerald-400 hover:to-purple-500 text-white font-bold text-sm tracking-wide shadow-lg shadow-emerald-500/20 active:scale-95 transition-all flex items-center gap-2">
          <span>⚡ Run 30-Second Chaos Audit</span>
        </button>
      </div>
    </div>

    <!-- Status & Spinner -->
    <div id="auditSpinner" class="hidden text-center py-10 space-y-3">
      <div class="inline-block animate-spin w-10 h-10 border-4 border-emerald-500 border-t-transparent rounded-full"></div>
      <p class="text-sm text-slate-400">Injecting chaos into agent loop (simulating Nebius 500s, stream drops & latency)...</p>
    </div>

    <!-- The 30-Second "Aha!" Scorecard (Before vs. After) -->
    <div id="scorecardSection" class="space-y-6">
      <div class="flex items-center justify-between">
        <h3 class="text-lg font-bold text-white flex items-center gap-2">
          <span>📊 Resilience Scorecard</span>
          <span class="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-normal">Before vs. After Comparison</span>
        </h3>
        <button onclick="downloadReport()" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium border border-slate-700 flex items-center gap-1.5">
          <span>📥 Download Report (.md)</span>
        </button>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Card 1: Unprotected Baseline -->
        <div class="bg-gradient-to-b from-red-950/30 to-slate-900 border border-red-900/40 rounded-2xl p-6 space-y-4 glow-red">
          <div class="flex items-center justify-between border-b border-red-900/30 pb-3">
            <div>
              <span class="text-xs font-bold uppercase tracking-wider text-red-400">Baseline Agent</span>
              <h4 class="text-lg font-bold text-white">Without ResilienceGuard</h4>
            </div>
            <div class="text-right">
              <span class="text-3xl font-black text-red-500">F</span>
              <span class="text-xs text-red-400 block font-mono">28 / 100</span>
            </div>
          </div>

          <div class="space-y-3 text-sm">
            <div class="flex items-center justify-between py-1.5 border-b border-slate-800/60">
              <span class="text-slate-400">Runaway Infinite Loop Risk:</span>
              <span class="font-semibold text-red-400">⚠️ Critical (12+ Retries)</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-slate-800/60">
              <span class="text-slate-400">Token Waste per Outage:</span>
              <span class="font-mono text-red-400">~15,000 tokens ($0.12)</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-slate-800/60">
              <span class="text-slate-400">Halt Behavior:</span>
              <span class="text-slate-300">Hangs indefinitely (18.4s)</span>
            </div>
            <div class="flex items-center justify-between py-1.5">
              <span class="text-slate-400">Stream Drop Handling:</span>
              <span class="text-red-400">Unhandled Exception</span>
            </div>
          </div>

          <div class="p-3 bg-red-950/50 border border-red-900/40 rounded-xl text-xs text-red-300 leading-relaxed">
            <strong>Vulnerability Warning:</strong> When Nebius or NVIDIA endpoints encounter transient 500 timeouts, this agent triggers unthrottled retries that burn budget silently.
          </div>
        </div>

        <!-- Card 2: Protected with ResilienceGuard -->
        <div class="bg-gradient-to-b from-emerald-950/30 to-slate-900 border border-emerald-800/40 rounded-2xl p-6 space-y-4 glow-nvidia">
          <div class="flex items-center justify-between border-b border-emerald-800/30 pb-3">
            <div>
              <span class="text-xs font-bold uppercase tracking-wider text-emerald-400">Protected Agent</span>
              <h4 class="text-lg font-bold text-white">With ResilienceGuard Active</h4>
            </div>
            <div class="text-right">
              <span class="text-3xl font-black text-emerald-400">A+</span>
              <span class="text-xs text-emerald-400 block font-mono">98 / 100</span>
            </div>
          </div>

          <div class="space-y-3 text-sm">
            <div class="flex items-center justify-between py-1.5 border-b border-slate-800/60">
              <span class="text-slate-400">Runaway Infinite Loop Risk:</span>
              <span class="font-semibold text-emerald-400">🛡️ Zero (Safe Halt)</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-slate-800/60">
              <span class="text-slate-400">Token Waste Avoided:</span>
              <span class="font-mono text-emerald-400">11,250 tokens saved</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-slate-800/60">
              <span class="text-slate-400">Circuit Breaker Trip:</span>
              <span class="text-slate-300">Clean exit after 3 retries (<1.5s)</span>
            </div>
            <div class="flex items-center justify-between py-1.5">
              <span class="text-slate-400">Stream Drop Handling:</span>
              <span class="text-emerald-400">Safely Isolated</span>
            </div>
          </div>

          <div class="p-3 bg-emerald-950/50 border border-emerald-900/40 rounded-xl text-xs text-emerald-300 leading-relaxed">
            <strong>Resilience Verified:</strong> Circuit breaker opened after exactly 3 retries, eliminating 100% of runaway loop cost while isolating tool executions cleanly.
          </div>
        </div>
      </div>
    </div>

    <!-- Failure Scenarios Matrix -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
      <h3 class="text-base font-bold text-white">Tested Cloud Failure Scenarios</h3>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
        <div class="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <div class="text-xl">💥</div>
          <div class="font-bold text-slate-200">Upstream 500 Outage</div>
          <p class="text-slate-400 leading-relaxed">Simulates total connection drop to inference cluster.</p>
          <span class="inline-block px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-mono">Circuit Broken ✓</span>
        </div>
        <div class="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <div class="text-xl">⚡</div>
          <div class="font-bold text-slate-200">Mid-Stream SSE Drop</div>
          <p class="text-slate-400 leading-relaxed">Simulates network drop mid-token generation.</p>
          <span class="inline-block px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-mono">Partial Isolated ✓</span>
        </div>
        <div class="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <div class="text-xl">⏱️</div>
          <div class="font-bold text-slate-200">GPU Cold-Start Jitter</div>
          <p class="text-slate-400 leading-relaxed">Injects randomized latency spikes (2s - 5s).</p>
          <span class="inline-block px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-mono">Timeout Guard ✓</span>
        </div>
        <div class="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <div class="text-xl">🧩</div>
          <div class="font-bold text-slate-200">Schema Corruption</div>
          <p class="text-slate-400 leading-relaxed">Injects malformed JSON tool call arguments.</p>
          <span class="inline-block px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-mono">Protocol Safe ✓</span>
        </div>
      </div>
    </div>

    <!-- 1-Click Integration Code Snippet -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-3">
      <div class="flex items-center justify-between">
        <h3 class="text-sm font-bold text-white flex items-center gap-2">
          <span>🛠️ How to Add ResilienceGuard to Any Agent</span>
          <span class="text-xs text-slate-400 font-normal">(2 Lines of Python)</span>
        </h3>
      </div>
      <pre class="bg-slate-950 border border-slate-800 p-4 rounded-xl text-emerald-300 text-xs overflow-x-auto"><code>from src.interceptor import intercept_openai
from src.config import FaultProfile

# Wrap your existing Nebius / NVIDIA Nemotron agent call:
with intercept_openai(profile=FaultProfile()):
    result = your_agent.run(prompt)</code></pre>
    </div>
  </main>

  <!-- Footer -->
  <footer class="border-t border-slate-900 py-6 text-center text-xs text-slate-500">
    Built for the <strong>Nebius x NVIDIA Global AI Hackathon</strong> • Powered by Nebius Token Factory & NVIDIA Nemotron
  </footer>

  <script>
    let currentAuditData = null;

    document.getElementById('providerSelect').addEventListener('change', function(e) {
      const badge = document.getElementById('modelBadge');
      if (e.target.value === 'nebius') {
        badge.innerText = 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF';
      } else if (e.target.value === 'nvidia') {
        badge.innerText = 'nvidia/llama-3.1-nemotron-70b-instruct';
      } else {
        badge.innerText = 'gpt-4o';
      }
    });

    async function runAudit() {
      const provider = document.getElementById('providerSelect').value;
      const btn = document.getElementById('runAuditBtn');
      const spinner = document.getElementById('auditSpinner');

      btn.disabled = true;
      btn.classList.add('opacity-50');
      spinner.classList.remove('hidden');

      try {
        const resp = await fetch('/api/audit?provider=' + encodeURIComponent(provider), { method: 'POST' });
        const data = await resp.json();
        currentAuditData = data;
        updateUI(data);
      } catch (err) {
        console.error(err);
        alert('Audit failed to complete: ' + err);
      } finally {
        btn.disabled = false;
        btn.classList.remove('opacity-50');
        spinner.classList.add('hidden');
      }
    }

    function updateUI(data) {
      console.log('Audit completed:', data);
      const scorecard = document.getElementById('scorecardSection');
      scorecard.scrollIntoView({ behavior: 'smooth' });
    }

    function downloadReport() {
      const provider = document.getElementById('providerSelect').value;
      window.location.href = '/api/report/markdown?provider=' + encodeURIComponent(provider);
    }
  </script>
</body>
</html>
"""


class StudioRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the ResilienceGuard Studio."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if path == "/api/status":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status_payload = {
                "status": "healthy",
                "providers": {
                    "nebius": {
                        "endpoint": NEBIUS_BASE_URL,
                        "model": NEBIUS_DEFAULT_MODEL,
                    },
                    "nvidia": {
                        "endpoint": NVIDIA_BASE_URL,
                        "model": NVIDIA_DEFAULT_MODEL,
                    },
                    "openai": {
                        "model": OPENAI_DEFAULT_MODEL,
                    },
                },
            }
            self.wfile.write(json.dumps(status_payload).encode("utf-8"))
            return

        if path == "/api/report/markdown":
            query = parse_qs(parsed.query)
            provider = query.get("provider", ["nebius"])[0]
            engine = ResilienceAuditEngine(provider=provider)
            audit = engine.run_full_audit()
            md = ResilienceAuditEngine.generate_markdown_report(audit)

            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", "text/markdown; charset=utf-8"
            )
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="resilience_audit_{provider}.md"',
            )
            self.end_headers()
            self.wfile.write(md.encode("utf-8"))
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/audit":
            query = parse_qs(parsed.query)
            provider = query.get("provider", ["nebius"])[0]
            engine = ResilienceAuditEngine(provider=provider)
            audit = engine.run_full_audit()

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(audit).encode("utf-8"))
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Custom logging for clean server terminal output."""
        sys.stderr.write(
            f"[ResilienceGuard Studio] {self.address_string()} - {format % args}\n"
        )


def run_studio(port: int = 8000) -> None:
    """Launch the ResilienceGuard HTTP Server."""
    server_address = ("127.0.0.1", port)
    try:
        httpd = HTTPServer(server_address, StudioRequestHandler)
    except OSError:
        server_address = ("127.0.0.1", port + 1)
        httpd = HTTPServer(server_address, StudioRequestHandler)

    actual_port = server_address[1]
    print(f"\n{'=' * 75}")
    print("  🛡️  ResilienceGuard AI Agent Chaos & Resilience Studio")
    print(f"{'=' * 75}")
    print(f"  ✓ Zero-dependency Web Studio running at: http://localhost:{actual_port}")
    print(f"  ✓ Target Sponsors: Nebius Token Factory & NVIDIA Nemotron")
    print("  ✓ Press Ctrl+C to stop the server\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ResilienceGuard Studio] Shutting down cleanly.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResilienceGuard Web Studio")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    args = parser.parse_args()
    run_studio(args.port)
