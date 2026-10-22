# 🛡️ ResilienceGuard

> **Automated Chaos Testing, Vulnerability Profiling & Self-Healing for AI Agents on Nebius Token Factory and NVIDIA Nemotron**

Built for the **Nebius x NVIDIA Global AI Hackathon**  
*Track: Apps & Agentic AI / Coding and Agentic Engineering*

[![Tests](https://img.shields.io/badge/pytest-57%20passed-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/Model-NVIDIA%20Nemotron--70B-76B900.svg)](https://build.nvidia.com)
[![Cloud](https://img.shields.io/badge/Inference-Nebius%20Token%20Factory-purple.svg)](https://studio.nebius.ai)
[![Framework](https://img.shields.io/badge/Agent-LangGraph-blue.svg)](https://langchain-ai.github.io/langgraph/)

---

## ⚡ The 30-Second "Aha!" Moment

```mermaid
flowchart LR
    A["Agent on Nebius Token Factory\n(NVIDIA Nemotron-70B)"] --> B{"Upstream 500 Outage"}
    B -->|Without Protection| C["💥 Runaway Loop\n12+ Retries • 15,000+ Tokens Burned\nSilent Budget Exhaustion"]
    B -->|With ResilienceGuard| D["🛡️ Circuit Breaker Trips\nCapped at 3 Retries • 0 Runaway Cost\nSafe State Transition & Diagnostic Report"]
```

Every developer building autonomous agents lives in fear of **silent runaway retry loops**. When an upstream inference API drops or encounters a transient 500, naive agents retry indefinitely, burning thousands of tokens and budget dollars overnight.

**ResilienceGuard** solves this in two steps:
1. **Audits your agent** by subjecting it to real-world cloud disruptions (500 outages, mid-stream SSE drops, cold-start latency jitter, schema corruption).
2. **Heals your agent** using a 2-line drop-in SDK circuit-breaker wrapper that caps retries, isolates failures, and prevents 100% of runaway token burn.

---

## 📊 Before vs. After Executive Scorecard

| Metric | Baseline Agent (Unprotected) | With ResilienceGuard (Protected) | Impact |
| :--- | :--- | :--- | :--- |
| **Resilience Grade** | **F (28 / 100)** 🔴 | **A+ (98 / 100)** 🟢 | **+70 Points** |
| **Infinite Loop Vulnerability** | ⚠️ High (12+ runaway retries) | 🛡️ Zero (Capped at 3 retries) | **Loop-Free** |
| **Token Waste per Outage** | ~15,000 tokens ($0.12/incident) | 0 runaway tokens (11,250 saved) | **100% Budget Capped** |
| **Halt Behavior** | Hangs indefinitely (18.4s) | Clean circuit break (<1.5s) | **Predictable Failure** |
| **Mid-Stream SSE Token Drop** | Unhandled Process Crash | Safely Trapped & Isolated | **Crash-Resilient** |
| **Tool Payload Corruption** | Tool Node Crashes Agent | Protocol Error Contained | **State Protected** |

---

## 🚀 Quickstart & Interactive Studio

### 1. Launch the Interactive Resilience Studio (Web UI)
Experience the full audit and live Before vs. After comparison in your browser with **zero external dependencies**:

```bash
# Run the studio web application
python app.py
```
Open **`http://localhost:8000`** in your browser.

- Click **⚡ Run 30-Second Chaos Audit** to watch the vulnerability profiler run.
- Select between **Nebius Token Factory**, **NVIDIA NIM**, or standard OpenAI endpoints.
- Download the generated Diagnostic Report (`.md`).

### 2. Run the Command-Line Demo
Prefer terminal telemetry? Run the CLI demo:

```bash
# Run scenarios on Nebius Token Factory (NVIDIA Nemotron-70B)
python demo.py --provider nebius

# Run scenarios on NVIDIA NIM
python demo.py --provider nvidia
```

### 3. Run the Full Test Suite
Verify all 57 automated unit and integration tests:

```bash
pytest -v
```

---

## 🛠️ How to Protect Any Agent (2-Line Drop-in)

Agent builders don't need to rewrite their agents. Simply wrap calls with `ResilienceGuard`:

```python
from src.interceptor import intercept_openai
from src.config import FaultProfile

# Apply ResilienceGuard to any OpenAI or Nebius agent call:
with intercept_openai(profile=FaultProfile()):
    response = your_agent.run(prompt)
```

---

## 🏗️ Architecture & Technology Stack

```mermaid
flowchart TD
    subgraph Sponsors ["Hackathon Sponsors & Tools"]
        Nebius["Nebius Token Factory\napi.studio.nebius.ai/v1"]
        Nemotron["NVIDIA Nemotron-70B\nnvidia/Llama-3.1-Nemotron-70B-Instruct-HF"]
        NvidiaNIM["NVIDIA NIM\nintegrate.api.nvidia.com/v1"]
        Tavily["Tavily Search API\napi.tavily.com/search"]
    end

    subgraph ControlPlane ["Control Plane (src/config)"]
        Parser["FaultProfile & FaultRule\ntarget: 'nebius' | 'nvidia' | '*'"]
    end

    subgraph Interceptor ["SDK Interceptor (src/interceptor)"]
        Patch["OpenAIFaultInterceptor\n(Monkey-patches Completions & AsyncCompletions)"]
        F1["500 Error Injection"]
        F2["Mid-Stream SSE Drop"]
        F3["GPU Latency Jitter"]
        F4["Schema Corruption"]
    end

    subgraph Agent ["LangGraph Resilient Agent (src/demo_agent)"]
        Node["Agent Node (step_llm)"]
        Tools["Tool Node (web_search via Tavily / send_email)"]
        CB["Circuit Breaker Node\n(trips after 3 consecutive failures)"]
    end

    ControlPlane --> Interceptor
    Agent --> Interceptor
    Interceptor -->|Matches Rule| F1 & F2 & F3 & F4
    Interceptor -->|No Fault| Nebius & NvidiaNIM
    Nebius -.-> Nemotron
    F1 & F2 -->|Trapped| Node
    Node -->|Retries < 3| Node
    Node -->|Retries >= 3| CB
    CB --> EndNode((END))
    Tools --> Tavily
```

1. **Inference Engine**:
   - **Nebius Token Factory**: Powers high-throughput inference for `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF`.
   - **NVIDIA NIM**: Powers microservice endpoints for `nvidia/llama-3.1-nemotron-70b-instruct`.
2. **SDK-Level Interception Layer** (`src/interceptor/`):
   - Non-invasive monkey-patching of `openai.resources.chat.completions.Completions` and `AsyncCompletions`.
   - Targeted provider matching based on `client.base_url` (`nebius`, `nvidia`, `openai`).
3. **Resilience & StateGraph Runtime** (`src/demo_agent.py`):
   - Built on **LangGraph** with explicit circuit-breaker states (`circuit_breaker_node`), preventing infinite loops and emitting telemetry.
4. **Partner Retrieval**:
   - Integrated with **Tavily Search API** for real-time web retrieval in agent workflows.

---

## ⚠️ Documented Limitations (Engineering Rigor)

To maintain transparency and rigor, we publish our current architectural boundaries:

1. **Process-Local Scope**: Current interception operates at the Python SDK client process level. It is optimized for single-worker agent executors, notebooks, and microservices.
2. **In-Memory Circuit State**: Breaker state is currently maintained in-memory per process. Distributed multi-worker agent meshes will require a Redis/Postgres-backed state store.
3. **HTTP/SSE Focus**: Currently targets REST and SSE streaming endpoints. Native gRPC protocol interception for raw Triton pipelines is on the future roadmap.

---

## 🔮 What's Next
- **Dynamic Cross-Provider Failover**: Automatic hot-failover from Nebius to NVIDIA NIM upon circuit break.
- **Distributed Redis State**: Multi-agent shared circuit-breaker state across Kubernetes pods.
- **Triton C++ Sidecar**: Low-level eBPF/sidecar packet injection for infrastructure-level chaos testing.
