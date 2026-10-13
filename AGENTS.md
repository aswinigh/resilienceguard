# Agent Directives: Fault Injection System

## Project Context
We are building a Python SDK-level fault injection system for AI agents, specifically targeted for the Nebius x NVIDIA Global AI Hackathon. 

## Architectural Constraints
* **Language & Framework:** Use Python. We are building SDK monkey-patching for the OpenAI client and using `LangGraph` for the baseline agent execution.
* **Scope:** Do NOT scaffold Kubernetes operators or Triton sidecars yet. Stick strictly to the local Python SDK interception MVP.
* **Testing:** Use `pytest`. The primary test case must prove that the agent hits the simulated 500 error and circuit-breaks instead of looping infinitely.

## Verification
* Use your browser tools to read the `LangGraph` documentation if you need to recall how to construct the graph state.
* You must verify your own code by running `pytest` in the terminal before asking for my approval to merge diffs.