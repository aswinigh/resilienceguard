# Agent Directives: ResilienceGuard

## Project Context
ResilienceGuard runs fire drills on AI agents that use Nebius Token Factory: a fault-injection proxy in
front of Token Factory, an audit that runs unmodified agents through six drills, and a fix loop where
NVIDIA Nemotron writes fixes that are re-audited until they pass. Built for the Nebius x NVIDIA Global AI
Hackathon (Coding and Agentic Engineering track).

## Architecture
* `src/proxy.py`: OpenAI-compatible fault proxy (faults, call log, guard, kill switch).
* `src/rg_shim/`: startup shim that reroutes Python agents' Token Factory traffic through the proxy.
* `src/agent_audit.py`: the six drills and scoring. `src/fixer.py`: the Nemotron fix loop.
* `app.py` + `studio/index.html`: the studio (standard library server, no page dependencies).
* `src/interceptor/` and `src/guard.py`: in-process library mode.

## Constraints
* Python, standard library where possible. Agents under test must run unmodified.
* Never touch the original agent folder: every drill and every candidate fix works in a fresh copy.
* Never write API keys into files other than the gitignored `.env`; saved runs must stay free of keys and machine paths.
* Scores must come from what runs actually did (exit codes, proxy logs, tool results), never be hardcoded.

## Verification
* Use `pytest`. Tests must not need network access: use `tests/fake_model.py` and the stand-in agents in `tests/fixtures/`.
* Run the full suite before asking for approval to merge changes.
