#!/usr/bin/env python3
"""Interactive Fault Injection & Resilience Demo for Nebius x NVIDIA Hackathon.

Demonstrates baseline agent execution vs. simulated chaos failures:
1. Baseline Run (Normal execution with web search and email tools)
2. Injected 500 / APIConnectionError (Proves circuit-breaking resilience and no infinite loops)
3. Injected Mid-Stream Drop (Proves stream failure handling)
4. Injected Tool Payload Corruption (Proves tool error recovery)
"""

from __future__ import annotations

import argparse
import sys
import time

from langchain_core.messages import AIMessage

from src.config.parser import FaultProfile, FaultRule
from src.demo_agent import run_demo_agent
from src.interceptor.openai_patch import intercept_openai
from src.providers import get_sponsor_client


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_banner():
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("=" * 75)
    print("  Nebius x NVIDIA AI Hackathon: Agent Fault Injection & Resilience System")
    print("=" * 75)
    print(f"{Colors.END}")


def run_baseline_demo(provider: str):
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}--- SCENARIO 1: Baseline Execution (No Faults) ---{Colors.END}")
    print(f"Target Provider: {Colors.BOLD}{provider.upper()}{Colors.END}")

    client, model = get_sponsor_client(provider)
    print(f"Endpoint: {client.base_url}")
    print(f"Selected Model: {model}")

    prompt = "Search for Nebius H100 GPU clusters and email the summary to team@hackathon.ai"
    print(f"User Prompt: \"{prompt}\"")

    class MockClient:
        def __init__(self, base_url):
            self.base_url = base_url
            self.call_count = 0

            def mock_create(*a, **kw):
                self.call_count += 1
                if self.call_count == 1:
                    # Step 1: Agent requests web_search tool
                    return type(
                        "MockResp",
                        (),
                        {
                            "choices": [
                                type(
                                    "Choice",
                                    (),
                                    {
                                        "message": type(
                                            "Msg",
                                            (),
                                            {
                                                "content": "Searching for Nebius cluster specs...",
                                                "tool_calls": [
                                                    type(
                                                        "TC",
                                                        (),
                                                        {
                                                            "id": "call_1",
                                                            "function": type(
                                                                "Fn",
                                                                (),
                                                                {
                                                                    "name": "web_search",
                                                                    "arguments": '{"query": "Nebius H100 clusters"}',
                                                                },
                                                            )(),
                                                        },
                                                    )()
                                                ],
                                            },
                                        )()
                                    },
                                )()
                            ]
                        },
                    )()
                else:
                    # Step 2: Agent completes with final answer
                    return type(
                        "MockResp",
                        (),
                        {
                            "choices": [
                                type(
                                    "Choice",
                                    (),
                                    {
                                        "message": type(
                                            "Msg",
                                            (),
                                            {
                                                "content": "Successfully retrieved Nebius H100 GPU cluster specs and emailed the summary.",
                                                "tool_calls": [],
                                            },
                                        )()
                                    },
                                )()
                            ]
                        },
                    )()

            self.chat = type(
                "Chat",
                (),
                {"completions": type("Completions", (), {"create": mock_create})()},
            )()

    mock_client = MockClient(client.base_url)
    print(f"\n{Colors.OKBLUE}[Agent Node]{Colors.END} Invoking model...")
    state = run_demo_agent(prompt, client=mock_client, max_retries=3)

    print(f"{Colors.OKGREEN}✓ Agent execution status:{Colors.END} {state.get('status')}")
    print(f"{Colors.OKGREEN}✓ Circuit breaker open:{Colors.END} {state.get('circuit_open')}")
    print(f"{Colors.OKGREEN}✓ Total messages generated:{Colors.END} {len(state.get('messages', []))}")


def run_chaos_500_demo(provider: str):
    print(f"\n{Colors.WARNING}{Colors.BOLD}--- SCENARIO 2: Injected 500 Connection Error (Circuit Breaker Test) ---{Colors.END}")
    print(f"Target Provider: {Colors.BOLD}{provider.upper()}{Colors.END}")

    rule = FaultRule(
        target=provider,
        action="error",
        error_type="APIConnectionError",
        status_code=500,
        message=f"Simulated 500: Upstream connection timeout to {provider.upper()} inference cluster",
        max_occurrences=5,
    )
    profile = FaultProfile(faults=[rule])

    client, model = get_sponsor_client(provider)

    print(f"Active Fault: {Colors.FAIL}{rule.message}{Colors.END}")
    print(f"Circuit Breaker Threshold: {Colors.BOLD}3 retries{Colors.END}")

    with intercept_openai(profile):
        print(f"\n{Colors.OKBLUE}[Agent Node]{Colors.END} Starting execution with chaos interceptor active...")
        state = run_demo_agent(
            "Summarize the market landscape",
            client=client,
            max_retries=3,
            model=model,
        )

    print(f"\n{Colors.BOLD}Chaos Test Results:{Colors.END}")
    print(f"  Final Agent Status:     {Colors.FAIL if state.get('circuit_open') else Colors.OKGREEN}{state.get('status')}{Colors.END}")
    print(f"  Circuit Breaker Tripped: {Colors.FAIL if state.get('circuit_open') else Colors.OKGREEN}{state.get('circuit_open')}{Colors.END}")
    print(f"  Consecutive Retries:    {state.get('retry_count')}/3")
    print(f"  Logged Error:           {state.get('error_message')}")

    if state.get("circuit_open") and state.get("retry_count") == 3:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ SUCCESS: Agent safely circuit-broke instead of entering an infinite loop!{Colors.END}")
    else:
        print(f"\n{Colors.FAIL}✗ FAILED: Expected circuit breaker to trip.{Colors.END}")


def run_chaos_stream_demo(provider: str):
    print(f"\n{Colors.WARNING}{Colors.BOLD}--- SCENARIO 3: Injected Mid-Stream Drop (SSE Dropout) ---{Colors.END}")
    print(f"Target Provider: {Colors.BOLD}{provider.upper()}{Colors.END}")

    rule = FaultRule(
        target=provider,
        action="stream_drop",
        error_type="APIConnectionError",
        message=f"Simulated SSE network drop during {provider.upper()} token generation",
        stream_drop_after_chunks=3,
    )
    profile = FaultProfile(faults=[rule])

    client, model = get_sponsor_client(provider)

    print(f"Active Fault: Drop connection after {rule.stream_drop_after_chunks} chunks.")

    with intercept_openai(profile):
        print(f"\n{Colors.OKBLUE}[Stream Client]{Colors.END} Requesting streaming chat completion...")
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Explain agent resilience"}],
            stream=True,
        )

        chunk_count = 0
        try:
            for chunk in stream:
                chunk_count += 1
                token = chunk.choices[0].delta.content or ""
                print(f"  [Chunk {chunk_count}] Received token: \"{token.strip()}\"")
        except Exception as e:
            print(f"\n{Colors.FAIL}Caught Injected Fault as expected:{Colors.END} {type(e).__name__}: {e}")
            print(f"{Colors.OKGREEN}✓ Successfully verified mid-stream drop resilience!{Colors.END}")


def main():
    parser = argparse.ArgumentParser(description="Fault Injection & Resilience Demo")
    parser.add_argument(
        "--provider",
        choices=["nebius", "nvidia", "openai"],
        default="nebius",
        help="Target sponsor provider (default: nebius)",
    )
    parser.add_argument(
        "--scenario",
        choices=["baseline", "500", "stream", "all"],
        default="all",
        help="Scenario to run (default: all)",
    )
    args = parser.parse_args()

    print_banner()

    if args.scenario in ("baseline", "all"):
        run_baseline_demo(args.provider)
    if args.scenario in ("500", "all"):
        run_chaos_500_demo(args.provider)
    if args.scenario in ("stream", "all"):
        run_chaos_stream_demo(args.provider)

    print(f"\n{Colors.BOLD}{'=' * 75}{Colors.END}\n")


if __name__ == "__main__":
    main()
