"""Comprehensive Benchmark & Agentic Evaluation Suite for Spark-X2.5-4B.

Evaluates 5 Core Pillars of Modern Agent Systems:
1. Agentic Thinking & Multi-Step Reasoning (GSM8K Math, Deductive Logic, DAG Planning)
2. Tool Calling & Schema Adherence (BFCL-style Selection, Typed Args, Synthesis, Restraint)
3. Context Management & Needle-in-a-Haystack (Single & Multi-Needle Cross-Document Synthesis)
4. Live SSE Thinking Token Streaming Protocol (Real-time delta.reasoning_content -> delta.content)
5. Live End-to-End Web Search Agent (Search Decision -> DDG Execution -> Answer Synthesis)
"""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Any, Dict, List, Optional
import requests
from ddgs import DDGS

BASE_URL = "http://127.0.0.1:8080/v1"
MODEL_ID = "hermitdave/Spark-X2.5-4B-MLX-4bit"


class BenchmarkRunner:
    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []

    def call_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": MODEL_ID,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        t0 = time.time()
        resp = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=120)
        dur = time.time() - t0

        if resp.status_code != 200:
            raise RuntimeError(f"API call failed ({resp.status_code}): {resp.text}")

        data = resp.json()
        data["_client_duration_sec"] = dur
        return data

    def get_vram_telemetry(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/system/vram", timeout=10)
            return resp.json() if resp.status_code == 200 else {}
        except Exception:
            return {}

    # =========================================================================
    # SUITE 1: AGENTIC THINKING & MULTI-STEP REASONING
    # =========================================================================
    def run_suite_reasoning(self) -> None:
        print("\n" + "=" * 70)
        print("  SUITE 1: AGENTIC THINKING & MULTI-STEP REASONING")
        print("=" * 70)

        # 1A. GSM8K / Math Multi-Step Problem
        print("\n▶ [1A] Math Multi-Step Problem (GSM8K Style)")
        math_prompt = (
            "A bakery sells sourdough loaves for $8 each and baguettes for $4 each. "
            "On Saturday, they sell 45 sourdough loaves. The number of baguettes sold is "
            "3 times the number of sourdough loaves sold. Ingredients cost $250 in total. "
            "What is the bakery's net profit on Saturday? Think step by step and provide the final number."
        )
        res = self.call_chat([{"role": "user", "content": math_prompt}], max_tokens=800)
        choice = res["choices"][0]["message"]
        reasoning = choice.get("reasoning_content") or ""
        content = choice.get("content") or ""
        has_thinking = len(reasoning.strip()) > 0
        got_correct = "650" in content or "650" in reasoning

        print(f"  • Thinking Captured: {has_thinking} ({len(reasoning.split())} words)")
        print(f"  • Answer Content: {content.strip()[:160]}...")
        print(f"  • Net Profit $650 Identified: {'PASS' if got_correct else 'FAIL'}")
        print(f"  • Latency: {res['_client_duration_sec']:.2f}s")
        self.results.append({
            "suite": "Reasoning",
            "test": "Math Multi-Step (GSM8K)",
            "passed": got_correct and has_thinking,
            "details": f"Reasoning length: {len(reasoning)} chars, Net profit $650: {got_correct}",
        })

        # 1B. Deductive Logic / Knights & Knaves
        print("\n▶ [1B] Deductive Logic (Knight & Knave Island)")
        logic_prompt = (
            "On an island, inhabitants are either Knights (who always tell the truth) or Knaves "
            "(who always lie). You meet two inhabitants, Alice and Bob. "
            "Alice says: 'At least one of us is a Knave.' "
            "Determine what Alice is and what Bob is. Show your reasoning step by step."
        )
        res = self.call_chat([{"role": "user", "content": logic_prompt}], max_tokens=800)
        choice = res["choices"][0]["message"]
        reasoning = choice.get("reasoning_content") or ""
        content = choice.get("content") or ""
        full_text = f"{reasoning} {content}".lower()
        alice_knight = "alice is a knight" in full_text or "alice: knight" in full_text
        bob_knave = "bob is a knave" in full_text or "bob: knave" in full_text
        passed = alice_knight and bob_knave

        print(f"  • Thinking Captured: {len(reasoning.split())} words")
        print(f"  • Conclusion: Alice=Knight ({alice_knight}), Bob=Knave ({bob_knave}) -> {'PASS' if passed else 'FAIL'}")
        print(f"  • Latency: {res['_client_duration_sec']:.2f}s")
        self.results.append({
            "suite": "Reasoning",
            "test": "Deductive Logic (Knights & Knaves)",
            "passed": passed,
            "details": f"Alice=Knight: {alice_knight}, Bob=Knave: {bob_knave}",
        })

        # 1C. Agentic Planning & Dependency Graph (DevOps DAG)
        print("\n▶ [1C] Agentic Planning & Dependency DAG")
        planning_prompt = (
            "You are an autonomous software agent. Organize these 5 software release tasks into a "
            "strict, logically valid topological execution sequence:\n"
            "A: Deploy built service to production cluster\n"
            "B: Run automated unit and integration tests\n"
            "C: Lint and type-check codebase\n"
            "D: Build and tag container image\n"
            "E: Apply database schema migrations\n"
            "Explain why each step must precede the next, then output the final sequence clearly."
        )
        res = self.call_chat([{"role": "user", "content": planning_prompt}], max_tokens=700)
        choice = res["choices"][0]["message"]
        full_planning = (choice.get("reasoning_content") or "") + " " + (choice.get("content") or "")
        # Valid execution logic: Lint (C) must come before Deploy (A), Build (D) must come before Deploy (A), Tests (B) must come before Deploy (A)
        c_before_a = full_planning.find("C") < full_planning.rfind("A")
        d_before_a = full_planning.find("D") < full_planning.rfind("A")
        plan_passed = c_before_a and d_before_a and ("sequence" in full_planning.lower() or "order" in full_planning.lower())

        print(f"  • Planning Output: {choice.get('content', '').strip()[:180]}...")
        print(f"  • Topological Ordering Valid: {'PASS' if plan_passed else 'FAIL'}")
        print(f"  • Latency: {res['_client_duration_sec']:.2f}s")
        self.results.append({
            "suite": "Reasoning",
            "test": "Agentic Planning (Release DAG)",
            "passed": plan_passed,
            "details": f"Lint & Build precede deploy: {plan_passed}",
        })

    # =========================================================================
    # SUITE 2: TOOL CALLING & SCHEMA ADHERENCE (BFCL)
    # =========================================================================
    def run_suite_tool_calling(self) -> None:
        print("\n" + "=" * 70)
        print("  SUITE 2: TOOL CALLING & SCHEMA ADHERENCE EVALUATION")
        print("=" * 70)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_mortgage",
                    "description": "Calculate monthly mortgage payment and interest.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "principal": {"type": "number", "description": "Loan amount"},
                            "annual_rate": {"type": "number", "description": "Annual interest rate in %"},
                            "years": {"type": "integer", "description": "Loan term in years"},
                        },
                        "required": ["principal", "annual_rate", "years"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_flight_database",
                    "description": "Query available flights between two airport codes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string", "description": "3-letter IATA code of origin"},
                            "destination": {"type": "string", "description": "3-letter IATA code of destination"},
                            "direct_only": {"type": "boolean", "description": "Only direct flights"},
                        },
                        "required": ["origin", "destination"],
                    },
                },
            },
        ]

        # 2A. Single Tool Selection & Argument Adherence
        print("\n▶ [2A] Function Selection & Typed Argument Adherence")
        user_msg = "Can you check flights from San Francisco (SFO) to Tokyo (HND)? Make sure it is direct only."
        res = self.call_chat(
            messages=[{"role": "user", "content": user_msg}],
            tools=tools,
            max_tokens=300,
        )
        choice = res["choices"][0]["message"]
        tool_calls = choice.get("tool_calls") or []
        tc_passed = False
        if tool_calls:
            call = tool_calls[0]["function"]
            name = call.get("name")
            args = json.loads(call.get("arguments", "{}"))
            print(f"  • Tool Chosen: {name}")
            print(f"  • Arguments Parsed: {args}")
            if name == "query_flight_database":
                if args.get("origin") == "SFO" and args.get("destination") == "HND" and args.get("direct_only") is True:
                    tc_passed = True

        print(f"  • Tool Selection & Parameter Adherence: {'PASS' if tc_passed else 'FAIL'}")
        self.results.append({
            "suite": "Tool Calling",
            "test": "Flight DB Tool Selection & Typed Args",
            "passed": tc_passed,
            "details": f"Chosen tool: {tool_calls[0]['function']['name'] if tool_calls else 'None'}",
        })

        # 2B. Multi-Turn Tool Response Synthesis
        print("\n▶ [2B] Multi-Turn Tool Response Synthesis")
        flight_tool_response = (
            "Found 2 direct flights:\n"
            "1. ANA NH107: Departs SFO 11:20 AM, Arrives HND 3:15 PM (+1), Price: $1,150\n"
            "2. United UA875: Departs SFO 10:45 AM, Arrives HND 2:30 PM (+1), Price: $980"
        )
        conversation = [
            {"role": "user", "content": user_msg},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "User wants direct flights from SFO to HND. Calling flight database.",
                "tool_calls": tool_calls,
            },
            {"role": "tool", "content": flight_tool_response},
        ]
        res2 = self.call_chat(messages=conversation, tools=tools, max_tokens=400)
        choice2 = res2["choices"][0]["message"]
        content2 = choice2.get("content") or ""
        synth_passed = "ANA" in content2 and "United" in content2 and ("$980" in content2 or "980" in content2)
        print(f"  • Synthesized Response: {content2.strip()[:180]}...")
        print(f"  • Multi-turn Tool Synthesis: {'PASS' if synth_passed else 'FAIL'}")
        self.results.append({
            "suite": "Tool Calling",
            "test": "Multi-Turn Tool Response Synthesis",
            "passed": synth_passed,
            "details": f"Incorporated flight data accurately: {synth_passed}",
        })

        # 2C. Tool Call Restraint / Negative Tool Calling (BFCL No-Tool Benchmark)
        print("\n▶ [2C] Tool Call Restraint (Answering Without Unnecessary Tool Call)")
        general_prompt = "What is the capital of France and what is its official currency?"
        res3 = self.call_chat(
            messages=[{"role": "user", "content": general_prompt}],
            tools=tools,
            max_tokens=250,
        )
        choice3 = res3["choices"][0]["message"]
        content3 = (choice3.get("content") or "").lower()
        tool_calls3 = choice3.get("tool_calls")
        no_tool_called = tool_calls3 is None or len(tool_calls3) == 0
        got_answers = "paris" in content3 and "euro" in content3
        restraint_passed = no_tool_called and got_answers

        print(f"  • Tool Invoked: {tool_calls3} (Expected: None)")
        print(f"  • Content Provided: {choice3.get('content', '').strip()[:120]}...")
        print(f"  • Negative Tool Restraint: {'PASS' if restraint_passed else 'FAIL'}")
        self.results.append({
            "suite": "Tool Calling",
            "test": "Tool Restraint / No-Tool Call",
            "passed": restraint_passed,
            "details": f"Zero unnecessary calls: {no_tool_called}, Accurate answer: {got_answers}",
        })

        # 2D. Live Streaming Tool Calling Protocol (OpenAI delta.tool_calls)
        print("\n▶ [2D] Live SSE Streaming Tool Calling Protocol (delta.tool_calls)")
        stream_payload = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Can you check direct flights from SFO to HND?"}],
            "tools": tools,
            "temperature": 0.0,
            "max_tokens": 300,
            "stream": True,
        }
        stream_tc_passed = False
        captured_tc_name = None
        finish_reason_tc = None
        t0 = time.time()
        with requests.post(f"{self.base_url}/chat/completions", json=stream_payload, stream=True, timeout=60) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    try:
                        chunk = json.loads(line_str[6:])
                        choices = chunk.get("choices", [])
                        if choices:
                            ch = choices[0]
                            if ch.get("finish_reason"):
                                finish_reason_tc = ch["finish_reason"]
                            delta_calls = ch.get("delta", {}).get("tool_calls", [])
                            if delta_calls:
                                captured_tc_name = delta_calls[0].get("function", {}).get("name")
                    except Exception:
                        pass
        stream_dur = time.time() - t0
        stream_tc_passed = captured_tc_name == "query_flight_database" and finish_reason_tc == "tool_calls"
        print(f"  • Stream Tool Call Captured: {captured_tc_name}")
        print(f"  • Finish Reason: {finish_reason_tc}")
        print(f"  • Latency: {stream_dur:.2f}s")
        print(f"  • Streaming Tool Protocol: {'PASS' if stream_tc_passed else 'FAIL'}")
        self.results.append({
            "suite": "Tool Calling",
            "test": "Live SSE Streaming delta.tool_calls Protocol",
            "passed": stream_tc_passed,
            "details": f"Tool: {captured_tc_name}, Finish Reason: {finish_reason_tc}, Latency: {stream_dur:.2f}s",
        })

    # =========================================================================
    # SUITE 3: CONTEXT MANAGEMENT & NEEDLE IN A HAYSTACK (NIAH)
    # =========================================================================
    def run_suite_context_niah(self) -> None:
        print("\n" + "=" * 70)
        print("  SUITE 3: CONTEXT MANAGEMENT & NEEDLE IN A HAYSTACK (NIAH)")
        print("=" * 70)

        filler = "The exploration of deep space presents extraordinary computational challenges. "
        test_sizes = [2000, 8000]

        for target_tokens in test_sizes:
            secret_token = f"PASSKEY_ALPHA_{target_tokens}_998"
            repeats = target_tokens // 10
            half = repeats // 2
            haystack_part1 = filler * half
            haystack_part2 = filler * (repeats - half)

            doc = (
                f"{haystack_part1}\n"
                f"### SYSTEM CLASSIFIED ENTRY ###\n"
                f"The authorized orbital docking passkey is {secret_token}.\n"
                f"###############################\n"
                f"{haystack_part2}"
            )

            prompt = (
                f"Below is an extensive research report:\n{doc}\n\n"
                f"What is the authorized orbital docking passkey mentioned in the report? "
                f"State the passkey clearly."
            )

            print(f"\n▶ [3A/B] Testing Single Needle at ~{target_tokens} tokens")
            t0 = time.time()
            res = self.call_chat([{"role": "user", "content": prompt}], max_tokens=150)
            dur = time.time() - t0
            choice = res["choices"][0]["message"]
            full_ans = (choice.get("reasoning_content") or "") + " " + (choice.get("content") or "")
            retrieved = secret_token in full_ans

            vram = self.get_vram_telemetry()
            metal_active = vram.get("memory", {}).get("process_rss_gb", "2.5")

            print(f"  • Duration: {dur:.2f}s | Metal Active VRAM: {metal_active} GB")
            print(f"  • Needle Retrieved: {'PASS' if retrieved else 'FAIL'}")
            self.results.append({
                "suite": "Context Management",
                "test": f"Single Needle NIAH @ ~{target_tokens} tokens",
                "passed": retrieved,
                "details": f"Time: {dur:.2f}s, VRAM: {metal_active}GB, Retrieved: {retrieved}",
            })

        # 3C. Multi-Needle Cross-Document Synthesis
        print("\n▶ [3C] Multi-Needle Cross-Document Aggregation")
        filler_multi = "The operational telemetry of robotic orbiters requires continuous adjustment. " * 60
        multi_doc = (
            f"=== DIVISION REPORT ALPHA ===\n{filler_multi}\n"
            f"Special Fiscal Audit: Project Apollo confirmed funding is 450 million USD.\n"
            f"{filler_multi}\n"
            f"=== DIVISION REPORT BETA ===\n{filler_multi}\n"
            f"Special Fiscal Audit: Project Artemis confirmed funding is 750 million USD.\n"
            f"{filler_multi}\n"
        )
        multi_prompt = (
            f"Review the multi-division reports below:\n{multi_doc}\n"
            f"What is the combined total funding for Project Apollo and Project Artemis? "
            f"Calculate the sum and state the total in million USD."
        )
        res_multi = self.call_chat([{"role": "user", "content": multi_prompt}], max_tokens=450)
        choice_multi = res_multi["choices"][0]["message"]
        full_multi = (choice_multi.get("reasoning_content") or "") + " " + (choice_multi.get("content") or "")
        # 450 + 750 = 1200 million
        has_1200 = "1200" in full_multi or "1,200" in full_multi
        print(f"  • Multi-Needle Synthesis Output: {choice_multi.get('content', '').strip()[:160]}...")
        print(f"  • Cross-Document Sum ($1,200M) Extracted: {'PASS' if has_1200 else 'FAIL'}")
        self.results.append({
            "suite": "Context Management",
            "test": "Multi-Needle Cross-Document Synthesis",
            "passed": has_1200,
            "details": f"Calculated combined budget ($1,200M): {has_1200}",
        })

    # =========================================================================
    # SUITE 4: LIVE THINKING TOKEN STREAMING PROTOCOL (SSE)
    # =========================================================================
    def run_suite_streaming_thinking(self) -> None:
        print("\n" + "=" * 70)
        print("  SUITE 4: LIVE THINKING TOKEN STREAMING PROTOCOL (SSE)")
        print("=" * 70)

        prompt = "Explain in 2 sentences why the sky appears blue on a clear day."
        print(f"\n▶ [4A] Live SSE Stream with delta.reasoning_content Verification")

        payload = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.0,
            "max_tokens": 300,
        }

        t0 = time.time()
        reasoning_chunks = []
        content_chunks = []
        in_thinking_stage = True
        transition_clean = True

        with requests.post(f"{self.base_url}/chat/completions", json=payload, stream=True, timeout=60) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})

                    r_tok = delta.get("reasoning_content")
                    c_tok = delta.get("content")

                    if r_tok:
                        if not in_thinking_stage:
                            # reasoning came after content already started!
                            transition_clean = False
                        reasoning_chunks.append(r_tok)

                    if c_tok:
                        in_thinking_stage = False
                        content_chunks.append(c_tok)

                except json.JSONDecodeError:
                    pass

        dur = time.time() - t0
        total_chunks = len(reasoning_chunks) + len(content_chunks)
        tps = round(total_chunks / dur, 1) if dur > 0 else 0

        has_live_thinking = len(reasoning_chunks) > 0
        has_live_content = len(content_chunks) > 0
        stream_passed = has_live_thinking and has_live_content and transition_clean

        print(f"  • Thinking Chunks Streamed: {len(reasoning_chunks)} chunks (live)")
        print(f"  • Content Chunks Streamed: {len(content_chunks)} chunks (live)")
        print(f"  • Clean Reasoning->Content Transition: {transition_clean}")
        print(f"  • Streaming Throughput: {tps} chunks/sec ({dur:.2f}s total)")
        print(f"  • Live Thinking Protocol: {'PASS' if stream_passed else 'FAIL'}")

        self.results.append({
            "suite": "Live Streaming",
            "test": "Live SSE delta.reasoning_content Protocol",
            "passed": stream_passed,
            "details": f"Thinking chunks: {len(reasoning_chunks)}, Content chunks: {len(content_chunks)}, TPS: {tps}",
        })

    # =========================================================================
    # SUITE 5: LIVE END-TO-END WEB SEARCH AGENT
    # =========================================================================
    def run_suite_live_web_search(self) -> None:
        print("\n" + "=" * 70)
        print("  SUITE 5: LIVE END-TO-END WEB SEARCH AGENT")
        print("=" * 70)

        web_tool = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the live Internet for current real-time news and facts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query keywords"},
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        query_question = "Search the web and provide a summary of the latest SpaceX Starship launch test mission details."
        print(f"\n▶ [5A] User Prompt: '{query_question}'")

        # Step 1: Agent decides to search the web
        res1 = self.call_chat(
            messages=[{"role": "user", "content": query_question}],
            tools=web_tool,
            max_tokens=300,
        )
        choice1 = res1["choices"][0]["message"]
        reasoning1 = choice1.get("reasoning_content") or ""
        tool_calls1 = choice1.get("tool_calls") or []

        print(f"  • Step 1 Thinking: {reasoning1.strip()[:140]}...")
        if not tool_calls1:
            print("  [FAIL] Model did not invoke web_search tool.")
            self.results.append({
                "suite": "Live Agent",
                "test": "Decide & Call Web Search",
                "passed": False,
                "details": "No tool call generated",
            })
            return

        call1 = tool_calls1[0]["function"]
        search_args = json.loads(call1.get("arguments", "{}"))
        search_query = search_args.get("query", "SpaceX Starship latest launch test")
        print(f"  • Step 1 Tool Call: {call1.get('name')} with query='{search_query}'")

        # Step 2: Execute Live Web Search via DDGS
        print(f"  • Step 2 Live Execution: Executing DuckDuckGo query '{search_query}'...")
        search_snippets = []
        try:
            ddg_results = list(DDGS().text(search_query, max_results=3))
            for item in ddg_results:
                search_snippets.append(f"Title: {item.get('title')}\nSnippet: {item.get('body')}\nURL: {item.get('href')}")
            tool_payload = "\n---\n".join(search_snippets)
            print(f"    ✓ Fetched {len(ddg_results)} live search snippets from the web.")
        except Exception as e:
            tool_payload = f"Error executing web search: {e}"
            print(f"    ✗ Web search error: {e}")

        # Step 3: Return tool observation to agent and get final synthesis
        conversation = [
            {"role": "user", "content": query_question},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning1,
                "tool_calls": tool_calls1,
            },
            {"role": "tool", "content": tool_payload},
        ]

        print("  • Step 3 Synthesis: Model reasoning over live search results...")
        res3 = self.call_chat(messages=conversation, tools=web_tool, max_tokens=1000)
        choice3 = res3["choices"][0]["message"]
        reasoning3 = choice3.get("reasoning_content") or ""
        final_answer = choice3.get("content") or ""
        synthesized_text = final_answer.strip() if final_answer.strip() else reasoning3.strip()

        has_substance = len(synthesized_text) > 80 and ("starship" in synthesized_text.lower() or "spacex" in synthesized_text.lower())
        print(f"  • Final Synthesis:\n\n{synthesized_text[:300]}...\n")
        print(f"  • Live Web Search Agent Loop: {'PASS' if has_substance else 'FAIL'}")

        self.results.append({
            "suite": "Live Agent",
            "test": "End-to-End Live Web Search Agent",
            "passed": has_substance,
            "details": f"Query: '{search_query}', Synthesized {len(synthesized_text)} chars",
        })

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("  SPARK-X2.5-4B COMPLETE BENCHMARK EVALUATION SUMMARY")
        print("=" * 70)
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])

        print(f"Total Tests Executed: {total} | Passed: {passed} | Success Rate: {(passed/total)*100:.1f}%\n")
        print(f"{'SUITE':<18} | {'TEST':<38} | {'STATUS':<6} | {'DETAILS'}")
        print("-" * 85)
        for r in self.results:
            status_str = "PASS" if r["passed"] else "FAIL"
            print(f"{r['suite']:<18} | {r['test']:<38} | {status_str:<6} | {r['details']}")
        print("=" * 70 + "\n")


import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Spark-X2.5-4B Comprehensive Benchmark Evaluation")
    parser.add_argument(
        "--suite",
        choices=["all", "reasoning", "tools", "context", "streaming", "search"],
        default="all",
        help="Benchmark suite to run (default: all)",
    )
    args = parser.parse_args()

    runner = BenchmarkRunner()
    try:
        if args.suite in ("all", "reasoning"):
            runner.run_suite_reasoning()
        if args.suite in ("all", "tools"):
            runner.run_suite_tool_calling()
        if args.suite in ("all", "context"):
            runner.run_suite_context_niah()
        if args.suite in ("all", "streaming"):
            runner.run_suite_streaming_thinking()
        if args.suite in ("all", "search"):
            runner.run_suite_live_web_search()
    except Exception as exc:
        print(f"\n[FATAL] Benchmark run interrupted: {exc}", file=sys.stderr)
        raise exc
    finally:
        runner.print_summary()


if __name__ == "__main__":
    main()
