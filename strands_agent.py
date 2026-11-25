"""
Step 7: the same cost agent, rebuilt in Strands
===============================================

agent_loop.py + cost_tools.py is the agent built by hand. This file is the same agent
built with Strands Agents (AWS's open-source agent SDK), on purpose reusing the same
tool functions, prompt and model, so the ONLY difference is who runs the loop.

  python strands_agent.py "How much have I spent on AWS so far this month?"
  python evals.py --engine strands        (same eval suite, Strands engine)

pip install strands-agents==1.64.0   (pinned: frameworks change fast, pin and upgrade on purpose)

What Strands gives you for free (things you hand-built in Steps 1-6):
  - the loop itself (call model -> run tools -> send results -> repeat)
  - tool specs generated from type hints + docstrings (no hand-written JSON schema)
  - tool exceptions turned into error results the model can read (your safe_run_tool)
  - retries with backoff on throttling / transient model errors (your RETRYABLE logic)
  - conversation memory across calls on the same Agent object, token metrics, tracing

What it does NOT give you (you still own these):
  - a turn limit: there is no max_turns setting, so we add one with a hook below
  - the tools themselves: validation, caching, exact names, doing the arithmetic in code
  - the prompt, the calendar, the model choice, and the evals
"""

import json
import os
import sys
import time

from strands import Agent, tool
from strands.hooks import (AfterModelCallEvent, AfterToolCallEvent, BeforeModelCallEvent,
                           BeforeToolCallEvent, HookProvider, HookRegistry)
from strands.models import BedrockModel

import cost_tools as ct
import telemetry
from agent_loop import INFERENCE_CONFIG, strip_thinking

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID")
REGION = os.environ.get("AWS_REGION", "us-east-1")
MAX_TURNS = 6


# ---------------------------------------------------------------------------
# Tools: the SAME functions from cost_tools.py, wrapped with @tool.
# Strands builds each tool spec from the type hints and the docstring:
# the summary becomes the description, the "Args:" lines become parameter descriptions.
# Compare with the ~70 lines of hand-written JSON in COST_TOOL_CONFIG.
# ---------------------------------------------------------------------------

@tool
def get_monthly_spend(month: str) -> dict:
    """Net AWS bill for the WHOLE ACCOUNT (all services) for one calendar month, split by charge type (Usage, Credit, Tax, Refund...). Use it for 'how much did we spend' questions. Never quote it as one service's spend. partial_month=true means the month isn't over yet.

    Args:
        month: Calendar month in YYYY-MM format, e.g. 2026-10
    """
    return ct.get_monthly_spend(month)


@tool
def get_spend_by_service(month: str, top_n: int = 8) -> dict:
    """Usage cost for one calendar month broken down by AWS service, largest first (top_n services plus an 'All other services' bucket). Use it to find WHICH services drive the bill or a change in it.

    Args:
        month: Calendar month in YYYY-MM format, e.g. 2026-10
        top_n: How many services to list (1-20, default 8)
    """
    return ct.get_spend_by_service(month, top_n)


@tool
def get_daily_spend(start_date: str, end_date: str, service: str = None) -> dict:
    """Usage cost per day for up to 31 days, optionally for ONE service. Use it to find WHEN a change started. Prefer exact service names from get_spend_by_service (e.g. 'Amazon Bedrock'). A near-miss is resolved and reported in 'interpreted_as'; an unknown name returns an error listing the services that had usage.

    Args:
        start_date: First day, YYYY-MM-DD
        end_date: Last day (inclusive), YYYY-MM-DD
        service: Optional exact service name
    """
    return ct.get_daily_spend(start_date, end_date, service)


@tool
def compare_periods(start_month: str, end_month: str, top_n: int = 5) -> dict:
    """Compares the AWS bill between two calendar months and everything in between, with all arithmetic done for you: bill change, usage change, what made up the difference (usage, tax, credits, one-off charges) and the services that moved most. Use it for ANY question about how spend changed over time. Quote its numbers; do not recalculate them.

    Args:
        start_month: First month of the period, YYYY-MM
        end_month: Last month of the period, YYYY-MM
        top_n: How many top-moving services to list (1-10, default 5)
    """
    return ct.compare_periods(start_month, end_month, top_n)


TOOLS = [get_monthly_spend, get_spend_by_service, get_daily_spend, compare_periods]


# ---------------------------------------------------------------------------
# Hooks: Strands' extension points. Two things we need that it doesn't do by itself.
# ---------------------------------------------------------------------------

class TurnLimit(HookProvider):
    """Our max_turns guardrail. Strands has no built-in turn cap, so before each model
    call we count, and past the limit we cancel the call. The agent stops cleanly."""

    def __init__(self, max_turns: int):
        self.max_turns, self.turns = max_turns, 0

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeModelCallEvent, self.before_model_call)

    def before_model_call(self, event: BeforeModelCallEvent) -> None:
        self.turns += 1
        if self.turns > self.max_turns:
            event.cancel = (f"[Agent stopped: hit max_turns={self.max_turns} without finishing. "
                            "Check the tool descriptions or the prompt.]")


def parse_tool_result(tool_result) -> dict:
    """Strands hands back a ToolResult; turn it back into the dict our tool returned."""
    text = "".join(c.get("text", "") for c in tool_result.get("content", []))
    if tool_result.get("status") == "error":
        return {"error": text}  # Strands turned the exception into an error result
    try:
        return json.loads(text)  # Strands JSON-encodes dict returns as text
    except ValueError:
        return {"text": text}


class TraceRecorder(HookProvider):
    """Records every tool call in the same shape agent_loop.run_agent uses, so evals.py
    can grade the Strands agent with exactly the same checks."""

    def __init__(self, trace: list):
        self.trace = trace

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AfterToolCallEvent, self.after_tool_call)

    def after_tool_call(self, event: AfterToolCallEvent) -> None:
        result = parse_tool_result(event.result)
        name = event.tool_use["name"]
        self.trace.append({"type": "tool", "name": name, "input": event.tool_use["input"], "result": result})
        print(f"    tool: {name}({event.tool_use['input']}) -> {result}")


class TelemetryHook(HookProvider):
    """Writes the same structured events as the hand-built loop (see telemetry.py),
    from Strands' lifecycle hooks. This is how you observe a loop you don't own."""

    def __init__(self, log: telemetry.RunLog):
        self.log, self.turn, self._t_model, self._ce_before = log, 0, None, {}

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeModelCallEvent, self.before_model)
        registry.add_callback(AfterModelCallEvent, self.after_model)
        registry.add_callback(BeforeToolCallEvent, self.before_tool)
        registry.add_callback(AfterToolCallEvent, self.after_tool)

    def before_model(self, event: BeforeModelCallEvent) -> None:
        self.turn += 1
        self._t_model = time.perf_counter()

    def after_model(self, event: AfterModelCallEvent) -> None:
        latency = (time.perf_counter() - self._t_model) * 1000 if self._t_model else 0
        stop = event.stop_response.stop_reason if event.stop_response else f"exception: {event.exception}"
        # Per-call token counts aren't exposed on this event; run totals are logged at run_end.
        self.log.model_call(self.turn, latency, stop)

    def before_tool(self, event: BeforeToolCallEvent) -> None:
        self._ce_before[event.tool_use["toolUseId"]] = telemetry.ce_calls()

    def after_tool(self, event: AfterToolCallEvent) -> None:
        tool_use = event.tool_use
        # If Strands runs several tools at once, these per-tool counts can overlap;
        # the run total in run_end is always exact.
        ce_requests = telemetry.ce_calls() - self._ce_before.pop(tool_use["toolUseId"], telemetry.ce_calls())
        latency = event.duration * 1000 if event.duration is not None else None
        self.log.tool_call(self.turn, tool_use["name"], tool_use["input"], parse_tool_result(event.result),
                           latency, ce_requests)


# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------

def build_model() -> BedrockModel:
    extra = {}
    if "nova" in (MODEL_ID or "").lower():
        extra["additional_request_fields"] = {"inferenceConfig": {"topK": 1}}  # Nova greedy decoding, as in Round 2
    return BedrockModel(
        model_id=MODEL_ID,
        region_name=REGION,
        temperature=INFERENCE_CONFIG["temperature"],
        max_tokens=INFERENCE_CONFIG["maxTokens"],
        **extra,
    )


def build_agent(trace: list, max_turns: int = MAX_TURNS, log: telemetry.RunLog = None) -> Agent:
    """A fresh Agent per question. (An Agent object keeps its conversation between calls,
    which is what you want in a chat app and NOT what you want in an eval.)"""
    return Agent(
        model=build_model(),
        tools=TOOLS,
        system_prompt=ct.cost_system_prompt()[0]["text"],
        hooks=[TurnLimit(max_turns), TraceRecorder(trace)] + ([TelemetryHook(log)] if log else []),
        callback_handler=None,  # silence token-by-token streaming; our hook prints the tool calls
    )


def run_strands(question: str, trace: list = None, max_turns: int = MAX_TURNS) -> str:
    trace = trace if trace is not None else []
    log = telemetry.RunLog(question, engine="strands", model=MODEL_ID)
    agent = build_agent(trace, max_turns, log)
    try:
        result = agent(question)  # <- the whole loop from Steps 1-4, in one line
    except BaseException as e:
        log.end(status="error", error=" ".join(f"{type(e).__name__}: {e}".split()))
        raise

    usage = result.metrics.accumulated_usage
    turns = result.metrics.cycle_count
    print(f"\nTotals: {turns} turns, {usage.get('inputTokens')} input tokens, "
          f"{usage.get('outputTokens')} output tokens")
    trace.append({"type": "usage", "turns": turns, "input_tokens": usage.get("inputTokens"),
                  "output_tokens": usage.get("outputTokens"), "stop": result.stop_reason})

    text = str(result)
    if result.stop_reason == "max_tokens":
        text += "\n\n[WARNING: answer was cut off by maxTokens]"
    answer = strip_thinking(text)
    log.end(answer=answer, turns=turns, input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            status="ok" if result.stop_reason == "end_turn" else result.stop_reason)
    return answer


if __name__ == "__main__":
    if not MODEL_ID:
        sys.exit("Set BEDROCK_MODEL_ID first.")
    question = sys.argv[1] if len(sys.argv) > 1 else (
        "How has my AWS spend changed over the last three full months, and what's driving it?"
    )
    print("QUESTION ->", question)
    answer = run_strands(question)
    print("\nFINAL ANSWER ->\n" + answer)
    ct.print_ce_bill()
