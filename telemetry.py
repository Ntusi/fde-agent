"""
Structured logging for the agent: one JSON line per event
=========================================================

print() output disappears when the terminal closes, can't be searched, and can't tell
two runs apart. This writes every event to logs/agent.jsonl as one JSON object per line,
all sharing a run_id, so any single run can be replayed and any question ("which runs
cost the most?", "how slow are tool calls?") can be answered later.

Events per run:  run_start -> model_call (per turn) -> tool_call (per tool) -> run_end

What is logged, and what is NOT (decide this with the customer, it's a data decision):
  - always:   run_id, timings, tokens, tool names and inputs, status, errors, cost estimate
  - always:   the question and the answer (needed to debug and to mine new eval cases)
  - NOT by default: full tool results (billing data can be sensitive). Set
                AGENT_LOG_FULL=1 to include them while debugging.
  - never:    credentials or keys

Same events, same format, whichever engine runs the loop (hand-built or Strands).
In production the same lines go to stdout and CloudWatch Logs collects them.

  AGENT_LOG_PATH   where to write   (default: logs/agent.jsonl next to this file)
  AGENT_LOG_FULL   1 = include full tool results
"""

import datetime as dt
import json
import os
import sys
import time
import uuid
from pathlib import Path

LOG_PATH = Path(os.environ.get("AGENT_LOG_PATH", Path(__file__).with_name("logs") / "agent.jsonl"))
LOG_FULL_RESULTS = os.environ.get("AGENT_LOG_FULL", "0") == "1"
RUN_TAG = os.environ.get("AGENT_RUN_TAG", "manual")  # evals.py sets "eval"

# USD per 1M tokens (input, output). Prices change: check the model's Pricing tab.
MODEL_PRICES = {
    "us.amazon.nova-2-lite-v1:0": (0.33, 2.75),
    "us.amazon.nova-pro-v1:0": (0.80, 3.20),
}
CE_PRICE_PER_REQUEST = 0.01


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def ce_calls():
    """Paid Cost Explorer requests so far, read from cost_tools if it's loaded."""
    ct = sys.modules.get("cost_tools")
    return getattr(ct, "CE_CALLS", 0) if ct else 0


def estimate_cost(model, input_tokens, output_tokens, ce_requests):
    price_in, price_out = MODEL_PRICES.get(model, (None, None))
    model_cost = None if price_in is None else (input_tokens * price_in + output_tokens * price_out) / 1e6
    tools_cost = ce_requests * CE_PRICE_PER_REQUEST
    return {
        "model_usd": round(model_cost, 6) if model_cost is not None else None,  # None = unknown price
        "tools_usd": round(tools_cost, 4),
        "total_usd": round((model_cost or 0) + tools_cost, 6),
    }


def summarise_result(result):
    """Enough to debug without copying billing data into logs."""
    if not isinstance(result, dict):
        return {"type": type(result).__name__}
    summary = {"keys": sorted(result)[:12], "size_bytes": len(json.dumps(result, default=str))}
    if "error" in result:
        summary["error"] = str(result["error"])[:300]  # errors are always worth keeping
    return summary


class RunLog:
    """One run = one question. Every event it writes carries the same run_id."""

    def __init__(self, question: str, engine: str, model: str):
        self.run_id = uuid.uuid4().hex[:12]
        self.engine, self.model = engine, model
        self._t0 = time.perf_counter()
        self._ce0 = ce_calls()
        self.tool_calls = 0
        self.emit("run_start", question=question, tag=RUN_TAG)

    def emit(self, event: str, **fields):
        line = {"ts": _now(), "run_id": self.run_id, "event": event,
                "engine": self.engine, "model": self.model, **fields}
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a") as f:
                f.write(json.dumps(line, default=str) + "\n")
        except OSError as e:  # logging must never break the agent
            print(f"[telemetry] could not write log: {e}", file=sys.stderr)

    def model_call(self, turn, latency_ms, stop_reason, input_tokens=None, output_tokens=None):
        self.emit("model_call", turn=turn, latency_ms=round(latency_ms), stop_reason=stop_reason,
                  input_tokens=input_tokens, output_tokens=output_tokens)

    def tool_call(self, turn, name, tool_input, result, latency_ms, ce_requests):
        self.tool_calls += 1
        fields = {"turn": turn, "tool": name, "input": tool_input,
                  "status": "error" if isinstance(result, dict) and "error" in result else "ok",
                  "latency_ms": round(latency_ms) if latency_ms is not None else None,
                  "ce_requests": ce_requests, "result": summarise_result(result)}
        if LOG_FULL_RESULTS:
            fields["result_full"] = result
        self.emit("tool_call", **fields)

    def end(self, answer=None, turns=None, input_tokens=0, output_tokens=0, status="ok", error=None):
        ce_requests = ce_calls() - self._ce0
        self.emit("run_end", status=status, error=error, answer=answer, turns=turns,
                  input_tokens=input_tokens, output_tokens=output_tokens, tool_calls=self.tool_calls,
                  ce_requests=ce_requests, latency_ms=round((time.perf_counter() - self._t0) * 1000),
                  est_cost=estimate_cost(self.model, input_tokens or 0, output_tokens or 0, ce_requests))
        print(f"[run {self.run_id}] logged to {LOG_PATH}")
