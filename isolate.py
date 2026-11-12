"""
Isolate a model bug: same question, change ONE thing at a time
==============================================================

Nova crashes with 'Model produced invalid sequence as part of ToolUse' on
"How much have I spent on AWS so far this month?". We already ruled out the date
(spelling out 2026-11 still crashes). This script keeps the question fixed and varies
the tool menu, the prompt and the decoding settings, one at a time.

It only makes the FIRST model call (where the crash happens) and never runs the tools,
so it costs no Cost Explorer requests, only a few cents of tokens.

  python isolate.py
  python isolate.py "any other question that crashes"
"""

import sys

from botocore.exceptions import ClientError

import agent_loop as al
import cost_tools as ct

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "How much have I spent on AWS so far this month?"
SPECS = {t["toolSpec"]["name"]: t for t in ct.COST_TOOL_CONFIG["tools"]}
ALL = list(SPECS)

# (label, tool names, extra toolConfig, use system prompt?, use model extras (topK)?)
VARIANTS = [
    ("Baseline: everything as in the agent", ALL, None, True, True),
    ("Only get_monthly_spend", ["get_monthly_spend"], None, True, True),
    ("Without get_daily_spend", ["get_monthly_spend", "get_spend_by_service"], None, True, True),
    ("Without get_spend_by_service", ["get_monthly_spend", "get_daily_spend"], None, True, True),
    ("Forced to call a tool (toolChoice any)", ALL, {"toolChoice": {"any": {}}}, True, True),
    ("No system prompt", ALL, None, False, True),
    ("No topK (temperature 0 only)", ALL, None, True, False),
]


def attempt(client, tools, extra, use_system, use_extras):
    tool_config = {"tools": [SPECS[n] for n in tools]}
    if extra:
        tool_config.update(extra)
    kwargs = dict(
        modelId=al.MODEL_ID,
        messages=[{"role": "user", "content": [{"text": QUESTION}]}],
        toolConfig=tool_config,
        inferenceConfig=al.INFERENCE_CONFIG,
    )
    if use_system:
        kwargs["system"] = ct.cost_system_prompt()
    if use_extras:
        kwargs.update(al.model_extras())
    try:
        r = client.converse(**kwargs)  # no retries on purpose: we want to see the raw behaviour
    except ClientError as e:
        return "CRASH", e.response["Error"]["Code"]
    calls = [f"{b['toolUse']['name']}({b['toolUse']['input']})"
             for b in r["output"]["message"]["content"] if "toolUse" in b]
    return "OK", "; ".join(calls) or f"answered without a tool (stopReason={r['stopReason']})"


def main():
    client = al.get_client()
    print(f"Model:    {al.MODEL_ID}\nQuestion: {QUESTION}\n")
    for label, tools, extra, use_system, use_extras in VARIANTS:
        status, detail = attempt(client, tools, extra, use_system, use_extras)
        print(f"{'✅' if status == 'OK' else '💥'} {label:<40} {detail}")
    print("\nRead it like a doctor: the variant that turns 💥 into ✅ points at the cause.")


if __name__ == "__main__":
    main()
