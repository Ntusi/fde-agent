"""
Raw tool-use loop with the Bedrock Converse API: Steps 1-5
==========================================================

What this file does:
  Step 1: a plain Converse call, so you can study the response shape.
  Step 2: one tool defined, so you can watch the model ASK to use it.
  Step 3: run the requested tools and send the results back for a final answer.
  Step 4: the loop. Keeps calling tools until the model is done (two tools, guardrails).
  Step 5: same loop, REAL Cost Explorer data (tools in cost_tools.py).

Setup:
  pip install boto3
  export AWS_PROFILE=<your-profile>          # creds with bedrock:InvokeModel
  export AWS_REGION=us-east-1                # a region where your Claude model is enabled
  export BEDROCK_MODEL_ID=<model or inference profile ID from the Bedrock console>

Run:
  python agent_loop.py 1
  python agent_loop.py 2
  python agent_loop.py 3
  python agent_loop.py 4
  python agent_loop.py 4 "your own question here"
  python agent_loop.py 5
  python agent_loop.py 5 "your own question here"
"""

import json
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_client():
    if not MODEL_ID:
        sys.exit(
            "Set BEDROCK_MODEL_ID first. Find it in the Bedrock console under "
            "Model catalog (or Cross-region inference for an inference profile ID)."
        )
    # "bedrock-runtime" is the data-plane client (calling models).
    # "bedrock" (no -runtime) is the control plane (listing models, etc.).
    return boto3.client("bedrock-runtime", region_name=REGION)


def show(title, obj):
    print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=2, default=str))


# Errors worth retrying: the request was fine, the service or model hiccuped.
# (Retrying a ValidationException or AccessDenied is pointless: it fails the same way.)
RETRYABLE = {"ThrottlingException", "ModelErrorException", "ServiceUnavailableException",
             "InternalServerException", "ModelNotReadyException"}


def call_converse(client, retries: int = 2, **kwargs):
    """Wraps converse(): retries transient errors, and explains setup errors."""
    for attempt in range(retries + 1):
        try:
            response = client.converse(modelId=MODEL_ID, **kwargs)
            break
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in RETRYABLE and attempt < retries:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s...
                print(f"  {code}: retrying in {wait}s (attempt {attempt + 2} of {retries + 1})")
                time.sleep(wait)
                continue
            hints = {
                "AccessDeniedException": "Model access not enabled in this region, or your IAM role lacks bedrock:InvokeModel.",
                "ValidationException": "Usually a wrong model ID for this region, or a malformed message/tool structure.",
                "ResourceNotFoundException": "Model ID doesn't exist in this region. Check the console.",
                "ThrottlingException": "Still throttled after retries. Wait a minute and run again.",
                "ModelErrorException": "The model kept producing a malformed response (e.g. a broken tool call).",
            }
            sys.exit(f"\n{code}: {e.response['Error']['Message']}\nHint: {hints.get(code, 'See error above.')}")
    response.pop("ResponseMetadata", None)  # HTTP noise; hide it so the useful shape stands out
    return response


# ---------------------------------------------------------------------------
# STEP 1: A plain Converse call
# ---------------------------------------------------------------------------

def step1(client):
    # `messages` IS the conversation. Converse is stateless: the model only knows
    # what you send in this list on every single call.
    # Each message = a role ("user" or "assistant") + a LIST of content blocks.
    # Even plain text is wrapped as a block: {"text": "..."}.
    messages = [
        {
            "role": "user",
            "content": [{"text": "In two sentences, what is an AWS Cost and Usage Report?"}],
        }
    ]

    response = call_converse(
        client,
        system=[{"text": "You are a concise AWS FinOps assistant."}],  # system prompt is separate from messages
        messages=messages,
        inferenceConfig={"maxTokens": 300, "temperature": 0.2},
    )

    show("Full response", response)

    # The three fields you'll use in every agent you ever build:
    print("\nstopReason ->", response["stopReason"])  # WHY the model stopped. Drives the loop later.
    print("usage      ->", response["usage"])          # tokens in/out, which is your cost
    content_blocks = response["output"]["message"]["content"]
    text = "".join(block.get("text", "") for block in content_blocks)
    print("\nModel said ->", text)

    # Try this: drop maxTokens to 20 and rerun. Watch stopReason become "max_tokens".
    # In production, a silently truncated answer is a real bug. Always check it.


# ---------------------------------------------------------------------------
# STEP 2: Define one tool and watch the model request it
# ---------------------------------------------------------------------------

# Fake data for now. In Step 5 this becomes a real Cost Explorer call.
FAKE_SPEND = {"2026-08": 4210.55, "2026-09": 4380.10, "2026-10": 6925.40}


def get_monthly_spend(month: str) -> dict:
    """The actual Python function. The model will NEVER call this itself."""
    if month not in FAKE_SPEND:
        return {"error": f"No data for {month}. Available: {sorted(FAKE_SPEND)}"}
    return {"month": month, "total_usd": FAKE_SPEND[month]}


# Maps tool name -> Python function. The loop in Step 4 will dispatch through this.
TOOLS = {"get_monthly_spend": get_monthly_spend}

# What the MODEL sees. It never sees your code, only this name, description
# and schema. The description is effectively a prompt: write it like one.
TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_monthly_spend",
                "description": (
                    "Returns the total AWS spend in USD for one calendar month. "
                    "Use it whenever the user asks about spend for a specific month. "
                    "Data exists for 2026-08 through 2026-10."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "month": {
                                "type": "string",
                                "description": "Month in YYYY-MM format, e.g. 2026-10",
                            }
                        },
                        "required": ["month"],
                    }
                },
            }
        }
    ]
}


def step2(client):
    messages = [
        {
            "role": "user",
            "content": [{"text": "Did our AWS spend jump in October 2026 compared to September?"}],
        }
    ]

    response = call_converse(
        client,
        system=[{"text": "You are a concise AWS FinOps assistant. Use tools for any spend figures; never guess numbers."}],
        messages=messages,
        toolConfig=TOOL_CONFIG,
        inferenceConfig={"maxTokens": 500, "temperature": 0},
    )

    show("Full response", response)
    print("\nstopReason ->", response["stopReason"])

    if response["stopReason"] != "tool_use":
        print("\nModel answered without a tool. Try making the system prompt or tool description stricter.")
        return

    # The model can return text AND tool requests in the same message,
    # and may ask for SEVERAL tools at once (here, probably Sep and Oct together).
    for block in response["output"]["message"]["content"]:
        if "text" in block:
            print("\nModel's thinking-out-loud ->", block["text"])
        if "toolUse" in block:
            tool_use = block["toolUse"]
            print("\nModel REQUESTS a tool:")
            print("  toolUseId ->", tool_use["toolUseId"], "  (you must echo this back with the result)")
            print("  name      ->", tool_use["name"])
            print("  input     ->", tool_use["input"])
            # Preview what we WOULD send back. We're not sending it yet.
            print("  if we ran it ->", TOOLS[tool_use["name"]](**tool_use["input"]))

    print(
        "\nNotice: the model has NOT seen any spend numbers. It stopped and asked you to run something.\n"
        "Step 3: run the function, send the result back as a toolResult, and let it finish the answer."
    )


# ---------------------------------------------------------------------------
# STEP 3: Run the tools and send the results back (one round trip)
# ---------------------------------------------------------------------------

FINOPS_SYSTEM = [{"text": "You are a concise AWS FinOps assistant. Use tools for any spend figures; never guess numbers."}]


def run_tool(tool_use: dict) -> dict:
    """Look up the requested tool by name and call it with the model's input."""
    fn = TOOLS.get(tool_use["name"])
    if fn is None:
        # Models occasionally invent tool names. Tell it, don't crash.
        return {"error": f"Unknown tool '{tool_use['name']}'. Available: {list(TOOLS)}"}
    return fn(**tool_use["input"])


def step3(client):
    messages = [
        {
            "role": "user",
            "content": [{"text": "Did our AWS spend jump in October 2026 compared to September?"}],
        }
    ]

    # ---- Round 1: the model asks for tools (same as Step 2) ----
    response = call_converse(
        client,
        system=FINOPS_SYSTEM,
        messages=messages,
        toolConfig=TOOL_CONFIG,
        inferenceConfig={"maxTokens": 500, "temperature": 0},
    )
    print("Round 1 stopReason ->", response["stopReason"])
    if response["stopReason"] != "tool_use":
        print("Model answered without a tool this time. Run it again.")
        return

    # GOTCHA #1: put the assistant's message back into history EXACTLY as received,
    # toolUse blocks included. Rebuild or edit it and Bedrock rejects the next call.
    assistant_message = response["output"]["message"]
    messages.append(assistant_message)

    # Run EVERY tool it asked for, and collect one toolResult per toolUse.
    tool_results = []
    for block in assistant_message["content"]:
        if "toolUse" in block:
            tool_use = block["toolUse"]
            result = run_tool(tool_use)
            print(f"  ran {tool_use['name']}({tool_use['input']}) -> {result}")
            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],  # GOTCHA #2: must match the request
                        "content": [{"json": result}],      # a {"text": "..."} block works too
                    }
                }
            )

    # GOTCHA #3: tool results go back in a USER message, ALL in ONE message.
    # (Two separate user messages in a row = ValidationException.)
    messages.append({"role": "user", "content": tool_results})

    show("The full conversation we send back (this IS the agent's memory)", messages)

    # ---- Round 2: the model now has the numbers ----
    # GOTCHA #4: pass toolConfig again. History contains tool blocks, so Bedrock requires it.
    response = call_converse(
        client,
        system=FINOPS_SYSTEM,
        messages=messages,
        toolConfig=TOOL_CONFIG,
        inferenceConfig={"maxTokens": 500, "temperature": 0},
    )
    print("\nRound 2 stopReason ->", response["stopReason"])
    print("usage ->", response["usage"], " (notice inputTokens grew: we resent the whole history)")

    text = "".join(b.get("text", "") for b in response["output"]["message"]["content"])
    print("\nFINAL ANSWER ->", text)

    if response["stopReason"] == "tool_use":
        print(
            "\nIt asked for MORE tools. We only coded one round trip, so we stop here.\n"
            "Handling 'keep going until it's done' is Step 4: the loop."
        )


# ---------------------------------------------------------------------------
# STEP 4: The loop. This is the agent.
# ---------------------------------------------------------------------------

# A second tool, so the agent has to work in stages:
# totals first -> spot the jump -> drill into services -> explain.
FAKE_SPEND_BY_SERVICE = {
    "2026-08": {"Amazon EC2": 2050.00, "Amazon RDS": 975.55, "Amazon S3": 440.00, "Amazon Bedrock": 220.00, "Other": 525.00},
    "2026-09": {"Amazon EC2": 2100.00, "Amazon RDS": 980.10, "Amazon S3": 450.00, "Amazon Bedrock": 300.00, "Other": 550.00},
    "2026-10": {"Amazon EC2": 2180.00, "Amazon RDS": 990.40, "Amazon S3": 465.00, "Amazon Bedrock": 2690.00, "Other": 600.00},
}


def get_spend_by_service(month: str) -> dict:
    if month not in FAKE_SPEND_BY_SERVICE:
        return {"error": f"No data for {month}. Available: {sorted(FAKE_SPEND_BY_SERVICE)}"}
    return {"month": month, "by_service_usd": FAKE_SPEND_BY_SERVICE[month]}


AGENT_TOOLS = {
    "get_monthly_spend": get_monthly_spend,
    "get_spend_by_service": get_spend_by_service,
}

AGENT_TOOL_CONFIG = {
    "tools": [
        TOOL_CONFIG["tools"][0],  # reuse get_monthly_spend from Step 2
        {
            "toolSpec": {
                "name": "get_spend_by_service",
                "description": (
                    "Returns AWS spend in USD for one calendar month, broken down by service. "
                    "Use it to find WHICH services drove a change in total spend. "
                    "Data exists for 2026-08 through 2026-10."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "month": {"type": "string", "description": "Month in YYYY-MM format, e.g. 2026-10"}
                        },
                        "required": ["month"],
                    }
                },
            }
        },
    ]
}

# The prompt tweak from Monday, folded in. Behaviour lives in the prompt.
AGENT_SYSTEM = [{"text": (
    "You are a concise AWS FinOps assistant. Use tools for any spend figures; never guess numbers. "
    "When comparing periods, give the absolute change AND the percentage change, and say whether it "
    "looks significant enough to investigate. If total spend changed significantly, find which "
    "services drove it before answering."
)}]


def safe_run_tool(tool_use: dict, tools: dict) -> dict:
    """Run one tool. Never let a tool crash the loop: turn failures into results the model can read."""
    fn = tools.get(tool_use["name"])
    if fn is None:
        return {"error": f"Unknown tool '{tool_use['name']}'. Available: {list(tools)}"}
    try:
        return fn(**tool_use["input"])
    except TypeError as e:
        # The model passed the wrong arguments (missing, extra or misnamed).
        return {"error": f"Bad arguments for {tool_use['name']}: {e}"}
    except Exception as e:
        # A real tool (e.g. a Cost Explorer call) can fail. Report it so the model can adapt.
        return {"error": f"{type(e).__name__}: {e}"}


def strip_thinking(text: str) -> str:
    """Nova wraps reasoning in <thinking> tags. Users shouldn't see it. A guarantee, so it's code, not prompt."""
    import re
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)  # drop reasoning entirely
    text = re.sub(r"</?answer>", "", text)  # keep the answer, drop just the tags
    return text.strip()


# Generation settings for the agent loop, in ONE place (evals fingerprint these too).
INFERENCE_CONFIG = {"maxTokens": 2000, "temperature": 0}


def model_extras() -> dict:
    """Model-specific settings that Converse doesn't standardise.

    Nova kept failing with 'Model produced invalid sequence as part of ToolUse'. AWS's Nova
    tool-use troubleshooting guide says: use greedy decoding (temperature 0 AND topK 1) and
    leave enough maxTokens. topK goes through additionalModelRequestFields, in Nova's own
    format, so it must only be sent to Nova (other providers expect a different shape).
    """
    if "nova" in (MODEL_ID or "").lower():
        return {"additionalModelRequestFields": {"inferenceConfig": {"topK": 1}}}
    return {}


def run_agent(client, question: str, max_turns: int = 6,
              tools: dict = None, tool_config: dict = None, system: list = None,
              trace: list = None) -> str:
    # Step 5 refactor: the loop no longer cares WHICH tools it runs. Pass in any
    # set (fake, real, a customer's) and the same engine drives them. Defaults = Step 4.
    tools = tools or AGENT_TOOLS
    tool_config = tool_config or AGENT_TOOL_CONFIG
    system = system or AGENT_SYSTEM
    # Step 6 (evals): pass a list as `trace` and the loop records what it did, so tests
    # can check HOW the agent got its answer (which tools, which inputs), not just the answer.
    trace = trace if trace is not None else []

    messages = [{"role": "user", "content": [{"text": question}]}]
    total_in = total_out = 0

    for turn in range(1, max_turns + 1):  # GUARD: never loop forever
        response = call_converse(
            client,
            system=system,
            messages=messages,
            toolConfig=tool_config,
            inferenceConfig=INFERENCE_CONFIG,
            **model_extras(),
        )
        stop = response["stopReason"]
        usage = response["usage"]
        total_in += usage["inputTokens"]
        total_out += usage["outputTokens"]
        print(f"\n--- Turn {turn}: stopReason={stop}  in={usage['inputTokens']} out={usage['outputTokens']}")

        assistant_message = response["output"]["message"]
        messages.append(assistant_message)  # exactly as received (Gotcha #1)

        if stop == "tool_use":
            tool_results = []
            for block in assistant_message["content"]:
                if "toolUse" in block:
                    tool_use = block["toolUse"]
                    result = safe_run_tool(tool_use, tools)
                    trace.append({"type": "tool", "name": tool_use["name"], "input": tool_use["input"], "result": result})
                    print(f"    tool: {tool_use['name']}({tool_use['input']}) -> {result}")
                    tool_results.append(
                        {"toolResult": {"toolUseId": tool_use["toolUseId"], "content": [{"json": result}]}}
                    )
            messages.append({"role": "user", "content": tool_results})  # all in ONE message (Gotcha #3)
            continue  # back to the top: call the model again with the new results

        # Not tool_use -> the model is done (or something went wrong). Either way, we stop.
        text = "".join(b.get("text", "") for b in assistant_message["content"])
        if stop == "max_tokens":
            text += "\n\n[WARNING: answer was cut off by maxTokens]"
        elif stop != "end_turn":
            text += f"\n\n[WARNING: unexpected stopReason '{stop}']"
        print(f"\nTotals: {turn} turns, {total_in} input tokens, {total_out} output tokens")
        trace.append({"type": "usage", "turns": turn, "input_tokens": total_in, "output_tokens": total_out, "stop": stop})
        return strip_thinking(text)

    print(f"\nTotals: hit max_turns={max_turns}, {total_in} input tokens, {total_out} output tokens")
    trace.append({"type": "usage", "turns": max_turns, "input_tokens": total_in, "output_tokens": total_out, "stop": "max_turns"})
    return "[Agent stopped: hit max_turns without finishing. Check the tool descriptions or the prompt.]"


def step4(client):
    question = sys.argv[2] if len(sys.argv) > 2 else (
        "Did our AWS spend jump in October 2026 compared to September? If so, what drove it?"
    )
    print("QUESTION ->", question)
    answer = run_agent(client, question)
    print("\nFINAL ANSWER ->\n" + answer)


# ---------------------------------------------------------------------------
# STEP 5: Same loop, REAL data. Tools live in cost_tools.py.
# ---------------------------------------------------------------------------

def step5(client):
    import cost_tools as ct  # imported here so Steps 1-4 still run without it

    question = sys.argv[2] if len(sys.argv) > 2 else (
        "How has my AWS spend changed over the last three full months, and what's driving it?"
    )
    print("QUESTION ->", question)
    answer = run_agent(
        client,
        question,
        tools=ct.COST_TOOLS,
        tool_config=ct.COST_TOOL_CONFIG,
        system=ct.cost_system_prompt(),
    )
    print("\nFINAL ANSWER ->\n" + answer)
    ct.print_ce_bill()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "1"
    client = get_client()
    {"1": step1, "2": step2, "3": step3, "4": step4, "5": step5}.get(step, step1)(client)
