"""
Step 6: Evals. Automatic, scored tests for the cost agent
=========================================================

Last week the agent gave two confident wrong answers and nothing crashed:
  - "No spend on Bedrock" (it filtered on 'Bedrock'; Cost Explorer needs 'Amazon Bedrock')
  - "Up 33% over three months" (it compared Sep->Oct and called that three months)
An eval turns failures like those into tests that run every time you change a prompt,
a tool or the model, so they can never come back unnoticed.

How the checks work: the agent's answer is compared against what the TOOLS actually
returned during that same run (recorded in the trace). So the tests stay valid as your
real bill changes: no hard-coded dollar amounts.

Run:
  python evals.py                     # all cases, hand-built agent (agent_loop.py)
  python evals.py bedrock             # only cases whose name contains "bedrock"
  python evals.py --engine strands    # same cases, Strands agent (strands_agent.py)

Cost: a full run is roughly 10-20 Cost Explorer requests (~$0.10-0.20) plus a few
cents of Bedrock. Results are saved in eval_results/ so you can compare runs.
"""

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import agent_loop as al
import cost_tools as ct

RESULTS_DIR = Path(__file__).with_name("eval_results")

# Budgets every case must stay inside. Cost and speed are part of "correct".
MAX_TURNS = 5
MAX_CE_REQUESTS = 6
MAX_ANSWER_LINES = 6  # the system prompt asks for at most 6 lines


# ---------------------------------------------------------------------------
# Facts from the trace: what the tools ACTUALLY returned in this run
# ---------------------------------------------------------------------------

def ok_results(trace, tool_name):
    """Successful results (no 'error' key) from one tool, in call order."""
    return [e["result"] for e in trace
            if e["type"] == "tool" and e["name"] == tool_name and "error" not in e["result"]]


def usage(trace):
    return next((e for e in reversed(trace) if e["type"] == "usage"), {})


# ---------------------------------------------------------------------------
# Answer helpers: does the text say what the data says?
# ---------------------------------------------------------------------------

def numbers_in(text):
    """All numbers in the text: '$1,234.50' -> 1234.5, '58.1%' -> 58.1."""
    found = re.findall(r"(?<![\w.-])\$?(\d[\d,]*(?:\.\d+)?)", text)
    return [float(n.replace(",", "")) for n in found]


def mentions_amount(answer, value, rel=0.02, abs_tol=0.006):
    """True if the answer contains a number close to value (allows normal rounding)."""
    target = abs(value)
    return any(abs(n - target) <= max(abs_tol, rel * target) for n in numbers_in(answer))


MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


def mentions_date(answer, iso_date):
    """'2026-11-04' matches '2026-11-04', 'November 4', 'Nov 4th', '4 November'."""
    d = dt.date.fromisoformat(iso_date)
    month = MONTHS[d.month - 1]
    text = answer.lower()
    patterns = [
        re.escape(iso_date),
        rf"\b{month[:3]}[a-z]*\.?\s+{d.day}(st|nd|rd|th)?\b",
        rf"\b{d.day}(st|nd|rd|th)?\s+(of\s+)?{month[:3]}",
    ]
    return any(re.search(p, text) for p in patterns)


SERVICE_ALIASES = {
    "AWS Key Management Service": ["kms"],
    "Amazon Elastic Compute Cloud - Compute": ["ec2"],
    "Amazon Simple Storage Service": ["s3"],
    "Amazon Relational Database Service": ["rds"],
    "Amazon Virtual Private Cloud": ["vpc"],
}


def mentions_service(answer, service):
    text = answer.lower()
    return service.lower() in text or any(re.search(rf"\b{a}\b", text) for a in SERVICE_ALIASES.get(service, []))


DOWN_WORDS = r"\b(decrease[ds]?|down|fell|fall|drop(ped)?|lower|declin(e|ed)|reduc(ed|tion))\b"
UP_WORDS = r"\b(increase[ds]?|up|rose|rise|grew|growth|higher|jump(ed)?)\b"


# ---------------------------------------------------------------------------
# Dates the cases depend on (computed, so the suite works any month)
# ---------------------------------------------------------------------------

TODAY = ct._today()


def month_offset(n):
    """n=0 -> current month, n=1 -> last month, ... as 'YYYY-MM'."""
    y, m = TODAY.year, TODAY.month - n
    while m <= 0:
        y, m = y - 1, m + 12
    return f"{y:04d}-{m:02d}"


CURRENT_MONTH = month_offset(0)
LAST_MONTH = month_offset(1)
LAST_3_FULL = [month_offset(3), month_offset(2), month_offset(1)]  # oldest first


# ---------------------------------------------------------------------------
# The cases. Each check returns (passed, detail).
# ---------------------------------------------------------------------------

def c_bedrock_this_month(answer, trace):
    starts = [e["input"].get("start_date", "") for e in trace if e["type"] == "tool" and e["name"] == "get_daily_spend"]
    ok = any(d.startswith(CURRENT_MONTH) for d in starts)
    return ok, f"'this month' = {CURRENT_MONTH}; daily ranges started on {starts or 'nothing'}"


def c_bedrock_exact_name(answer, trace):
    hits = [r for r in ok_results(trace, "get_daily_spend") if r.get("service") == "Amazon Bedrock"]
    return bool(hits), "get_daily_spend ran on 'Amazon Bedrock'" if hits else "never queried 'Amazon Bedrock' successfully"


def c_bedrock_total(answer, trace):
    hits = [r for r in ok_results(trace, "get_daily_spend") if r.get("service") == "Amazon Bedrock"]
    if not hits:
        return False, "no Bedrock data to compare against"
    total = hits[-1]["total_usd"]
    return mentions_amount(answer, total), f"tool total ${total}"


def c_bedrock_peak(answer, trace):
    hits = [r for r in ok_results(trace, "get_daily_spend") if r.get("service") == "Amazon Bedrock"]
    if not hits:
        return False, "no Bedrock data to compare against"
    peak = hits[-1]["peak_day"]
    if peak is None:
        return True, "no spend, so no peak day to name"
    return mentions_date(answer, peak), f"tool peak day {peak}"


def _period_facts(trace):
    """The truth for the three-month question, taken from what the tools returned in this run.

    Accepts either route: Round 4's compare_periods, or the older one-month-at-a-time tools.
    Returns None if the first and last full months were never fetched.
    """
    first, last = LAST_3_FULL[0], LAST_3_FULL[-1]
    for r in reversed(ok_results(trace, "compare_periods")):
        if r["period"] == f"{first} to {last}":
            movers = r["top_service_changes"]
            return {"route": "compare_periods", "breakdowns": True,
                    "bill": (r["bill_change"]["from_usd"], r["bill_change"]["to_usd"]),
                    "usage": (r["usage_change"]["from_usd"], r["usage_change"]["to_usd"]),
                    "usage_pct": r["usage_change"]["change_pct"],
                    "one_offs": r["one_off_charges"],
                    "top_mover": movers[0]["service"] if movers else None}

    monthly = {r["month"]: r for r in ok_results(trace, "get_monthly_spend")}
    if first not in monthly or last not in monthly:
        return None
    f, l = monthly[first], monthly[last]
    usage = (f["by_charge_type_usd"].get("Usage", 0.0), l["by_charge_type_usd"].get("Usage", 0.0))
    one_offs = [{"month": m["month"], "charge_type": t, "usd": v}
                for m in (f, l) for t, v in m["by_charge_type_usd"].items()
                if t not in ("Usage", "Tax") and abs(v) >= 0.5]
    services = {r["month"]: r["by_service_usd"] for r in ok_results(trace, "get_spend_by_service")}
    top_mover = None
    breakdowns = first in services and last in services
    if breakdowns:
        names = set(services[first]) | set(services[last])
        deltas = {n: services[last].get(n, 0.0) - services[first].get(n, 0.0) for n in names if not n.startswith("All other")}
        deltas = {n: d for n, d in deltas.items() if abs(d) >= 0.005}  # nothing moved -> no "biggest mover"
        top_mover = max(deltas, key=lambda n: abs(deltas[n])) if deltas else None
    return {"route": "monthly tools", "breakdowns": breakdowns,
            "bill": (f["account_net_total_usd"], l["account_net_total_usd"]),
            "usage": usage,
            "usage_pct": round((usage[1] - usage[0]) / usage[0] * 100, 1) if usage[0] else None,
            "one_offs": one_offs, "top_mover": top_mover}


def _direction(a, b):
    return "up" if b > a + 0.005 else "down" if b < a - 0.005 else "flat"


def c_right_period(answer, trace):
    facts = _period_facts(trace)
    if not facts:
        return False, f"never fetched both {LAST_3_FULL[0]} and {LAST_3_FULL[-1]}"
    return True, f"covered {LAST_3_FULL[0]} to {LAST_3_FULL[-1]} via {facts['route']}"


def c_bill_amounts(answer, trace):
    facts = _period_facts(trace)
    if not facts:
        return False, "no period data to compare against"
    a, b = facts["bill"]
    return mentions_amount(answer, a) and mentions_amount(answer, b), f"bill should read ${a} -> ${b}"


def c_bill_direction(answer, trace):
    facts = _period_facts(trace)
    if not facts:
        return False, "no period data to compare against"
    a, b = facts["bill"]
    d = _direction(a, b)
    words = DOWN_WORDS if d == "down" else UP_WORDS if d == "up" else r"flat|unchanged|same"
    return bool(re.search(words, answer.lower())), f"the bill went {d.upper()} (${a} -> ${b})"


def c_usage_story(answer, trace):
    facts = _period_facts(trace)
    if not facts:
        return False, "no period data to compare against"
    (ba, bb), (ua, ub) = facts["bill"], facts["usage"]
    if _direction(ba, bb) == _direction(ua, ub):
        return True, "usage moved the same way as the bill; nothing extra to say"
    said = (mentions_amount(answer, ua) and mentions_amount(answer, ub)) or (
        facts["usage_pct"] is not None and mentions_amount(answer, facts["usage_pct"], rel=0.03, abs_tol=0.6))
    return said, f"bill went {_direction(ba, bb)} but usage went {_direction(ua, ub)} (${ua} -> ${ub}); say both"


def c_one_off(answer, trace):
    facts = _period_facts(trace)
    if not facts:
        return False, "no period data to compare against"
    if not facts["one_offs"]:
        return True, "no one-off charges in the period"
    named = any(mentions_amount(answer, o["usd"]) for o in facts["one_offs"]) or bool(
        re.search(r"one.?off|one.?time|non.?recurring", answer.lower()))
    return named, f"should name the one-off: {facts['one_offs']}"


def c_top_mover(answer, trace):
    facts = _period_facts(trace)
    if not facts or not facts["breakdowns"]:
        return False, "never got a service breakdown for both ends of the period"
    if not facts["top_mover"]:
        return True, "no service changed between the two months"
    return mentions_service(answer, facts["top_mover"]), f"biggest usage mover: {facts['top_mover']}"


def c_no_invented_amounts(answer, trace):
    fetched = any(e["type"] == "tool" and "error" not in e["result"] for e in trace)
    has_dollars = bool(re.search(r"\$\s?\d", answer))
    return not fetched and not has_dollars, "no data exists, so no $ figures should appear"


def c_says_unavailable(answer, trace):
    pattern = (r"not available|no data|unavailable|older than|only (keeps|retains|covers|goes back|has)|"
               r"13.month|history|can.?t|cannot|unable|outside|beyond|not possible")
    return bool(re.search(pattern, answer.lower())), "should say the data isn't available"


def c_fetched_last_month_services(answer, trace):
    hits = [r for r in ok_results(trace, "get_spend_by_service") if r["month"] == LAST_MONTH]
    return bool(hits), f"get_spend_by_service({LAST_MONTH})"


def _top_service(trace):
    hits = [r for r in ok_results(trace, "get_spend_by_service") if r["month"] == LAST_MONTH]
    if not hits:
        return None
    services = [(s, v) for s, v in hits[-1]["by_service_usd"].items() if not s.startswith("All other")]
    return services[0] if services else None


def c_names_top_service(answer, trace):
    top = _top_service(trace)
    if not top:
        return False, "no breakdown to compare against"
    return mentions_service(answer, top[0]), f"top service: {top[0]}"


def c_top_service_amount(answer, trace):
    top = _top_service(trace)
    if not top:
        return False, "no breakdown to compare against"
    return mentions_amount(answer, top[1]), f"top service amount ${top[1]}"


def c_fetched_current_month(answer, trace):
    hits = [r for r in ok_results(trace, "get_monthly_spend") if r["month"] == CURRENT_MONTH]
    return bool(hits), f"get_monthly_spend({CURRENT_MONTH})"


def c_mtd_amount(answer, trace):
    hits = [r for r in ok_results(trace, "get_monthly_spend") if r["month"] == CURRENT_MONTH]
    if not hits:
        return False, "no current-month data"
    total = hits[-1]["account_net_total_usd"]
    return mentions_amount(answer, total), f"tool total ${total}"


def c_flags_partial(answer, trace):
    pattern = r"so far|partial|to date|month-to-date|mtd|not over|isn.?t over|as of|through|until|incomplete"
    return bool(re.search(pattern, answer.lower())), "should say the month isn't finished"


CASES = [
    {
        "name": "bedrock_this_month",
        "why": "Regression: the silent $0 bug from the 'Bedrock' vs 'Amazon Bedrock' name mismatch.",
        "question": "What did I spend on Bedrock this month, and on which day was it highest?",
        "checks": [("Used the current month for 'this month'", c_bedrock_this_month),
                   ("Queried Bedrock by its exact name", c_bedrock_exact_name),
                   ("States the total the tool returned", c_bedrock_total),
                   ("Names the peak day the tool returned", c_bedrock_peak)],
    },
    {
        "name": "three_full_months",
        "why": ("Regression: wrong months, wrong sign, usage reported as the bill, and August's one-off "
                "$8 missed. The right answer needs the bill change AND the usage change."),
        "question": "How has my AWS spend changed over the last three full months, and what's driving it?",
        "checks": [("Covered the right period", c_right_period),
                   ("States the bill at both ends", c_bill_amounts),
                   ("Gets the bill direction right", c_bill_direction),
                   ("Also gives the usage change when it differs", c_usage_story),
                   ("Names the one-off charge", c_one_off),
                   ("Names the biggest service mover", c_top_mover)],
    },
    {
        "name": "out_of_range",
        "why": "Refusal: data older than Cost Explorer's history must not be invented.",
        "question": "What was my AWS spend in January 2023?",
        "checks": [("No invented dollar figures", c_no_invented_amounts),
                   ("Says the data isn't available", c_says_unavailable)],
    },
    {
        "name": "top_service_last_month",
        "why": "Basic lookup: the most common question a customer asks.",
        "question": "Which AWS service cost me the most last month?",
        "checks": [("Fetched last month's service breakdown", c_fetched_last_month_services),
                   ("Names the actual top service", c_names_top_service),
                   ("States its amount", c_top_service_amount)],
    },
    {
        "name": "month_to_date",
        "why": "Partial month: the answer must not read like a finished month.",
        "question": "How much have I spent on AWS so far this month?",
        "checks": [("Fetched the current month", c_fetched_current_month),
                   ("States the amount the tool returned", c_mtd_amount),
                   ("Says the month isn't over", c_flags_partial)],
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def budget_checks(answer, trace, ce_requests):
    u = usage(trace)
    lines = [l for l in answer.splitlines() if l.strip()]
    return [
        ("Budget: turns", u.get("turns", 99) <= MAX_TURNS, f"{u.get('turns')} turns (max {MAX_TURNS})"),
        ("Budget: Cost Explorer requests", ce_requests <= MAX_CE_REQUESTS, f"{ce_requests} paid (max {MAX_CE_REQUESTS})"),
        ("Format: answer length", len(lines) <= MAX_ANSWER_LINES, f"{len(lines)} lines (max {MAX_ANSWER_LINES})"),
    ]


def git_version():
    """The code version: last commit, plus '+changes' if files were edited since. 'nogit' outside a repo."""
    here = Path(__file__).parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "*.py"], cwd=here,
                               capture_output=True, text=True).stdout.strip()
        return commit + ("+changes" if dirty else "")
    except Exception:
        return "nogit"


ENGINE = "handbuilt"  # or "strands"; set from the command line


def config_id():
    """Fingerprint of everything that shapes behaviour, so you know WHAT each score was measured on.

    Covers model, prompt, tool menu, generation settings AND the code (via git). Without the
    code, a fix inside a tool function would score under the same id as the old version.
    """
    blob = json.dumps({"model": al.MODEL_ID, "system": ct.COST_PROMPT_TEMPLATE,
                       "tools": ct.COST_TOOL_CONFIG, "inference": al.INFERENCE_CONFIG,
                       "extras": al.model_extras(), "code": git_version(), "engine": ENGINE}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def previous_scores():
    if not RESULTS_DIR.exists():
        return {}
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        return {}
    data = json.loads(files[-1].read_text())
    return {c["name"]: (c["passed"], c["total"]) for c in data["cases"]}


def make_runner():
    """Same question in, same (answer, trace) out, whichever engine runs the loop."""
    if ENGINE == "strands":
        import strands_agent as sa  # imported here so the hand-built path doesn't need Strands installed
        return lambda question, trace: sa.run_strands(question, trace)
    client = al.get_client()
    return lambda question, trace: al.run_agent(client, question, tools=ct.COST_TOOLS,
                                                tool_config=ct.COST_TOOL_CONFIG,
                                                system=ct.cost_system_prompt(), trace=trace)


def run(filter_text=None):
    run_case = make_runner()
    cases = [c for c in CASES if not filter_text or filter_text in c["name"]]
    if not cases:
        sys.exit(f"No case name contains '{filter_text}'. Cases: {[c['name'] for c in CASES]}")
    before = previous_scores()
    report = {"when": dt.datetime.now().isoformat(timespec="seconds"), "model": al.MODEL_ID, "engine": ENGINE,
              "code": git_version(), "config_id": config_id(), "cases": []}

    for case in cases:
        print(f"\n{'=' * 70}\nCASE {case['name']}: {case['question']}\n{'=' * 70}")
        trace = []
        ce_before = ct.CE_CALLS
        crashed = None
        try:
            answer = run_case(case["question"], trace)
        except (Exception, SystemExit) as e:
            # One broken case must not kill the whole suite. Record it as a failure and move on.
            crashed = " ".join(f"{type(e).__name__}: {e}".split())  # one line, readable in the report
            answer = f"[CRASHED] {crashed}"
        ce_used = ct.CE_CALLS - ce_before
        print("\nANSWER ->\n" + answer + "\n")

        results = [("Completed without crashing", crashed is None, crashed or "ok")]
        for label, fn in case["checks"]:
            try:
                passed, detail = fn(answer, trace)
            except Exception as e:  # a broken check is a failed check, not a crashed suite
                passed, detail = False, f"check crashed: {type(e).__name__}: {e}"
            results.append((label, passed, detail))
        results += budget_checks(answer, trace, ce_used)

        for label, passed, detail in results:
            print(f"  {'PASS' if passed else 'FAIL'}  {label}  ({detail})")

        u = usage(trace)
        report["cases"].append({
            "name": case["name"], "question": case["question"], "answer": answer,
            "passed": sum(p for _, p, _ in results), "total": len(results),
            "checks": [{"label": l, "passed": p, "detail": d} for l, p, d in results],
            "turns": u.get("turns"), "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"), "ce_requests": ce_used, "trace": trace,
        })

    # ---- Summary ----
    print(f"\n{'=' * 70}\nSUMMARY  engine={ENGINE}  model={al.MODEL_ID}  code={report['code']}  config={report['config_id']}\n{'=' * 70}")
    total_p = total_t = 0
    for c in report["cases"]:
        total_p += c["passed"]
        total_t += c["total"]
        prev = before.get(c["name"])
        delta = f"   (last run {prev[0]}/{prev[1]})" if prev else ""
        mark = "✅" if c["passed"] == c["total"] else "❌"
        print(f"{mark} {c['name']:<26} {c['passed']}/{c['total']}{delta}")
    tokens_in = sum(c["input_tokens"] or 0 for c in report["cases"])
    tokens_out = sum(c["output_tokens"] or 0 for c in report["cases"])
    print(f"\nScore: {total_p}/{total_t} checks ({100 * total_p / total_t:.0f}%)")
    print(f"Tokens: {tokens_in} in / {tokens_out} out")
    ct.print_ce_bill()

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{report['config_id']}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved: {out}")
    return total_p == total_t


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--engine" in args:
        i = args.index("--engine")
        ENGINE = args[i + 1]
        del args[i:i + 2]
        if ENGINE not in ("handbuilt", "strands"):
            sys.exit("--engine must be handbuilt or strands")
    all_passed = run(args[0] if args else None)
    sys.exit(0 if all_passed else 1)  # non-zero exit = a CI pipeline would block the change
