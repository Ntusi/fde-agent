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
  python evals.py                 # all cases
  python evals.py bedrock         # only cases whose name contains "bedrock"

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


def c_three_months_fetched(answer, trace):
    got = {r["month"] for r in ok_results(trace, "get_monthly_spend")}
    missing = [m for m in LAST_3_FULL if m not in got]
    return not missing, f"needed {LAST_3_FULL}" + (f", missing {missing}" if missing else "")


def _first_last(trace):
    by_month = {r["month"]: r["account_net_total_usd"] for r in ok_results(trace, "get_monthly_spend")}
    if LAST_3_FULL[0] in by_month and LAST_3_FULL[-1] in by_month:
        return by_month[LAST_3_FULL[0]], by_month[LAST_3_FULL[-1]]
    return None


def c_compares_ends(answer, trace):
    fl = _first_last(trace)
    if not fl:
        return False, "first or last month not fetched"
    first, last = fl
    ok = mentions_amount(answer, first) and mentions_amount(answer, last)
    return ok, f"should state {LAST_3_FULL[0]} ${first} and {LAST_3_FULL[-1]} ${last}"


def c_direction(answer, trace):
    fl = _first_last(trace)
    if not fl:
        return False, "first or last month not fetched"
    first, last = fl
    text = answer.lower()
    if last < first:
        return bool(re.search(DOWN_WORDS, text)), f"spend went DOWN (${first} -> ${last})"
    if last > first:
        return bool(re.search(UP_WORDS, text)), f"spend went UP (${first} -> ${last})"
    return True, "flat"


def c_drivers_both_ends(answer, trace):
    months = {r["month"] for r in ok_results(trace, "get_spend_by_service")}
    ok = LAST_3_FULL[0] in months and LAST_3_FULL[-1] in months
    return ok, f"service breakdown fetched for {sorted(months) or 'no months'}; a change needs both ends"


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
        "why": "Regression: it compared only Sep->Oct and called that 'three months'.",
        "question": "How has my AWS spend changed over the last three full months, and what's driving it?",
        "checks": [("Fetched all three full months", c_three_months_fetched),
                   ("Compares first month to last month", c_compares_ends),
                   ("Gets the direction right (up/down)", c_direction),
                   ("Looked at service drivers at both ends", c_drivers_both_ends)],
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


def config_id():
    """Fingerprint of everything that shapes behaviour, so you know WHAT each score was measured on.

    Covers model, prompt, tool menu, generation settings AND the code (via git). Without the
    code, a fix inside a tool function would score under the same id as the old version.
    """
    blob = json.dumps({"model": al.MODEL_ID, "system": ct.COST_PROMPT_TEMPLATE,
                       "tools": ct.COST_TOOL_CONFIG, "inference": al.INFERENCE_CONFIG,
                       "extras": al.model_extras(), "code": git_version()}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def previous_scores():
    if not RESULTS_DIR.exists():
        return {}
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        return {}
    data = json.loads(files[-1].read_text())
    return {c["name"]: (c["passed"], c["total"]) for c in data["cases"]}


def run(filter_text=None):
    client = al.get_client()
    cases = [c for c in CASES if not filter_text or filter_text in c["name"]]
    if not cases:
        sys.exit(f"No case name contains '{filter_text}'. Cases: {[c['name'] for c in CASES]}")
    before = previous_scores()
    report = {"when": dt.datetime.now().isoformat(timespec="seconds"), "model": al.MODEL_ID,
              "code": git_version(), "config_id": config_id(), "cases": []}

    for case in cases:
        print(f"\n{'=' * 70}\nCASE {case['name']}: {case['question']}\n{'=' * 70}")
        trace = []
        ce_before = ct.CE_CALLS
        crashed = None
        try:
            answer = al.run_agent(client, case["question"], tools=ct.COST_TOOLS,
                                  tool_config=ct.COST_TOOL_CONFIG, system=ct.cost_system_prompt(), trace=trace)
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
    print(f"\n{'=' * 70}\nSUMMARY  model={al.MODEL_ID}  code={report['code']}  config={report['config_id']}\n{'=' * 70}")
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
    all_passed = run(sys.argv[1] if len(sys.argv) > 1 else None)
    sys.exit(0 if all_passed else 1)  # non-zero exit = a CI pipeline would block the change
