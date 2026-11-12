"""
Step 5: real Cost Explorer tools for the agent
==============================================

Same agent loop as Step 4. The only thing that changes is WHAT the tools do:
instead of reading a fake dict, they call the Cost Explorer API on your account.

  python agent_loop.py 5
  python agent_loop.py 5 "Which service costs me the most this month?"

Things worth noticing in this file (all FDE habits):
  1. ONE choke point for every API call (_get_cost_and_usage): cache, count, cost.
  2. Validate the model's inputs BEFORE spending money on an API call.
  3. Shape tool output for the model: top N + "everything else", rounded numbers.
     Raw API responses waste tokens and bury the signal.
  4. The model doesn't know today's date. We tell it, in the system prompt.
"""

import datetime as dt
import difflib
import json
import re

import boto3

CE_PRICE_PER_REQUEST_USD = 0.01  # Cost Explorer API pricing: each paginated request is billed
HISTORY_MONTHS = 13              # Cost Explorer keeps roughly the last 13 months
MAX_DAILY_RANGE_DAYS = 31

_ce_client = None
_cache = {}
CE_CALLS = 0
CACHE_HITS = 0


# ---------------------------------------------------------------------------
# Plumbing: one door for every Cost Explorer call
# ---------------------------------------------------------------------------

def _ce():
    global _ce_client
    if _ce_client is None:
        # Cost Explorer is a global service whose API endpoint lives in us-east-1,
        # whatever region your workloads run in.
        _ce_client = boto3.client("ce", region_name="us-east-1")
    return _ce_client


def _get_cost_and_usage(**kwargs) -> list:
    """Call GetCostAndUsage with caching and pagination. Returns all ResultsByTime entries.

    Why a choke point: an agent in a loop may ask for the same month twice. Every real
    request costs $0.01, so we cache identical requests and count the ones we pay for.
    """
    global CE_CALLS, CACHE_HITS
    key = json.dumps(kwargs, sort_keys=True)
    if key in _cache:
        CACHE_HITS += 1
        return _cache[key]

    results, token = [], None
    while True:  # Cost Explorer paginates large responses; each page is a billed request
        args = dict(kwargs)
        if token:
            args["NextPageToken"] = token
        CE_CALLS += 1
        page = _ce().get_cost_and_usage(**args)
        results.extend(page["ResultsByTime"])
        token = page.get("NextPageToken")
        if not token:
            break

    _cache[key] = results
    return results


def print_ce_bill():
    print(
        f"\nCost Explorer: {CE_CALLS} paid request(s), {CACHE_HITS} served from cache "
        f"(~${CE_CALLS * CE_PRICE_PER_REQUEST_USD:.2f})"
    )


# ---------------------------------------------------------------------------
# Input validation: reject bad requests before they cost anything
# ---------------------------------------------------------------------------

def _today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()  # Cost Explorer works in UTC


def _month_range(month: str):
    """'2026-10' -> (start, end_exclusive, is_partial). Raises ValueError with a message the model can act on."""
    if not isinstance(month, str) or not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError(f"month must look like YYYY-MM (e.g. 2026-10), got {month!r}")
    year, mon = int(month[:4]), int(month[5:])
    if not 1 <= mon <= 12:
        raise ValueError(f"month number must be 01-12, got {month!r}")

    today = _today()
    start = dt.date(year, mon, 1)
    if start > today:
        raise ValueError(f"{month} is in the future. Today is {today.isoformat()}.")
    months_back = (today.year - year) * 12 + (today.month - mon)
    if months_back > HISTORY_MONTHS - 1:
        raise ValueError(f"{month} is older than Cost Explorer's ~{HISTORY_MONTHS}-month history.")

    end = dt.date(year + (mon == 12), mon % 12 + 1, 1)  # first day of next month (End is exclusive)
    is_partial = end > today
    if is_partial:
        end = today + dt.timedelta(days=1)  # current month: only up to today
    return start, end, is_partial


def _parse_date(value: str, name: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be YYYY-MM-DD, got {value!r}")


def _usd(x: float) -> float:
    # Sandbox bills are often cents. Keep small amounts readable instead of rounding them to 0.
    return round(x, 2) if abs(x) >= 1 else round(x, 4)


USAGE_ONLY = {"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Usage"]}}


def _service_names(start: dt.date, end_excl: dt.date) -> list:
    """Every service with usage in the period, as Cost Explorer spells it. One cached request."""
    results = _get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": end_excl.isoformat()},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        Filter=USAGE_ONLY,
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )
    return sorted({g["Keys"][0] for period in results for g in period.get("Groups", [])})


def _resolve_service(service: str, start: dt.date, end_excl: dt.date):
    """Turn the model's service name into Cost Explorer's exact name.

    Step 5 bug: the model sent 'Bedrock', Cost Explorer only knows 'Amazon Bedrock',
    the filter silently matched nothing, and the agent said 'no spend'. Never again.
    Returns (exact_name, note). note says what we assumed, so the model knows.
    """
    names = _service_names(start, end_excl)
    if service in names:
        return service, None  # already exact: nothing to fix

    # 1. "contains" match, ignoring case: 'Bedrock' is inside 'Amazon Bedrock'
    contains = [n for n in names if service.lower() in n.lower()]
    # 2. spelling match: 'Amazon Bedrok' is close to 'Amazon Bedrock'.
    #    cutoff 0.8, not difflib's default 0.6: at 0.6 the shared 'Amazon ' prefix made
    #    'Amazon Bedrock' look "close" to 'Amazon CloudFront' (eval run, 12 Nov). Typos score ~0.95.
    lower = {n.lower(): n for n in names}
    close = [lower[m] for m in difflib.get_close_matches(service.lower(), list(lower), n=3, cutoff=0.8)]
    candidates = list(dict.fromkeys(contains + close))  # merge, drop duplicates, keep order

    if len(candidates) == 1:
        # One clear match: use it, but SAY so. Never correct silently.
        return candidates[0], f"Interpreted '{service}' as '{candidates[0]}'."
    if candidates:
        raise ValueError(f"Ambiguous service '{service}'. Did you mean one of: {candidates}?")
    raise ValueError(
        f"No usage recorded for a service named '{service}' in this period. "
        f"Services with usage: {names[:25]}"
    )


# ---------------------------------------------------------------------------
# The tools (plain Python functions; the model never sees this code)
# ---------------------------------------------------------------------------

def get_monthly_spend(month: str) -> dict:
    start, end, partial = _month_range(month)
    results = _get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "RECORD_TYPE"}],  # Usage vs Credit vs Tax, in ONE request
    )
    by_type = {}
    for period in results:
        for g in period.get("Groups", []):
            by_type[g["Keys"][0]] = by_type.get(g["Keys"][0], 0.0) + float(g["Metrics"]["UnblendedCost"]["Amount"])

    return {
        "month": month,
        # Eval finding (12 Nov): asked about Bedrock, the model quoted THIS number (the whole bill)
        # as Bedrock's spend, twice. A name that says what it covers is harder to misuse.
        "scope": "whole account, all services",
        "account_net_total_usd": _usd(sum(by_type.values())),
        "by_charge_type_usd": {k: _usd(v) for k, v in sorted(by_type.items(), key=lambda kv: -abs(kv[1]))},
        "partial_month": partial,
        "data_through": (end - dt.timedelta(days=1)).isoformat(),
        "estimated": any(p.get("Estimated") for p in results),
    }


def get_spend_by_service(month: str, top_n: int = 8) -> dict:
    start, end, partial = _month_range(month)
    top_n = max(1, min(int(top_n), 20))
    results = _get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        Filter=USAGE_ONLY,  # usage cost BEFORE credits, so drivers show even on a credit-funded account
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )
    totals = {}
    for period in results:
        for g in period.get("Groups", []):
            totals[g["Keys"][0]] = totals.get(g["Keys"][0], 0.0) + float(g["Metrics"]["UnblendedCost"]["Amount"])

    ranked = sorted(((s, v) for s, v in totals.items() if abs(v) >= 0.005), key=lambda kv: -kv[1])
    top, rest = ranked[:top_n], ranked[top_n:]
    by_service = {s: _usd(v) for s, v in top}
    if rest:
        by_service[f"All other services ({len(rest)})"] = _usd(sum(v for _, v in rest))

    return {
        "month": month,
        "usage_total_usd": _usd(sum(totals.values())),
        "by_service_usd": by_service,
        "note": "Usage charges only, before credits, refunds and tax.",
        "partial_month": partial,
    }


def get_daily_spend(start_date: str, end_date: str, service: str = None) -> dict:
    start = _parse_date(start_date, "start_date")
    end_incl = _parse_date(end_date, "end_date")
    today = _today()
    if end_incl < start:
        raise ValueError("end_date is before start_date")
    if end_incl > today:
        end_incl = today
    if (end_incl - start).days + 1 > MAX_DAILY_RANGE_DAYS:
        raise ValueError(f"Range too long: max {MAX_DAILY_RANGE_DAYS} days per call.")

    end_excl = end_incl + dt.timedelta(days=1)
    flt, interpreted = USAGE_ONLY, None
    if service:
        # Step 6 fix: check the name against what Cost Explorer actually has BEFORE filtering.
        service, interpreted = _resolve_service(service, start, end_excl)
        flt = {"And": [USAGE_ONLY, {"Dimensions": {"Key": "SERVICE", "Values": [service]}}]}

    results = _get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": end_excl.isoformat()},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
        Filter=flt,
    )
    days = {p["TimePeriod"]["Start"]: _usd(float(p["Total"]["UnblendedCost"]["Amount"])) for p in results}
    # Step 6 fix: all-zero days have no peak. Don't name a "peak day" that isn't one.
    peak_day = max(days, key=days.get) if any(days.values()) else None
    result = {
        "service": service or "all services",
        "scope": f"{service} only" if service else "whole account, all services",
        "daily_usd": days,
        "total_usd": _usd(sum(days.values())),
        "peak_day": peak_day,
        "note": "Usage charges only, before credits.",
    }
    if interpreted:
        result["interpreted_as"] = interpreted
    return result


# ---------------------------------------------------------------------------
# What the MODEL sees: the menu
# ---------------------------------------------------------------------------

COST_TOOLS = {
    "get_monthly_spend": get_monthly_spend,
    "get_spend_by_service": get_spend_by_service,
    "get_daily_spend": get_daily_spend,
}


def _month_schema():
    return {"type": "string", "description": "Calendar month in YYYY-MM format, e.g. 2026-10"}


COST_TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_monthly_spend",
                "description": (
                    "Net AWS bill for the WHOLE ACCOUNT (all services) for one calendar month, split by "
                    "charge type (Usage, Credit, Tax, Refund...). Use it for 'how much did we spend' questions. "
                    "Never quote it as one service's spend. "
                    "partial_month=true means the month isn't over yet."
                ),
                "inputSchema": {"json": {"type": "object", "properties": {"month": _month_schema()}, "required": ["month"]}},
            }
        },
        {
            "toolSpec": {
                "name": "get_spend_by_service",
                "description": (
                    "Usage cost for one calendar month broken down by AWS service, largest first "
                    "(top_n services plus an 'All other services' bucket). Use it to find WHICH services "
                    "drive the bill or a change in it."
                ),
                "inputSchema": {"json": {
                    "type": "object",
                    "properties": {
                        "month": _month_schema(),
                        "top_n": {"type": "integer", "description": "How many services to list (1-20, default 8)"},
                    },
                    "required": ["month"],
                }},
            }
        },
        {
            "toolSpec": {
                "name": "get_daily_spend",
                "description": (
                    "Usage cost per day for up to 31 days, optionally for ONE service. Use it to find WHEN "
                    "a change started. Prefer exact service names from get_spend_by_service (e.g. 'Amazon Bedrock'). "
                    "A near-miss is resolved and reported in 'interpreted_as'; an unknown name returns an error "
                    "listing the services that had usage."
                ),
                "inputSchema": {"json": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "First day, YYYY-MM-DD"},
                        "end_date": {"type": "string", "description": "Last day (inclusive), YYYY-MM-DD"},
                        "service": {"type": "string", "description": "Optional exact service name"},
                    },
                    "required": ["start_date", "end_date"],
                }},
            }
        },
    ]
}


# The prompt is a TEMPLATE: code fills in the dates. Evals fingerprint the template itself,
# so a wording change gets a new config id while the date changing every day doesn't.
COST_PROMPT_TEMPLATE = (
    "You are a concise AWS FinOps assistant analysing the user's REAL AWS bill through Cost Explorer.\n"
    "Calendar (computed by code; use these exact months, do not work out dates yourself):\n"
    "- Today: {today} (UTC)\n"
    "- This month: {current_month} (PARTIAL, not over yet)\n"
    "- Last month: {last_month}\n"
    "- Last three full months: {last_3_full}\n"
    "Use tools for every figure; never guess numbers. "
    "For 'last N months' use full months only; never compare a partial month with a full month "
    "unless the question is about this month. "
    "If credits offset the bill, say so and analyse usage cost. "
    "Each Cost Explorer request costs money: never request data you already have.\n"
    "When asked about ONE service, report only that service's figures.\n"
    "When asked how spend changed over a period, compare the FIRST and LAST month of that period and state "
    "both amounts. Explain the change by comparing get_spend_by_service for those same two months.\n"
    "Answer format:\n"
    "1. First line: the headline with amounts in USD and % change.\n"
    "2. The main driver(s) with their share of the change.\n"
    "3. One line on what to check next.\n"
    "Maximum 6 lines."
)


def calendar_facts() -> dict:
    """Date arithmetic belongs in code. Eval run (12 Nov): told 'last three full months' and
    today's date, Nova still picked September and the half-finished November."""
    today = _today()
    months, y, m = [], today.year, today.month
    for _ in range(4):  # this month + the three before it
        months.append(f"{y:04d}-{m:02d}")
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    current, last, before_last, oldest = months
    return {
        "today": today.isoformat(),
        "current_month": current,
        "last_month": last,
        "last_3_full": f"{oldest}, {before_last}, {last}",
    }


def cost_system_prompt() -> list:
    return [{"text": COST_PROMPT_TEMPLATE.format(**calendar_facts())}]
