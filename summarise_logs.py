"""
Your first agent dashboard: what did each run cost, how long did it take, what failed?
=====================================================================================

Reads logs/agent.jsonl (written by telemetry.py) and answers the questions you'd
otherwise have to guess at.

  python summarise_logs.py              # last 20 runs + totals
  python summarise_logs.py --last 50
  python summarise_logs.py --tag manual # only real questions, not eval runs
  python summarise_logs.py --run 3f9c2a1b7d4e   # replay one run, event by event
"""

import json
import statistics
import sys
from collections import defaultdict

from telemetry import LOG_PATH


def load():
    if not LOG_PATH.exists():
        sys.exit(f"No log file yet at {LOG_PATH}. Run the agent first.")
    runs = defaultdict(list)
    with LOG_PATH.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue  # a half-written line shouldn't break the dashboard
            runs[e["run_id"]].append(e)
    return runs


def replay(events):
    """The full story of one run: the thing you want when someone asks 'why did it say that?'"""
    for e in events:
        t = e["ts"][11:23]
        if e["event"] == "run_start":
            print(f"{t}  START   [{e['engine']} / {e['model']}]  {e['question']}")
        elif e["event"] == "model_call":
            tokens = f"  in={e['input_tokens']} out={e['output_tokens']}" if e.get("input_tokens") is not None else ""
            print(f"{t}  MODEL   turn {e['turn']}  {e['latency_ms']} ms  stop={e['stop_reason']}{tokens}")
        elif e["event"] == "tool_call":
            err = f"  ERROR: {e['result'].get('error')}" if e["status"] == "error" else ""
            print(f"{t}  TOOL    turn {e['turn']}  {e['tool']}({e['input']})  {e['latency_ms']} ms  "
                  f"CE={e['ce_requests']}{err}")
        elif e["event"] == "run_end":
            c = e["est_cost"]
            print(f"{t}  END     status={e['status']}  {e['latency_ms']} ms  turns={e['turns']}  "
                  f"tokens={e['input_tokens']}/{e['output_tokens']}  CE={e['ce_requests']}  "
                  f"cost≈${c['total_usd']:.4f} (model ${c['model_usd']}, tools ${c['tools_usd']})")
            if e.get("error"):
                print(f"          error: {e['error']}")
            if e.get("answer"):
                print(f"          answer: {e['answer'][:300]}")


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))] if values else 0


def main():
    args = sys.argv[1:]
    runs = load()

    if "--run" in args:
        run_id = args[args.index("--run") + 1]
        if run_id not in runs:
            sys.exit(f"No run {run_id} in {LOG_PATH}")
        return replay(runs[run_id])

    last = int(args[args.index("--last") + 1]) if "--last" in args else 20
    tag = args[args.index("--tag") + 1] if "--tag" in args else None

    rows = []
    for run_id, events in runs.items():
        start = next((e for e in events if e["event"] == "run_start"), None)
        end = next((e for e in events if e["event"] == "run_end"), None)
        if not start or not end or (tag and start.get("tag") != tag):
            continue
        tool_errors = sum(1 for e in events if e["event"] == "tool_call" and e["status"] == "error")
        rows.append({"run_id": run_id, "ts": start["ts"], "engine": start["engine"], "tag": start.get("tag"),
                     "question": start["question"], "status": end["status"], "latency_ms": end["latency_ms"],
                     "turns": end["turns"], "tokens_in": end["input_tokens"], "tokens_out": end["output_tokens"],
                     "tools": end["tool_calls"], "tool_errors": tool_errors, "ce": end["ce_requests"],
                     "cost": end["est_cost"]["total_usd"]})
    rows.sort(key=lambda r: r["ts"])
    rows = rows[-last:]
    if not rows:
        sys.exit("No complete runs match.")

    print(f"{'when (UTC)':<17} {'run_id':<13} {'engine':<9} {'tag':<6} {'status':<8} {'secs':>5} {'turns':>5} "
          f"{'tok in':>7} {'tools':>5} {'CE':>3} {'cost $':>8}  question")
    for r in rows:
        status = r["status"] + ("!" if r["tool_errors"] else "")
        print(f"{r['ts'][:16]:<17} {r['run_id']:<13} {r['engine']:<9} {str(r['tag']):<6} {status:<8} "
              f"{r['latency_ms'] / 1000:>5.1f} {str(r['turns']):>5} {r['tokens_in']:>7} {r['tools']:>5} "
              f"{r['ce']:>3} {r['cost']:>8.4f}  {r['question'][:50]}")

    lat = [r["latency_ms"] / 1000 for r in rows]
    costs = [r["cost"] for r in rows]
    failed = [r for r in rows if r["status"] != "ok"]
    print(f"\n{len(rows)} runs | latency p50 {pct(lat, 50):.1f}s, p95 {pct(lat, 95):.1f}s | "
          f"cost avg ${statistics.mean(costs):.4f}, total ${sum(costs):.4f} | "
          f"{len(failed)} not ok | {sum(r['tool_errors'] for r in rows)} tool errors (marked !)")
    by_engine = defaultdict(list)
    for r in rows:
        by_engine[r["engine"]].append(r)
    for engine, rs in sorted(by_engine.items()):
        print(f"  {engine:<9} {len(rs)} runs, avg {statistics.mean(r['latency_ms'] for r in rs) / 1000:.1f}s, "
              f"avg ${statistics.mean(r['cost'] for r in rs):.4f}")
    print("\nReplay any run:  python summarise_logs.py --run <run_id>")


if __name__ == "__main__":
    main()
