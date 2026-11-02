"""Paired comparison of two systems on COBOLEval (per CLAUDE.md stats rules).

  - pass@1 per problem (all tests pass), compile rate, total cost, p50 latency, per system
  - McNemar exact test on per-problem pass (two-sided binomial on the discordant pairs)
  - bootstrap over PROBLEMS (not tests), 5,000 resamples, 95% CI on the pass-rate difference
  - failure breakdown: blocked / compile error / wrong answer / pass

Run: .venv/bin/python -m eval.compare qwen38-27b-base qwen38-27b-sft
"""

import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITERS, SEED = 5000, 0


def load(system):
    return {r["id"]: r for r in map(json.loads, open(os.path.join(ROOT, f"results/{system}.jsonl")))}


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def outcome(r):
    if r["pass"]:
        return "pass"
    if r.get("error") == "blocked":
        return "blocked"
    return "wrong answer" if r["compiled"] else "compile error"


def main(a_name, b_name):
    a, b = load(a_name), load(b_name)
    ids = sorted(set(a) & set(b))
    pa = [int(a[i]["pass"]) for i in ids]
    pb = [int(b[i]["pass"]) for i in ids]
    n = len(ids)
    only_b = sum(1 for x, y in zip(pa, pb) if y and not x)
    only_a = sum(1 for x, y in zip(pa, pb) if x and not y)
    rng = random.Random(SEED)
    diffs = []
    for _ in range(ITERS):
        s = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(pb[i] - pa[i] for i in s) / n)
    diffs.sort()
    lo, hi = diffs[int(0.025 * ITERS)], diffs[int(0.975 * ITERS) - 1]

    def stats(rows):
        lat = sorted(r["latency_s"] for r in rows)
        out = {}
        for r in rows:
            out[outcome(r)] = out.get(outcome(r), 0) + 1
        return {"pass@1": sum(r["pass"] for r in rows) / len(rows), "passed": sum(r["pass"] for r in rows),
                "compiled": sum(r["compiled"] for r in rows) / len(rows), "cost": sum(r["cost"] for r in rows),
                "p50_latency_s": lat[len(lat) // 2], "outcomes": out}

    summary = {
        "n_problems": n,
        a_name: stats([a[i] for i in ids]),
        b_name: stats([b[i] for i in ids]),
        "diff_pass@1": (sum(pb) - sum(pa)) / n,
        "bootstrap_95ci": [lo, hi],
        "mcnemar": {"b_only_pass": only_b, "a_only_pass": only_a, "p_value": mcnemar_exact(only_b, only_a)},
    }
    out = os.path.join(ROOT, f"results/compare_{a_name}_vs_{b_name}.json")
    json.dump(summary, open(out, "w"), indent=1)

    sa, sb = summary[a_name], summary[b_name]
    print(f"| System | pass@1 | compiled | cost | p50 latency |\n| --- | --- | --- | --- | --- |")
    for name, s in ((a_name, sa), (b_name, sb)):
        print(f"| {name} | {s['passed']}/{n} ({s['pass@1']:.1%}) | {s['compiled']:.1%} | ${s['cost']:.2f} | {s['p50_latency_s']:.1f}s |")
    m = summary["mcnemar"]
    print(f"\ndiff {summary['diff_pass@1']:+.1%} (95% bootstrap CI {lo:+.1%} to {hi:+.1%}); "
          f"McNemar: {b_name} only {m['b_only_pass']}, {a_name} only {m['a_only_pass']}, p = {m['p_value']:.2g}")
    print(f"outcomes {a_name}: {sa['outcomes']}\noutcomes {b_name}: {sb['outcomes']}\nwrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
