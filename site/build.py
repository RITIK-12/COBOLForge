"""Build the project page (GitHub Pages) from the repo's own files.

Every number on the page comes from results/, data/, sft/logs/ and rl/ at build time, so the page
follows the repo: when GRPO rows land in results/qwen38-27b-grpo*.jsonl they show up on the next push.
Stdlib only, so the Pages workflow needs no installs.

Run: python site/build.py   ->  _site/index.html (open it, or serve _site/)
"""

import glob
import html
import json
import os
import random
import re
import shutil
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")
OUT = os.path.join(ROOT, "_site")
sys.path.insert(0, ROOT)

from eval.compare import mcnemar_exact, outcome  # noqa: E402

BASE, SFT = "qwen38-27b-base", "qwen38-27b-sft"
LABELS = {BASE: "Qwen 3.8 27B base", SFT: "+ SFT"}
EXAMPLE = "HumanEval/138"  # base: 'MOD' is not defined; SFT: FUNCTION MOD, 8/8 tests
ITERS, SEED = 5000, 0  # same bootstrap as eval/compare.py

# Frontier open-weights models run through our own harness (eval/run.py: prompt.py, temperature 0,
# 4,096 max tokens, same grader as base/SFT). (results id, label, size, thinking level if not off, logo)
FRONTIER = [
    ("muse-spark-open", "Muse Spark Open", "816B", "low", "meta-color"),
    ("qwen3p8-max", "Qwen 3.8 Max", "2.4T", None, "qwen-color"),
    ("glm-5p3", "GLM-5.3", "743B", "low", "zai"),
    ("deepseek-v4p1-flash-1017", "DeepSeek V4.1 Flash 1017", "755B", None, "deepseek-color"),
    ("kimi-k3", "Kimi K3", "2.8T", None, "kimi"),  # kimi-color is near-white on the light badge
]
OUR_ICON = "qwen-color"
ICONS = "https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@1.95.1/icons/{}.svg"  # LobeHub Icons, pinned
# Published COBOLEval pass@1, each from its own harness (COBOL-Coder arXiv 2604.03986 Table 3;
# zorse-project/COBOLEval README)
PUBLISHED = [
    ("COBOL-Coder 14B", 49.33, "paper"), ("COBOL-Coder 7B", 44.70, "paper"), ("GPT-4o", 16.40, "paper"),
    ("GPT-4", 15.75, "paper"), ("GPT-4", 10.27, "COBOLEval README"), ("mAInframer 13B", 11.64, "paper"),
    ("gpt-oss-120b", 4.11, "paper"), ("Qwen2.5-Coder 14B", 2.74, "paper"),
]


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def results(system):
    return {r["id"]: r for r in rows(os.path.join(ROOT, f"results/{system}.jsonl"))}


def system_stats(res):
    rs = list(res.values())
    out = {"pass": 0, "wrong answer": 0, "compile error": 0, "blocked": 0}
    for r in rs:
        out[outcome(r)] += 1
    return {"n": len(rs), "passed": out["pass"], "pass_rate": out["pass"] / len(rs),
            "compiled": sum(r["compiled"] for r in rs) / len(rs), "cost": sum(r["cost"] for r in rs),
            "p50_latency_s": statistics.median(r["latency_s"] for r in rs), "outcomes": out}


def paired(a, b):
    ids = sorted(set(a) & set(b))
    pa, pb = [int(a[i]["pass"]) for i in ids], [int(b[i]["pass"]) for i in ids]
    n = len(ids)
    only_b = sum(1 for x, y in zip(pa, pb) if y and not x)
    only_a = sum(1 for x, y in zip(pa, pb) if x and not y)
    rng = random.Random(SEED)
    diffs = []
    for _ in range(ITERS):
        s = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(pb[i] - pa[i] for i in s) / n)
    diffs.sort()
    return {"n": n, "diff": (sum(pb) - sum(pa)) / n, "ci": [diffs[int(0.025 * ITERS)], diffs[int(0.975 * ITERS) - 1]],
            "b_only": only_b, "a_only": only_a, "p": mcnemar_exact(only_b, only_a),
            "cells": [("both" if x and y else "new" if y else "lost" if x else "none") for x, y in zip(pa, pb)],
            "ids": ids}


def error_kind(r):
    e = r.get("error") or ""
    if r["compiled"] or r["pass"] or e == "blocked":
        return None
    m = re.search(r"\.cbl:\d+: error: ([^\n|]*)", e)
    msg = m.group(1) if m else ""
    if "MOD" in msg and "not defined" in msg:
        return "MOD used as an operator"
    if "syntax error" in msg:
        return "syntax error"
    if "subscript" in msg:
        return "missing subscript"
    if "not defined" in msg or ("FUNCTION" in msg and "unknown" in msg):
        return "unknown name or FUNCTION"
    return "other"


def error_kinds(res):
    out = {}
    for r in res.values():
        k = error_kind(r)
        if k:
            out[k] = out.get(k, 0) + 1
    return out


def funnel():
    teacher = rows(os.path.join(ROOT, "data/teacher_samples.jsonl"))
    problems = rows(os.path.join(ROOT, "data/mbpp_cobol.jsonl"))
    train = rows(os.path.join(ROOT, "data/train.jsonl"))
    return [
        ["MBPP Python problems", sum(1 for _ in open(os.path.join(ROOT, "data/raw/mbpp.jsonl")))],
        ["converted to COBOL", len(problems)],
        ["Kimi K3 programs", len(teacher)],
        ["compile", sum(1 for t in teacher if t["grade"]["compiled"])],
        ["pass every test", sum(1 for t in teacher if t["grade"]["passed"])],
        ["kept for SFT", len(train)],
    ], sum(len(p["tests"]) for p in problems), len({t["id"].split("#")[0] for t in train})


TRAIN_RATE = 4.103  # Serverless Training, Qwen 3.8 27B, $/1M train tokens (docs, 2026-11-02)


def sft_loss():
    """Per-step CE loss and total trained tokens from the committed SFT launch log."""
    logs = sorted(glob.glob(os.path.join(ROOT, "sft/logs/launch-*.log")))
    pts, tokens = [], 0
    for line in open(logs[-1]):
        if line.startswith('{"type": "metrics"'):
            m = json.loads(line)
            pts.append([m["step"], round(m["ce_loss"], 5)])
        elif m := re.search(r"Step \d+/\d+ \| .*tokens: (\d+)", line):
            tokens += int(m.group(1))
    return pts, tokens


def grpo():
    pool = rows(os.path.join(ROOT, "rl/pool.jsonl")) if os.path.exists(os.path.join(ROOT, "rl/pool.jsonl")) else []
    runs = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "rl/runs/*/metrics.jsonl"))):
        name = os.path.basename(os.path.dirname(path))
        curve = [[m["step"], round(m["rollout/raw_reward"], 4)] for m in rows(path) if "rollout/raw_reward" in m]
        if curve:
            runs[name] = curve
    def summary(name):
        path = os.path.join(ROOT, f"rl/runs/{name}/summary.json")
        return json.load(open(path)) if name and os.path.exists(path) else None

    main = max((k for k in runs if k != "smoke"), key=lambda k: len(runs[k]), default=None)
    return {"pool": len(pool), "runs": runs, "main": main, "smoke": summary("smoke"), "summary": summary(main)}


KEYWORDS = set("""IDENTIFICATION DIVISION PROGRAM-ID ENVIRONMENT INPUT-OUTPUT SECTION DATA WORKING-STORAGE
LINKAGE PROCEDURE USING IF ELSE END-IF MOVE TO GOBACK END PROGRAM PERFORM VARYING FROM BY UNTIL
END-PERFORM COMPUTE FUNCTION AND OR NOT OCCURS TIMES INDEXED PIC VALUE""".split())


def cobol_html(code, mark=None):
    """Minimal COBOL highlighter: comments ('*' indicator), keywords, names, literals.

    The shared left margin (COBOL's sequence area) is removed so code fits on a phone."""
    raw = code.rstrip("\n").split("\n")
    pad = min(len(l) - len(l.lstrip()) for l in raw if l.strip())
    lines = []
    for line in raw:
        line = line[pad:]
        if line.lstrip().startswith("*"):
            body = f'<span class="c">{html.escape(line)}</span>'
        else:
            parts = []
            for tok in re.findall(r"'[^']*'|\"[^\"]*\"|[A-Za-z0-9][A-Za-z0-9-]*(?:\([0-9]+\))?|\s+|.", line):
                t = html.escape(tok)
                if tok.upper() in KEYWORDS:
                    parts.append(f'<span class="k">{t}</span>')
                elif re.match(r"^(L-|WS-|RESULT\b)", tok):
                    parts.append(f'<span class="v">{t}</span>')
                elif re.match(r"^S?9|^X\(|^\d", tok):
                    parts.append(f'<span class="n">{t}</span>')
                else:
                    parts.append(t)
            body = "".join(parts)
        cls = ' class="hl"' if mark and mark in line else ""
        lines.append(f"<span{cls}>{body or ' '}</span>")
    return "".join(lines)  # spans are display:block; a newline between them would add blank lines in <pre>


def strip_fence(text):
    m = re.search(r"```(?:cobol)?\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def procedure(code):
    lines = code.rstrip().split("\n")
    i = next(k for k, l in enumerate(lines) if l.strip().startswith("PROCEDURE DIVISION"))
    return "\n".join(lines[i:])


def example():
    safe = re.sub(r"[^A-Za-z0-9]+", "_", EXAMPLE)
    prob = next(p for p in rows(os.path.join(ROOT, "eval/data/CobolEval.jsonl")) if p["task_id"] == EXAMPLE)
    base = strip_fence(json.load(open(os.path.join(ROOT, f"eval/cache/{BASE}/{safe}.json")))["text"])
    sft = strip_fence(json.load(open(os.path.join(ROOT, f"eval/cache/{SFT}/{safe}.json")))["text"])
    err = results(BASE)[EXAMPLE]["error"].replace("compile: ", "").split("\n")
    prompt = "\n".join(l for l in prob["prompt"].rstrip().split("\n"))
    return {
        "prompt": cobol_html(prompt),
        "base": cobol_html(procedure(base), mark=" MOD "),
        "sft": cobol_html(procedure(sft), mark="FUNCTION MOD"),
        "error": html.escape("\n".join(err[:4])),
        "tests": results(SFT)[EXAMPLE]["n_tests"],
    }


def harness(sft, systems_stats):
    """Same-harness leaderboard: ours + every frontier model with all 146 rows, paired vs SFT."""
    fix_path = os.path.join(ROOT, "results/section_fix.json")
    fix = json.load(open(fix_path)) if os.path.exists(fix_path) else {}
    out = []
    for sid, label, size, think, icon in FRONTIER:
        path = os.path.join(ROOT, f"results/{sid}.jsonl")
        if not os.path.exists(path):
            continue
        res = results(sid)
        if len(res) < len(sft):
            continue
        st, pr = system_stats(res), paired(sft, res)
        entry = {"label": label, "size": size, "think": think, "icon": ICONS.format(icon), "outcomes": st["outcomes"],
                 "n": st["n"], "compiled": st["compiled"], "pass": 100 * st["pass_rate"],
                 "fixed": 100 * fix.get(sid, {}).get("fixed", st["passed"]) / st["n"],
                 "cost": st["cost"] / st["n"], "diff": 100 * pr["diff"], "p": pr["p"], "ours": False,
                 "p_fix": None, "diff_fix": None}
        if sid in fix and SFT in fix:  # same McNemar, both sides scored with the section-order fix
            o, s = set(fix[sid]["fixed_ids"]), set(fix[SFT]["fixed_ids"])
            entry["p_fix"], entry["diff_fix"] = mcnemar_exact(len(o - s), len(s - o)), 100 * (len(o) - len(s)) / st["n"]
        out.append(entry)
    for s, st in systems_stats.items():  # ours; the fine-tuned ones are highlighted, base stays muted
        label = LABELS[s] if s == BASE else "Qwen 3.8 27B " + LABELS[s]
        out.append({"label": label, "size": "27B", "think": None, "icon": ICONS.format(OUR_ICON), "outcomes": st["outcomes"],
                    "n": st["n"], "compiled": st["compiled"], "pass": 100 * st["pass_rate"],
                    "fixed": 100 * fix.get(s, {}).get("fixed", st["passed"]) / st["n"],
                    "cost": st["cost"] / st["n"], "diff": None, "p": None, "ours": True, "hl": s != BASE})
    for e in out:
        e["params"] = size_b(e)
    return out


def size_b(entry):
    """'743B' / '2.4T' -> billions, for ordering."""
    return float(entry["size"][:-1]) * (1000 if entry["size"].endswith("T") else 1)


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def sci(p):
    m, e = f"{p:.1e}".split("e")
    return f"{m}×10<sup>{int(e)}</sup>"


def main():
    base, sft = results(BASE), results(SFT)
    systems = [BASE, SFT]
    grpo_files = sorted(glob.glob(os.path.join(ROOT, "results/qwen38-27b-grpo*.jsonl")))
    if grpo_files:
        g = os.path.basename(grpo_files[-1])[:-6]
        systems.append(g)
        LABELS[g] = "+ SFT + GRPO"
    res = {s: results(s) for s in systems}
    stats = {s: system_stats(res[s]) for s in systems}
    pair = paired(base, sft)
    steps, n_tests, n_train_problems = funnel()
    loss, sft_tokens = sft_loss()
    rl = grpo()
    ex = example()
    grpo_pair = paired(sft, res[systems[2]]) if grpo_files else None
    board = harness(sft, stats)
    others = [b for b in board if not b["ours"]]
    ahead = [b["label"] for b in others if b["p"] < 0.05 and b["diff"] > 0]
    tied = [b["label"] for b in others if b["p"] >= 0.05]
    behind = [b["label"] for b in others if b["p"] < 0.05 and b["diff"] < 0]
    fix_ahead = [b["label"] for b in others if b["label"] not in ahead
                 and b["p_fix"] is not None and b["p_fix"] < 0.05 and b["diff_fix"] > 0]
    fix_note = (f" With COBOLEval's section-order fix (hollow rings), {' and '.join(fix_ahead)} also move significantly ahead."
                if fix_ahead else "")

    data = {
        "systems": [{"id": s, "label": LABELS[s], **stats[s]} for s in systems],
        "waffle": [{"id": i, "state": c} for i, c in zip(pair["ids"], pair["cells"])],
        "errors": {LABELS[s]: error_kinds(res[s]) for s in systems},
        "funnel": steps,
        "harness": board,
        "loss": loss,
        "grpo": rl,
        "grpo_eval": ({k: v for k, v in grpo_pair.items() if k not in ("cells", "ids")} if grpo_pair else None),
        "published": [[f"{n} ({src})" if src != "paper" else n, v, src] for n, v, src in PUBLISHED]
        + [["Qwen 3.8 27B + SFT (ours)", round(100 * stats[SFT]["pass_rate"], 2), "ours"],
           ["Qwen 3.8 27B base (ours)", round(100 * stats[BASE]["pass_rate"], 2), "ours"]],
    }

    sb, ss = stats[BASE], stats[SFT]
    loss_first = statistics.mean(v for _, v in loss[:5])
    loss_last = statistics.mean(v for _, v in loss[-10:])
    smoke = rl["smoke"] or {}
    smoke_curve = rl["runs"].get("smoke", [[0, 0], [0, 0]])
    run = rl["summary"]
    if run:  # finished GRPO run: its size and cost
        grpo_t2 = (f"<b>{len(rl['runs'][rl['main']])} steps</b>"
                   f"<span>{run['rollouts']:,} rollouts · {run['minutes']:.0f} min</span>")
        grpo_t3 = f"<b>${run['est_cost_usd_upper']:.2f}</b><span>training + rollouts (upper bound)</span>"
    else:  # only the smoke test so far
        grpo_t2 = (f"<b>{smoke_curve[0][1]:.2f} → {smoke_curve[-1][1]:.2f}</b>"
                   "<span>mean reward, 2-step smoke test</span>")
        grpo_t3 = f"<b>${smoke.get('est_cost_usd_upper', 1.0) / 2:.2f}</b><span>per step · 64 rollouts</span>"
    if grpo_files:
        gs = stats[systems[2]]
        grpo_status = (f"<b>{pct(gs['pass_rate'])}%</b><span>GRPO pass@1 ({gs['passed']}/{gs['n']}); "
                       f"{grpo_pair['diff'] * 100:+.1f} pts vs SFT, McNemar p = {grpo_pair['p']:.2g}</span>")
    elif run:
        grpo_status = "<b>eval next</b><span>COBOLEval score lands here on the next push</span>"
    else:
        grpo_status = "<b>running</b><span>results land here on the next push</span>"
    vals = {
        "base_pct": pct(sb["pass_rate"]), "sft_pct": pct(ss["pass_rate"]),
        "base_passed": sb["passed"], "sft_passed": ss["passed"], "n": sb["n"],
        "base_compile": pct(sb["compiled"], 0), "sft_compile": pct(ss["compiled"], 0),
        "diff": f"{100 * pair['diff']:+.1f}", "ci_lo": f"{100 * pair['ci'][0]:+.1f}", "ci_hi": f"{100 * pair['ci'][1]:+.1f}",
        "b_only": pair["b_only"], "a_only": pair["a_only"], "p_html": sci(pair["p"]),
        "base_cost": f"{sb['cost']:.2f}", "sft_cost": f"{ss['cost']:.2f}",
        "both": pair["cells"].count("both"), "neither": pair["cells"].count("none"),
        "mbpp": steps[0][1], "problems": steps[1][1], "mbpp_tests": f"{n_tests:,}",
        "teacher_n": steps[2][1], "teacher_pass": steps[4][1], "teacher_pass_pct": pct(steps[4][1] / steps[2][1], 0),
        "train_rows": steps[5][1], "train_problems": n_train_problems,
        "sft_steps": loss[-1][0], "loss_first": f"{loss_first:.3f}", "loss_last": f"{loss_last:.3f}",
        "sft_tokens": f"{sft_tokens / 1e6:.2f}M", "sft_cost": f"{sft_tokens * TRAIN_RATE / 1e6:.2f}",
        "pool": rl["pool"], "grpo_t2": grpo_t2, "grpo_t3": grpo_t3, "grpo_status": grpo_status,
        "ex_prompt": ex["prompt"], "ex_base": ex["base"], "ex_sft": ex["sft"], "ex_error": ex["error"],
        "ex_tests": ex["tests"], "ex_id": EXAMPLE,
        "n_frontier": len(others), "n_tied": len(tied), "n_ahead": len(ahead), "n_behind": len(behind),
        "tied_names": ", ".join(tied) or "none", "ahead_names": ", ".join(ahead) or "none",
        "behind_names": ", ".join(behind) or "none", "fix_note": fix_note,
        "frontier_sizes": f"{min(others, key=size_b)['size']}–{max(others, key=size_b)['size']}" if others else "",
        "sft_cost_pp": f"{ss['cost'] / ss['n']:.4f}",
    }

    page = open(os.path.join(SITE, "index.html")).read()
    page = re.sub(r"\{\{(\w+)\}\}", lambda m: str(vals[m.group(1)]), page)
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    open(os.path.join(OUT, "index.html"), "w").write(page)
    for f in ("style.css", "app.js", "favicon.svg"):
        shutil.copy(os.path.join(SITE, f), OUT)
    with open(os.path.join(OUT, "data.js"), "w") as f:
        f.write("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"built {OUT}: base {vals['base_pct']}% -> SFT {vals['sft_pct']}%"
          f"{', GRPO ' + pct(stats[systems[2]]['pass_rate']) + '%' if grpo_files else ''}; "
          f"{len(loss)} loss points; GRPO runs {list(rl['runs'])}")


if __name__ == "__main__":
    main()
