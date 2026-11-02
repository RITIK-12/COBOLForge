"""Secondary scoring: re-grade cached outputs with COBOLEval's section-order fix.

The headline metric is strict: grade(construct(prompt, text)), exactly as base and SFT were scored.
Some models (notably Kimi K3) copy the skeleton's order and put WORKING-STORAGE after LINKAGE,
which GnuCOBOL rejects. COBOLEval's own chat harness always runs swap_sections, so this script
also scores every system with swap_sections applied to full programs. No model is called:
it reads the cached responses only.

Run: .venv/bin/python -m eval.regrade_swap qwen38-27b-base qwen38-27b-sft kimi-k3 ...
Writes results/section_fix.json: {system: {n, strict, fixed, fixed_ids}}.
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from grader.construct import construct, extract_code_block, swap_sections  # noqa: E402
from grader.grade import grade  # noqa: E402


def cached_text(system, task_id):
    safe = re.sub(r"[^A-Za-z0-9]+", "_", task_id)
    for d in (f"eval/cache/{system}", f"results/raw/{system}"):  # run_insession.py, run.py
        path = os.path.join(ROOT, d, f"{safe}.json")
        if os.path.exists(path):
            return json.load(open(path))["text"]
    return None


def fixed_program(prompt, text):
    sol = extract_code_block(text)
    return swap_sections(sol) if "IDENTIFICATION DIVISION" in sol.upper() else construct(prompt, text)


def main(systems):
    problems = {json.loads(l)["task_id"]: json.loads(l) for l in open(os.path.join(ROOT, "eval/data/CobolEval.jsonl"))}
    out_path = os.path.join(ROOT, "results/section_fix.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for system in systems:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, f"results/{system}.jsonl"))]

        def one(r):
            text = cached_text(system, r["id"])
            if text is None:
                return r["id"], r["pass"]
            return r["id"], grade(fixed_program(problems[r["id"]]["prompt"], text), problems[r["id"]])["passed"]

        with ThreadPoolExecutor(max(1, (os.cpu_count() or 4) - 1)) as ex:
            fixed = dict(ex.map(one, rows))
        out[system] = {"n": len(rows), "strict": sum(r["pass"] for r in rows), "fixed": sum(fixed.values()),
                       "fixed_ids": sorted(i for i, ok in fixed.items() if ok)}
        print(f"{system:26s} n={len(rows)} strict {out[system]['strict']:3d}  with section fix {out[system]['fixed']:3d}")
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
