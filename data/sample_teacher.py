"""Rejection sampling: teacher writes K samples per MBPP problem, grade() keeps the passers.

Every API response is cached in data/cache/<model>/ (never re-called). Re-running with a larger
--samples only fetches the new sample indices; --only-unsolved skips problems that already have
--keep passing programs. Outputs:
  data/teacher_samples.jsonl  every sample with its grade (failures = DPO rejects / RL pool)
  data/train.jsonl            {id, messages: [system, user, assistant]}, <= --keep per problem

Run: .venv/bin/python -m data.sample_teacher --samples 2
"""

import argparse
import asyncio
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from grader.construct import construct
from grader.grade import grade
from prompt import build_messages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://api.fireworks.ai/inference/v1/chat/completions"
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")


def cache_path(model, task_id, k):
    d = os.path.join(ROOT, "data/cache", model.split("/")[-1])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{re.sub(r'[^A-Za-z0-9]+', '_', task_id)}_{k}.json")


async def call(client, sem, args, problem, k):
    path = cache_path(args.model, problem["task_id"], k)
    if os.path.exists(path):
        return json.load(open(path))
    body = {"model": args.model, "messages": build_messages(problem["prompt"]),
            "temperature": args.temperature, "max_tokens": args.max_tokens}
    async with sem:
        for attempt in range(30):
            t = time.time()
            try:
                r = await client.post(URL, json=body)
                if r.status_code == 429:  # generated-token rate limit: wait as told, don't count toward giving up
                    await asyncio.sleep(float(r.headers.get("retry-after", 5)) + random.random() * 3)
                    continue
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                r.raise_for_status()
                d = r.json()
                c = d["choices"][0]
                row = {"task_id": problem["task_id"], "k": k, "model": args.model,
                       "content": c["message"].get("content") or "", "finish_reason": c.get("finish_reason"),
                       "usage": d.get("usage"), "latency_s": round(time.time() - t, 2)}
                json.dump(row, open(path, "w"))
                return row
            except (httpx.HTTPError, KeyError, ValueError) as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    print(f"  {problem['task_id']}#{k}: HTTP {status} {e.response.text[:200]}")
                    return None
                await asyncio.sleep(min(60, 2 ** attempt + random.random()))
        print(f"  {problem['task_id']}#{k}: gave up after retries")
        return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="accounts/fireworks/models/kimi-k3")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--only-unsolved", action="store_true")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    problems = [json.loads(l) for l in open(os.path.join(ROOT, "data/mbpp_cobol.jsonl"))]
    if args.limit:
        problems = problems[: args.limit]
    by_id = {p["task_id"]: p for p in problems}

    samples_path = os.path.join(ROOT, "data/teacher_samples.jsonl")
    graded = {}
    if os.path.exists(samples_path):
        for l in open(samples_path):
            s = json.loads(l)
            graded[(s["task_id"], s["k"])] = s

    def n_passed(tid):
        return sum(1 for (t, _), s in graded.items() if t == tid and s["grade"]["passed"])

    todo = [(p, k) for p in problems for k in range(args.samples)
            if (p["task_id"], k) not in graded
            and not (args.only_unsolved and n_passed(p["task_id"]) >= args.keep)]
    print(f"{len(problems)} problems, {len(todo)} samples to fetch+grade (model {args.model})")

    sem = asyncio.Semaphore(args.concurrency)
    pool = ThreadPoolExecutor(max(2, (os.cpu_count() or 4) - 1))
    loop = asyncio.get_running_loop()
    headers = {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}
    t0, done = time.time(), 0

    async def one(client, p, k):
        nonlocal done
        row = await call(client, sem, args, p, k)
        if row is None:
            return
        program = construct(p["prompt"], row["content"])
        row["program"] = program
        row["grade"] = await loop.run_in_executor(pool, grade, program, p)
        graded[(p["task_id"], k)] = row
        done += 1
        if done % 50 == 0:
            ok = sum(s["grade"]["passed"] for s in graded.values())
            print(f"  {done}/{len(todo)} done, {ok}/{len(graded)} samples pass, {time.time() - t0:.0f}s")

    async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(900.0)) as client:
        await asyncio.gather(*(one(client, p, k) for p, k in todo))

    rows = sorted(graded.values(), key=lambda s: (s["task_id"], s["k"]))
    with open(samples_path, "w") as f:
        for s in rows:
            f.write(json.dumps(s) + "\n")

    train, seen = [], {}
    for s in rows:
        p = by_id.get(s["task_id"])
        if not p or not s["grade"]["passed"]:
            continue
        progs = seen.setdefault(s["task_id"], set())
        norm = re.sub(r"\s+", " ", s["program"]).strip()
        if len(progs) >= args.keep or norm in progs:
            continue
        progs.add(norm)
        answer = "```cobol\n" + s["program"].strip("\n") + "\n```"
        train.append({"id": f"{s['task_id']}#{s['k']}",
                      "messages": build_messages(p["prompt"]) + [{"role": "assistant", "content": answer}]})
    with open(os.path.join(ROOT, "data/train.jsonl"), "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")

    n = len(rows)
    print(f"samples {n}: compiled {sum(s['grade']['compiled'] for s in rows)}, "
          f"passed {sum(s['grade']['passed'] for s in rows)}, "
          f"truncated {sum(s['finish_reason'] == 'length' for s in rows)}")
    print(f"problems solved {len(seen)}/{len(problems)} -> data/train.jsonl {len(train)} examples")


if __name__ == "__main__":
    asyncio.run(main())
