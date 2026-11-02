"""Run one system on COBOLEval and append one row per problem to results/{system}.jsonl.

    python -m eval.run --system kimi-k3 --subset 30
    python -m eval.run --system qwen3p8-27b-base            # all 146
    python -m eval.run --summary kimi-k3 glm-5p3 --subset 30

Rules (CLAUDE.md): temperature 0, max_tokens 4096, prompt.py for every system, async with
bounded concurrency, retries with backoff on 429/5xx, every raw response cached to
results/raw/{system}/ and never re-requested; ids already in results/ are skipped.
Sampler systems need the fireworks-ai[training] SDK and the cookbook `training` package.
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.systems import SYSTEMS  # noqa: E402
from grader.construct import construct  # noqa: E402
from grader.grade import grade  # noqa: E402
from prompt import build_messages  # noqa: E402

TEMPERATURE = 0.0
MAX_TOKENS = 4096
CONCURRENCY = 16
GRADE_WORKERS = max(1, (os.cpu_count() or 4) - 1)
CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
SAMPLER_URL = "https://api.fireworks.ai/training/v1/serverless"
DATA = ROOT / "eval/data/CobolEval.jsonl"
SUBSET30 = ROOT / "eval/data/subset30.json"
RESULTS = ROOT / "results"


def load_env():
    env = ROOT / ".env"
    if "FIREWORKS_API_KEY" not in os.environ and env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def load_problems(subset):
    problems = [json.loads(line) for line in DATA.open()]
    if not subset:
        return problems
    if subset != 30:
        raise SystemExit("only --subset 30 (fixed ids) is supported")
    if not SUBSET30.exists():  # fixed seed, written once and committed
        ids = sorted(random.Random(0).sample([p["task_id"] for p in problems], 30), key=lambda t: int(t.split("/")[1]))
        SUBSET30.write_text(json.dumps(ids, indent=1) + "\n")
    keep = set(json.loads(SUBSET30.read_text()))
    return [p for p in problems if p["task_id"] in keep]


def safe(task_id):
    return task_id.replace("/", "_")


def category(text, truncated, g):
    if g["passed"]:
        return "pass"
    has_fence = "```" in text
    looks_cobol = "IDENTIFICATION DIVISION" in text.upper() or "PROCEDURE DIVISION" in text.upper()
    if truncated and not (has_fence and looks_cobol):
        return "truncated"
    if not looks_cobol:
        return "no_code"
    if g["error"] == "blocked":
        return "blocked"
    if not g["compiled"]:
        return "compile_error"
    return "wrong_answer"


# ---------- model calls (each returns a raw dict that is cached verbatim) ----------

async def call_chat(cfg, messages, client):
    body = {"model": cfg["model"], "messages": messages, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
    if cfg.get("reasoning_effort") is not None:
        body["reasoning_effort"] = cfg["reasoning_effort"]
    headers = {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}
    for attempt in range(7):
        t = time.time()
        try:
            r = await client.post(CHAT_URL, json=body, headers=headers, timeout=600)
        except Exception as e:  # network error: retry
            err = f"{type(e).__name__}: {e}"
        else:
            if r.status_code == 200:
                d = r.json()
                ch = d["choices"][0]
                u = d.get("usage") or {}
                return {
                    "text": ch["message"].get("content") or "",
                    "reasoning_chars": len(ch["message"].get("reasoning_content") or ""),
                    "finish_reason": ch.get("finish_reason"),
                    "prompt_tokens": u.get("prompt_tokens", 0),
                    "completion_tokens": u.get("completion_tokens", 0),
                    "latency_s": time.time() - t,
                    "request": {k: v for k, v in body.items() if k != "messages"},
                }
            err = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code not in (408, 409, 425, 429) and r.status_code < 500:
                raise RuntimeError(err)  # 4xx config error: don't retry, don't cache
        await asyncio.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
    raise RuntimeError(f"gave up after retries: {err}")


class Sampler:
    """Serverless-training sampler; blocking SDK calls are run in threads."""

    def __init__(self, cfg):
        from fireworks.training import sdk
        from fireworks.training.sdk import FiretitanServiceClient
        from training.renderer import get_renderer, get_text_content
        from training.renderer.tokenizer import get_tokenizer

        self.get_text_content = get_text_content
        self.tokenizer = get_tokenizer(cfg["tokenizer"])
        self.renderer = get_renderer(cfg["renderer"], self.tokenizer)
        self.service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url=SAMPLER_URL)
        src = {"model_path": cfg["model_path"]} if cfg.get("model_path") else {"base_model": cfg["base_model"]}
        self.client = self.service.create_sampling_client(tokenizer=self.tokenizer, **src)
        self.params = sdk.SamplingParams(
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stop=self.renderer.get_stop_sequences()
        )
        self.cfg = cfg

    def _sample(self, messages):
        prompt = self.renderer.build_generation_prompt(messages)
        last = None
        for attempt in range(5):
            t = time.time()
            try:
                res = self.client.sample(prompt=prompt, num_samples=1, sampling_params=self.params).result()
                break
            except Exception as e:
                last = e
                time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
        else:
            raise RuntimeError(f"gave up after retries: {last}")
        out = list(res.sequences[0].tokens)
        msg, term = self.renderer.parse_response(out)
        return {
            "text": self.get_text_content(msg),
            "finish_reason": "stop" if term.is_clean else "length",
            "prompt_tokens": len(prompt.to_ints()),
            "completion_tokens": len(out),
            "latency_s": time.time() - t,
            "request": {"renderer": self.cfg["renderer"], "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                        **{k: self.cfg[k] for k in ("base_model", "model_path") if self.cfg.get(k)}},
        }

    async def __call__(self, messages, _client=None):
        return await asyncio.to_thread(self._sample, messages)

    def close(self):
        for c in (self.client, self.service):
            try:
                c.close()
            except Exception:
                pass


# ---------- main loop ----------

async def run(system, subset):
    cfg = SYSTEMS[system]
    problems = load_problems(subset)
    out_path = RESULTS / f"{system}.jsonl"
    raw_dir = RESULTS / "raw" / system
    raw_dir.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in out_path.open()} if out_path.exists() else set()
    todo = [p for p in problems if p["task_id"] not in done]
    print(f"{system}: {len(problems)} problems, {len(done & {p['task_id'] for p in problems})} already done, {len(todo)} to run")
    if not todo:
        return

    import httpx

    sem, grade_sem = asyncio.Semaphore(CONCURRENCY), asyncio.Semaphore(GRADE_WORKERS)
    lock = asyncio.Lock()
    sampler = Sampler(cfg) if cfg["kind"] == "sampler" and any(
        not (raw_dir / f"{safe(p['task_id'])}.json").exists() for p in todo) else None
    failures = []

    async def one(p, client):
        raw_path = raw_dir / f"{safe(p['task_id'])}.json"
        if raw_path.exists():
            raw = json.loads(raw_path.read_text())  # cached: never re-call the model
        else:
            messages = build_messages(p["prompt"])
            async with sem:
                try:
                    raw = await (sampler(messages) if sampler else call_chat(cfg, messages, client))
                except Exception as e:
                    failures.append((p["task_id"], str(e)[:300]))
                    return
            raw_path.write_text(json.dumps(raw))
        async with grade_sem:
            g = await asyncio.to_thread(grade, construct(p["prompt"], raw["text"]), p)
        truncated = raw.get("finish_reason") == "length"
        cost = (raw["prompt_tokens"] * cfg["price_in"] + raw["completion_tokens"] * cfg["price_out"]) / 1e6
        row = {
            "id": p["task_id"], "system": system, "pass": bool(g["passed"]), "compiled": bool(g["compiled"]),
            "tokens": raw["prompt_tokens"] + raw["completion_tokens"],
            "latency_s": round(raw["latency_s"], 3), "cost": round(cost, 6),
            # extra fields (additive; contract fields above unchanged)
            "prompt_tokens": raw["prompt_tokens"], "completion_tokens": raw["completion_tokens"],
            "n_pass": g["n_pass"], "n_tests": g["n_tests"], "truncated": truncated,
            "category": category(raw["text"], truncated, g), "error": (g["error"] or "")[:500],
        }
        async with lock:
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    try:
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=CONCURRENCY)) as client:
            await asyncio.gather(*(one(p, client) for p in todo))
    finally:
        if sampler:
            sampler.close()
    for tid, err in failures:
        print(f"  FAILED {tid}: {err}")


def summarize(systems, subset):
    ids = {p["task_id"] for p in load_problems(subset)}
    hdr = f"{'system':24s} {'n':>4s} {'pass@1':>7s} {'compile':>8s} {'$/problem':>10s} {'p50 lat':>8s} {'p50 out tok':>11s}  failures"
    print(hdr)
    for s in systems:
        path = RESULTS / f"{s}.jsonl"
        rows = [r for r in map(json.loads, path.open()) if r["id"] in ids] if path.exists() else []
        if not rows:
            print(f"{s:24s} (no rows)")
            continue
        n = len(rows)
        cats = {}
        for r in rows:
            cats[r.get("category", "?")] = cats.get(r.get("category", "?"), 0) + 1
        print(f"{s:24s} {n:4d} {sum(r['pass'] for r in rows)/n:7.1%} {sum(r['compiled'] for r in rows)/n:8.1%} "
              f"{statistics.mean(r['cost'] for r in rows):10.5f} {statistics.median(r['latency_s'] for r in rows):7.1f}s "
              f"{statistics.median(r['completion_tokens'] for r in rows):11.0f}  "
              + ", ".join(f"{k}={v}" for k, v in sorted(cats.items()) if k != "pass"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=sorted(SYSTEMS))
    ap.add_argument("--subset", type=int, default=0, help="30 = fixed 30-problem check; 0 = all 146")
    ap.add_argument("--summary", nargs="*", help="print summary table for these systems")
    a = ap.parse_args()
    load_env()
    if a.system:
        asyncio.run(run(a.system, a.subset))
    if a.summary is not None:
        summarize(a.summary or [a.system], a.subset)


if __name__ == "__main__":
    main()
