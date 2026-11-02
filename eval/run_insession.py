"""COBOLEval for Qwen 3.8 27B base vs SFT, both sampled in one Serverless Training session.

Qwen 3.8 27B has no serverless inference endpoint, so both systems go through the training
sampler with identical settings: renderer `qwen3_8_disable_thinking_interleaved` (thinking off),
temperature 0, max_tokens 4096, seed 0, prompt.build_messages. Base = the base model's sampler;
SFT = the final SFT training checkpoint loaded into this session, saved for sampling.

Raw outputs are cached in eval/cache/<system>/; results/<system>.jsonl gets one row per problem
({id, system, pass, compiled, tokens, latency_s, cost} + n_pass, n_tests, stop_reason, error).
Ids already in results/ are never re-sampled.

Run: .venv-train/bin/python -m eval.run_insession [--limit N] [--systems base,sft]
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get(
    "FW_COOKBOOK", os.path.expanduser("~/.claude/plugins/cache/fw-ai-cookbook/fireworks-training/2.3.0")))
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

from grader.construct import construct  # noqa: E402
from grader.grade import grade  # noqa: E402
from prompt import build_messages  # noqa: E402

BASE_MODEL = "accounts/fireworks/models/qwen3p8-27b"
TOKENIZER = "Qwen/Qwen3.8-27B"
RENDERER = "qwen3_8_disable_thinking_interleaved"
SFT_STATE = "ritik-7xa24uxitcg9/run-94dffbdc3914486d9d362a614f824d0c/step-90"
SYSTEMS = {"base": "qwen38-27b-base", "sft": "qwen38-27b-sft", "base-think": "qwen38-27b-base-think"}
# base-think: context row only. Base model with thinking ON (cookbook default renderer `qwen3_8`, the HF
# serving default) and a large budget so reasoning isn't cut off; passes using <= 4096 completion
# tokens are what the standard 4096 budget would give (temperature 0).
THINK_RENDERER, THINK_MAX_TOKENS = "qwen3_8", 32768
MAX_TOKENS, TEMPERATURE, SEED, CONCURRENCY = 4096, 0.0, 0, 16
# Serverless Training sampler rates, $/1M tokens (docs, 2026-11-02); prefill billed uncached (upper bound)
PREFILL_RATE, SAMPLE_RATE = 1.86, 5.595


def safe(task_id):
    return re.sub(r"[^A-Za-z0-9]+", "_", task_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--systems", default="base,sft")
    ap.add_argument("--state", action="append", default=[], metavar="NAME=CHECKPOINT",
                    help="extra tuned system scored like sft, e.g. grpo=<account>/<run>/<step>; "
                         "rows go to results/qwen38-27b-NAME.jsonl")
    args = ap.parse_args()
    states = {"sft": SFT_STATE}
    for spec in args.state:
        name, ref = spec.split("=", 1)
        SYSTEMS[name] = f"qwen38-27b-{name}"
        states[name] = ref

    from fireworks.training import tinker
    from fireworks.training.sdk import FiretitanServiceClient
    from training.renderer import get_text_content
    from training.utils.supervised import build_renderer
    from training.utils.tokenizers import load_tokenizer

    problems = [json.loads(l) for l in open(os.path.join(ROOT, "eval/data/CobolEval.jsonl"))]
    if args.limit:
        problems = problems[: args.limit]
    tok = load_tokenizer(TOKENIZER)

    service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                     base_url="https://api.fireworks.ai/training/v1/serverless")
    try:
        for key in args.systems.split(","):
            system = SYSTEMS[key]
            out_path = os.path.join(ROOT, f"results/{system}.jsonl")
            cache_dir = os.path.join(ROOT, f"eval/cache/{system}")
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            done = set()
            if os.path.exists(out_path):
                done = {json.loads(l)["id"] for l in open(out_path)}
            todo = [p for p in problems if p["task_id"] not in done]
            print(f"[{system}] {len(todo)} to run ({len(done)} already in results)", flush=True)
            if not todo:
                continue

            think = key == "base-think"
            renderer = build_renderer(tok, TOKENIZER, THINK_RENDERER if think else RENDERER)
            params = tinker.SamplingParams(max_tokens=THINK_MAX_TOKENS if think else MAX_TOKENS,
                                           temperature=TEMPERATURE, seed=SEED, stop=renderer.get_stop_sequences())
            if key in ("base", "base-think"):
                sampler = service.create_sampling_client(base_model=BASE_MODEL, tokenizer=tok)
            else:
                tc = service.create_training_client_from_state(states[key])
                path = tc.save_weights_for_sampler(f"eval-{key}"[:17]).result().path
                print(f"[{system}] loaded {states[key]} -> sampler {path}", flush=True)
                sampler = service.create_sampling_client(model_path=path, tokenizer=tok, training_client=tc)

            def run(p):
                cache = os.path.join(cache_dir, f"{safe(p['task_id'])}.json")
                if os.path.exists(cache):
                    c = json.load(open(cache))
                else:
                    prompt = renderer.build_generation_prompt(build_messages(p["prompt"]))
                    t = time.time()
                    resp = sampler.sample(prompt, num_samples=1, sampling_params=params).result(timeout=900)
                    seq = resp.sequences[0]
                    msg, _ = renderer.parse_response(list(seq.tokens))
                    raw = tok.decode(list(seq.tokens))
                    c = {"id": p["task_id"], "text": get_text_content(msg), "raw": raw[-6000:] if think else raw,
                         "prompt_tokens": prompt.length, "completion_tokens": len(seq.tokens),
                         "stop_reason": str(seq.stop_reason), "latency_s": round(time.time() - t, 2)}
                    json.dump(c, open(cache, "w"))
                g = grade(construct(p["prompt"], c["text"]), p)
                return {"id": p["task_id"], "system": system, "pass": g["passed"], "compiled": g["compiled"],
                        "tokens": c["prompt_tokens"] + c["completion_tokens"], "latency_s": c["latency_s"],
                        "cost": round((c["prompt_tokens"] * PREFILL_RATE + c["completion_tokens"] * SAMPLE_RATE) / 1e6, 6),
                        "n_pass": g["n_pass"], "n_tests": g["n_tests"], "stop_reason": c["stop_reason"],
                        "error": (g["error"] or "")[:300]}

            n = 0
            with ThreadPoolExecutor(CONCURRENCY) as ex, open(out_path, "a") as f:
                for row in ex.map(run, todo):
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    n += 1
                    if n % 25 == 0:
                        print(f"[{system}] {n}/{len(todo)}", flush=True)
            rows = [json.loads(l) for l in open(out_path)]
            print(f"[{system}] pass@1 {sum(r['pass'] for r in rows)}/{len(rows)}, "
                  f"compiled {sum(r['compiled'] for r in rows)}/{len(rows)}, "
                  f"cost ${sum(r['cost'] for r in rows):.2f}", flush=True)
    finally:
        service.close()


if __name__ == "__main__":
    main()
