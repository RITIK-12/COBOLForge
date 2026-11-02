"""Pick the GRPO problem pool: MBPP problems where the SFT model's rewards vary.

Samples the SFT checkpoint K times per MBPP problem at the GRPO temperature, grades each sample,
and writes rl/pool_samples.jsonl (every sample) and rl/pool.jsonl (problems whose K rewards are
not all equal: always-pass / always-fail groups give GRPO zero advantage). The session it opens
is deleted at the end.

    .venv-fw/bin/python -m rl.select_pool --k 4
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from prompt import build_messages
from rl.common import (PRICE, RENDERER, ROOT, SFT_STATE, TOKENIZER, delete_session,
                       load_problems, reward, use_sft_account_key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    use_sft_account_key()

    import os
    from fireworks.training import sdk
    from fireworks.training.sdk import FiretitanServiceClient
    from training.renderer import get_renderer, get_text_content
    from training.renderer.tokenizer import get_tokenizer

    problems = load_problems()
    out_samples = ROOT / "rl/pool_samples.jsonl"
    done = {json.loads(l)["task_id"] for l in out_samples.open()} if out_samples.exists() else set()
    todo = [p for t, p in problems.items() if t not in done]
    print(f"{len(problems)} MBPP problems, {len(done)} already sampled, {len(todo)} to go", flush=True)

    tok = get_tokenizer(TOKENIZER)
    renderer = get_renderer(RENDERER, tok)
    params = sdk.SamplingParams(max_tokens=a.max_tokens, temperature=a.temperature, stop=renderer.get_stop_sequences())
    service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                     base_url="https://api.fireworks.ai/training/v1/serverless")
    session = None
    tokens = {"prefill": 0, "sample": 0}
    try:
        if todo:
            tc = service.create_training_client_from_state(SFT_STATE)
            session = getattr(service, "training_session_name", None)
            path = tc.save_weights_for_sampler("rl-pool").result().path
            print(f"session {session}; SFT {SFT_STATE} -> sampler {path}", flush=True)
            sampler = service.create_sampling_client(model_path=path, tokenizer=tok, training_client=tc)

            def run(p):
                prompt = renderer.build_generation_prompt(build_messages(p["prompt"]))
                for attempt in range(4):
                    try:
                        res = sampler.sample(prompt=prompt, num_samples=a.k, sampling_params=params).result()
                        break
                    except Exception as e:
                        if attempt == 3:
                            print(f"  FAILED {p['task_id']}: {str(e)[:200]}", flush=True)
                            return None
                        time.sleep(2 ** attempt * 3)
                rows = []
                for j, seq in enumerate(res.sequences):
                    out = list(seq.tokens)
                    msg, term = renderer.parse_response(out)
                    r = reward(get_text_content(msg), p)
                    rows.append({"task_id": p["task_id"], "k": j, "reward": r, "completion_tokens": len(out),
                                 "truncated": not term.is_clean})
                tokens["prefill"] += len(prompt.to_ints())  # one prefill per problem; K samples share it
                tokens["sample"] += sum(r["completion_tokens"] for r in rows)
                return rows

            with ThreadPoolExecutor(a.workers) as ex, out_samples.open("a") as f:
                for i, rows in enumerate(ex.map(run, todo), 1):
                    if rows:
                        f.writelines(json.dumps(r) + "\n" for r in rows)
                        f.flush()
                    if i % 50 == 0 or i == len(todo):
                        print(f"  {i}/{len(todo)} problems sampled", flush=True)
    finally:
        try:
            service.close()
        finally:
            delete_session(session)

    by = {}
    for r in map(json.loads, out_samples.open()):
        by.setdefault(r["task_id"], []).append(r["reward"])
    pool = [{"task_id": t, "rewards": rs, "mean": sum(rs) / len(rs)} for t, rs in by.items() if len(set(rs)) > 1]
    (ROOT / "rl/pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sorted(pool, key=lambda r: r["task_id"])))
    allpass = sum(1 for rs in by.values() if min(rs) == 1.0)
    allzero = sum(1 for rs in by.values() if max(rs) == 0.0)
    cost = (tokens["prefill"] * PRICE["prefill"] + tokens["sample"] * PRICE["sample"]) / 1e6
    print(json.dumps({"problems": len(by), "pool (reward varies)": len(pool), "always 1.0": allpass,
                      "always 0": allzero, "mean reward": round(sum(sum(r) for r in by.values()) /
                                                                sum(len(r) for r in by.values()), 3),
                      "this_run_cost_usd": round(cost, 2), "tokens": tokens}, indent=1))


if __name__ == "__main__":
    main()
