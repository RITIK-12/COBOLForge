"""GRPO from the SFT checkpoint on Fireworks serverless training (Qwen 3.8 27B, LoRA r16).

Cookbook recipe `training.recipes.experiment.async_rl_loop_serverless` (built-in clipped GRPO
loss, sampler snapshot published after every optimizer step) with our pieces plugged in:
  prompt    prompt.build_messages (frozen), renderer qwen3_8_disable_thinking_interleaved
  reward    grade(): 1.0 all tests pass, 0.2 compiles, 0.0 otherwise
  rows      rl/pool.jsonl (MBPP problems where SFT rewards vary; rl/select_pool.py)
  init      SFT training checkpoint step-90 (sft/RUN.md), loaded with optimizer state
Training checkpoints (DCP) every --dcp-every steps + final are run-scoped and outlive the
session; evaluate them with create_training_client_from_state like eval/run_insession.py.

    .venv-fw/bin/python -m rl.grpo --steps 2 --run-name smoke     # smoke test
    .venv-fw/bin/python -m rl.grpo --steps 30 --run-name grpo-v1
"""

import argparse
import asyncio
import json
import random
import time

from prompt import build_messages
from rl.common import (BASE_MODEL, LORA_ALPHA, LORA_RANK, PRICE, RENDERER, ROOT, SFT_STATE, TOKENIZER,
                       load_problems, reward, use_sft_account_key)

STATS = {"rollouts": 0, "completion_tokens": 0, "prompt_tokens": 0, "rewards": []}


def make_rollout_fn_factory(problems):
    from training.renderer import get_text_content
    from training.utils.rl.rollout.renderer import single_turn_renderer_rollout
    from training.utils.supervised import build_renderer

    def factory(setup):
        renderer = build_renderer(setup.tokenizer, setup.tokenizer_id, RENDERER, load_image_processor=False)
        sampler = setup.sampler  # owned by the recipe

        async def grade_fn(row, message, _parse_success):
            text = get_text_content(message)
            r = await asyncio.to_thread(reward, text, problems[row["task_id"]])
            STATS["rollouts"] += 1
            STATS["rewards"].append(r)
            STATS["completion_tokens"] += len(setup.tokenizer.encode(text))
            return r

        async def rollout_fn(row):
            return await single_turn_renderer_rollout(
                row,
                renderer=renderer,
                sample_with_prompt_tokens=sampler.sample_with_prompt_tokens,
                message_builder=lambda r: r["messages"],
                reward_fn=grade_fn,
                sample_kwargs=setup.sample_kwargs,
                tokenizer=setup.tokenizer,
            )

        return rollout_fn

    return factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True, help="optimizer steps (= rows / groups-per-step)")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--groups-per-step", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=8, help="completions per prompt (group size)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--dcp-every", type=int, default=5)
    ap.add_argument("--init", default=SFT_STATE, help="training checkpoint to start from")
    a = ap.parse_args()
    use_sft_account_key()

    from training.recipes.experiment import async_rl_loop_serverless as recipe
    from training.utils.rl.losses import has_reward_variance

    problems = load_problems()
    pool = [json.loads(l)["task_id"] for l in (ROOT / "rl/pool.jsonl").open()]
    random.Random(0).shuffle(pool)
    n_rows = a.steps * a.groups_per_step
    if n_rows > len(pool):
        raise SystemExit(f"{a.steps} steps x {a.groups_per_step} groups needs {n_rows} rows; pool has {len(pool)}")
    rows = [{"task_id": t, "messages": build_messages(problems[t]["prompt"])} for t in pool[:n_rows]]

    run_dir = ROOT / "rl/runs" / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = recipe.Config(
        base_model=BASE_MODEL, tokenizer_model=TOKENIZER,
        learning_rate=a.lr, completions_per_prompt=a.rollouts, prompt_groups_per_step=a.groups_per_step,
        max_completion_tokens=a.max_tokens, max_seq_len=8192, temperature=a.temperature,
        epochs=1, shuffle=False, seed=0, max_rows=n_rows, min_group_size=a.rollouts,
        lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA, init_from_checkpoint=a.init,
        snapshot_prefix=a.run_name[:8], metrics_file=str(run_dir / "metrics.jsonl"),
        dcp_save_interval=a.dcp_every, save_final_checkpoint=True,
    )
    (run_dir / "config.json").write_text(json.dumps(
        {**{k: getattr(cfg, k) for k in ("base_model", "learning_rate", "completions_per_prompt",
                                         "prompt_groups_per_step", "max_completion_tokens", "temperature",
                                         "max_rows", "lora_rank", "lora_alpha", "init_from_checkpoint",
                                         "dcp_save_interval")},
         "renderer": RENDERER, "reward": "1.0 pass / 0.2 compiles / 0", "steps": a.steps,
         "pool_size": len(pool)}, indent=1))
    print(f"GRPO {a.run_name}: {a.steps} steps x {a.groups_per_step} groups x {a.rollouts} rollouts, "
          f"lr {a.lr}, init {a.init}", flush=True)

    t0 = time.time()
    try:
        result = asyncio.run(recipe.main(
            cfg, rollout_fn_factory=make_rollout_fn_factory(problems),
            dynamic_filter_fn=has_reward_variance, rows=rows,
        ))
    finally:
        rs = STATS["rewards"]
        # cost upper bound: every prompt prefilled uncached; train tokens = prompt + completion
        prompt_tok = STATS["rollouts"] * 470
        est = (prompt_tok * PRICE["prefill"] + STATS["completion_tokens"] * PRICE["sample"]
               + (prompt_tok + STATS["completion_tokens"]) * PRICE["train"]) / 1e6
        summary = {"minutes": round((time.time() - t0) / 60, 1), "rollouts": STATS["rollouts"],
                   "mean_reward": round(sum(rs) / len(rs), 3) if rs else None,
                   "pass_rate": round(sum(r == 1.0 for r in rs) / len(rs), 3) if rs else None,
                   "completion_tokens": STATS["completion_tokens"], "est_cost_usd_upper": round(est, 2)}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1), flush=True)
    (run_dir / "result.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
