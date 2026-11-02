"""Serverless LoRA SFT of Qwen 3.8 27B on data/train.jsonl via the Fireworks cookbook's sft_loop.

Uses the maintained recipe `training/recipes/sft_loop.py` (Config(serverless=True)) from the
fw-ai cookbook, which owns batching, DCP (resumable) checkpoints, and promotion.

Thinking is OFF for training and must be OFF for eval: renderer `qwen3_8_disable_thinking_interleaved`
(empty think block in the generation prompt). Train targets are code only, so base and tuned are
both evaluated in non-thinking mode with this renderer, temperature 0, max_tokens 4096.

  check   render every row locally: exact token counts, loss mask, no API calls
  train   run SFT (billable). Keeps the session open so KJ can load the final training
          checkpoint (create_training_client_from_state) for GRPO / in-session eval.

Run with the cookbook venv:  .venv-train/bin/python -m sft.train_serverless check
"""

import argparse
import json
import logging
import os
import sys
import time

COOKBOOK = os.environ.get(
    "FW_COOKBOOK", os.path.expanduser("~/.claude/plugins/cache/fw-ai-cookbook/fireworks-training/2.3.0"))
sys.path.insert(0, COOKBOOK)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = "accounts/fireworks/models/qwen3p8-27b"
TOKENIZER = "Qwen/Qwen3.8-27B"
RENDERER = "qwen3_8_disable_thinking_interleaved"
MAX_SEQ_LEN = 4096
PARAMS = dict(lora_rank=16, lora_alpha=32, learning_rate=1e-4, epochs=2, batch_size=16,
              warmup_steps=5, grad_clip_norm=1.0, seed=0)
WANDB_PROJECT = "cobolforge"


def wandb_config(run_name):
    """W&B run in the logged-in account's default entity (override with WANDB_ENTITY).

    The cookbook falls back to offline mode unless WANDB_API_KEY is in the environment, so hand it
    wandb's own stored credential (never printed)."""
    import wandb
    from training.utils import WandBConfig

    api = wandb.Api()
    if not os.environ.get("WANDB_API_KEY") and api.api_key:
        os.environ["WANDB_API_KEY"] = api.api_key
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    return WandBConfig(entity=entity, project=WANDB_PROJECT, run_name=run_name)


def messages_only(src, dst):
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            g.write(json.dumps({"messages": json.loads(line)["messages"]}) + "\n")
    return dst


def check():
    from training.utils.supervised import build_renderer, render_messages_to_datum
    from training.utils.tokenizers import load_tokenizer

    tok = load_tokenizer(TOKENIZER)
    renderer = build_renderer(tok, TOKENIZER, RENDERER)
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/train.jsonl"))]
    lens, trained = [], []
    for r in rows:
        rendered = render_messages_to_datum(r["messages"], renderer=renderer, max_seq_len=MAX_SEQ_LEN)
        d = rendered.datum
        n = d.model_input.length
        w = d.loss_fn_inputs["weights"].data
        lens.append(n)
        trained.append(sum(1 for x in w if x > 0))
    ex = rows[0]["messages"]
    gen = renderer.build_generation_prompt(ex[:-1])
    print("generation prompt tail:", repr(tok.decode(gen.to_ints())[-160:]))
    total = sum(lens)
    print(f"rows {len(rows)}; tokens/row median {sorted(lens)[len(lens) // 2]}, max {max(lens)} (limit {MAX_SEQ_LEN}); "
          f"total {total}; loss-weighted tokens {sum(trained)} ({sum(trained) / total:.0%})")
    print(f"training tokens x{PARAMS['epochs']} epochs = {total * PARAMS['epochs']}; "
          f"steps ~ {len(rows) * PARAMS['epochs'] // PARAMS['batch_size']}")
    return total


def train(args):
    import training.recipes.sft_loop as sft_loop

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    log_path = os.path.join(ROOT, f"sft/logs/{stamp}")
    os.makedirs(log_path, exist_ok=True)
    dataset = messages_only(os.path.join(ROOT, "data/train.jsonl"), os.path.join(log_path, "train_messages.jsonl"))
    cfg = sft_loop.Config(
        serverless=True,
        log_path=log_path,
        base_model=BASE_MODEL,
        tokenizer_model=TOKENIZER,
        renderer_name=RENDERER,
        dataset=dataset,
        max_seq_len=MAX_SEQ_LEN,
        dcp_save_interval=args.dcp_save_interval,
        save_final_checkpoint=True,
        close_session_on_exit=False,
        output_model_id=args.output_model_id or None,
        wandb=wandb_config(f"qwen38-sft-{stamp}"),
        **PARAMS,
    )
    print(json.dumps({"log_path": log_path, "base_model": BASE_MODEL, "renderer": RENDERER,
                      "max_seq_len": MAX_SEQ_LEN, "wandb": f"{cfg.wandb.entity}/{cfg.wandb.project}/{cfg.wandb.run_name}",
                      **PARAMS}, indent=1), flush=True)
    return sft_loop.main(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "train"])
    ap.add_argument("--dcp-save-interval", type=int, default=20)
    ap.add_argument("--output-model-id", default="")
    a = ap.parse_args()
    if a.cmd == "check":
        check()
    else:
        print(train(a))
