# SFT run: qwen38-sft-20261102-203901

Approved by Ritik in chat on 2026-11-02 ("Launch now", after choosing "Qwen 3.8 27B" and "reasoning off").

| | |
| --- | --- |
| Surface | Fireworks Serverless Training API, cookbook `training/recipes/sft_loop.py` (fireworks-training 2.3.0, fireworks-ai 1.2.30) |
| Account | `ritik-7xa24uxitcg9` |
| Base model | `accounts/fireworks/models/qwen3p8-27b` |
| Tokenizer / renderer | `Qwen/Qwen3.8-27B` / `qwen3_8_disable_thinking_interleaved` (thinking OFF) |
| Data | `data/train.jsonl` @ 804e6ad: 709 rows, 413 MBPP problems, no eval split; loss on assistant tokens only (51%) |
| Tokens | 564,848 per epoch (exact, rendered), 1,129,696 trained over 2 epochs |
| LoRA | rank 16, alpha 32 |
| Optimizer | AdamW lr 1e-4 constant after 5 warmup steps, beta1 0.9, beta2 0.95, eps 1e-12, wd 0, grad clip 1.0, seed 0 |
| Batch / epochs / steps | 16 rows / 2 / ~88 |
| Max seq len | 4096 (longest row 1,611) |
| Checkpoints | DCP (resumable) every 20 steps + final; session left open (`close_session_on_exit=False`) |
| Promotion | final sampler checkpoint -> model `cobol-sft-qwen38-202611022038` |
| Session / run | `ts-b0aaaacfcf6a4661a9e617a5063a65ad` / `run-94dffbdc3914486d9d362a614f824d0c` |
| Checkpoint refs | `ritik-7xa24uxitcg9/run-94dffbdc3914486d9d362a614f824d0c/<checkpoint>` (for `init_from_checkpoint` / `create_training_client_from_state`) |
| W&B | https://forge.coreweave.com/wandb/visualaiblog/cobolforge/runs/58xcqtcd |
| Log | `sft/logs/launch-202611022038.log`, recipe state in `sft/logs/20261102-203901/` |
| Cost estimate | 1.13M train tokens x $4.103/M (Serverless Training docs, 2026-11-02) = ~$4.64; no sampling during SFT |

## Eval contract (base vs tuned, KJ)

Both systems through the Serverless Training sampler with the **same** renderer
(`qwen3_8_disable_thinking_interleaved`), temperature 0, max_tokens 4096, `prompt.build_messages`.
Base = a fresh LoRA client on `qwen3p8-27b` saved for sampling before any step; tuned = this run's
final checkpoint (load the training checkpoint with `create_training_client_from_state`, then
`save_weights_for_sampler`). Qwen3.8 27B has no serverless inference endpoint (404), so do not use
`/inference/v1/chat/completions` for it.
