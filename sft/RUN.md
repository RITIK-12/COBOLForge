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

## Result (2026-11-02 12:39-12:45 PT, ~6 min)

| | |
| --- | --- |
| State | **Completed**: 90 optimizer steps (89 full batches + 1 partial), no errors |
| Tokens trained | 1,129,696 (matches the local render count) -> ~$4.64 at $4.103/M |
| Loss (token-weighted CE, assistant tokens) | first 5 steps 0.135 -> epoch 1 avg 0.098 -> epoch 2 avg 0.069 -> last 10 steps 0.065 (final 0.062) |
| Grad norm | final 1.22 pre-clip (clipped to 1.0); no spikes on W&B |
| Checkpoints (resumable DCP) | `step-20`, `step-40`, `step-60`, `step-80`, `step-90` (final) |
| Sampler checkpoint | `run-94dffbdc3914486d9d362a614f824d0c-step-90-3c990e18` |
| Promoted model | `accounts/ritik-7xa24uxitcg9/models/cobol-sft-qwen38-202611022038`: READY, `HF_PEFT_ADDON`, r=16, base `qwen3p8-27b` |
| Session | `ts-b0aaaacfcf6a4661a9e617a5063a65ad` left open for KJ (holds one serverless run slot; no idle compute charge) |

Low starting loss is expected: about half of each answer copies the prompt skeleton (divisions,
LINKAGE, comments). Train loss alone says nothing about COBOLEval; that needs the eval below.

## Eval contract (base vs tuned, KJ)

Both systems through the Serverless Training sampler with the **same** renderer
(`qwen3_8_disable_thinking_interleaved`), temperature 0, max_tokens 4096, `prompt.build_messages`.
Base = a fresh LoRA client on `qwen3p8-27b` saved for sampling before any step; tuned = this run's
final checkpoint (load the training checkpoint with `create_training_client_from_state`, then
`save_weights_for_sampler`). Qwen3.8 27B has no serverless inference endpoint (404), so do not use
`/inference/v1/chat/completions` for it.
