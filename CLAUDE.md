# COBOL Copilot — Buildathon (Mon Nov 2, 2026, submit by 4:30 PM PT)

## Goal
Fine-tune a small open model on Fireworks to write correct COBOL. Show a statistically
significant gain on COBOLEval (146 problems, 821 tests) vs the same base model.
Frontier models (Claude, GPT) are optional context, not the required baseline.

## Stack
- Student: Nemotron Lightning 3.5 30B on Fireworks (fallback: Qwen3.8 27B)
- Teacher: strongest open coder on Fireworks serverless (chosen at 11:45 sync)
- Held-out test: COBOLEval — NEVER used for training, prompt tuning, or data generation
- Training source: MBPP (filtered to ints, strings, flat lists), rejection-sampled via GnuCOBOL
- Post-training: SFT (LoRA r16, alpha 32, lr 1e-4, 2 epochs, batch 16) → GRPO or expert iteration
- Grader: GnuCOBOL (`cobc`) + COBOLEval harness, run locally on macOS

## Contracts (do not change without telling both teammates)
- `prompt.py`: single system prompt + template for ALL training and eval. Frozen after 11:45.
- `grader/grade.py`: `grade(program: str, problem: dict) -> {compiled, passed, n_pass, n_tests, error}`
  Same function = data filter, eval metric, RL reward. Timeout 10s per program run.
- `data/train.jsonl`: `{id, messages: [system, user, assistant]}`
- `results/{system}.jsonl`: one row per problem
  `{id, system, pass, compiled, tokens, latency_s, cost}`

## Repo layout
grader/  data/  sft/  rl/  eval/  results/  charts/
KJ owns: grader/, eval/, rl/, results/ stats. Teammate owns: data/, sft/, README.

## Rules
- Eval settings: temperature 0, max_tokens 4096, identical prompt for every system.
- Cache every API result to disk; never re-call a model for a row already in results/.
- Async with bounded concurrency (semaphore ~16); retries with backoff on 429/5xx.
- Do not invent Fireworks model IDs or API params — check Fireworks docs or `firectl` output.
- Do not guess COBOLEval's harness format — read the repo's source and follow it exactly.
- Secrets only from env (`FIREWORKS_API_KEY`, `OPENROUTER_API_KEY`); keep `.env.example` updated.
- Small, frequent commits to main. Prefer working code now over clean code later.
- Stats: McNemar on per-problem pass; bootstrap resampling PROBLEMS (not tests), 5,000 iters, 95% CI.
