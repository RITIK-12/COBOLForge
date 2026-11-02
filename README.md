# COBOLForge

Fine-tune a small open model on Fireworks to write correct COBOL, and measure the gain on
**COBOLEval** (146 problems, 821 tests; held out, never trained on). Plan: [PLAN.md](PLAN.md).
Project page: https://ritik-12.github.io/COBOLForge/ (built from this repo's results by `site/build.py`
and `.github/workflows/pages.yml` on every push to main).

## Results: COBOLEval, Qwen 3.8 27B base vs SFT

Same model, same prompt (`prompt.py`), same sampler (Serverless Training session), thinking off,
temperature 0, max_tokens 4096, one sample per problem. Graded by `grader/grade.py` (COBOLEval harness).

| System | pass@1 | Compiles | Sampling cost (146 problems) | p50 latency |
| --- | --- | --- | --- | --- |
| Qwen 3.8 27B base | 19/146 (13.0%) | 32.9% | $0.74 | 3.9s |
| **Qwen 3.8 27B + SFT** | **75/146 (51.4%)** | **78.1%** | $0.60 | 6.0s |

- **Gain: +38.4 points**, 95% bootstrap CI over problems [+29.5, +46.6] (5,000 resamples).
- **McNemar exact**: SFT-only passes 59, base-only passes 3, p = 1.7e-14.
- Failure breakdown, base -> SFT: compile errors 97 -> 32, wrong answers 29 -> 39, blocked 1 -> 0.
- **Contamination check**: MBPP and HumanEval share a few tasks (gcd, largest prime factor). Excluding
  every COBOLEval problem whose description has word-Jaccard >= 0.3 with any training problem (13 dropped),
  the gain is +34.6 points (12.8% -> 47.4%, n=133, CI [+25.6, +43.6], p = 1e-11); at >= 0.25 (22 dropped)
  it is +32.3 points (p = 5e-10).
- Caveats: one training run, one seed, n=146; HumanEval/74 and /160 can't pass for any system (broken
  dataset callers); base evaluated with thinking off (same mode as SFT); costs are upper bounds (prefill
  billed as uncached).
- Thinking-on base (context row, paused): renderer `qwen3_8`, 32k budget; ~10-22k tokens and 70-140s per
  problem, ~$12 for all 146. 11/146 responses cached in `eval/cache/qwen38-27b-base-think/`. Resume (skips
  cached work): `.venv-train/bin/python -m eval.run_insession --systems base-think`.

Per-problem rows: `results/qwen38-27b-{base,sft}.jsonl`; raw model outputs: `eval/cache/`; stats:
`results/compare_qwen38-27b-base_vs_qwen38-27b-sft.json` (`eval/compare.py`). Reproduce with
`.venv-train/bin/python -m eval.run_insession` then `.venv/bin/python -m eval.compare qwen38-27b-base qwen38-27b-sft`.

## Status (team sync)

| Piece | State | Owner |
| --- | --- | --- |
| Grader `grader/grade.py` | Done; self-test passes (144/146 callers build, HumanEval/74 and /160 are broken upstream) | KJ |
| Frozen prompt `prompt.py` | Done: `build_messages(problem["prompt"])` -> [system, user]; model answers a full program in one ```` ```cobol ```` block | Ritik |
| MBPP -> COBOLEval converter `data/convert_mbpp.py` | Done: 459 problems, 1,371 tests | Ritik |
| Teacher rejection sampling `data/sample_teacher.py` | Done (`kimi-k3`): 709 SFT rows | Ritik |
| SFT | **Done**: serverless LoRA SFT on **Qwen 3.8 27B** (thinking off), 90 steps, loss 0.135 -> 0.062; model `cobol-sft-qwen38-202611022038`. Details, checkpoints, W&B: [sft/RUN.md](sft/RUN.md) | Ritik |
| Eval base vs SFT | **Done** (results above): `eval/run_insession.py`, `eval/compare.py` | Ritik |
| GRPO / expert iteration, frontier context rows, charts | Next; GRPO can start from SFT checkpoint `step-90` ([sft/RUN.md](sft/RUN.md)) | KJ |

### Student switched to Qwen 3.8 27B (2026-11-02)

We need SFT **and** GRPO. Checked against the live catalog, the Serverless Training docs and the cookbook:

| Model | SFT | GRPO | Serverless Training | Serverless inference (base) |
| --- | --- | --- | --- | --- |
| Nemotron Lightning 3.5 30B A3B | managed only | no (`rlLoraTunable`/`rlTunable` false) | no | yes |
| **Qwen 3.8 27B** (`qwen3p8-27b`) | yes | yes | yes | no (404) -> sample in-session |
| Muse Glimmer 30B | yes | yes | yes | no (deprecated 2026-09-25) |

Qwen 3.8 27B also leads Muse and Nemotron on every coding/reasoning benchmark their model cards share.
Teacher data and `prompt.py` are student-independent, so nothing upstream changes.

### Eval contract (base vs tuned)

- Serverless Training sampler for **both** systems, renderer `qwen3_8_disable_thinking_interleaved`
  (thinking off; training targets are code only), temperature 0, max_tokens 4096, `prompt.build_messages`.
- Base = fresh LoRA client on `accounts/fireworks/models/qwen3p8-27b`, `save_weights_for_sampler` before any step.
- Tuned = load this run's final training checkpoint (`create_training_client_from_state`, refs in
  [sft/RUN.md](sft/RUN.md)), then `save_weights_for_sampler`. The final weights are also promoted to a model.
- Optional context row: base with thinking on (shows the gain isn't from a weakened baseline).

### Other Fireworks facts (2026-11-02)

- `kimi-k3` (teacher) rejects `n > 1` and caps generated tokens at 72k/min (429 + `Retry-After`).
  It returns reasoning in `reasoning_content`, so `content` is just the answer.
- No fine-tuned LoRA runs on serverless inference (`supportsServerlessLora: false` everywhere); use
  in-session sampling or an on-demand deployment.

## Setup (macOS, Apple Silicon)

```bash
/opt/homebrew/bin/brew install gnucobol      # GnuCOBOL 3.2.0 (the Intel brew in /usr/local fails)
uv venv .venv --python /opt/homebrew/bin/python3.11
uv pip install --python .venv/bin/python httpx numpy python-dotenv "fireworks-ai>=1.2.30,<2"
export FIREWORKS_API_KEY=...                  # see .env.example
.venv/bin/python -m grader.grade --self-test  # grade.py needs Python >= 3.10

# training: separate venv with the fw-ai cookbook's dependencies (torch, transformers, wandb, ...)
uv venv .venv-train --python /opt/homebrew/bin/python3.11
uv pip install --python .venv-train/bin/python -r <cookbook>/training/pyproject.toml deps
.venv-train/bin/python -m sft.train_serverless check   # renders every row, exact tokens, no API calls
.venv-train/bin/python -m sft.train_serverless train --output-model-id cobol-sft-qwen38-<stamp>
```

`sft/train_serverless.py` finds the cookbook at `~/.claude/plugins/cache/fw-ai-cookbook/fireworks-training/2.3.0`
(override with `FW_COOKBOOK`). W&B logs to the logged-in account's default entity (override `WANDB_ENTITY`).

## Run order

```bash
.venv/bin/python -m data.convert_mbpp                            # data/mbpp_cobol.jsonl
.venv/bin/python -m data.sample_teacher --samples 2              # data/teacher_samples.jsonl, data/train.jsonl
.venv/bin/python -m data.sample_teacher --samples 8 --only-unsolved   # top up problems with no pass yet
```

Every teacher response is cached in `data/cache/` (gitignored) and never re-requested.

## Training data

Source: MBPP ([google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp), CC BY 4.0; 974 Python problems; tests only, Python solutions discarded), converted to the exact
COBOLEval harness format (`src/main.rs` @ 0bb96c3: skeleton prompt, LINKAGE layout, caller programs,
output records). Types are inferred from the test asserts because MBPP has no type hints.

Dropped as faulty (not repaired):

- types other than int / str / bool return / flat list of int or str
- list inputs whose length differs across tests (one fixed `OCCURS` per argument); empty lists
- ints outside +-10^9 (the harness parser only handles a leading-zero sign overpunch)
- strings > 100 chars, non-ASCII, quotes, or leading/trailing spaces (`PIC X(100)` is space-padded)
- tests expecting `[]` or `""` (any output passes them because the parser truncates/strips)
- problems left with < 2 tests, or where every test expects the same value (a constant program passes)
- names too long for COBOL, and near-duplicates of COBOLEval problems (same entry point or word
  overlap >= 0.5 with a COBOLEval description; COBOLEval is used only to *remove* data)

Then every caller is compiled against a stub solution; problems whose callers fail are dropped.

| Stage | Count |
| --- | --- |
| MBPP rows | 974 |
| Converted + callers compile | 459 problems, 1,371 tests |
| Teacher (`kimi-k3`, T=0.7, 2 per problem) samples: compiled / passed | 918: 832 / 718 (78%); 47 hit the 8,192-token limit |
| `train.jsonl` | **709 rows from 413 of 459 problems** (≤ 2 distinct programs per problem); ~1.3M training tokens over 2 epochs |

`train.jsonl` rows are `{id, messages: [system, user, assistant]}` with system + user from
`prompt.py` and assistant = the passing program in one ```` ```cobol ```` block, at most 2 distinct
programs per problem. Failing samples stay in `data/teacher_samples.jsonl` (DPO rejects / expert-iteration pool).

## Disclosure

Built with AI coding assistants (Claude Code).
