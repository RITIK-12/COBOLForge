# COBOLForge

Fine-tune a small open model on Fireworks to write correct COBOL, and measure the gain on
**COBOLEval** (146 problems, 821 tests; held out, never trained on). Plan: [PLAN.md](PLAN.md).

## Status (team sync)

| Piece | State | Owner |
| --- | --- | --- |
| Grader `grader/grade.py` | Done; self-test passes (144/146 callers build, HumanEval/74 and /160 are broken upstream) | KJ |
| Frozen prompt `prompt.py` | Done: `build_messages(problem["prompt"])` -> [system, user]; model answers a full program in one ```` ```cobol ```` block | Ritik |
| MBPP -> COBOLEval converter `data/convert_mbpp.py` | Done: 459 problems, 1,371 tests | Ritik |
| Teacher rejection sampling `data/sample_teacher.py` | Done (`kimi-k3`): 709 SFT rows | Ritik |
| SFT | Managed LoRA SFT on Nemotron (see below) | Ritik |
| Baselines / eval / stats | `eval/`, `results/` | KJ |

### Fireworks facts checked on the live catalog (2026-11-02)

For `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b`:

- `supervisedLoraTunable: true` -> **Managed LoRA SFT works.**
- `rlLoraTunable: false`, `rlTunable: false`, `supportsServerlessTraining: false` -> **no GRPO / RL and no
  Serverless Training API on this student.** Post-training after SFT = expert iteration (another round of
  rejection-sampled SFT from the SFT model's own passing samples). `qwen3p8-27b` has `rlLoraTunable: true`
  if RL on a different student is ever needed (would need new baselines).
- `supportsServerlessLora: false` -> the tuned LoRA can't run on serverless inference; evaluating it
  needs an **on-demand deployment**. Evaluate the base model on that same deployment too, so base and
  tuned go through the same serving path.
- `kimi-k3` (teacher) rejects `n > 1`; one request per sample. It returns reasoning in
  `reasoning_content`, so `content` is just the answer.

## Setup (macOS, Apple Silicon)

```bash
/opt/homebrew/bin/brew install gnucobol      # GnuCOBOL 3.2.0 (the Intel brew in /usr/local fails)
uv venv .venv --python /opt/homebrew/bin/python3.11
uv pip install --python .venv/bin/python httpx numpy python-dotenv "fireworks-ai>=1.2.30,<2"
export FIREWORKS_API_KEY=...                  # see .env.example
.venv/bin/python -m grader.grade --self-test  # grade.py needs Python >= 3.10
```

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
