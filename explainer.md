# COBOLForge: how the whole pipeline works

What we built, in the order it happens, with the real numbers from this repo (as of 2026-11-02, ~1 PM PT).
For the plan and deadlines see [PLAN.md](PLAN.md); for headline results see [README.md](README.md).

---

## The one-paragraph version

We take a mid-sized open model (**Qwen 3.8 27B**, the *student*) that writes poor COBOL, and teach it
to write correct COBOL. To get training data we take ~460 simple Python problems (MBPP), turn each one
into a COBOL problem with real test programs, and ask a much larger model (**Kimi K3**, the *teacher*)
to solve them in COBOL. A real COBOL compiler (GnuCOBOL) runs every answer against its tests; only the
answers that pass become training examples. We fine-tune the student on those (**SFT**), then plan to
improve it further by letting it practice and rewarding it when its programs pass (**GRPO**). We measure
everything on **COBOLEval**, a separate, held-out set of 146 COBOL problems the model never trains on.

**Result so far:** the student went from **13.0% to 51.4%** of COBOLEval problems solved after SFT
(+38.4 points, 95% CI +29.5 to +46.6, McNemar p = 1.7e-14).

---

## The pipeline at a glance

```
                       ┌──────────────────────── grader/grade.py (GnuCOBOL) ────────────────────────┐
                       │      the same judge is used for data filtering, evaluation and RL reward   │
                       └───────────────▲────────────────────────▲─────────────────────────▲────────┘
                                       │ keep passers           │ score                   │ reward
MBPP (974 Python problems)             │                        │                         │
   │ 1. convert                        │                        │                         │
   ▼                                   │                        │                         │
459 COBOL problems ──2. teacher──► 918 Kimi K3 programs ──► 709 verified examples          │
 (data/mbpp_cobol.jsonl)            (2 per problem)          (data/train.jsonl)            │
                                                                   │ 4. SFT                │
                                                                   ▼                       │
Qwen 3.8 27B base ─────────────────────────────────────► Qwen 3.8 27B + SFT ──5. GRPO──► + GRPO
       │                                                           │                       │
       └──────────── 3/6. evaluate on COBOLEval (146 held-out problems, never trained on) ──┘
```

---

## Step 0: The judge (`grader/`), built first because everything depends on it

Every stage needs a trustworthy answer to "is this COBOL program correct?". `grader/grade.py` gives one:

- **What it does.** It takes a COBOL program and a problem, then for each test: compiles the program together with
  that test's *caller* program (`cobc -w -fformat=variable -x`), runs it, reads the result file the
  caller writes, and compares it with the expected value. A problem **passes only if every test passes**.
- **Faithful to COBOLEval.** The compile command, output-file naming, value parsing (including the `p`/`y`
  prefix for negative numbers) and float tolerance (0.001) are copied from the official COBOLEval harness
  (`zorse-project/COBOLEval` @ `0bb96c3`).
- **Safe to run untrusted code.** Programs are model-written, so:
  - a *static guard* rejects programs that use `CALL` (except their own entry point), file I/O (`SELECT…ASSIGN`,
    `OPEN`, `DELETE FILE`) or `EXEC` before compiling;
  - the run step uses macOS `sandbox-exec`: no network, and no file writes outside a throwaway temp folder;
  - a 10-second timeout per test, after which the problem's remaining tests are skipped.
- **Self-tested.** `python -m grader.grade --self-test` checks that:
  - a correct solution passes;
  - wrong, broken, blocked and infinite-loop programs fail;
  - the sandbox blocks escapes;
  - all 146 COBOLEval callers build.
- **Known dataset bug.** HumanEval/74 and /160 have broken callers in COBOLEval itself, so they can't pass
  for any model. The maximum possible score is therefore 144/146.

The key design choice: **the same function is the data filter, the evaluation metric and the RL reward.**
So "correct" means the same thing at every stage.

## Step 0b: One frozen prompt (`prompt.py`)

Every model, in training and in evaluation, sees exactly the same instructions:
- the system prompt: "you are a GnuCOBOL 3.2 programmer…", plus the rules: keep the LINKAGE SECTION,
  no files, `CALL` or `DISPLAY`, end with `GOBACK`;
- the user message: the COBOLEval-style program skeleton.

The model must answer with one complete program in a ```` ```cobol ```` block; `grader/construct.py`
extracts the last such block. Freezing this keeps comparisons fair.

---

## Step 1: Data conversion, MBPP → COBOL problems (`data/convert_mbpp.py`)

**Why.** COBOLEval is held out, so we need *other* COBOL problems to train on. There is no ready-made COBOL
training set with tests, so we build one.

**Source.** MBPP ("Mostly Basic Python Problems", Google): 974 short Python tasks, each with a description and
assert-style tests. We keep only the problem text and tests; the Python solutions are thrown away.

**How.** The converter mirrors COBOLEval's own Python→COBOL translator (its `src/main.rs`), so the training
problems look exactly like the test problems:
- Each MBPP function becomes a COBOL subprogram skeleton: a LINKAGE SECTION with `L-*` inputs and a `RESULT`,
  plus the task text as comments.
- Each MBPP assert becomes a COBOL *caller* program that fills the inputs, calls the solution, and writes the
  result to a file.
- Types are inferred from the test values, since MBPP has no type hints.

**What gets dropped, not repaired.** Problems that can't be represented faithfully:
- floats, nested lists or `None`;
- lists whose length varies across tests;
- very large ints or long strings;
- tests that any program would pass (empty expected values, every test expecting the same answer);
- names too long for COBOL;
- callers that don't compile;
- **near-duplicates of COBOLEval problems**, which are decontaminated here.

**Output.** `data/mbpp_cobol.jsonl`: **459 problems, 1,371 tests**, all in COBOLEval's exact format, so
`grade()` works on them unchanged.

---

## Step 2: The teacher, Kimi K3 (`data/sample_teacher.py`)

### Where the teacher fits

The student is **the model we fine-tune and ship**. The **teacher** is never trained and never shipped.
Its only job is to **write the example solutions the student learns from**.

We need it because:
- **The student can't produce its own training data yet.** Base Qwen solves only 13% of COBOLEval, mostly
  failing to compile. Sampling it on MBPP would yield few passing programs, mostly on the easiest problems.
- **A big model can.** Kimi K3 (2.8T parameters, open weights) solves most problems. Its correct
  programs show the student working GnuCOBOL patterns.
- **The grader makes the teacher safe.** We don't trust the teacher blindly: every teacher program is
  compiled and tested, and **only programs that pass all tests are kept** ("rejection sampling"). Teacher
  mistakes lower the *yield*, never the *quality* of the data.
- **GRPO needs it indirectly.** GRPO learns by comparing several attempts at the same problem. If every
  attempt scores zero, there is nothing to compare. Teacher data, through SFT, lifts the student to where
  it sometimes succeeds, so GRPO has a signal to work with (Step 5).
- **It provides the "large model" comparison.** It gives the "small model vs. very large model" cost and
  quality point for the pitch.

### What we ran

- **Settings.** Kimi K3, 2 samples per problem, temperature 0.7, up to 8,192 tokens, on all 459 problems.
  Every response is cached (`data/cache/`) and never re-requested.
- **Yield:**

| | Count |
|---|---|
| Teacher samples | 918 |
| Compiled | 832 (91%) |
| **Passed all tests** | **718 (78%)** |
| Problems with ≥1 passing program | 413 / 459 |
| **Training examples kept** (≤2 per problem) | **709** → `data/train.jsonl` |

- **Format.** `data/train.jsonl` holds `{id, messages: [system, user, assistant]}`. The assistant message is
  the verified program, and the system and user messages come from the same `prompt.py` used in evaluation.
- **Failed samples are kept** in `data/teacher_samples.jsonl` for possible later use, as DPO rejected
  examples or a hard-problem pool.
- **Cost.** About $33 at Kimi's list price ($3 in / $15 out per 1M tokens). This is an upper bound, and it was
  billed to Ritik's account.

### How we chose the teacher (the 30-problem check)

PLAN.md allows exactly one use of COBOLEval outside final evaluation: a **30-problem check** to choose the
teacher and to sanity-check settings. The 30 problem IDs are fixed (seed 0, `eval/data/subset30.json`), and
every model got the same prompt, temperature 0 and 4,096 max tokens (`eval/run.py`, `eval/systems.py`).

| System | pass@1, strict | pass@1, section-order fix* | $/problem |
|---|---|---|---|
| Qwen 3.8 27B base (student) | 16.7% | 16.7% | $0.0051 |
| **Kimi K3** | 26.7% | **76.7%** | $0.0099 |
| DeepSeek V4.1 Flash | 63.3% | 73.3% | $0.0007 |
| GLM-5.3 (reasoning "low") | 60.0% | 60.0% | $0.0025 |
| gpt-oss-120b (reasoning "low") | 0.0% | 3.3% | $0.0005 |

\* Kimi often copies the skeleton's section order, with WORKING-STORAGE *after* LINKAGE, which GnuCOBOL
rejects. COBOLEval's official chat harness fixes this automatically (`swap_sections`); our strict grader does
not. Whether to apply the fix in evaluation is still an **open decision**: it doesn't change base Qwen's score.

Notes:
- Kimi's teacher run on MBPP passed 78%, with no fix needed. It ran with Kimi's default reasoning, while the
  check ran with reasoning off, so with reasoning on Kimi mostly gets the section order right.
- GLM-5.3 and gpt-oss can't turn reasoning off (the API rejects `"none"`), so they ran at `"low"`.
- The whole check cost **$0.56**.

---

## Step 3: Baseline, measuring the student before training

**Which student, and why it changed.** The brief named Nemotron Lightning 3.5 30B. We checked against the
Fireworks API, the live docs and the training cookbook:

| Model | SFT | GRPO on serverless training | Serverless inference |
|---|---|---|---|
| Nemotron Lightning 3.5 30B | managed SFT only | **no** (no RL support; not in the serverless training pool) | yes |
| **Qwen 3.8 27B** | **yes** | **yes** (in the serverless pool, RL-tunable) | no → evaluate through the training sampler |
| Muse Glimmer 30B | yes (free until Nov 30) | yes | no |

We need both SFT and GRPO, so the student became **Qwen 3.8 27B** (`accounts/fireworks/models/qwen3p8-27b`).

**Reasoning off.** Qwen 3.8 "thinks" before answering by default. In a test it spent all 4,096 tokens
reasoning and never wrote code. With the reasoning-off renderer (`qwen3_8_disable_thinking_interleaved`)
it answers in about 400 tokens and 3 seconds. **Reasoning is off everywhere**: training data rendering, SFT,
GRPO and every evaluation.

**How Qwen is evaluated.** Qwen 3.8 27B has no regular serverless chat endpoint, so both base and fine-tuned
versions are sampled through the **Serverless Training sampler**. Same renderer, temperature 0, 4,096 max
tokens, same prompt. Base and tuned are scored through exactly the same path.

**Baseline result (all 146 problems):** **19/146 = 13.0%** pass@1, 32.9% compile. Most failures are compile
errors: real GnuCOBOL mistakes such as `MOD` used as a variable, `FROM WS-I + 1`, or `EXIT PERFORM` outside
an inline `PERFORM`. This is the gap training should close.

---

## Step 4: SFT, supervised fine-tuning (`sft/`)

**What it is.** Show the student the 709 verified teacher programs and train it to reproduce them
(next-token prediction). Loss is computed only on the assistant's program, not on the prompt.

**How it ran** (details in [sft/RUN.md](sft/RUN.md)):

| | |
|---|---|
| Where | Fireworks **Serverless Training API** (LoRA; no GPUs to provision) |
| Model | Qwen 3.8 27B, reasoning off |
| LoRA | rank 16, alpha 32 |
| Optimizer | AdamW, lr 1e-4, 2 epochs, batch 16, seed 0 → 90 steps |
| Data | 709 rows, 1.13M training tokens |
| Time / cost | ~6 minutes / ~$4.64 |
| Loss | 0.135 → 0.062 |
| Output | checkpoint `step-90`, promoted to model `cobol-sft-qwen38-202611022038` (Ritik's account) |

**Result on COBOLEval (146 problems, same settings as baseline):**

| System | pass@1 | Compiles | Eval cost |
|---|---|---|---|
| Qwen 3.8 27B base | 19/146 (13.0%) | 32.9% | $0.74 |
| **Qwen 3.8 27B + SFT** | **75/146 (51.4%)** | **78.1%** | $0.60 |

- **Gain: +38.4 points.** 95% bootstrap CI (resampling problems, 5,000×) is [+29.5, +46.6]. McNemar exact
  p = 1.7e-14: SFT solves 59 problems base doesn't, base solves 3 that SFT doesn't.
- **What improved.** Compiling (33% → 78%) *and* correctness among programs that compile (40% → 66%).
- **Independently re-checked.** The numbers reproduce from the per-problem rows. Base Qwen agrees on 29/30
  problems with a separate run on the 30-problem subset.
- **Contamination.** MBPP and HumanEval share a few classic tasks (e.g. gcd). Dropping every COBOLEval problem
  that resembles a training problem still leaves a gain of **+34.6 points** (n=133, p = 1e-11), so the result
  isn't explained by overlap.

SFT alone already meets the project goal: a statistically significant gain on held-out COBOLEval.

---

## Step 5: GRPO, reinforcement learning with a verifiable reward (`rl/`, running)

**What it is.** SFT teaches the student to imitate. GRPO (Group Relative Policy Optimization) teaches it
from its *own* attempts:

1. Give the student a problem and let it write **8 different programs** (temperature 1.0).
2. Score each with `grade()`: **1.0** passes all tests, **0.2** compiles but fails, **0** doesn't compile.
3. Programs that scored **above the group's average** are reinforced; those below are discouraged.
4. Repeat across many problems and steps.

This is "RL with verifiable rewards": the reward comes from actually compiling and running the code, not
from a human or a learned judge.

**Why it comes after SFT.** GRPO only learns when attempts *differ* in score. Base Qwen fails almost
everything, so most groups would score all-zero and give nothing to learn from. After SFT, the student passes
about half the time, which is the range where GRPO works best.

**Plan:**
- **Start** from the SFT checkpoint, on the Serverless Training API (`async_rl_loop_serverless`, built-in
  clipped GRPO loss), same renderer and prompt.
- **Problem pool:** MBPP problems where the SFT model passes some but not all of 4 samples.
  Always-pass and always-fail problems carry no signal.
- **Steps:** 8 problems × 8 rollouts per step. 2-step smoke test, then train until 3:00 PM. Hard dollar
  cap in the script.
- **Evaluate** the GRPO checkpoint on all 146 problems, through the same sampler and settings, compared
  against its SFT starting point (McNemar + bootstrap).
- **Fallbacks:** if GRPO isn't converging by 2:45, switch to expert iteration (sample, keep passers,
  retrain). If that fails too, SFT is the final model and GRPO is reported as an experiment.

**Cost estimate** (on KJ's credits; Qwen rates $1.86 prefill / $5.595 sample / $4.103 train per 1M tokens;
~470 prompt + ~500 output tokens measured):

| Item | Estimate |
|---|---|
| SFT rerun on KJ's account (see below) | $4.64 |
| Problem-pool selection | ~$2.50–5.50 |
| Smoke test (2 steps) | ~$1 |
| GRPO, ~$0.45/step × 15–30 steps | **~$7–14** |
| Evals (146 problems each) | ~$0.60 each |
| **Total** | **~$20–26** (worst case ~$40) |

**Accounts.** The SFT checkpoint lives in Ritik's Fireworks account, and checkpoints can only be
sampled or trained by the account that owns them. So GRPO runs **on Ritik's account**, starting directly from
his SFT `step-90`. Scripts read his key from `.env` as `RITIK_FIREWORKS_API_KEY`.

**What has run so far (2026-11-02):**

| Stage | Result |
|---|---|
| Problem pool (`rl/select_pool.py`) | SFT sampled 4× per MBPP problem: **331/459 problems** have varying rewards (87 always pass, 22 always 0); SFT mean reward 0.477; $4.64 |
| Smoke test (2 steps) | Pipeline works end to end; ~$0.50/step, ~50 s/step |
| Full run `grpo-v1` (`rl/grpo.py`) | 36 steps × 8 problems × 8 rollouts, lr 2e-5, LoRA r16 from SFT `step-90`; running, est. ~$18. Training is stable (grad norm 0.06–0.22, clip ≈0.1%, no length blow-up) |
| Eval | Final checkpoint → all 146 problems via `eval/run_insession.py --state grpo=<checkpoint>`, McNemar vs SFT |

**What counts as success.** SFT is 75/146 (51.4%). For the gain over SFT to be significant at n=146 (McNemar
p < 0.05), GRPO needs roughly **+7–9 net problems with few regressions, about ≥57% pass@1**:
- 54–57% is positive but unproven;
- 50–53% means no effect, and SFT ships as the final model.

**If GRPO is flat.** The next lever is a compiler-feedback repair round. 29 of SFT's failures are genuine
compile errors, mostly a single error each. It would be reported as its own system, not as pass@1.

---

## Rules that keep the results honest

- **COBOLEval is held out.** It's never used for training, prompt tuning or data generation. The only
  exception is the 30-problem check, used to pick the teacher.
- **Same everything for every system.** Same `prompt.py`, temperature 0, 4,096 max tokens, same renderer
  (reasoning off) for the student, and the same grader.
- **Cache every model response.** Results can be re-graded without re-calling models
  (`results/raw/`, `eval/cache/`).
- **Statistics.** McNemar on per-problem pass/fail; bootstrap that resamples *problems*, not tests,
  5,000 times, 95% CI.
- **Report everything.** That includes losses and settings that differ: GLM and gpt-oss ran at reasoning
  "low", and the two broken COBOLEval problems are noted.

## Caveats

- One training run, one seed, 146 problems.
- COBOLEval is algorithmic COBOL (HumanEval translated), not enterprise COBOL with files, copybooks or CICS.
- The large models still score higher on the 30-problem check (Kimi 77% with the section fix). The pitch is
  quality per dollar and the size of the gain, not beating them.
- Costs are estimates from list prices; prefill is counted as uncached, so they are upper bounds.

## Where the files are

| Path | What |
|---|---|
| `grader/grade.py`, `grader/construct.py` | Judge + answer extraction |
| `prompt.py` | Frozen prompt |
| `data/convert_mbpp.py` → `data/mbpp_cobol.jsonl` | MBPP → COBOL problems |
| `data/sample_teacher.py` → `data/train.jsonl` | Teacher rejection sampling |
| `sft/train_serverless.py`, `sft/RUN.md` | SFT run |
| `eval/run.py`, `eval/systems.py` | 30-problem check (chat models + sampler) |
| `eval/run_insession.py`, `eval/compare.py` | Base vs SFT eval + statistics |
| `results/` | One row per problem per system |
| `rl/select_pool.py`, `rl/grpo.py`, `rl/pool.jsonl` | GRPO pool selection + training |
