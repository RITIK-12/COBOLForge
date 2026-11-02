# COBOLForge — Project Plan

Buildathon: Mon Nov 2, 2026 · Build 10:30–4:30 PT · Start submitting by 3:30 · Uploads must finish by 4:30
Repo: RITIK-12/COBOLForge · Team: KJ (RL / post-training) + Ritik (SFT)

---

## 1. Pitch

A small open model, fine-tuned on Fireworks, that writes correct COBOL.
Banks, insurers and governments still run on COBOL, the people who write it are retiring,
and general models are weak at it. We show a statistically significant gain on a published,
held-out COBOL benchmark at a fraction of frontier cost and latency.

**Headline claim:** "Base 30B goes from X% to Y% pass@1 on COBOLEval (McNemar p, bootstrap 95% CI),
at $Z per problem vs $W for frontier models."
Beating frontier models on correctness is a bonus slide, not the claim.

---

## 2. Benchmark (held-out test)

- **COBOLEval** (zorse-project): 146 problems, 821 tests, HumanEval translated to COBOL.
- Never used for training, prompt tuning, data generation or teacher selection beyond the 30-problem check.
- Metric: **pass@1 per problem** (all of a problem's tests pass = 1), temperature 0.
- Secondary: compile rate, cost per problem, p50 latency.

### Harness details grade.py must copy exactly
- Build: `cobc -w -fformat=variable -x <caller.cbl> <solution.cbl>`; each test has its own caller program.
- Output: caller writes its result to `<ENTRY-POINT>.TXT` (name upper-cased, `_` → `-`).
- Parsing by result type: Bool, Int, Float, String, List. Negative numbers carry a `p`/`y` prefix.
  Floats compared with tolerance 0.001. Use the harness's own parser.
- Prompt assembly: model code is appended to the problem prompt, then WORKING-STORAGE and
  LINKAGE sections are swapped back into order.
- Upstream disables the run step for safety; we enable it only inside the sandbox, 10s timeout.
- Target GnuCOBOL 3.2.0.

---

## 3. Systems compared

| System | Role | Required? |
| --- | --- | --- |
| Base student (Nemotron Lightning 3.5 30B; fallback Qwen3.8 27B) | Baseline | **Yes** |
| Student + SFT | Main result | **Yes** |
| Student + SFT + GRPO (or expert iteration) | Post-training result | If time allows |
| Teacher (strongest open coder on Fireworks) | Context, data source | Yes (cheap) |
| One Claude + one GPT model via OpenRouter | Frontier context | Optional |

All systems: same `prompt.py`, temperature 0, max_tokens 4096. No weakened baselines.

---

## 4. Grader (KJ)

- **Sandbox: native + static guard + sandbox-exec.** GnuCOBOL 3.2.0 from Homebrew, run directly on macOS.
  - *Static guard* (before compiling): reject any `CALL` except the problem's own entry point
    (the only target COBOLEval callers use), `SELECT … ASSIGN`, `OPEN`, `DELETE FILE`, `EXEC`.
    Matched with comments stripped and literals blanked. Returns `compiled=false, passed=false, error="blocked"`.
  - *sandbox-exec* on the run step: no network, no file writes outside the per-grade temp dir
    (`GRADER_SANDBOX=0` turns it off, e.g. on Linux).
  - 10s run timeout; after a problem's first timeout its remaining tests are skipped. Limits on CPU, file size and core dumps.
  - Each `grade()` call gets its own temp dir, so calls run in parallel (`grade_many`).
- **API:** `grade(program, problem) -> {compiled, passed, n_pass, n_tests, error}`.
  Same function is the data filter, the eval metric and the RL reward. Safe for concurrent calls.
- **Self-test:** `python -m grader.grade --self-test`: the reference passes; wrong, broken, blocked and hanging
  programs fail as expected; the sandbox blocks escapes; all 146 callers build against a stub.
  HumanEval/74 and /160 have broken dataset callers (out-of-bounds subscripts) and can't pass for any system.
- Measured: ~1s per problem (one compile + run per test), ~25s for all 146 in parallel.

---

## 5. Training data (Ritik)

Source: **MBPP problems and tests only** (the Python solutions are discarded).

1. Filter MBPP to signatures with ints, strings, flat lists of ints/strings → expect ~400–500 problems.
2. Convert each problem to COBOLEval's harness format (signature → COBOL skeleton, asserts → caller programs).
   Reuse COBOLEval's own conversion code where possible.
3. **Rejection sampling:** teacher writes 8 samples per problem at temperature 0.7; keep only programs
   that compile and pass all tests via `grade()`.
4. Keep at most 2 passing programs per problem; target 300–600 examples.
5. Dedupe near-matches against HumanEval / COBOLEval.
6. Save `data/train.jsonl` and yield stats (tried / compiled / passed) for the README.
7. Save failing samples too: they become DPO rejected examples and the GRPO problem pool.

**Fallback:** if the teacher is no better than the student, sample from the student itself
(self-distillation / rejection-sampling fine-tuning).

---

## 6. SFT (Ritik)

- LoRA rank 16, alpha 32, lr 1e-4, 2 epochs, batch 16, max length 4096, loss on completion only.
- Format: system = eval system prompt; user = COBOLEval-style skeleton + task; assistant = full COBOL program, no prose.
- Prefer Fireworks serverless training (no queue; score via `save_weights_for_sampler`, no deployment).
  Fallback: managed SFT (`firectl sftj create`).
- **Launch by 12:45, hard deadline 1:00.** Time the first 3 steps and extrapolate the finish time.
- Estimated total (setup → trained → evaluated): 45–90 min, done ~1:30–2:15.

---

## 7. Post-training (KJ)

Verifiable reward from `grade()`: **1.0** all tests pass · **0.2** compiles but fails · **0** does not compile.

**Primary — GRPO** (if SFT is evaluated by ~2:00)
- Start from the SFT checkpoint.
- Problem pool: MBPP problems where SFT passes 1–6 of 8 samples (always-pass / always-fail give no signal).
- 8 rollouts per problem, group-relative advantages, policy-gradient loss in the Fireworks training loop
  (`forward_backward` with a custom loss instead of cross-entropy).
- Steps = minutes left before 3:00 ÷ measured minutes per step (expect ~15; 30 if fast). Stop at 3:00.
- Smoke-test 2 steps on the base model with 8 problems during the 12:45–2:00 wait.

**Fallback — expert iteration** (if SFT lands after 2:15): sample SFT 4× on failed MBPP problems,
keep passers, retrain. Labelled honestly as filtered-imitation RLVR.

**Option — DPO on verifier pairs** (only if Fireworks supports DPO for the student):
chosen = passing program, rejected = failing program on the same problem.
Called "preference optimization with verifier labels", not RLHF.

**No RLHF:** no human preference data or reward model; not feasible in the time.

---

## 8. Evaluation & statistics (KJ)

- `eval/run.py`: async, bounded concurrency (~16), retries with backoff, caches every row,
  skips ids already in `results/{system}.jsonl`.
- Row format: `{id, system, pass, compiled, tokens, latency_s, cost}`.
- **McNemar** on per-problem pass/fail (fine-tune vs base; GRPO vs SFT; vs frontier).
- **Bootstrap:** resample problems (not tests — tests within a problem are clustered), 5,000 iterations, 95% CI.
- Call a gain real only when both agree. ~10+ point gain needed for significance at n=146.
- Report every result, including losses and non-significant runs.

### Charts
1. Leaderboard: pass@1 with 95% CI, compile rate, cost per problem, latency.
2. McNemar disagreement bars per comparison.
3. Bootstrap CI chart per comparison.
4. Failure breakdown: compile errors vs wrong answers, before vs after.

---

## 9. Shared contracts

| Contract | Spec | Owner |
| --- | --- | --- |
| `prompt.py` | One system prompt + template for all training and eval; frozen at 11:45 | Both |
| `grader/grade.py` | `grade()` API above | KJ |
| `data/train.jsonl` | `{id, messages: [system, user, assistant]}` | Ritik |
| `results/{system}.jsonl` | One row per problem (format above) | Both |

Repo layout: `grader/ data/ sft/ rl/ eval/ results/ charts/`. Each person works in their own folders,
small commits, push to main often. Secrets only from env (`FIREWORKS_API_KEY`, `OPENROUTER_API_KEY`).

---

## 10. Timeline

| Time | KJ (RL / post-training) | Ritik (SFT) | Sync / gate |
| --- | --- | --- | --- |
| 11:21–11:45 | Native grader, `grade.py`, self-test; Fireworks setup; base + teacher on 30 problems | Confirm student trainable; filter + convert MBPP | |
| **11:45** | | | **Sync 1:** go/no-go, pick teacher, freeze prompt |
| 11:45–12:45 | Full 146 baselines (base, teacher, frontier); `stats.py` | Rejection sampling, dedupe, `train.jsonl`, **launch SFT** | |
| **12:45** | | | **Sync 2:** data yield, baseline numbers, SFT job ID |
| 12:45–2:00 | GRPO loop + reward; 2-step smoke test | Monitor SFT; backup data (expert iteration, DPO pairs); README skeleton; charts; video clip of base failing | |
| **~2:00** | | | **Sync 3:** SFT checkpoint. Not ready by 2:15 → skip GRPO, run expert iteration |
| 2:00–3:00 | Filter pool, run GRPO | Eval SFT on 146, stats, charts | |
| **3:00** | | | **Sync 4:** stop GRPO, take latest checkpoint |
| 3:00–3:30 | Eval GRPO checkpoint, final stats | Leaderboard, charts, rubric table | |
| 3:30–4:15 | Record and edit video | Repo check in incognito, README, submit | |
| 4:15–4:30 | Buffer | Buffer | Uploads processed by 4:30 |

---

## 11. Go / no-go gates

| Check | Action |
| --- | --- |
| Nemotron not trainable on our account | Switch to Qwen3.8 27B at Sync 1 |
| Base student > ~50% on 30-problem check | Expect a small gain; lean on cost/latency story |
| Teacher ≤ base student | Sample from the student instead |
| Frontier > 70% on 30 problems | Pitch is "same quality at a fraction of cost and latency" |
| < 300 verified examples at 12:45 | Train on what we have; don't extend the data phase |
| SFT not evaluated by 2:15 | Expert iteration instead of GRPO |
| GRPO not converging by 2:45 | Stop, report it as tried, use SFT as the final model |

**Hard stops:** SFT launch ≤ 1:00 · GRPO stops 3:00 · nothing new after 3:30.
**Blocked > 10 min:** say so and pick up the other person's backup task.

---

## 12. Risks

| Risk | Mitigation |
| --- | --- |
| MBPP → harness conversion is slow or lossy | Filter to simple types; reuse COBOLEval converter; 90-min cap |
| Fireworks queue / deployment time | Serverless training; score in-session |
| Grader too slow for RL rollouts | Native (no container overhead), parallel `grade_many`, early stop on timeout |
| Gain not significant at n=146 | Report honestly; show compile-rate gain and failure breakdown |
| Contamination concerns | Dedupe MBPP vs HumanEval; never touch COBOLEval for training |
| GRPO degrades the model | Keep SFT as final; report GRPO as an experiment |

---

## 13. Submission checklist

- [ ] Public repo, checked in an incognito window
- [ ] README: run order, cost per step, data yield, results table, rubric mapping, caveats, disclosure
- [ ] `results/` saved per system so judges can read without rerunning
- [ ] Charts: leaderboard, McNemar, bootstrap, failure breakdown
- [ ] Video under 60s: problem → base model fails to compile → fine-tune passes → significance chart → number + one caveat
- [ ] `.env.example` with `FIREWORKS_API_KEY`, `OPENROUTER_API_KEY`; dependencies pinned (`uv.lock`)
- [ ] Disclosure: AI coding assistants used (Claude Code)

---

## 14. Rubric mapping

| Criterion | Weight | What we show | Honest caveat |
| --- | --- | --- | --- |
| Benchmark evaluation | 40% | Same model before/after on 146 held-out problems; McNemar + problem-level bootstrap; all runs reported | One training run, one seed; n=146 |
| Business value | 25% | Legacy COBOL in banking, insurance, government; cost and latency vs frontier | Algorithmic tasks, not enterprise COBOL (file I/O, copybooks, CICS) |
| Technical execution | 20% | Verified rejection-sampled data, SFT + RLVR on a sandboxed execution reward | No hyperparameter sweep |
| Presentation & demo | 10% | Compile-fail → pass demo; significance charts | |
| Documentation | 5% | README with run order, cached results, pinned deps | |
