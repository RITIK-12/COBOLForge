# COBOLForge

A 27B open model, fine-tuned on Fireworks to write COBOL that compiles and passes.

[![Project page](https://img.shields.io/badge/Project_page-COBOLForge-e8662a?style=for-the-badge&logo=githubpages&logoColor=white)](https://ritik-12.github.io/COBOLForge/)

## Results

COBOLEval: 146 held-out problems (HumanEval in COBOL), 821 tests, never used for training. Every system gets the
same prompt (`prompt.py`), temperature 0, 4,096 max tokens, one sample, and the same grader (GnuCOBOL 3.2).

| System | Size | pass@1 | 95% CI | $/problem |
| --- | --- | --- | --- | --- |
| Muse Spark Open (thinking low) | 816B | 112/146 (76.7%) | [69.9%, 83.6%] | $0.0055 |
| Qwen 3.8 Max | 2.4T | 87/146 (59.6%) | [52.1%, 67.8%] | $0.0051 |
| GLM-5.3 (thinking low) | 743B | 80/146 (54.8%) | [46.6%, 63.0%] | $0.0029 |
| **Our SFT** (Qwen 3.8 27B) | 27B | **75/146 (51.4%)** | [43.2%, 59.6%] | $0.0041 |
| **Our GRPO** (from SFT) | 27B | 73/146 (50.0%) | [42.5%, 58.2%] | |
| DeepSeek V4.1 Flash 1017 | 755B | 73/146 (50.0%) | [41.8%, 58.2%] | $0.0008 |
| Kimi K3 (strict) | 2.8T | 28/146 (19.2%) | [13.0%, 25.3%] | $0.0104 |
| Qwen 3.8 27B base | 27B | 19/146 (13.0%) | [7.5%, 19.2%] | $0.0051 |

- **SFT: 13.0% → 51.4%**, +38.4 points (95% CI [+29.5, +46.6]), McNemar 59 vs 3, p = 1.7e-14.
  Programs that compile: 33% → 78%.
- **vs frontier open models:** our 27B is statistically tied with Qwen 3.8 Max (p = 0.08), GLM-5.3 (p = 0.56)
  and DeepSeek V4.1 Flash 1017 (p = 0.88). Only Muse Spark Open, which can't turn thinking off, is ahead.
- **Kimi K3** mostly fails by putting WORKING-STORAGE after LINKAGE. With COBOLEval's own section-order fix it
  scores 74.0% (DeepSeek 66.4%; everyone else unchanged): `results/section_fix.json`.
- **GRPO** (37 steps, 2,536 rollouts, reward 1 / 0.2 / 0) did not improve on SFT.
- **Contamination check:** dropping the 13 COBOLEval problems most similar to any training problem still
  leaves +34.6 points (p = 1e-11).
- Caveats: one training run, one seed, n = 146; HumanEval/74 and /160 can't pass for any model (broken
  callers upstream); costs are list-price upper bounds.

## How it works

1. **Grader** (`grader/`): GnuCOBOL 3.2 + the COBOLEval harness, sandboxed. One `grade()` is the data filter,
   the eval metric and the RL reward.
2. **Data** (`data/`): 974 MBPP Python problems → 459 COBOL problems with 1,371 test callers → Kimi K3 writes
   918 programs → **709 that pass every test** become `data/train.jsonl`.
3. **SFT** (`sft/`): LoRA r16 on Qwen 3.8 27B, Fireworks Serverless Training, 90 steps, ~6 min, $4.64.
   Run record: [sft/RUN.md](sft/RUN.md).
4. **GRPO** (`rl/`): 8 attempts per problem from the SFT checkpoint, group-relative advantages.
5. **Eval** (`eval/`): `run_insession.py` (base, SFT, GRPO), `run.py` (frontier models), `compare.py` (McNemar +
   bootstrap over problems).

Full walkthrough: [explainer.md](explainer.md). Per-problem rows: `results/`; raw outputs: `eval/cache/`, `results/raw/`.

## Reproduce

```bash
/opt/homebrew/bin/brew install gnucobol
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python httpx numpy python-dotenv "fireworks-ai>=1.2.30,<2"
export FIREWORKS_API_KEY=...                                   # see .env.example
.venv/bin/python -m grader.grade --self-test
.venv/bin/python -m data.convert_mbpp && .venv/bin/python -m data.sample_teacher --samples 2
.venv-train/bin/python -m sft.train_serverless train --output-model-id cobol-sft-qwen38-<stamp>   # cookbook venv
.venv-train/bin/python -m eval.run_insession                    # base + SFT
.venv/bin/python -m eval.run --system muse-spark-open           # any frontier model in eval/systems.py
.venv/bin/python -m eval.compare qwen38-27b-base qwen38-27b-sft
```

Every model response is cached and never re-requested. The project page is built from `results/` by
`site/build.py` on every push to main.

Data: MBPP (CC BY 4.0), COBOLEval (MIT). Built with AI coding assistants (Claude Code).
