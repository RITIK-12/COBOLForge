"""Shared bits for rl/: account key, SFT checkpoint, reward, session cleanup.

GRPO runs on the account that owns the SFT checkpoint (Ritik's). Put his key in .env as
RITIK_FIREWORKS_API_KEY; it is copied into FIREWORKS_API_KEY for this process only, because the
cookbook recipe reads that name.
"""

import json
import os
import sys
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grader.construct import construct  # noqa: E402
from grader.grade import GRADE_WORKERS_DEFAULT, grade  # noqa: E402

BASE_MODEL = "accounts/fireworks/models/qwen3p8-27b"
TOKENIZER = "Qwen/Qwen3.8-27B"
RENDERER = "qwen3_8_disable_thinking_interleaved"  # reasoning off, same as SFT and every eval
SFT_STATE = "ritik-7xa24uxitcg9/run-94dffbdc3914486d9d362a614f824d0c/step-90"  # sft/RUN.md
SFT_ACCOUNT = SFT_STATE.split("/")[0]
LORA_RANK, LORA_ALPHA = 16, 32  # must match the SFT adapter
PRICE = {"prefill": 1.86, "sample": 5.595, "train": 4.103}  # $/1M, Qwen 3.8 27B serverless training docs


def use_sft_account_key():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    key = os.environ.get("RITIK_FIREWORKS_API_KEY")
    if not key:
        raise SystemExit("RITIK_FIREWORKS_API_KEY missing from .env (needed: the SFT checkpoint is in his account)")
    os.environ["FIREWORKS_API_KEY"] = key


def load_problems():
    return {p["task_id"]: p for p in map(json.loads, (ROOT / "data/mbpp_cobol.jsonl").open())}


_grade_slots = threading.BoundedSemaphore(GRADE_WORKERS_DEFAULT)


def reward(text: str, problem: dict) -> float:
    """1.0 all tests pass, 0.2 compiles but fails, 0.0 otherwise (PLAN.md §7)."""
    with _grade_slots:  # 64 concurrent rollouts must not fork 64 cobc at once
        g = grade(construct(problem["prompt"], text), problem)
    return 1.0 if g["passed"] else 0.2 if g["compiled"] else 0.0


def delete_session(session_name: str):
    if not session_name:
        return
    req = urllib.request.Request(f"https://api.fireworks.ai/v1/{session_name}", method="DELETE",
                                 headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        print(f"deleted session {session_name} -> HTTP {r.status}", flush=True)
