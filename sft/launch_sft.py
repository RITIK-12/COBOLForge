"""Managed LoRA SFT on Fireworks for the student (Nemotron Lightning 3.5 30B A3B).

  check   validate data/train.jsonl locally, estimate tokens and cost (no API writes)
  launch  upload the dataset and create the SFT job (billable; run only after plan approval)
  watch   poll the job: state, progress, status message; prints render-sample + metrics URLs

Run state is appended to sft/runs.jsonl. Run: .venv/bin/python -m sft.launch_sft check
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "data/train.jsonl")
RUNS = os.path.join(ROOT, "sft/runs.jsonl")
ACCOUNT = os.environ.get("FIREWORKS_ACCOUNT_ID", "ritik-7xa24uxitcg9")
BASE_MODEL = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
PARAMS = dict(lora_rank=16, epochs=2, learning_rate=1e-4, batch_size_samples=16, max_context_length=4096)
RATE_PER_M = 3.00  # $/1M training tokens, LoRA SFT, 16.1B-80B (fireworks.ai/pricing, 2026-11-02)
CHARS_PER_TOKEN = 3.0  # rough; COBOL has many short tokens. Real count is in the job's metrics.


def load_rows():
    rows, bad = [], []
    for i, line in enumerate(open(TRAIN), 1):
        try:
            r = json.loads(line)
            m = r["messages"]
            assert [x["role"] for x in m] == ["system", "user", "assistant"], "roles"
            assert all(isinstance(x["content"], str) and x["content"].strip() for x in m), "empty content"
            assert m[2]["content"].startswith("```cobol\n") and m[2]["content"].rstrip().endswith("```"), "fence"
            rows.append(r)
        except Exception as e:
            bad.append((i, str(e)))
    return rows, bad


def check():
    rows, bad = load_rows()
    lens = sorted(sum(len(x["content"]) for x in r["messages"]) / CHARS_PER_TOKEN for r in rows)
    problems = {r["id"].split("#")[0] for r in rows}
    total = sum(lens)
    train_tokens = total * PARAMS["epochs"]
    print(f"rows {len(rows)} from {len(problems)} problems; bad rows {len(bad)} {bad[:5]}")
    print(f"approx tokens/row: median {lens[len(lens) // 2]:.0f}, max {lens[-1]:.0f} (limit {PARAMS['max_context_length']})")
    print(f"approx training tokens {train_tokens / 1e6:.2f}M -> ~${train_tokens / 1e6 * RATE_PER_M:.2f} "
          f"(range x0.7-x1.5: ${train_tokens / 1e6 * RATE_PER_M * 0.7:.2f}-${train_tokens / 1e6 * RATE_PER_M * 1.5:.2f})")
    print(f"steps ~ {len(rows) * PARAMS['epochs'] / PARAMS['batch_size_samples']:.0f}; params {PARAMS}")
    return rows, bad


def client():
    from fireworks import Fireworks
    return Fireworks(account_id=ACCOUNT, timeout=120)


def launch():
    rows, bad = check()
    if bad:
        sys.exit("fix bad rows first")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    dataset_id, job_id = f"cobol-mbpp-sft-{stamp}", f"cobol-sft-{stamp}"
    upload = os.path.join(ROOT, f"sft/{dataset_id}.jsonl")
    with open(upload, "w") as f:  # messages only; ids stay in data/train.jsonl
        for r in rows:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")

    c = client()
    c.datasets.create(dataset_id=dataset_id, dataset={"display_name": dataset_id, "format": "CHAT",
                                                      "example_count": str(len(rows))})
    with open(upload, "rb") as f:
        c.datasets.upload(dataset_id=dataset_id, file=f)
    for _ in range(60):
        d = c.datasets.get(dataset_id=dataset_id).model_dump()
        if str(d.get("state", "")).endswith("READY"):
            break
        time.sleep(5)
    print("dataset", d.get("name"), d.get("state"), d.get("example_count"))

    job = c.supervised_fine_tuning_jobs.create(
        supervised_fine_tuning_job_id=job_id,
        base_model=BASE_MODEL,
        dataset=f"accounts/{ACCOUNT}/datasets/{dataset_id}",
        output_model=f"accounts/{ACCOUNT}/models/{job_id}",
        display_name=job_id,
        eval_auto_carveout=False,
        **PARAMS,
    )
    j = job.model_dump()
    rec = {"time": stamp, "job": j.get("name"), "dataset": dataset_id, "output_model": f"accounts/{ACCOUNT}/models/{job_id}",
           "base_model": BASE_MODEL, "params": PARAMS, "rows": len(rows), "state": j.get("state")}
    with open(RUNS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


def watch(job_id, once=False):
    c = client()
    while True:
        j = c.supervised_fine_tuning_jobs.get(supervised_fine_tuning_job_id=job_id).model_dump()
        st = j.get("status") or {}
        prog = j.get("job_progress") or {}
        print(time.strftime("%H:%M:%S"), j.get("state"), f"{prog.get('percent')}%", f"epoch {prog.get('epoch')}",
              (st.get("message") or "")[:160], flush=True)
        if once or str(j.get("state", "")).split("_")[-1] in ("COMPLETED", "FAILED", "CANCELLED", "EXPIRED"):
            for k in ("render_samples_signed_url", "metrics_file_signed_url"):
                if j.get(k):
                    print(k, "available")
            return j
        time.sleep(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "launch", "watch"])
    ap.add_argument("--job")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    {"check": check, "launch": launch}.get(a.cmd, lambda: watch(a.job, a.once))()
