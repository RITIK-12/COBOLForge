"""COBOLEval-compatible grader. Same function is the data filter, eval metric and RL reward.

Mirrors zorse-project/COBOLEval scripts/evaluation.py @ 0bb96c3 exactly where it matters:
  - compile: cobc -w -fformat=variable -x <caller.cbl> <solution.cbl>   (one build per test)
  - run the caller; non-zero exit = test fails
  - read <ENTRY-POINT>.TXT (upper-case, '_' -> '-'), parse by result type_, compare
  - floats: abs_tol=0.001; lists truncated to len(expected) before comparing
Deviations (none change pass/fail): ast.literal_eval instead of eval for expected values;
each grade runs in its own temp dir so calls are safe to run in parallel; we stop after the
first compile failure (the solution is shared by every test, so the rest would fail too);
`compiled` reports real compile status (upstream folds runtime errors into "not compiled").

problem dict needs: entry_point, tests=[{test: <caller COBOL>, result: {value, type_}}].
MBPP-derived training problems must use the same shape.
"""

import ast
import math
import os
import resource
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

RUN_TIMEOUT_S = 10
COMPILE_TIMEOUT_S = 30
COBC = os.environ.get("COBC") or shutil.which("cobc") or "/opt/homebrew/bin/cobc"


class ParseError(Exception):
    pass


def _parse_int(res: str) -> int:
    res = res.strip()
    if res.startswith("p") or res.startswith("y"):
        return -int(res[1:])
    return int(res)


def _parse_float(res: str) -> float:
    res = res.strip()
    if res.startswith("p") or res.startswith("y"):
        return -float(res[1:])
    return float(res)


def _parse(lines, type_, true):
    try:
        match type_:
            case "Bool":
                return lines[0].strip() == "1"
            case "Int":
                return _parse_int(lines[0])
            case "Float":
                return _parse_float(lines[0])
            case "String":
                return lines[0].strip()
            case {"List": "Int"}:
                return [_parse_int(x) for x in lines][: len(true)]
            case {"List": "Float"}:
                return [_parse_float(x) for x in lines][: len(true)]
            case {"List": "String"}:
                return [x.strip() for x in lines][: len(true)]
            case _:
                raise ParseError(f"invalid result type {type_}")
    except ParseError:
        raise
    except Exception as e:
        raise ParseError(f"result {lines[:5]} of type {type_} failed: {e}")


def _is_equal(type_, result, true) -> bool:
    match type_:
        case "Float":
            return math.isclose(result, true, abs_tol=0.001)
        case {"List": "Float"}:
            # upstream zips (no length check); keep identical semantics
            return all(math.isclose(r, t, abs_tol=0.001) for r, t in zip(result, true))
        case _:
            return result == true


def _limits():
    # Best-effort sandboxing for untrusted programs: CPU, output-file size, no core dumps.
    for lim, val in (
        (resource.RLIMIT_CPU, (RUN_TIMEOUT_S + 1, RUN_TIMEOUT_S + 1)),
        (resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024)),
        (resource.RLIMIT_CORE, (0, 0)),
    ):
        try:
            resource.setrlimit(lim, val)
        except (ValueError, OSError):
            pass


def _expected(test):
    true = ast.literal_eval(test["result"]["value"])
    return list(true) if isinstance(true, tuple) else true


def grade(program: str, problem: dict) -> dict:
    name, tests = problem["entry_point"], problem["tests"]
    result_file = f"{name.upper().replace('_', '-')}.TXT"
    out = {"compiled": False, "passed": False, "n_pass": 0, "n_tests": len(tests), "error": None}
    if not program or not program.strip():
        out["error"] = "empty program"
        return out

    with tempfile.TemporaryDirectory(prefix="cobgrade_") as d:
        sol = os.path.join(d, f"{name}.cbl")
        caller = os.path.join(d, f"call_{name}.cbl")
        exe = os.path.join(d, f"call_{name}")
        with open(sol, "w") as f:
            f.write(program)

        n_pass, errors = 0, []
        for i, test in enumerate(tests):
            with open(caller, "w") as f:
                f.write(test["test"])
            for p in (exe, os.path.join(d, result_file)):
                if os.path.exists(p):
                    os.remove(p)

            try:
                c = subprocess.run(
                    [COBC, "-w", "-fformat=variable", "-x", "-o", exe, caller, sol],
                    cwd=d, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                out["error"] = "compile timeout"
                break
            if c.returncode != 0:
                msg = (c.stderr or c.stdout).replace(d + "/", "").strip()
                if f"{name}.cbl:" not in msg.replace(f"call_{name}.cbl:", ""):
                    # error is in the dataset's caller (HumanEval/74, /160 have out-of-bounds
                    # subscripts), not the solution: fail just this test, like upstream
                    errors.append(f"test {i}: caller compile error")
                    continue
                out["error"] = "compile: " + msg[:2000]
                break
            out["compiled"] = True

            try:
                r = subprocess.run(
                    [exe], cwd=d, capture_output=True, text=True,
                    timeout=RUN_TIMEOUT_S, preexec_fn=_limits, stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired:
                errors.append(f"test {i}: run timeout")
                continue
            if r.returncode != 0:
                errors.append(f"test {i}: runtime rc={r.returncode} {(r.stderr or r.stdout).strip()[:300]}")
                continue

            try:
                with open(os.path.join(d, result_file), errors="replace") as f:
                    lines = f.readlines()
                if not lines:
                    errors.append(f"test {i}: empty result file")
                    continue
                true, type_ = _expected(test), test["result"]["type_"]
                got = _parse(lines, type_, true)
                if _is_equal(type_, got, true):
                    n_pass += 1
                else:
                    errors.append(f"test {i}: expected {true!r} got {got!r}")
            except FileNotFoundError:
                errors.append(f"test {i}: no result file")
            except Exception as e:
                errors.append(f"test {i}: {e}")

        out["n_pass"] = n_pass
        out["passed"] = out["compiled"] and n_pass == len(tests)
        if out["error"] is None and errors:
            out["error"] = "; ".join(errors)[:2000]
    return out


def grade_many(pairs, workers: int = None):
    """pairs: iterable of (program, problem). Returns list of grade dicts in order."""
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda pp: grade(*pp), pairs))


def _self_test() -> int:
    """Reference solution passes, mutants fail, every COBOLEval caller builds against a stub."""
    import json, time
    from grader.construct import construct

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    problems = [json.loads(l) for l in open(os.path.join(root, "eval/data/CobolEval.jsonl"))]
    p0 = problems[0]
    good = open(os.path.join(here, "selftest/has_close_elements.cbl")).read()
    cases = [
        ("reference", good, dict(compiled=True, passed=True, n_pass=4)),
        ("wrong answer", good.replace("MOVE 1 TO RESULT", "MOVE 0 TO RESULT"), dict(compiled=True, passed=False, n_pass=2)),
        ("compile error", good.replace("END-PERFORM\n           GOBACK", "GOBACK"), dict(compiled=False, passed=False, n_pass=0)),
        ("infinite loop", good.replace("MOVE 0 TO RESULT\n", "MOVE 0 TO RESULT\n           PERFORM UNTIL 1 = 2 CONTINUE END-PERFORM\n", 1), dict(compiled=True, passed=False, n_pass=0)),
    ]
    ok = True
    for label, prog, want in cases:
        t = time.time()
        r = grade(prog, p0)
        good_case = all(r[k] == v for k, v in want.items())
        ok &= good_case
        print(f"{'PASS' if good_case else 'FAIL'}  {label:14s} {time.time()-t:5.1f}s  {r}")

    stub = "```cobol\n       PROCEDURE DIVISION.\n           GOBACK.\n```"
    progs = [construct(p["prompt"], stub) + f"\n       END PROGRAM {p['entry_point'].upper().replace('_', '-')}.\n" for p in problems]
    t = time.time()
    rs = grade_many(list(zip(progs, problems)))
    known_bad = {"HumanEval/74", "HumanEval/160"}  # dataset callers subscript out of bounds
    broken = [p["task_id"] for p, r in zip(problems, rs) if not r["compiled"] and p["task_id"] not in known_bad]
    leaked = [p["task_id"] for p, r in zip(problems, rs) if r["passed"]]
    stub_ok = not broken and not leaked
    ok &= stub_ok
    print(f"{'PASS' if stub_ok else 'FAIL'}  stub x146      {time.time()-t:5.1f}s  "
          f"compiled {sum(r['compiled'] for r in rs)}/146 (known-bad callers: {sorted(known_bad)}), "
          f"stub passes {len(leaked)}, unexpected compile failures {broken}")
    print("SELF-TEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print("usage: python -m grader.grade --self-test")
