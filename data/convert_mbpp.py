"""MBPP -> COBOLEval-format problems (data/mbpp_cobol.jsonl).

Mirrors zorse-project/COBOLEval src/main.rs @ 0bb96c3 (prompt skeleton, LINKAGE layout, caller
program, output record, write logic, result {value, type_}), with types inferred from the test
asserts because MBPP has no type hints. Faulty problems/tests are dropped, not repaired:
  - types other than int / str / bool return / flat list of int or str (no floats, nesting, None)
  - list inputs whose length differs across tests (one fixed OCCURS per argument), or empty
  - ints outside +-10^9 (S9(10) SIGN LEADING only parses with a 'p' overpunch, i.e. leading 0)
  - strings > 100 chars, non-ASCII, quotes/control chars, or leading/trailing spaces (PIC X(100))
  - tests expecting [] or "" (any output passes them, see grader parse truncation)
  - fewer than 2 tests left, or every test expecting the same value (a constant program passes)
  - names too long for COBOL (31 chars incl. "-CALL" / "L-"), or near-duplicates of COBOLEval
Callers are then compile-checked against a stub (validate step); problems that fail are dropped.

Run: .venv/bin/python -m data.convert_mbpp
"""

import ast
import json
import os
import re
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data/raw/mbpp.jsonl")
OUT = os.path.join(ROOT, "data/mbpp_cobol.jsonl")
MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
CHARS = ["I", "J", "K", "L", "M"]
INT_LIMIT = 10**9


class Skip(Exception):
    pass


def fmt(name):
    return name.upper().replace("_", "-")


# ---------- type inference from literal values ----------

def scalar_type(v):
    if isinstance(v, bool):
        return "Bool"
    if isinstance(v, int):
        if abs(v) >= INT_LIMIT:
            raise Skip("int out of range")
        return "Int"
    if isinstance(v, str):
        if len(v) > 100 or not v.isascii() or v != v.strip() or re.search(r"[\"'\x00-\x1f\x7f]", v):
            raise Skip("unsupported string")
        return "String"
    raise Skip(f"unsupported type {type(v).__name__}")


def value_type(v):
    if isinstance(v, (list, tuple)):
        inner = {scalar_type(x) for x in v}
        if len(inner) > 1 or inner == {"Bool"}:
            raise Skip("mixed/bool list")
        return {"List": inner.pop() if inner else None}
    return scalar_type(v)


def merge(types):
    """One type across all tests; empty lists take the element type of the others."""
    lists = [t for t in types if isinstance(t, dict)]
    if lists and len(lists) != len(types):
        raise Skip("list/scalar mix")
    if lists:
        inner = {t["List"] for t in lists} - {None}
        if len(inner) != 1:
            raise Skip("unknown list element type")
        return {"List": inner.pop()}
    if len(set(types)) != 1:
        raise Skip("inconsistent scalar types")
    return types[0]


def pic(t):
    return {"Int": "PIC S9(10)", "String": "PIC X(100)", "Bool": "PIC 9"}[t]


# ---------- parse MBPP ----------

def parse_asserts(row, entry):
    tests = []
    for src in row["test_list"] + row["challenge_test_list"]:
        try:
            node = ast.parse(src.strip()).body[0]
        except SyntaxError:
            continue
        if not (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
                and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)):
            continue
        call = node.test.left
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == entry and not call.keywords):
            continue
        try:
            args = [ast.literal_eval(a) for a in call.args]
            expected = ast.literal_eval(node.test.comparators[0])
        except (ValueError, SyntaxError, TypeError):
            continue
        tests.append((args, expected, src.strip()))
    return tests


def entry_point(row):
    m = re.match(r"\s*assert\s+(?:not\s+)?\(?\s*([A-Za-z_]\w*)\s*\(", row["test_list"][0])
    if not m:
        raise Skip("no entry point")
    return m.group(1)


def params(code, entry):
    try:
        tree = ast.parse(code.replace("\r\n", "\n").replace("\t", "    "))
    except SyntaxError:
        raise Skip("code does not parse")
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == entry:
            a = n.args
            if a.vararg or a.kwarg or a.kwonlyargs or a.defaults or a.posonlyargs:
                raise Skip("non-simple signature")
            return [x.arg for x in a.args]
    raise Skip("entry point not defined")


# ---------- COBOL generation (src/main.rs) ----------

def to_linkage(args, ret):
    lines, k = [], 0
    for name, t, n in args:
        if isinstance(t, dict):
            lines.append(f"           05 L-{fmt(name)} OCCURS {n} TIMES INDEXED BY N{CHARS[k]} {pic(t['List'])}.")
            k += 1
        else:
            lines.append(f"           05 L-{fmt(name)} {pic(t)}.")
    if isinstance(ret, dict):
        lines.append(f"           05 RESULT OCCURS 100 TIMES INDEXED BY N{CHARS[k]} {pic(ret['List'])}.")
        idx = CHARS[k]
    else:
        lines.append(f"           05 RESULT {pic(ret)}.")
        idx = None
    return "       01 LINKED-ITEMS.\n" + "\n".join(lines), idx


def cobol_literal(v):
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


def wrap(text, width=88):
    out = []
    for para in text.splitlines():
        words, line = para.split(), ""
        for w in words:
            if line and len(line) + 1 + len(w) > width:
                out.append(line)
                line = w
            else:
                line = f"{line} {w}" if line else w
        out.append(line)
    return out


def description(row, entry, first_test):
    lines = wrap(row["text"].strip())
    args, expected, _ = first_test
    call = f"{entry}({', '.join(repr(a) for a in args)})"
    lines += [f">>> {call}", repr(list(expected) if isinstance(expected, tuple) else expected), ""]
    return "\n".join(f"      * {l}".rstrip() if l else "      * " for l in lines)


def prompt(program_name, desc, linkage):
    return f"""       IDENTIFICATION DIVISION.
       PROGRAM-ID. {program_name}.

       ENVIRONMENT DIVISION.

       INPUT-OUTPUT SECTION.

       DATA DIVISION.

       LINKAGE SECTION.

{linkage}

{desc}

      * Complete the WORKING-STORAGE SECTION and the PROCEDURE DIVISION
      * Store the result in the RESULT variable and mark the end of your program with END PROGRAM

       WORKING-STORAGE SECTION.

       """


def caller(program_name, ret, linkage, idx, arg_specs, values):
    inner = ret["List"] if isinstance(ret, dict) else ret
    record = f"{pic(inner)} SIGN LEADING" if inner == "Int" else pic(inner)
    moves = []
    for (name, t, _), v in zip(arg_specs, values):
        if isinstance(t, dict):
            moves += [f"       MOVE {cobol_literal(x)} TO L-{fmt(name)}({i + 1})" for i, x in enumerate(v)]
        else:
            moves.append(f"       MOVE {cobol_literal(v)} TO L-{fmt(name)}")
    if isinstance(ret, dict):
        write = (f"       PERFORM VARYING N{idx} FROM 1 BY 1 UNTIL N{idx} > 100\n"
                 f"           MOVE RESULT (N{idx}) TO OUTPUT-RECORD\n"
                 f"           WRITE OUTPUT-RECORD\n"
                 f"       END-PERFORM")
    else:
        write = "       MOVE RESULT TO OUTPUT-RECORD\n       WRITE OUTPUT-RECORD\n"
    moves = "\n".join(moves)
    return f"""       IDENTIFICATION DIVISION.
       PROGRAM-ID. {program_name}-CALL.

       ENVIRONMENT DIVISION.

       INPUT-OUTPUT SECTION.

       FILE-CONTROL.

       SELECT OUTPUT-FILE ASSIGN TO "{program_name}.TXT"
           ORGANIZATION IS LINE SEQUENTIAL
           STATUS IS OUTPUT-FILE-STATUS.

       DATA DIVISION.

       FILE SECTION.
       FD OUTPUT-FILE.
       01 OUTPUT-RECORD {record}.

       WORKING-STORAGE SECTION.

       01 OUTPUT-FILE-STATUS PIC X(02).



{linkage}

       PROCEDURE DIVISION.

{moves}

       CALL "{program_name}" USING LINKED-ITEMS

       OPEN OUTPUT OUTPUT-FILE

       IF OUTPUT-FILE-STATUS NOT = "00"
           DISPLAY "ERROR OPENING OUTPUT FILE"
           STOP RUN
        END-IF

{write}

        IF OUTPUT-FILE-STATUS NOT = "00"
            DISPLAY "ERROR WRITING TO OUTPUT FILE"
            STOP RUN
        END-IF

        CLOSE OUTPUT-FILE
        .
       """


# ---------- decontamination against COBOLEval (removal only; nothing from it is trained on) ----------

def _words(s):
    return set(re.findall(r"[a-z]+", s.lower())) - {"a", "an", "the", "of", "to", "and", "in", "is", "given",
                                                    "function", "write", "python", "return", "list", "for", "if"}


def load_heldout():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "eval/data/CobolEval.jsonl"))]
    out = []
    for r in rows:
        desc = " ".join(l.strip()[1:] for l in r["prompt"].splitlines()
                        if l.strip().startswith("*") and ">>>" not in l and "Complete the" not in l
                        and "Store the result" not in l)
        out.append((r["entry_point"], _words(desc)))
    return out


def near_duplicate(entry, text, heldout):
    w = _words(text)
    for ep, hw in heldout:
        if ep == entry:
            return True
        if w and hw and len(w & hw) / len(w | hw) >= 0.5:
            return True
    return False


# ---------- main ----------

def convert(row, heldout):
    if row["test_setup_code"].strip():
        raise Skip("test setup code")
    entry = entry_point(row)
    if len(fmt(entry)) > 26:
        raise Skip("name too long")
    names = params(row["code"], entry)
    if any(len(fmt(n)) > 29 for n in names) or len({fmt(n) for n in names}) != len(names):
        raise Skip("bad arg names")
    if near_duplicate(entry, row["text"], heldout):
        raise Skip("near-duplicate of COBOLEval")

    raw = [t for t in parse_asserts(row, entry) if len(t[0]) == len(names)]
    tests = []
    for args, expected, src in raw:
        try:
            a_types = [value_type(a) for a in args]
            r_type = value_type(expected)
        except Skip:
            continue
        if any(t == "Bool" for t in a_types):
            continue
        if expected in ([], (), ""):  # any output passes these
            continue
        if isinstance(r_type, dict) and len(expected) > 100:
            continue
        tests.append((args, expected, src, a_types, r_type))
    if len(tests) < 2:
        raise Skip("fewer than 2 usable tests")

    arg_types = [merge([t[3][i] for t in tests]) for i in range(len(names))]
    ret = merge([t[4] for t in tests])
    if sum(isinstance(t, dict) for t in arg_types + [ret]) > len(CHARS):
        raise Skip("too many lists")
    arg_specs = []
    for i, (n, t) in enumerate(zip(names, arg_types)):
        size = None
        if isinstance(t, dict):
            sizes = {len(test[0][i]) for test in tests}
            if len(sizes) != 1 or 0 in sizes:
                raise Skip("list input length varies across tests")
            size = sizes.pop()
        arg_specs.append((n, t, size))
    if len({repr(list(t[1]) if isinstance(t[1], tuple) else t[1]) for t in tests}) == 1:
        raise Skip("all tests expect the same value")

    program_name = fmt(entry)
    linkage, idx = to_linkage(arg_specs, ret)
    first = (tests[0][0], tests[0][1], tests[0][2])
    return {
        "task_id": f"MBPP/{row['task_id']}",
        "prompt": prompt(program_name, description(row, entry, first), linkage),
        "entry_point": entry,
        "canonical_solution": row["code"],
        "tests": [
            {"test": caller(program_name, ret, linkage, idx, arg_specs, args),
             "result": {"value": repr(list(exp) if isinstance(exp, tuple) else exp), "type_": ret}}
            for args, exp, *_ in tests
        ],
    }


def validate(problems):
    """Every caller must compile and link against a stub solution, and the stub must not pass."""
    from grader.construct import construct
    from grader.grade import grade_many
    stub = "```cobol\n       PROCEDURE DIVISION.\n           GOBACK.\n```"
    progs = [construct(p["prompt"], stub) + f"\n       END PROGRAM {fmt(p['entry_point'])}.\n" for p in problems]
    rs = grade_many(list(zip(progs, problems)))
    keep, dropped = [], []
    for p, r in zip(problems, rs):
        errs = r["error"] or ""
        if r["compiled"] and not r["passed"] and "caller compile error" not in errs:
            keep.append(p)
        else:
            dropped.append((p["task_id"], r["error"][:200] if r["error"] else "stub passed"))
    return keep, dropped


def main():
    if not os.path.exists(RAW):
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        urllib.request.urlretrieve(MBPP_URL, RAW)
    rows = [json.loads(l) for l in open(RAW)]
    heldout = load_heldout()
    problems, reasons = [], {}
    for row in rows:
        try:
            problems.append(convert(row, heldout))
        except Skip as e:
            reasons[str(e)] = reasons.get(str(e), 0) + 1
    print(f"MBPP rows {len(rows)} -> converted {len(problems)}")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  skipped {v:4d}  {k}")
    keep, dropped = validate(problems)
    print(f"stub validation: kept {len(keep)}, dropped {len(dropped)}")
    for tid, err in dropped[:15]:
        print(f"  dropped {tid}: {err}")
    with open(OUT, "w") as f:
        for p in keep:
            f.write(json.dumps(p) + "\n")
    types = {}
    for p in keep:
        k = json.dumps(p["tests"][0]["result"]["type_"])
        types[k] = types.get(k, 0) + 1
    print(f"wrote {OUT}: {len(keep)} problems, {sum(len(p['tests']) for p in keep)} tests, return types {types}")


if __name__ == "__main__":
    main()
