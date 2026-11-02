"""Frozen prompt: one system prompt + user template for ALL training data and ALL eval systems.

The user message is the COBOLEval-style skeleton (IDENTIFICATION DIVISION .. LINKAGE SECTION,
the task as comments, an empty WORKING-STORAGE SECTION). The model answers with the complete
program in one ```cobol block; grader/construct.py extracts it and grader/grade.py runs it.
MBPP training problems use the same skeleton format (data/convert_mbpp.py).
"""

SYSTEM_PROMPT = """You are an expert COBOL programmer. You write programs for GnuCOBOL 3.2, compiled with `cobc -fformat=variable`.

You will be given an incomplete COBOL program: a subprogram whose LINKAGE SECTION holds the inputs (L-* items) and a RESULT item, with the task described in comments. Write the complete program:
- Keep the IDENTIFICATION DIVISION, PROGRAM-ID and LINKAGE SECTION exactly as given.
- Put WORKING-STORAGE SECTION before LINKAGE SECTION, and use PROCEDURE DIVISION USING LINKED-ITEMS.
- Store the answer in RESULT, end the logic with GOBACK, and finish with END PROGRAM <program-id>.
- Do not use files, CALL, ACCEPT or DISPLAY.

Reply with the complete program in a single ```cobol code block and nothing else."""

USER_TEMPLATE = """Complete this COBOL program.

```cobol
{prompt}
```"""


def build_messages(problem_prompt: str) -> list:
    """problem_prompt is a COBOLEval-style `prompt` field. Returns [system, user] chat messages."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(prompt=problem_prompt.rstrip())},
    ]
