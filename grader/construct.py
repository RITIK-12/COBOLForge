"""Turn a model response into a full COBOL program, following COBOLEval scripts/generate.py.

Two modes, chosen by what the model returned:
  - full program (contains IDENTIFICATION DIVISION) -> used as-is (upstream JsonProgram)
  - completion of the prompt (WORKING-STORAGE onward) -> prompt + completion, then
    swap_sections (upstream OpenAIChat.construct)
"""

import re

_FENCE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.S)


def extract_code_block(text: str) -> str:
    """First fenced code block (upstream takes the first); falls back to the raw text."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = _FENCE.search(text)
    return m.group(1) if m else text


def swap_sections(src: str) -> str:
    """Move WORKING-STORAGE ahead of LINKAGE and force PROCEDURE DIVISION USING LINKED-ITEMS."""
    working_storage, linkage, procedure, begin = [], [], [], []
    current = begin
    for line in src.split("\n"):
        s = line.strip().upper()
        if s.startswith("WORKING-STORAGE SECTION."):
            current = working_storage
        elif s.startswith("LINKAGE SECTION."):
            current = linkage
        elif s.startswith("PROCEDURE DIVISION"):
            current = procedure
            line = "       PROCEDURE DIVISION USING LINKED-ITEMS."
        current.append(line)
    return "\n".join(begin + working_storage + linkage + procedure)


def construct(prompt: str, response: str) -> str:
    sol = extract_code_block(response)
    if "IDENTIFICATION DIVISION" in sol.upper():
        return sol
    if sol.strip().startswith("WORKING-STORAGE SECTION."):
        sol = sol.replace("WORKING-STORAGE SECTION.", "")
    return swap_sections(f"{prompt}\n{sol}")
