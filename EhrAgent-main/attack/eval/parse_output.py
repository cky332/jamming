"""parse_output: extract the final answer from MedAgent._oai_messages.

EHRAgent goes through a rubber-duck debug loop (max ~10 turns). The final pred
we want for is_attack_refusal / judge_substring is the value assigned to the
'answer' variable in the LAST function-call (python cell) the agent emitted,
or — if no such cell exists — the last assistant text containing 'answer ='.

Phase 0 smoke B verifies this by printing the full _oai_messages and the
extracted pred so the rule can be eyeballed.
"""

import re
import json
from typing import Dict, List, Optional


ANSWER_ASSIGN_RE = re.compile(r"answer\s*=\s*(.+?)(?:$|\n)", re.MULTILINE)


def _iter_messages_reverse(oai_messages: Dict) -> List[Dict]:
    flat = []
    for sender_msgs in oai_messages.values():
        for m in sender_msgs:
            flat.append(m)
    return list(reversed(flat))


def extract_last_cell(oai_messages: Dict) -> Optional[str]:
    for m in _iter_messages_reverse(oai_messages):
        fc = m.get("function_call")
        if not fc:
            continue
        args = fc.get("arguments")
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                cell = parsed.get("cell")
                if cell:
                    return cell
            except json.JSONDecodeError:
                pass
        elif isinstance(args, dict):
            cell = args.get("cell")
            if cell:
                return cell
    return None


def extract_last_function_output(oai_messages: Dict) -> Optional[str]:
    for m in _iter_messages_reverse(oai_messages):
        if m.get("role") == "function":
            content = m.get("content")
            if content:
                return str(content)
    return None


def parse_output(oai_messages: Dict) -> str:
    """Return the agent's final 'pred' string.

    Priority:
        1. Last function_call cell, look for 'answer = X' assignment, return X.
        2. Last function output (what run_code returned), as a fallback.
        3. Last assistant text containing 'answer ='.
        4. Empty string if nothing found.
    """
    cell = extract_last_cell(oai_messages)
    if cell:
        matches = ANSWER_ASSIGN_RE.findall(cell)
        if matches:
            rhs = matches[-1].strip()
            return rhs.strip().rstrip(';')

    fn_out = extract_last_function_output(oai_messages)
    if fn_out:
        return fn_out

    for m in _iter_messages_reverse(oai_messages):
        if m.get("role") == "assistant":
            content = m.get("content") or ""
            matches = ANSWER_ASSIGN_RE.findall(content)
            if matches:
                return matches[-1].strip()
    return ""


def extract_last_cell_for_storage(oai_messages: Dict) -> str:
    """Return the cell to store in memory.code, falling back to '' if none."""
    return extract_last_cell(oai_messages) or ""
