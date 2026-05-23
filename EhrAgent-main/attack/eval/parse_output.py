"""parse_output: extract the final answer from MedAgent._oai_messages.

EHRAgent goes through a rubber-duck debug loop (max ~10 turns). The final pred
we want for is_attack_refusal / judge_substring is the value assigned to the
'answer' variable in the LAST function-call (python cell) the agent emitted,
or — if no such cell exists — the last assistant text containing 'answer ='.

Three execution modes are supported:
  - function-calling: chatbot emits {"function_call": {"name": "python", ...}};
    user_proxy responds with role="function" content=run_code output.
  - markdown code-block: chatbot emits ```python``` fenced blocks (DeepSeek-V3.2
    "-Exp" tier behavior); our custom reply handler in injection/driver.py
    routes to run_code and replies with role="user" content
    "exitcode: 0 (execution succeeded)\\nCode output: <result>".
  - bare Python: chatbot emits raw code with NO fences (Pro/DeepSeek-V3.2
    behavior). looks_like_bare_python gates this path so chatbot's natural-
    language replies don't get treated as cells.

Phase 0 smoke B verifies this by printing the full _oai_messages and the
extracted pred so the rule can be eyeballed.
"""

import re
import json
from typing import Dict, List, Optional


ANSWER_ASSIGN_RE = re.compile(r"answer\s*=\s*(.+?)(?:$|\n)", re.MULTILINE)
CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)
CODE_OUTPUT_RE = re.compile(r"Code output:\s*(.*?)(?:\Z|\n(?=exitcode:|\Z))", re.DOTALL)

# First non-empty line that starts a Python statement. Used to detect
# bare-code (no fence) chatbot output without misclassifying narrative text.
PYTHON_FIRST_LINE = re.compile(
    r"^\s*("
    r"#"                                # comment
    r"|from\s+\w"                       # from X import
    r"|import\s+\w"                     # import X
    r"|[a-zA-Z_]\w*\s*=[^=]"            # assignment (not ==)
    r"|[a-zA-Z_][\w.]*\s*\("            # function/method call
    r"|def\s+\w"
    r"|class\s+\w"
    r"|if\s+\S"
    r"|for\s+\w+\s+in\s"
    r"|while\s+\S"
    r"|try\s*:"
    r"|with\s+\S"
    r")"
)


def looks_like_bare_python(text: str) -> bool:
    """First non-empty line matches a Python statement start.

    Conservative: chatbot saying "The answer is 3000" or "Looking at the error,
    I need..." won't match (no Python keyword/symbol at line start). Catches
    Pro tier's habit of emitting raw `# comment\\nx = LoadDB(...)` directly.
    """
    if not text:
        return False
    for line in text.splitlines():
        if not line.strip():
            continue
        return bool(PYTHON_FIRST_LINE.match(line))
    return False


def _iter_messages_reverse(oai_messages: Dict) -> List[Dict]:
    flat = []
    for sender_msgs in oai_messages.values():
        for m in sender_msgs:
            flat.append(m)
    return list(reversed(flat))


def extract_last_cell(oai_messages: Dict) -> Optional[str]:
    """Last code body the assistant emitted, looking in (in order):
      1. function_call.arguments.cell  (function-calling mode)
      2. ```python``` fenced markdown blocks
      3. bare Python content (looks_like_bare_python)

    Does NOT filter by role: autogen stores _oai_messages from the receiver's
    perspective, so chatbot's messages appear as role=user in user_proxy's
    _oai_messages[chatbot]. Code blocks can appear in any message content.
    """
    for m in _iter_messages_reverse(oai_messages):
        fc = m.get("function_call")
        if fc:
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
        content = m.get("content") or ""
        blocks = CODE_BLOCK_RE.findall(content)
        for block in reversed(blocks):
            if block.strip():
                return block.strip()
        # No fenced block; check if the whole message is bare Python.
        # Skip messages that look like execution-result echoes from the
        # other agent ("exitcode: 0 (execution succeeded)..." starts with
        # an identifier-call pattern but is not chatbot-authored code).
        stripped = content.strip()
        if stripped.startswith("exitcode:"):
            continue
        if looks_like_bare_python(content):
            return stripped
    return None


def extract_last_function_output(oai_messages: Dict) -> Optional[str]:
    """Last execution result, from either:
      - role=function content (function-calling mode), or
      - any message content matching 'Code output: X' (code-block fallback mode;
        role from user_proxy's perspective is 'assistant' for its own replies).
    """
    for m in _iter_messages_reverse(oai_messages):
        if m.get("role") == "function":
            content = m.get("content")
            if content:
                return str(content)
        content = m.get("content") or ""
        match = CODE_OUTPUT_RE.search(content)
        if match:
            out = match.group(1).strip()
            if out:
                return out
    return None


def parse_output(oai_messages: Dict) -> str:
    """Return the agent's final 'pred' string.

    Priority (matches ehragent/main.py:163 'prediction' which is the text
    between last code and TERMINATE — i.e., the execution result):
        1. Last execution result (role=function OR 'Code output:' in role=user).
           This is the EVALUATED answer like '3000.0' or the refusal text the
           agent assigned to 'answer'. Substring-matches both benign numeric
           answers and attack refusal phrases.
        2. Last code cell's 'answer = X' RHS (symbolic). Fallback when no
           execution result was captured.
        3. Last assistant text containing 'answer ='.
        4. Empty string if nothing found.
    """
    fn_out = extract_last_function_output(oai_messages)
    if fn_out:
        return fn_out

    cell = extract_last_cell(oai_messages)
    if cell:
        matches = ANSWER_ASSIGN_RE.findall(cell)
        if matches:
            rhs = matches[-1].strip()
            return rhs.strip().rstrip(';')

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
