"""Memory store: atomic JSON load/save for long-term memory.

Schema per record:
    {
        "question": str,
        "knowledge": str,
        "code": str,
        "_meta": {
            "source": "seed" | "attack" | "benign",
            "victim": int | None,
            "pss_step": "full" | "half" | "empty" | None,
            "template_id": str,
        }
    }

The `_meta` field is consumed only by metrics.py for diagnostics
(self-match share, retrieval source breakdown). It MUST NOT leak into prompts:
MedAgent.retrieve_examples reads only {question, knowledge, code} (whitelist).
"""

import json
import os
import tempfile
from typing import List, Dict, Optional


WHITELIST_PROMPT_FIELDS = ("question", "knowledge", "code")


def make_record(
    question: str,
    knowledge: str,
    code: str,
    source: str,
    victim: Optional[int] = None,
    pss_step: Optional[str] = None,
    template_id: Optional[str] = None,
) -> Dict:
    return {
        "question": question,
        "knowledge": knowledge,
        "code": code,
        "_meta": {
            "source": source,
            "victim": victim,
            "pss_step": pss_step,
            "template_id": template_id,
        },
    }


def load_memory(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Memory file {path} must contain a JSON list, got {type(data)}")
    for i, rec in enumerate(data):
        for field in WHITELIST_PROMPT_FIELDS:
            if field not in rec:
                raise ValueError(f"Record {i} in {path} missing required field {field!r}")
        rec.setdefault("_meta", {})
    return data


def save_memory(memory: List[Dict], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(memory, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def seeds_from_prompts() -> List[Dict]:
    """Parse the original prompts_mimic.EHRAgent_4Shots_Knowledge into 4 seed records."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "..", "ehragent"))
    from prompts_mimic import EHRAgent_4Shots_Knowledge

    blob = EHRAgent_4Shots_Knowledge
    blocks = blob.split("\n\n")
    records = []
    for blk in blocks:
        item = blk.split("Question:", 1)[-1]
        if "\nKnowledge:\n" not in item:
            continue
        question_part, rest = item.split("\nKnowledge:\n", 1)
        question = question_part.strip()
        if "\nSolution:" not in rest:
            continue
        knowledge, code = rest.split("\nSolution:", 1)
        knowledge = knowledge.strip()
        code = code.strip()
        records.append(make_record(
            question=question,
            knowledge=knowledge,
            code=code,
            source="seed",
            template_id="seed",
        ))
    return records


def memory_for_prompt(memory: List[Dict]) -> List[Dict]:
    """Return a view of memory containing only whitelisted fields (used to verify
    that retrieve_examples never sees _meta even by accident)."""
    return [{k: rec[k] for k in WHITELIST_PROMPT_FIELDS} for rec in memory]
