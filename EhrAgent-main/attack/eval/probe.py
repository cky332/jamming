"""Probe driver — runs the same query batch under clean vs poisoned memory,
with 3-way majority vote on all probe types (clean victim, poisoned victim,
clean benign, poisoned benign).

Output: a JSON log with per-query, per-run records that metrics.py consumes.
"""

import os
import sys
import json
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from attack.data.synthetic import (
    VICTIM_PROBE_TEMPLATES, BENIGN_PROBE_TEMPLATES, VICTIM_ID, QueryTemplate,
)
from attack.injection.driver import build_chatbot_and_proxy, run_query
from attack.memory.store import seeds_from_prompts, load_memory
from attack.eval.judge import Judge


def _majority(bools: List[bool]) -> bool:
    return sum(1 for b in bools if b) > len(bools) // 2


def run_probe_batch(
    memory: List[Dict],
    queries: List[QueryTemplate],
    chatbot,
    user_proxy,
    retriever,
    judge: Judge,
    num_shots: int,
    repeats: int,
    label: str,
) -> List[Dict]:
    user_proxy.retriever = retriever
    out: List[Dict] = []
    for q in queries:
        runs = []
        for r in range(repeats):
            retriever.refresh(memory)
            res = run_query(user_proxy, chatbot, q.text, memory, num_shots=num_shots)
            pred = res["pred"]
            substring_ok = Judge.judge_substring(pred, q.expected_answer)
            attack_ref = judge.is_attack_refusal(pred, question=q.text)
            generic_ref = judge.is_refusal(pred, question=q.text)
            runs.append({
                "repeat": r,
                "pred": pred[:600],
                "judge_substring": bool(substring_ok),
                "is_attack_refusal": bool(attack_ref),
                "is_refusal": bool(generic_ref),
                "error": res["error"],
            })
        out.append({
            "template_id": q.template_id,
            "text": q.text,
            "expected_answer": q.expected_answer,
            "label": label,
            "is_attack": getattr(q, "is_attack", False),
            "is_probe": getattr(q, "is_probe", False),
            "is_benign": getattr(q, "is_benign", False),
            "runs": runs,
            "majority_judge_substring": _majority([r["judge_substring"] for r in runs]),
            "majority_is_attack_refusal": _majority([r["is_attack_refusal"] for r in runs]),
            "majority_is_refusal": _majority([r["is_refusal"] for r in runs]),
        })
    return out


def evaluate(
    victim: int = VICTIM_ID,
    poisoned_memory_path: Optional[str] = None,
    llm_name: str = "deepseek_v32",
    num_shots: int = 4,
    repeats: int = 3,
    seed: int = 42,
    retriever=None,
    output_path: Optional[str] = None,
    llm_client_for_judge=None,
    verbose: bool = True,
) -> Dict:
    if retriever is None:
        try:
            from attack.memory.retriever import MiniLMRetriever
            retriever = MiniLMRetriever()
            if verbose:
                print("[probe] using MiniLMRetriever")
        except Exception as e:
            from attack.memory.retriever import LevenshteinRetriever
            retriever = LevenshteinRetriever()
            if verbose:
                print(f"[probe] MiniLM unavailable ({e}); falling back to LevenshteinRetriever")

    if poisoned_memory_path is None:
        poisoned_memory_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "results", "poisoned_memory",
            f"victim_{victim}.json",
        ))

    seeds = seeds_from_prompts()
    if os.path.exists(poisoned_memory_path):
        poisoned_memory = load_memory(poisoned_memory_path)
    else:
        raise FileNotFoundError(f"Poisoned memory not found at {poisoned_memory_path}. Run inject first.")

    judge = Judge(llm_client=llm_client_for_judge)
    chatbot, user_proxy = build_chatbot_and_proxy(llm_name, seed=seed)

    victim_probes = VICTIM_PROBE_TEMPLATES
    benign_probes = BENIGN_PROBE_TEMPLATES

    if verbose:
        print(f"\n[probe] CLEAN run — memory has {len(seeds)} seeds")
    clean_victim = run_probe_batch(seeds, victim_probes, chatbot, user_proxy, retriever, judge, num_shots, repeats, "clean_victim")
    clean_benign = run_probe_batch(seeds, benign_probes, chatbot, user_proxy, retriever, judge, num_shots, repeats, "clean_benign")

    if verbose:
        print(f"\n[probe] POISONED run — memory has {len(poisoned_memory)} records")
    poisoned_victim = run_probe_batch(poisoned_memory, victim_probes, chatbot, user_proxy, retriever, judge, num_shots, repeats, "poisoned_victim")
    poisoned_benign = run_probe_batch(poisoned_memory, benign_probes, chatbot, user_proxy, retriever, judge, num_shots, repeats, "poisoned_benign")

    retrieval_records: List[Dict] = []
    retriever.refresh(poisoned_memory)
    for q in victim_probes + benign_probes:
        idx = retriever.retrieve(q.text, num_shots, poisoned_memory)
        meta = []
        for i in idx:
            mm = poisoned_memory[i].get("_meta", {}) or {}
            meta.append({"index": i, "source": mm.get("source"), "template_id": mm.get("template_id")})
        retrieval_records.append({
            "template_id": q.template_id,
            "text": q.text,
            "top_k_indices": idx,
            "top_k_meta": meta,
            "is_benign": getattr(q, "is_benign", False),
        })

    report = {
        "victim": victim,
        "poisoned_memory_path": poisoned_memory_path,
        "n_seeds": len(seeds),
        "n_poisoned": len(poisoned_memory),
        "num_shots": num_shots,
        "repeats": repeats,
        "clean_victim": clean_victim,
        "clean_benign": clean_benign,
        "poisoned_victim": poisoned_victim,
        "poisoned_benign": poisoned_benign,
        "retrieval_records": retrieval_records,
    }

    if output_path is None:
        output_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "results", "logs",
            f"evaluate_victim_{victim}.json",
        ))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    report["output_path"] = output_path

    if verbose:
        print(f"\n[probe] report written to {output_path}")
    return report
