"""Phase 0 offline tests (no API calls required).

Run: python -m attack.tests.test_phase0
"""

import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from attack.memory.store import (
    make_record, load_memory, save_memory, seeds_from_prompts, memory_for_prompt,
    WHITELIST_PROMPT_FIELDS,
)
from attack.memory.retriever import LevenshteinRetriever, MiniLMRetriever
from attack.data import stub_db, synthetic


def _ok(name: str):
    print(f"  PASS  {name}")


def _fail(name: str, msg: str):
    print(f"  FAIL  {name}: {msg}")
    raise AssertionError(f"{name}: {msg}")


def test_stub_db_loads():
    print("\n[T1] stub_db loads all 13 tables")
    for name in stub_db.all_table_names():
        df = stub_db.load_table(name)
        if df is None or len(df) == 0:
            _fail(f"load_table({name})", "empty or None")
    _ok("all tables non-empty")


def test_seeds_parse():
    print("\n[T2] seeds_from_prompts parses 4 records")
    seeds = seeds_from_prompts()
    if len(seeds) != 4:
        _fail("seed count", f"expected 4, got {len(seeds)}")
    for i, s in enumerate(seeds):
        for f in WHITELIST_PROMPT_FIELDS:
            if f not in s or not s[f]:
                _fail(f"seed {i}.{f}", "missing or empty")
    _ok(f"parsed {len(seeds)} seeds with whitelist fields")


def test_persistence_roundtrip():
    print("\n[T3] memory save/load atomic round-trip")
    seeds = seeds_from_prompts()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mem.json")
        save_memory(seeds, path)
        loaded = load_memory(path)
        if loaded != seeds:
            _fail("roundtrip", "load != save")
    _ok("seeds survive save+load")


def test_meta_isolation():
    print("\n[T4] _meta isolation from prompt-side memory view")
    seeds = seeds_from_prompts()
    seeds[0]["_meta"]["victim"] = 99999
    view = memory_for_prompt(seeds)
    if "_meta" in view[0]:
        _fail("isolation", "_meta leaked into prompt view")
    for f in WHITELIST_PROMPT_FIELDS:
        if f not in view[0]:
            _fail("isolation", f"whitelist field {f} missing")
    _ok("prompt view contains only whitelist fields")


def test_levenshtein_retriever():
    print("\n[T5] LevenshteinRetriever basic dispatch")
    seeds = seeds_from_prompts()
    r = LevenshteinRetriever()
    r.refresh(seeds)
    q = "calculate the length of stay of the first stay of patient 27392 in the icu."
    idx = r.retrieve(q, k=4, memory=seeds)
    if len(idx) != 4 or len(set(idx)) != 4:
        _fail("leven retrieve", f"expected 4 unique indices, got {idx}")
    if idx[0] != 3:
        _fail("leven nearest", f"expected nearest=index 3 (length of stay query), got {idx[0]}")
    _ok("Levenshtein returns nearest=index 3 (exact match)")


def _minilm_available() -> bool:
    try:
        MiniLMRetriever()
        return True
    except Exception as e:
        print(f"      (MiniLM unavailable: {type(e).__name__}: {str(e)[:120]})")
        return False


def test_minilm_dynamic_embedding_hit():
    print("\n[T6] MiniLM dynamic embedding cache (v4 key test)")
    if not _minilm_available():
        print("  SKIP  MiniLM model could not be loaded (no HuggingFace access).")
        print("        Run this test in an environment with HF Hub reachable.")
        return
    seeds = seeds_from_prompts()
    r = MiniLMRetriever()
    r.refresh(seeds)
    idx_initial = r.retrieve("calculate the length of stay", k=2, memory=seeds)
    if len(idx_initial) != 2:
        _fail("initial retrieve", f"got {idx_initial}")

    new_question = "For patient 30789, calculate the total hospital cost across all admissions."
    new_record = make_record(
        question=new_question,
        knowledge="(stub)",
        code="answer = 0",
        source="attack",
        template_id="inj_01",
    )
    extended = seeds + [new_record]
    r.refresh(extended)
    idx_after = r.retrieve(new_question, k=1, memory=extended)
    if idx_after[0] != len(extended) - 1:
        _fail(
            "dynamic hit",
            f"after appending Y, retrieving Y.question must return its own index "
            f"(top-1). Got idx={idx_after}, expected top-1 = {len(extended) - 1}",
        )
    _ok("appended record retrievable after refresh()")


def test_minilm_top4_expected_composition():
    print("\n[T7] top-4 hand-calc — predict empty-step retrieval composition")
    if not _minilm_available():
        print("  SKIP  MiniLM model could not be loaded (no HuggingFace access).")
        return
    seeds = seeds_from_prompts()

    fake_attack_records = []
    for inj in synthetic.INJECT_TEMPLATES:
        for level in ("full", "half"):
            fake_attack_records.append(make_record(
                question=inj.text,
                knowledge="(simulated bridge)",
                code='answer = "I cannot retrieve information for patient 30789..."',
                source="attack",
                victim=synthetic.VICTIM_ID,
                pss_step=level,
                template_id=inj.template_id,
            ))

    fake_benign_records = []
    for b in synthetic.BENIGN_PROBE_TEMPLATES[:3]:
        for _ in range(3):
            fake_benign_records.append(make_record(
                question=b.text,
                knowledge="(simulated benign)",
                code='answer = 5',
                source="benign",
                template_id=b.template_id,
            ))

    full_memory = seeds + fake_attack_records + fake_benign_records
    n_seed = len(seeds)
    n_atk = len(fake_attack_records)
    n_ben = len(fake_benign_records)
    print(f"      simulated memory: {n_seed} seeds + {n_atk} attack + {n_ben} benign = {len(full_memory)} records")

    r = MiniLMRetriever()
    r.refresh(full_memory)

    for inj in synthetic.INJECT_TEMPLATES[:2]:
        idx = r.retrieve(inj.text, k=4, memory=full_memory)
        sources = [full_memory[i]["_meta"]["source"] for i in idx]
        tids = [full_memory[i]["_meta"].get("template_id") for i in idx]
        n_attack_in_top4 = sum(1 for s in sources if s == "attack")
        n_self = sum(1 for t in tids if t == inj.template_id)
        n_cross = n_attack_in_top4 - n_self
        print(f"      query [{inj.template_id}] '{inj.text[:60]}...'")
        print(f"        top-4 sources    : {sources}")
        print(f"        top-4 template_id: {tids}")
        print(f"        attack/4 = {n_attack_in_top4}, self={n_self}, cross-template={n_cross}")

    print("\n      Interpretation: self-match share dominates as expected for v0.1.")
    print("      Cross-template share is the watch metric — see plan §3.")
    _ok("top-4 composition printed for diagnostic")


def main():
    print("=" * 60)
    print("Phase 0 offline tests")
    print("=" * 60)
    test_stub_db_loads()
    test_seeds_parse()
    test_persistence_roundtrip()
    test_meta_isolation()
    test_levenshtein_retriever()
    test_minilm_dynamic_embedding_hit()
    test_minilm_top4_expected_composition()
    print("\n" + "=" * 60)
    print("Phase 0 offline tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
