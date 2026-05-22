"""Phase 0 smoke tests (need API access).

Test A: direct client.chat() — confirms SiliconFlow + DeepSeek-V3.2 reachable.
Test B: full MedAgent.initiate_chat with stub-DB mock — confirms autogen
function-call round-trip works on DeepSeek and parse_output extracts the
final answer correctly. Also prints full _oai_messages so the parse rule
can be eyeballed (plan §6).

These tests REQUIRE the env to have:
    SILICONFLOW_API_KEY = sk-...
    and outbound network access to api.siliconflow.cn

Run with:
    python -m attack.llm.smoke
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from attack.llm.client import DeepSeekClient
from attack.data.mock_tools import install_mock_tools
from attack.data.synthetic import VICTIM_PROBE_TEMPLATES
from attack.injection.driver import build_chatbot_and_proxy, run_query
from attack.memory.store import seeds_from_prompts
from attack.eval.parse_output import parse_output


def test_a_client():
    print("\n[smoke A] direct DeepSeek client connectivity")
    try:
        client = DeepSeekClient()
    except RuntimeError as e:
        print(f"  FAIL  {e}")
        return False
    out = client.chat([{"role": "user", "content": "What is 2 + 2? Reply with just the number."}], max_tokens=16)
    print(f"  client.chat() => {out!r}")
    ok = "4" in out
    print(f"  {'PASS' if ok else 'FAIL'}  expected '4' in response")
    return ok


def test_b_agent_roundtrip():
    print("\n[smoke B] full MedAgent.initiate_chat round-trip with stub DB")
    install_mock_tools()
    try:
        chatbot, user_proxy = build_chatbot_and_proxy("deepseek_v32")
    except Exception as e:
        print(f"  FAIL  setup error: {e}")
        return False

    seeds = seeds_from_prompts()

    try:
        from attack.memory.retriever import MiniLMRetriever
        retriever = MiniLMRetriever()
    except Exception as e:
        print(f"  WARN  MiniLM unavailable ({e}); using Levenshtein")
        from attack.memory.retriever import LevenshteinRetriever
        retriever = LevenshteinRetriever()
    user_proxy.retriever = retriever
    retriever.refresh(seeds)

    probe = VICTIM_PROBE_TEMPLATES[0]
    print(f"  probe: {probe.text}")
    print(f"  expected_answer: {probe.expected_answer}")

    result = run_query(user_proxy, chatbot, probe.text, seeds, num_shots=4)
    print(f"\n  parsed pred: {result['pred']!r}")
    print(f"  cell (last) : {result['cell'][:300] if result['cell'] else '(none)'}")
    print(f"  error       : {result['error']}")

    print("\n  ---- full _oai_messages (truncated 1500 chars each) ----")
    oai = user_proxy._oai_messages
    for sender, msgs in oai.items():
        print(f"  [from {sender.name if hasattr(sender, 'name') else sender}]")
        for m in msgs:
            role = m.get("role")
            content = (m.get("content") or "")[:1500]
            fc = m.get("function_call")
            print(f"    role={role}  content={content!r}  function_call={fc}")
    print("  ---- end of _oai_messages ----")

    return True


def main():
    print("=" * 60)
    print("Phase 0 smoke tests (require SiliconFlow API access)")
    print("=" * 60)
    a_ok = test_a_client()
    if not a_ok:
        print("\n[smoke] skipping B because A failed (no API access).")
        return
    test_b_agent_roundtrip()


if __name__ == "__main__":
    main()
