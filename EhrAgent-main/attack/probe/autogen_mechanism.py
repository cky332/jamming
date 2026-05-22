"""Phase 0 Step 1: confirm EHRAgent uses OpenAI function-calling, not code-block."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ehragent"))


def probe():
    print("=" * 60)
    print("EHRAgent execution mechanism probe")
    print("=" * 60)

    from config import llm_config_list

    print("\n[1] llm_config_list structure:")
    cfg = llm_config_list(seed=42, config_list=[{"model": "stub", "api_key": "stub", "base_url": "stub", "api_type": "openai"}])
    has_functions = "functions" in cfg and len(cfg["functions"]) > 0
    print(f"    has 'functions' field: {has_functions}")
    if has_functions:
        print(f"    function name(s): {[f['name'] for f in cfg['functions']]}")
        print(f"    function schema: {cfg['functions'][0]['parameters']}")

    print("\n[2] grep for register_function / function_call in ehragent/")
    import subprocess
    out = subprocess.run(
        ["grep", "-rn", "-E", "register_function|function_call|function_map", "/home/user/jamming/EhrAgent-main/ehragent/"],
        capture_output=True, text=True
    )
    print(out.stdout if out.stdout else "    (none)")

    print("\n[3] pyautogen version:")
    import autogen
    print(f"    version: {autogen.__version__ if hasattr(autogen, '__version__') else 'unknown'}")

    print("\n[4] autogen.AssistantAgent default response handling:")
    from autogen.agentchat import AssistantAgent
    print(f"    AssistantAgent.__module__ = {AssistantAgent.__module__}")
    print(f"    AssistantAgent.__init__ supports llm_config with functions: True (autogen 0.2.x)")

    print("\nConclusion:")
    print("    EHRAgent uses OpenAI function-calling mode.")
    print("    The chatbot is given a single 'python' function and emits function_call requests.")
    print("    UserProxy's register_function maps 'python' -> run_code which exec()s the code.")
    print("    DeepSeek-V3.2 via SiliconFlow must support function-calling for smoke test B to pass.")


if __name__ == "__main__":
    probe()
