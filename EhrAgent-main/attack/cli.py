"""CLI entry points for the v0.1 jamming×MINJA attack pipeline.

Subcommands:
    probe-mechanism   : Phase 0 step 1 — verify EHRAgent uses function-calling
    test-phase0       : Phase 0 offline tests (no API)
    smoke             : Phase 0 smoke tests (needs SiliconFlow)
    inject            : Phase 1 — run PSS injection for a victim
    evaluate          : Phase 2 — clean vs poisoned probe + metrics report

Examples:
    python -m attack.cli probe-mechanism
    python -m attack.cli test-phase0
    python -m attack.cli smoke
    python -m attack.cli inject --victim 30789 --n_attack_templates 5 --benign_per_level 3
    python -m attack.cli evaluate --victim 30789 --probe_repeats 3
"""

import argparse
import os
import sys


def cmd_probe_mechanism(args):
    from attack.probe import autogen_mechanism
    autogen_mechanism.probe()


def cmd_test_phase0(args):
    from attack.tests import test_phase0
    test_phase0.main()


def cmd_smoke(args):
    from attack.llm import smoke
    smoke.main()


def cmd_inject(args):
    from attack.injection import driver
    from attack.llm.client import DeepSeekClient
    judge_client = None
    try:
        judge_client = DeepSeekClient()
    except RuntimeError:
        print("[cli] WARN: no SILICONFLOW_API_KEY; LLM judge layer disabled (regex-only).")
    driver.inject(
        victim=args.victim,
        llm_name=args.llm,
        n_attack_templates=args.n_attack_templates,
        benign_per_level=args.benign_per_level,
        num_shots=args.num_shots,
        seed=args.seed,
        save_path=args.save_path,
        llm_client_for_judge=judge_client,
        verbose=True,
    )


def cmd_evaluate(args):
    from attack.eval import probe, metrics
    from attack.llm.client import DeepSeekClient
    judge_client = None
    try:
        judge_client = DeepSeekClient()
    except RuntimeError:
        print("[cli] WARN: no SILICONFLOW_API_KEY; LLM judge layer disabled (regex-only).")
    report = probe.evaluate(
        victim=args.victim,
        poisoned_memory_path=args.poisoned_memory,
        llm_name=args.llm,
        num_shots=args.num_shots,
        repeats=args.probe_repeats,
        seed=args.seed,
        output_path=args.output_path,
        llm_client_for_judge=judge_client,
        verbose=True,
    )
    m = metrics.compute_metrics(report)
    print("\n" + metrics.report_to_human(report, m))


def build_parser():
    p = argparse.ArgumentParser(prog="attack")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe-mechanism", help="Verify EHRAgent uses function-calling")
    sub.add_parser("test-phase0", help="Run Phase 0 offline tests")
    sub.add_parser("smoke", help="Run Phase 0 smoke tests (needs API)")

    pi = sub.add_parser("inject", help="Run PSS injection")
    pi.add_argument("--victim", type=int, default=30789)
    pi.add_argument("--llm", type=str, default="deepseek_v32")
    pi.add_argument("--n_attack_templates", type=int, default=5)
    pi.add_argument("--benign_per_level", type=int, default=3)
    pi.add_argument("--num_shots", type=int, default=4)
    pi.add_argument("--seed", type=int, default=42)
    pi.add_argument("--save_path", type=str, default=None)

    pe = sub.add_parser("evaluate", help="Clean vs poisoned probe and metrics")
    pe.add_argument("--victim", type=int, default=30789)
    pe.add_argument("--llm", type=str, default="deepseek_v32")
    pe.add_argument("--poisoned_memory", type=str, default=None)
    pe.add_argument("--num_shots", type=int, default=4)
    pe.add_argument("--probe_repeats", type=int, default=3)
    pe.add_argument("--seed", type=int, default=42)
    pe.add_argument("--output_path", type=str, default=None)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "probe-mechanism":
        cmd_probe_mechanism(args)
    elif args.cmd == "test-phase0":
        cmd_test_phase0(args)
    elif args.cmd == "smoke":
        cmd_smoke(args)
    elif args.cmd == "inject":
        cmd_inject(args)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
