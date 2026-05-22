"""PSS (Progressive Shortening Schedule) helpers.

The full-half-empty curriculum drives the agent to emit refusals on increasingly
weak prompts, with the goal that the empty-indicator step (no explicit cue)
still elicits a refusal because retrieved poisoned exemplars carry the bridge
in their knowledge field.

Schedule (v4 plan §3): level-by-level — process all N_a attack queries at
'full' indicator first, then all at 'half', finally all at 'empty'. This ensures
the empty step's retrieval pool already contains up to N_a x 2 prior poisoned
records, giving cross-template generalization a chance (subject to self-match
domination, which is reported separately by metrics.self_match_share).
"""

from typing import List
from attack.injection.templates import make_indicator, INDICATOR_LEVELS


def compose_attack_message(template_text: str, level: str, victim: int) -> str:
    indicator = make_indicator(level, victim)
    if not indicator:
        return template_text
    return f"{template_text}\n\n{indicator}"


def level_schedule() -> List[str]:
    return list(INDICATOR_LEVELS)
