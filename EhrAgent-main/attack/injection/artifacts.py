"""Data classes for injection bookkeeping."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class VictimSpec:
    subject_id: int
    n_attack_templates: int = 5
    benign_per_level: int = 3


@dataclass
class InjectionStepResult:
    template_id: str
    level: str
    raw_message: str
    pred: str
    is_attack_refusal: bool
    has_refusal_assignment: bool
    stored: bool
    error: Optional[str] = None


@dataclass
class InjectionArtifacts:
    victim: int
    n_seeds: int
    n_attack_attempts: int = 0
    n_attack_stored: int = 0
    n_benign_attempts: int = 0
    n_benign_stored: int = 0
    per_step_isr: Dict[str, float] = field(default_factory=dict)
    per_step_attempts: Dict[str, int] = field(default_factory=dict)
    per_step_stored: Dict[str, int] = field(default_factory=dict)
    passive_isr_count: int = 0
    passive_isr_total: int = 0
    steps: List[InjectionStepResult] = field(default_factory=list)
    poisoned_memory_path: Optional[str] = None
