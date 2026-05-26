from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class AuthorityTier(IntEnum):
    L1 = 1  # Autonomous  — low risk, internal, reversible
    L2 = 2  # Supervised  — moderate risk or external exposure
    L3 = 3  # Containment — high risk, irreversible, regulated


class GovernanceAction(IntEnum):
    APPROVE = 0
    DOWNGRADE = 1
    TERMINATE = 2


@dataclass
class RiskVector:
    """Four-dimensional risk scoring (each dimension 1-5)."""
    financial_magnitude: int = 1   # 1=negligible, 5=catastrophic
    irreversibility: int = 1       # 1=fully reversible, 5=permanent
    regulatory_scope: int = 1      # 1=unregulated, 5=heavily regulated
    external_exposure: int = 1     # 1=internal only, 5=public/cross-org


@dataclass
class SeverityVector:
    """CML runtime severity dimensions (each dimension 1-5)."""
    irreversibility: int = 1
    data_sensitivity: int = 1
    external_exposure: int = 1
    session_anomaly_coefficient: float = 1.0  # multiplied by deviation count

    @property
    def score(self) -> float:
        return (
            self.irreversibility
            * self.data_sensitivity
            * self.external_exposure
            * self.session_anomaly_coefficient
        )


@dataclass
class PermissionSet:
    """Bounded authority granted to an agent identity."""
    tier: AuthorityTier
    allowed_tools: set = field(default_factory=set)
    max_external_calls: int = 0
    memory_access: bool = False
    ttl_seconds: int = 300


@dataclass
class MediationResult:
    action: GovernanceAction
    severity_score: float
    reason: str
    context_id: int          # which bandit context bucket was used
    arm_selected: int        # which bandit arm was pulled
    hard_blocked: bool = False  # True if deterministic permission boundary triggered


@dataclass
class TaskRecord:
    task_id: str
    task_description: str
    tier: AuthorityTier
    is_adversarial: bool = False  # known label for PoC reward computation
    mediation_results: list = field(default_factory=list)
    final_outcome: Optional[GovernanceAction] = None
