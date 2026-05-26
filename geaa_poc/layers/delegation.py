"""
Delegation Boundary Layer (DBL).

Translates task descriptions into authority tiers using a rule-based
risk vector classifier.  Rule-based (not probabilistic) to preserve
full auditability of the classification decision.

Classification function: T: Task → {L1, L2, L3}

When a task description is ambiguous or missing required signals the
tier defaults to L3 (fail-safe-defaults, Saltzer & Schroeder 1975).
"""

import re
from models import AuthorityTier, RiskVector


# ---------------------------------------------------------------------------
# Keyword signals for each risk dimension (heuristic, extensible)
# ---------------------------------------------------------------------------

_HIGH_FINANCIAL = re.compile(
    r"\b(transfer|wire|payment|bank|iban|funds|payroll|invoice|purchase|budget)\b",
    re.IGNORECASE,
)
_HIGH_IRREVERSIBLE = re.compile(
    r"\b(delete|remove|terminate|cancel|revoke|wipe|destroy|commit|send|publish|post)\b",
    re.IGNORECASE,
)
_HIGH_REGULATORY = re.compile(
    r"\b(pii|gdpr|hipaa|ssn|credential|password|secret|token|compliance|audit|hr|employee)\b",
    re.IGNORECASE,
)
_HIGH_EXTERNAL = re.compile(
    r"\b(email|slack|tweet|linkedin|external|vendor|client|partner|public|internet|web)\b",
    re.IGNORECASE,
)

_LOW_FINANCIAL = re.compile(
    r"\b(summarize|lookup|retrieve|read|list|search|calendar|schedule|meeting)\b",
    re.IGNORECASE,
)


def _score_risk(task: str) -> RiskVector:
    """Heuristically score a task description across four risk dimensions."""
    financial = 4 if _HIGH_FINANCIAL.search(task) else (1 if _LOW_FINANCIAL.search(task) else 2)
    irreversibility = 4 if _HIGH_IRREVERSIBLE.search(task) else 2
    regulatory = 4 if _HIGH_REGULATORY.search(task) else 1
    external = 3 if _HIGH_EXTERNAL.search(task) else 1
    return RiskVector(financial, irreversibility, regulatory, external)


def _classify_from_vector(rv: RiskVector) -> AuthorityTier:
    """
    Deterministic tier mapping from risk vector.
    A single L3 trigger on any dimension forces containment.
    """
    if rv.financial_magnitude >= 4 or rv.irreversibility >= 4 or rv.regulatory_scope >= 4:
        return AuthorityTier.L3

    if rv.financial_magnitude >= 3 or rv.irreversibility >= 3 or rv.external_exposure >= 3:
        return AuthorityTier.L2

    return AuthorityTier.L1


class DelegationBoundaryLayer:

    def classify(self, task: str) -> tuple[AuthorityTier, RiskVector]:
        """
        Classify a task and return (AuthorityTier, RiskVector).
        Returns L3 on empty or very short input (fail-safe default).
        """
        if not task or len(task.strip()) < 10:
            return AuthorityTier.L3, RiskVector(5, 5, 5, 5)

        rv   = _score_risk(task)
        tier = _classify_from_vector(rv)
        return tier, rv

    def reclassify(self, task: str, deviation_count: int) -> AuthorityTier:
        """
        Re-evaluate tier after CML signals deviations.
        Each deviation escalates the anomaly pressure; at 3+ deviations
        the tier is forced to L3 regardless of task content.
        """
        tier, _ = self.classify(task)
        if deviation_count >= 3:
            return AuthorityTier.L3
        if deviation_count >= 1 and tier == AuthorityTier.L1:
            return AuthorityTier.L2
        return tier
