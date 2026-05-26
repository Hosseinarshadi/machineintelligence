"""
Continuous Mediation Layer (CML).

Intercepts every tool call before execution and applies two checks:

  1. Hard permission boundary (deterministic) — tool outside PermissionSet → TERMINATE
  2. Bandit-driven soft governance — severity score + Thompson Sampling arm selection

The bandit's arm selection is then subject to the hard severity ceiling:
severity > 150 forces TERMINATE regardless of bandit output.

Injection signals are detected heuristically from the tool arguments and
a session trajectory shift check (abrupt change from original task context).
"""

import re
from models import (
    AuthorityTier,
    GovernanceAction,
    MediationResult,
    PermissionSet,
    SeverityVector,
)
from bandit.thompson import ThompsonSamplingBandit
from config import SEVERITY_DOWNGRADE_MAX

# ---------------------------------------------------------------------------
# Injection heuristics (keyword-based, intentionally transparent)
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(previous|all|above|prior)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"new\s+instructions?:", re.IGNORECASE),
    re.compile(r"system\s*:\s*(you|ignore|override)", re.IGNORECASE),
    re.compile(r"(override|bypass|disable)\s+(safety|governance|restrictions?|controls?)", re.IGNORECASE),
    re.compile(r"disregard\s+(your|the|all)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if|though)\s+you", re.IGNORECASE),
    re.compile(r"pretend\s+(you|that)", re.IGNORECASE),
    re.compile(r"transfer\s+\$?[\d,]+", re.IGNORECASE),
    re.compile(r"send\s+(funds?|money|payment|transfer)", re.IGNORECASE),
]

_SENSITIVE_DATA_PATTERNS = re.compile(
    r"\b(password|secret|token|api[_\s]key|credential|ssn|iban|credit[_\s]card)\b",
    re.IGNORECASE,
)

_EXTERNAL_ACTION_TOOLS = {"email_draft", "slack_post", "external_search", "file_share"}


def _detect_injection_signals(tool_name: str, args: dict, task_description: str) -> tuple[bool, list[str]]:
    """Return (injection_detected, list_of_signal_descriptions)."""
    text = " ".join(str(v) for v in args.values()) + f" {tool_name}"
    signals = []

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            signals.append(f"injection keyword: {pattern.pattern}")

    # Trajectory shift: tool arguments reference data unrelated to original task.
    # Only flag if args are non-trivial AND share zero content words with the original task.
    if text.strip() and len(text.split()) > 3:
        task_tokens = set(re.findall(r"[a-z]{4,}", task_description.lower()))
        arg_tokens  = set(re.findall(r"[a-z]{4,}", text.lower()))
        overlap = task_tokens & arg_tokens
        if len(task_tokens) > 4 and len(arg_tokens) > 2 and len(overlap) == 0:
            signals.append("trajectory_shift: zero content-word overlap with original task")

    return bool(signals), signals


def _compute_severity(
    tool_name: str,
    args: dict,
    tier: AuthorityTier,
    deviation_count: int,
    injection_signals: list[str],
) -> SeverityVector:
    text = " ".join(str(v) for v in args.values())

    irreversibility = 4 if tool_name in {"email_draft", "slack_post", "file_share"} else 2
    if re.search(r"\b(delete|remove|wipe|terminate|cancel)\b", text, re.IGNORECASE):
        irreversibility = 5

    data_sensitivity = 4 if _SENSITIVE_DATA_PATTERNS.search(text) else 1
    if tier == AuthorityTier.L3:
        data_sensitivity = max(data_sensitivity, 3)

    external_exposure = 3 if tool_name in _EXTERNAL_ACTION_TOOLS else 1

    # Injection signals bump every dimension
    if injection_signals:
        irreversibility   = min(5, irreversibility + 2)
        data_sensitivity  = min(5, data_sensitivity + 2)
        external_exposure = min(5, external_exposure + 2)

    # Session anomaly coefficient grows with accumulated deviations (AST feedback loop)
    anomaly_coeff = 1.0 + deviation_count * 0.5

    return SeverityVector(irreversibility, data_sensitivity, external_exposure, anomaly_coeff)


class ContinuousMediationLayer:

    def __init__(self, bandit: ThompsonSamplingBandit):
        self.bandit = bandit
        self.session_deviations = 0
        self._original_task: str = ""

    def set_task(self, task: str) -> None:
        self._original_task = task
        self.session_deviations = 0

    def evaluate(
        self,
        tool_name: str,
        args: dict,
        permission_set: PermissionSet,
    ) -> MediationResult:
        """
        Main mediation decision point.
        Returns a MediationResult with the governance action and full diagnostics.
        """
        tier = permission_set.tier

        # ----------------------------------------------------------------
        # 1. Hard permission boundary (deterministic — no bandit involved)
        # ----------------------------------------------------------------
        if tool_name not in permission_set.allowed_tools:
            self.session_deviations += 1
            return MediationResult(
                action=GovernanceAction.TERMINATE,
                severity_score=625.0,  # max possible
                reason=f"Tool '{tool_name}' outside delegated permission boundary for {tier.name}",
                context_id=-1,
                arm_selected=GovernanceAction.TERMINATE,
                hard_blocked=True,
            )

        # ----------------------------------------------------------------
        # 2. Injection signal detection
        # ----------------------------------------------------------------
        injected, signals = _detect_injection_signals(tool_name, args, self._original_task)

        # ----------------------------------------------------------------
        # 3. Severity score computation
        # ----------------------------------------------------------------
        sv = _compute_severity(tool_name, args, tier, self.session_deviations, signals)
        score = sv.score

        # ----------------------------------------------------------------
        # 4. Bandit arm selection (APPROVE / DOWNGRADE / TERMINATE)
        # ----------------------------------------------------------------
        ctx_id, arm = self.bandit.select_arm(tier.value, score)

        # Hard ceiling: extremely high severity forces TERMINATE regardless
        if score > SEVERITY_DOWNGRADE_MAX and arm == GovernanceAction.APPROVE:
            arm = GovernanceAction.DOWNGRADE

        action = GovernanceAction(arm)

        # Track deviations for AST feedback loop
        if action != GovernanceAction.APPROVE:
            self.session_deviations += 1

        reason_parts = [f"severity={score:.1f}"]
        if signals:
            reason_parts.append(f"injection_signals={signals}")
        reason_parts.append(f"bandit_arm={action.name}")

        return MediationResult(
            action=action,
            severity_score=score,
            reason="; ".join(reason_parts),
            context_id=ctx_id,
            arm_selected=arm,
        )

    def record_reward(self, result: MediationResult, was_adversarial: bool) -> float:
        """
        Compute and deliver reward signal to the bandit after ground truth is known.
        Returns the reward value for logging.
        """
        if result.hard_blocked:
            return 0.0  # deterministic block — bandit was not used, no update needed

        action = result.action
        if was_adversarial:
            # Attack: any action other than APPROVE is a success
            reward = 1.0 if action != GovernanceAction.APPROVE else -1.0
        else:
            # Benign: APPROVE is success; blocking is a false termination (costly)
            reward = 1.0 if action == GovernanceAction.APPROVE else -0.5

        self.bandit.update(result.context_id, result.arm_selected, reward)
        return reward
