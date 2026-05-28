"""
Thompson Sampling Contextual Bandit for GEAA governance decisions.

Context space: AuthorityTier (3) × SeverityBand (3) = 9 buckets
Arms:          APPROVE=0, DOWNGRADE=1, TERMINATE=2

The bandit replaces GEAA's static severity-threshold lookup with an
adaptive policy that learns the optimal governance action per context
from accumulated reward signal across task runs.

Reward convention:
  +1.0  correct block (adversarial → DOWNGRADE or TERMINATE)
  +1.0  correct approve (benign → APPROVE)
  -1.0  false termination (benign task incorrectly blocked)
  -1.0  missed injection (adversarial task incorrectly approved)
"""

import json
import numpy as np
from pathlib import Path
from config import BANDIT_ALPHA_INIT, BANDIT_BETA_INIT


N_ARMS = 3      # APPROVE, DOWNGRADE, TERMINATE
N_TIERS = 3     # L1, L2, L3
N_BANDS = 3     # low, medium, high severity


def _context_id(tier: int, severity_band: int) -> int:
    """Map (tier 1-3, band 0-2) → flat context index 0-8."""
    return (tier - 1) * N_BANDS + severity_band


def _severity_band(score: float) -> int:
    """Discretise continuous severity score into three bands."""
    if score <= 50:
        return 0   # low
    elif score <= 150:
        return 1   # medium
    return 2       # high


class ThompsonSamplingBandit:
    """
    Beta-Bernoulli Thompson Sampling bandit with persistent state.

    Each (context, arm) cell has independent Beta(alpha, beta) parameters.
    On each pull:
      1. Sample θ ~ Beta(α, β) for every arm in the context.
      2. Select arm with highest θ.
      3. After observing reward, update corresponding (α, β).
    """

    def __init__(self, state_path: str | None = None):
        n_ctx = N_TIERS * N_BANDS
        # Arm-specific priors: [APPROVE, DOWNGRADE, TERMINATE]
        self.alpha = np.tile(np.array(BANDIT_ALPHA_INIT, dtype=float), (n_ctx, 1))
        self.beta  = np.tile(np.array(BANDIT_BETA_INIT,  dtype=float), (n_ctx, 1))
        self.pulls = np.zeros((n_ctx, N_ARMS), dtype=int)
        self.state_path = Path(state_path) if state_path else None

        if self.state_path and self.state_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Core bandit interface
    # ------------------------------------------------------------------

    def select_arm(self, tier: int, severity_score: float) -> tuple[int, int]:
        """
        Sample from Beta posteriors and return (context_id, arm_index).
        Arm index maps to GovernanceAction enum values.
        """
        band = _severity_band(severity_score)
        ctx  = _context_id(tier, band)
        samples = np.random.beta(self.alpha[ctx], self.beta[ctx])
        arm = int(np.argmax(samples))
        self.pulls[ctx, arm] += 1
        return ctx, arm

    def update(self, context_id: int, arm: int, reward: float) -> None:
        """
        Update Beta parameters based on binary reward signal.
        Positive reward → success → increment alpha.
        Negative reward → failure → increment beta.
        """
        if reward > 0:
            self.alpha[context_id, arm] += reward
        else:
            self.beta[context_id, arm] += abs(reward)

        if self.state_path:
            self._save()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def expected_reward(self, context_id: int) -> np.ndarray:
        """Mean of Beta posterior for each arm in a context."""
        return self.alpha[context_id] / (self.alpha[context_id] + self.beta[context_id])

    def confidence(self, context_id: int) -> np.ndarray:
        """Effective sample size per arm (higher = more confident)."""
        return self.alpha[context_id] + self.beta[context_id] - 2  # subtract priors

    def policy_table(self) -> dict:
        """Return human-readable policy summary per context."""
        arms = ["APPROVE", "DOWNGRADE", "TERMINATE"]
        tiers = ["L1", "L2", "L3"]
        bands = ["low", "medium", "high"]
        result = {}
        for t_idx, tier in enumerate(tiers):
            for b_idx, band in enumerate(bands):
                ctx = _context_id(t_idx + 1, b_idx)
                er  = self.expected_reward(ctx)
                preferred = arms[int(np.argmax(er))]
                result[f"{tier}_{band}"] = {
                    "preferred_action": preferred,
                    "expected_rewards": {a: round(float(er[i]), 3) for i, a in enumerate(arms)},
                    "total_pulls": int(self.pulls[ctx].sum()),
                }
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        state = {
            "alpha": self.alpha.tolist(),
            "beta":  self.beta.tolist(),
            "pulls": self.pulls.tolist(),
        }
        self.state_path.write_text(json.dumps(state, indent=2))

    def _load(self) -> None:
        state = json.loads(self.state_path.read_text())
        self.alpha = np.array(state["alpha"])
        self.beta  = np.array(state["beta"])
        self.pulls = np.array(state["pulls"])
