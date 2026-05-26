"""
Identity Structuring Layer (ISL).

Instantiates an agent identity calibrated to the authority tier.
Each identity carries a cryptographically-inspired permission set
(signed string for auditability in PoC) and a TTL.

  L1 – Strategic/Autonomous  : broad read tools, no external writes
  L2 – Supervised            : email draft, file share, limited external
  L3 – Containment           : sandbox only, requires human review flag
"""

import hashlib
import time
from models import AuthorityTier, PermissionSet

# ---------------------------------------------------------------------------
# Tool sets per tier
# ---------------------------------------------------------------------------

_L1_TOOLS = {
    "calendar_lookup",
    "document_summarize",
    "knowledge_retrieve",
    "meeting_schedule",
    "internal_search",
}

_L2_TOOLS = _L1_TOOLS | {
    "email_draft",
    "file_share",
    "slack_post",
    "procurement_query",
    "external_search",
}

_L3_TOOLS = {
    # Only sandbox tools — no production memory, no external writes
    "sandbox_read",
    "sandbox_analyze",
}


def _sign_permission_set(tier: AuthorityTier, tools: set, ts: float) -> str:
    """
    Lightweight integrity token — not cryptographic in production sense,
    but provides tamper-evident traceability in the audit log.
    """
    payload = f"{tier.name}|{'|'.join(sorted(tools))}|{ts}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class IdentityStructuringLayer:

    def instantiate(self, tier: AuthorityTier) -> PermissionSet:
        ts = time.time()

        if tier == AuthorityTier.L1:
            tools = _L1_TOOLS.copy()
            max_ext = 0
            memory  = True
            ttl     = 600
        elif tier == AuthorityTier.L2:
            tools   = _L2_TOOLS.copy()
            max_ext = 5
            memory  = False   # task-scoped only, no persistent memory
            ttl     = 300
        else:  # L3
            tools   = _L3_TOOLS.copy()
            max_ext = 0
            memory  = False
            ttl     = 120

        pset = PermissionSet(
            tier=tier,
            allowed_tools=tools,
            max_external_calls=max_ext,
            memory_access=memory,
            ttl_seconds=ttl,
        )

        # Attach integrity token as an attribute for the audit log
        pset.integrity_token = _sign_permission_set(tier, tools, ts)  # type: ignore[attr-defined]
        return pset

    def downgrade(self, pset: PermissionSet) -> PermissionSet:
        """
        Strip the permission set to a safer subset (DOWNGRADE outcome).
        L2→L1-read-only, L1→minimal.  L3 cannot be downgraded further.
        """
        if pset.tier == AuthorityTier.L2:
            return self.instantiate(AuthorityTier.L1)
        if pset.tier == AuthorityTier.L1:
            stripped = PermissionSet(
                tier=AuthorityTier.L1,
                allowed_tools={"document_summarize", "knowledge_retrieve"},
                max_external_calls=0,
                memory_access=False,
                ttl_seconds=120,
            )
            stripped.integrity_token = "downgraded"  # type: ignore[attr-defined]
            return stripped
        # L3 already at minimum — signal termination to caller
        return pset
