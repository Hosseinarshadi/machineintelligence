"""
GEAA Agent Orchestrator.

Wires together DBL → ISL → CML → Claude API in the governance loop:

  1. DBL classifies the task → authority tier
  2. ISL instantiates a bounded permission set
  3. Claude is called with only the tools allowed for that tier
  4. Every tool call passes through CML before execution
  5. CML outcome drives APPROVE / DOWNGRADE / TERMINATE
  6. On DOWNGRADE the permission set is tightened and Claude continues
  7. On TERMINATE execution halts and the incident is logged
  8. Reward is fed back to the bandit after each mediation decision
"""

import uuid
import time
from typing import Optional

import anthropic

from config import ANTHROPIC_API_KEY, MODEL
from models import AuthorityTier, GovernanceAction, PermissionSet, TaskRecord
from layers.delegation import DelegationBoundaryLayer
from layers.identity import IdentityStructuringLayer
from layers.mediation import ContinuousMediationLayer
from bandit.thompson import ThompsonSamplingBandit
from tools.registry import get_tools_for_tier, execute_tool


_SYSTEM_PROMPT = """\
You are a governed AI agent operating under strict authority boundaries.
You have access only to the tools listed. Do not attempt actions outside \
your delegated scope.
Complete the user's task efficiently using only the provided tools.
If you cannot complete the task within your permissions, say so clearly.
"""


class GEAAAgent:

    def __init__(self, bandit: Optional[ThompsonSamplingBandit] = None):
        self.client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.dbl     = DelegationBoundaryLayer()
        self.isl     = IdentityStructuringLayer()
        self.bandit  = bandit or ThompsonSamplingBandit(state_path="geaa_bandit_state.json")
        self.cml     = ContinuousMediationLayer(self.bandit)
        self.audit_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_task(
        self,
        task: str,
        is_adversarial: bool = False,
        max_tool_rounds: int = 6,
    ) -> TaskRecord:
        """
        Execute a task under GEAA governance.

        Args:
            task:            Natural-language task description.
            is_adversarial:  Ground truth label (PoC only; used for reward signal).
            max_tool_rounds: Hard cap on agentic tool-use loops.
        """
        task_id = str(uuid.uuid4())[:8]
        tier, risk_vector = self.dbl.classify(task)
        perm_set = self.isl.instantiate(tier)
        self.cml.set_task(task)

        record = TaskRecord(
            task_id=task_id,
            task_description=task,
            tier=tier,
            is_adversarial=is_adversarial,
        )

        self._log("TASK_START", task_id, {
            "tier": tier.name,
            "risk_vector": vars(risk_vector),
            "integrity_token": getattr(perm_set, "integrity_token", "n/a"),
        })

        messages = [{"role": "user", "content": task}]
        terminated = False

        for round_idx in range(max_tool_rounds):
            # Get only the tools allowed for the current (possibly downgraded) tier
            available_tools = get_tools_for_tier(perm_set.tier)

            response = self.client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                tools=available_tools,
                messages=messages,
            )

            # No tool calls → task complete
            if response.stop_reason == "end_turn":
                text = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                self._log("TASK_COMPLETE", task_id, {"response": text, "rounds": round_idx + 1})
                record.final_outcome = GovernanceAction.APPROVE
                return record

            # Process all tool calls in this response
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_args = block.input

                # ---- CML evaluation ----
                med_result = self.cml.evaluate(tool_name, tool_args, perm_set)
                reward = self.cml.record_reward(med_result, is_adversarial)
                record.mediation_results.append(med_result)

                self._log("TOOL_CALL_EVALUATED", task_id, {
                    "tool": tool_name,
                    "args": tool_args,
                    "action": med_result.action.name,
                    "severity": round(med_result.severity_score, 2),
                    "reason": med_result.reason,
                    "reward": reward,
                })

                if med_result.action == GovernanceAction.TERMINATE:
                    self._escalate(task_id, tool_name, med_result)
                    record.final_outcome = GovernanceAction.TERMINATE
                    terminated = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "[GOVERNANCE] Tool call TERMINATED. Escalated to human reviewer.",
                    })
                    break

                elif med_result.action == GovernanceAction.DOWNGRADE:
                    perm_set = self.isl.downgrade(perm_set)
                    # Re-evaluate tier after downgrade
                    tier = perm_set.tier
                    self._log("PERMISSION_DOWNGRADE", task_id, {"new_tier": tier.name})
                    # Still execute the tool under reduced permissions (if allowed post-downgrade)
                    if tool_name in perm_set.allowed_tools:
                        output = execute_tool(tool_name, tool_args)
                    else:
                        output = f"[GOVERNANCE] '{tool_name}' blocked after permission downgrade."
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    })

                else:  # APPROVE
                    output = execute_tool(tool_name, tool_args)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    })

            if terminated:
                break

            # Feed tool results back into conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        # DBL reclassification feedback loop if CML accumulated deviations
        if self.cml.session_deviations > 0:
            new_tier = self.dbl.reclassify(task, self.cml.session_deviations)
            if new_tier != tier:
                self._log("DBL_RECLASSIFY", task_id, {
                    "old_tier": tier.name,
                    "new_tier": new_tier.name,
                    "deviations": self.cml.session_deviations,
                })

        if record.final_outcome is None:
            record.final_outcome = GovernanceAction.APPROVE

        return record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, event: str, task_id: str, data: dict) -> None:
        entry = {"ts": time.time(), "event": event, "task_id": task_id, **data}
        self.audit_log.append(entry)

    def _escalate(self, task_id: str, tool_name: str, result) -> None:
        self._log("HUMAN_ESCALATION", task_id, {
            "tool_attempted": tool_name,
            "severity_score": result.severity_score,
            "reason": result.reason,
            "hard_blocked": result.hard_blocked,
        })
