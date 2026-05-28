"""
GEAA Pipeline Elements for AgentDojo.

Implements the three GEAA layers as AgentDojo BasePipelineElement subclasses
so they can be composed with AgentDojo's standard LLM and tool execution elements.

Flow:
  GEAADelegationElement  →  AnthropicLLM  →  ToolsExecutionLoop[AnthropicLLM + GEAAMediationExecutor]

The GEAADelegationElement runs once per task:
  - classifies task text → authority tier (DBL)
  - filters runtime.functions to tools allowed for that tier (ISL permission boundary)
  - initialises CML + bandit context in extra_args

The GEAAMediationExecutor replaces plain ToolsExecutor:
  - intercepts every tool call before runtime execution (CML hard check + severity + bandit)
  - logs decisions for post-run metric computation
  - on TERMINATE replaces tool result with a governance-block message
"""

import re
import sys
from ast import literal_eval
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import yaml

sys.path.insert(0, ".")

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatMessage,
    ChatToolResultMessage,
    text_content_block_from_string,
)

from bandit.thompson import ThompsonSamplingBandit
from layers.delegation import DelegationBoundaryLayer
from layers.identity import IdentityStructuringLayer
from layers.mediation import ContinuousMediationLayer, _detect_injection_signals, _compute_severity, _INJECTION_PATTERNS
from models import AuthorityTier, GovernanceAction, PermissionSet


# ---------------------------------------------------------------------------
# AgentDojo tool → GEAA tier mapping
# These patterns mirror the risk classification from Table 1 in the GEAA paper.
# ---------------------------------------------------------------------------

_L3_TOOL_NAMES = {
    "send_money",
    "schedule_transaction",
    "update_password",
    "update_user_info",
}

_L2_TOOL_NAMES = {
    "send_email",
    "create_calendar_event",
    "cancel_calendar_event",
    "reschedule_calendar_event",
    "add_calendar_event_participants",
    "delete_email",
    "share_file",
    "delete_file",
    "create_file",
    "append_to_file",
    "update_scheduled_transaction",
    "post_message",
    "send_direct_message",
    "invite_user_to_slack",
    "add_user_to_slack",
    "book_hotel",
    "book_flight",
    "cancel_hotel_reservation",
    "cancel_flight",
}


def _min_tier_for_tool(tool_name: str) -> AuthorityTier:
    if tool_name in _L3_TOOL_NAMES:
        return AuthorityTier.L3
    if tool_name in _L2_TOOL_NAMES:
        return AuthorityTier.L2
    return AuthorityTier.L1


def _filter_functions(runtime: FunctionsRuntime, tier: AuthorityTier) -> dict:
    """Return only the functions allowed for the given authority tier."""
    return {
        name: func
        for name, func in runtime.functions.items()
        if _min_tier_for_tool(name) <= tier
    }


def _tool_result_to_str(result: Any) -> str:
    if isinstance(result, dict):
        return yaml.safe_dump(result).strip()
    if isinstance(result, list):
        return yaml.safe_dump([r.model_dump() if hasattr(r, "model_dump") else r for r in result]).strip()
    if hasattr(result, "model_dump"):
        return yaml.safe_dump(result.model_dump()).strip()
    return str(result)


def _is_string_list(s: str) -> bool:
    try:
        return isinstance(literal_eval(s), list)
    except (ValueError, SyntaxError):
        return False


def _scan_text_for_injection(text: str) -> bool:
    """Return True if any GEAA injection pattern matches in the given text."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _extract_result_text(msg) -> str:
    """Pull all text out of a ChatToolResultMessage content block list."""
    parts = []
    for block in msg.get("content") or []:
        if isinstance(block, dict):
            parts.append(block.get("text", ""))
        else:
            parts.append(getattr(block, "text", ""))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Layer 1 — Delegation Boundary Layer pipeline element
# ---------------------------------------------------------------------------

class GEAADelegationElement(BasePipelineElement):
    """
    First pipeline element.  Classifies the task, filters the runtime,
    and stores GEAA governance context in extra_args for downstream elements.
    """

    def __init__(self, bandit: ThompsonSamplingBandit) -> None:
        self.dbl = DelegationBoundaryLayer()
        self.isl = IdentityStructuringLayer()
        self.bandit = bandit

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:

        tier, risk_vector = self.dbl.classify(query)

        # ISL: filter runtime to only tools for this tier
        allowed = _filter_functions(runtime, tier)
        runtime.update_functions(allowed)

        # Initialise CML for this task
        cml = ContinuousMediationLayer(self.bandit)
        cml.set_task(query)

        # Build permission set reflecting the allowed AgentDojo tools
        perm_set = PermissionSet(
            tier=tier,
            allowed_tools=set(allowed.keys()),
            max_external_calls=5 if tier >= AuthorityTier.L2 else 0,
            memory_access=(tier == AuthorityTier.L1),
            ttl_seconds=300,
        )

        extra_args = {
            **extra_args,
            "geaa_tier":              tier,
            "geaa_risk_vector":       risk_vector,
            "geaa_perm_set":          perm_set,
            "geaa_cml":               cml,
            "geaa_decisions":         [],    # list of MediationResult objects
            "geaa_injection_signals": [],    # flat list of detected signal strings
            "geaa_terminated":        False,
        }
        return query, runtime, env, messages, extra_args


# ---------------------------------------------------------------------------
# Layer 3 — Continuous Mediation Layer pipeline element
# ---------------------------------------------------------------------------

class GEAAMediationExecutor(BasePipelineElement):
    """
    Replaces plain ToolsExecutor.  Intercepts each tool call with GEAA CML
    before dispatching to runtime.run_function().

    Decision outcomes:
      APPROVE   — execute normally
      DOWNGRADE — execute, note escalated deviation count, tighten future decisions
      TERMINATE — block execution, return governance message, mark task as terminated
    """

    # Class-level log shared across all benchmark runs for metric aggregation
    execution_log: list[dict] = []

    def __init__(self) -> None:
        pass

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:

        if not messages or messages[-1].get("role") != "assistant":
            return query, runtime, env, messages, extra_args

        tool_calls: list[FunctionCall] = messages[-1].get("tool_calls") or []
        if not tool_calls:
            return query, runtime, env, messages, extra_args

        cml: ContinuousMediationLayer | None = extra_args.get("geaa_cml")
        perm_set: PermissionSet | None        = extra_args.get("geaa_perm_set")
        task_context: str                     = query

        # Consume result-injection flag from the previous loop iteration.
        # Set by scanning approved tool returns for embedded injection content.
        prior_injection: bool = extra_args.pop("geaa_result_injection", False)

        tool_results: list[ChatToolResultMessage] = []

        for tool_call in tool_calls:
            fn_name = tool_call.function
            fn_args = dict(tool_call.args)

            # ----------------------------------------------------------------
            # Guard: unknown or empty function name (AgentDojo convention)
            # ----------------------------------------------------------------
            if fn_name == EMPTY_FUNCTION_NAME:
                tool_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error="Empty function name.",
                    )
                )
                continue

            if fn_name not in runtime.functions:
                tool_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=f"Tool '{fn_name}' is not available.",
                    )
                )
                continue

            # ----------------------------------------------------------------
            # GEAA CML evaluation (carries prior result-injection context)
            # ----------------------------------------------------------------
            if cml and perm_set:
                med_result = cml.evaluate(fn_name, fn_args, perm_set, prior_injection=prior_injection)
                extra_args["geaa_decisions"].append(med_result)
                if med_result.reason:
                    sigs = [s for s in med_result.reason.split("; ") if "injection_signals" in s]
                    extra_args["geaa_injection_signals"].extend(sigs)

                # Heuristic reward signal so the bandit learns within the session.
                # Grounded in information available at call time (no ground truth needed).
                if not med_result.hard_blocked and med_result.context_id >= 0:
                    injection_detected = bool(sigs)
                    action = med_result.action
                    if injection_detected and action == GovernanceAction.TERMINATE:
                        reward = 0.9   # proactive block on detected signal
                    elif injection_detected and action == GovernanceAction.DOWNGRADE:
                        reward = 0.6   # partial mitigation
                    elif injection_detected and action == GovernanceAction.APPROVE:
                        reward = -0.4  # possibly letting injection through
                    elif not injection_detected and action == GovernanceAction.APPROVE:
                        reward = 0.7   # clean approval of benign-looking call
                    elif not injection_detected and action == GovernanceAction.DOWNGRADE:
                        reward = -0.1  # mild over-caution
                    else:  # not injected, TERMINATE
                        reward = -0.5  # likely false termination
                    cml.bandit.update(med_result.context_id, med_result.arm_selected, reward)

                # Log for post-run metric aggregation
                self.execution_log.append({
                    "task":           task_context[:80],
                    "tool":           fn_name,
                    "action":         med_result.action.name,
                    "severity":       med_result.severity_score,
                    "hard_blocked":   med_result.hard_blocked,
                    "injection_sigs": bool(sigs),
                    "context_id":     med_result.context_id,
                    "arm":            med_result.arm_selected,
                    "tier":           perm_set.tier.name,
                })

                if med_result.action == GovernanceAction.TERMINATE:
                    extra_args["geaa_terminated"] = True
                    tool_results.append(
                        ChatToolResultMessage(
                            role="tool",
                            content=[text_content_block_from_string(
                                f"[GEAA GOVERNANCE] Tool '{fn_name}' call TERMINATED. "
                                f"Severity={med_result.severity_score:.1f}. "
                                "Incident escalated to human reviewer. Task halted."
                            )],
                            tool_call_id=tool_call.id,
                            tool_call=tool_call,
                            error=f"Governance TERMINATE: {med_result.reason}",
                        )
                    )
                    continue  # skip execution, try next tool_call

            # ----------------------------------------------------------------
            # Allowed — coerce list args then execute
            # ----------------------------------------------------------------
            for k, v in fn_args.items():
                if isinstance(v, str) and _is_string_list(v):
                    tool_call.args[k] = literal_eval(v)

            result, error = runtime.run_function(env, fn_name, fn_args)
            formatted = _tool_result_to_str(result)
            result_msg = ChatToolResultMessage(
                role="tool",
                content=[text_content_block_from_string(formatted)],
                tool_call_id=tool_call.id,
                tool_call=tool_call,
                error=error,
            )
            tool_results.append(result_msg)

            # Scan the returned content for embedded injection patterns.
            # If found, set flag so the NEXT tool call's CML evaluation knows
            # prior content was poisoned (catches content-borne attacks like
            # important_instructions where injection lives in data, not args).
            if not error and _scan_text_for_injection(formatted):
                extra_args["geaa_result_injection"] = True

        return query, runtime, env, [*messages, *tool_results], extra_args
