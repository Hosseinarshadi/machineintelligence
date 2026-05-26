"""
Tool registry for the GEAA PoC.

Each entry defines the Anthropic-compatible tool schema and the minimum
authority tier required to execute it.  The ISL uses this mapping to
filter available tools per agent identity.
"""

from models import AuthorityTier

# ---------------------------------------------------------------------------
# Simulated tool implementations (PoC — no side effects)
# ---------------------------------------------------------------------------

def _calendar_lookup(query: str) -> str:
    return f"[SIMULATED] Calendar events matching '{query}': 09:00 Team standup, 14:00 Review meeting"


def _document_summarize(doc_id: str) -> str:
    return f"[SIMULATED] Summary of document {doc_id}: quarterly review showing 12% growth in Q3."


def _knowledge_retrieve(topic: str) -> str:
    return f"[SIMULATED] Knowledge base results for '{topic}': 3 articles found."


def _meeting_schedule(title: str, participants: str, time: str) -> str:
    return f"[SIMULATED] Meeting '{title}' scheduled with {participants} at {time}."


def _internal_search(query: str) -> str:
    return f"[SIMULATED] Internal search for '{query}': 7 results found."


def _email_draft(to: str, subject: str, body: str) -> str:
    return f"[SIMULATED] Email DRAFTED (not sent) to {to} re: {subject}."


def _file_share(file_id: str, recipient: str) -> str:
    return f"[SIMULATED] File {file_id} share link generated for {recipient}."


def _slack_post(channel: str, message: str) -> str:
    return f"[SIMULATED] Slack message DRAFTED (not posted) to #{channel}."


def _procurement_query(vendor: str, item: str) -> str:
    return f"[SIMULATED] Procurement query for {item} from {vendor}: availability confirmed."


def _external_search(query: str) -> str:
    return f"[SIMULATED] External search for '{query}': 12 web results."


def _sandbox_read(path: str) -> str:
    return f"[SIMULATED] Sandbox read of {path}: file contents returned (sandboxed)."


def _sandbox_analyze(data: str) -> str:
    return f"[SIMULATED] Sandbox analysis of provided data: no anomalies detected."


# ---------------------------------------------------------------------------
# Registry: tool_name → (handler, min_tier, anthropic_schema)
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict] = {
    "calendar_lookup": {
        "handler": _calendar_lookup,
        "min_tier": AuthorityTier.L1,
        "schema": {
            "name": "calendar_lookup",
            "description": "Look up calendar events for a given query or date range.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query or date"}},
                "required": ["query"],
            },
        },
    },
    "document_summarize": {
        "handler": _document_summarize,
        "min_tier": AuthorityTier.L1,
        "schema": {
            "name": "document_summarize",
            "description": "Summarize a document by ID.",
            "input_schema": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
        },
    },
    "knowledge_retrieve": {
        "handler": _knowledge_retrieve,
        "min_tier": AuthorityTier.L1,
        "schema": {
            "name": "knowledge_retrieve",
            "description": "Retrieve internal knowledge base articles on a topic.",
            "input_schema": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    },
    "meeting_schedule": {
        "handler": _meeting_schedule,
        "min_tier": AuthorityTier.L1,
        "schema": {
            "name": "meeting_schedule",
            "description": "Schedule a meeting with specified participants.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "title":        {"type": "string"},
                    "participants": {"type": "string"},
                    "time":         {"type": "string"},
                },
                "required": ["title", "participants", "time"],
            },
        },
    },
    "internal_search": {
        "handler": _internal_search,
        "min_tier": AuthorityTier.L1,
        "schema": {
            "name": "internal_search",
            "description": "Search internal company resources.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    "email_draft": {
        "handler": _email_draft,
        "min_tier": AuthorityTier.L2,
        "schema": {
            "name": "email_draft",
            "description": "Draft an email to an external recipient (not yet sent).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "to":      {"type": "string"},
                    "subject": {"type": "string"},
                    "body":    {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    "file_share": {
        "handler": _file_share,
        "min_tier": AuthorityTier.L2,
        "schema": {
            "name": "file_share",
            "description": "Generate a share link for a file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_id":   {"type": "string"},
                    "recipient": {"type": "string"},
                },
                "required": ["file_id", "recipient"],
            },
        },
    },
    "slack_post": {
        "handler": _slack_post,
        "min_tier": AuthorityTier.L2,
        "schema": {
            "name": "slack_post",
            "description": "Post a message to a Slack channel.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["channel", "message"],
            },
        },
    },
    "procurement_query": {
        "handler": _procurement_query,
        "min_tier": AuthorityTier.L2,
        "schema": {
            "name": "procurement_query",
            "description": "Query procurement system for vendor availability.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "vendor": {"type": "string"},
                    "item":   {"type": "string"},
                },
                "required": ["vendor", "item"],
            },
        },
    },
    "external_search": {
        "handler": _external_search,
        "min_tier": AuthorityTier.L2,
        "schema": {
            "name": "external_search",
            "description": "Search the public web for information.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    "sandbox_read": {
        "handler": _sandbox_read,
        "min_tier": AuthorityTier.L3,
        "schema": {
            "name": "sandbox_read",
            "description": "Read a file in the isolated sandbox environment.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    "sandbox_analyze": {
        "handler": _sandbox_analyze,
        "min_tier": AuthorityTier.L3,
        "schema": {
            "name": "sandbox_analyze",
            "description": "Analyze data in the sandbox (no production side effects).",
            "input_schema": {
                "type": "object",
                "properties": {"data": {"type": "string"}},
                "required": ["data"],
            },
        },
    },
}


def get_tools_for_tier(tier: AuthorityTier) -> list[dict]:
    """Return Anthropic-compatible tool schemas for a given authority tier."""
    return [
        entry["schema"]
        for entry in TOOL_REGISTRY.values()
        if entry["min_tier"] <= tier
    ]


def execute_tool(name: str, args: dict) -> str:
    """Dispatch tool call to the simulated handler."""
    entry = TOOL_REGISTRY.get(name)
    if not entry:
        return f"[ERROR] Unknown tool: {name}"
    return entry["handler"](**args)
