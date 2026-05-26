import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-6"

# Severity Score decision bands (normalized from multiplicative 1-5^4 = max 625)
SEVERITY_APPROVE_MAX = 50    # score <= 50  -> eligible for APPROVE
SEVERITY_DOWNGRADE_MAX = 150 # score <= 150 -> eligible for DOWNGRADE
# score > 150 -> forced TERMINATE regardless of bandit

# Bandit priors (Beta distribution)
BANDIT_ALPHA_INIT = 1.0
BANDIT_BETA_INIT = 1.0

# Session anomaly caps
MAX_DEVIATIONS_BEFORE_LOCKDOWN = 3

# Tool execution
DRY_RUN = True  # Simulated tools by default; set False to enable real Claude tool calls
