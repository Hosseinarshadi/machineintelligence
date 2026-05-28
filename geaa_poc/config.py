import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-6"

# Severity Score decision bands (normalized from multiplicative 1-5^4 = max 625)
SEVERITY_APPROVE_MAX = 50    # score <= 50  -> eligible for APPROVE
SEVERITY_DOWNGRADE_MAX = 150 # score <= 150 -> eligible for DOWNGRADE
# score > 150 -> forced TERMINATE regardless of bandit

# Bandit warm-start priors (arm-specific Beta parameters)
# Biased toward APPROVE at cold-start; skeptical about TERMINATE without evidence.
# APPROVE=0, DOWNGRADE=1, TERMINATE=2
BANDIT_ALPHA_INIT = [5.0, 2.0, 1.0]   # alpha per arm
BANDIT_BETA_INIT  = [1.0, 1.0, 5.0]   # beta per arm

# Session anomaly caps
MAX_DEVIATIONS_BEFORE_LOCKDOWN = 3

# Tool execution
DRY_RUN = True  # Simulated tools by default; set False to enable real Claude tool calls
