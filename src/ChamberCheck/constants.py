"""
Common constants and defaults for ChamberCheck.

This file stores all hardcoded defaults and configuration options.
"""

from datetime import datetime, timedelta

# ============================================================================
# SCRAPER DEFAULTS
# ============================================================================

DEFAULT_SUBREDDIT = "samharris"
DEFAULT_USER_AGENT = "ChamberCheck/0.1 Research"

# Default posts to fetch
DEFAULT_NUM_POSTS = 100 #API max is 101.
DEFAULT_SORT_METHOD = "top"

# Default date range (1 year back from 2025)
DEFAULT_START_DATE = datetime(2018, 1, 1)
DEFAULT_END_DATE = datetime(2025, 12, 31)

# Default comment limit per post
DEFAULT_COMMENTS_PER_POST = 2000

# Rate limiting
SCRAPER_RATE_LIMIT_DELAY = 2  # seconds between API requests
SCRAPER_REQUEST_TIMEOUT = 10  # seconds

# ============================================================================
# LLM PROVIDER DEFAULTS
# ============================================================================

# Default providers
DEFAULT_LLM_PROVIDER = "openai"

# Default models
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MINI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

# Temperature for analysis (lower = more consistent, deterministic)
LLM_TEMPERATURE = 0.3

# Max tokens for LLM response
LLM_MAX_TOKENS = 500

# ============================================================================
# ANALYSIS DEFAULTS
# ============================================================================

# Default number of comments to analyze (top by engagement)
DEFAULT_ANALYSIS_TOP_N = 400

# Filter method
DEFAULT_ANALYSIS_FILTER = "top_by_absolute_score"

# ============================================================================
# DATA DIRECTORIES
# ============================================================================

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_DIR = "data/output"

# ============================================================================
# FILE NAMING PATTERNS
# ============================================================================

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
SCRAPER_POSTS_FILENAME_PATTERN = "{subreddit}_posts_json{keyword_suffix}_{timestamp}.json"
SCRAPER_COMMENTS_FILENAME_PATTERN = "{subreddit}_comments_json{keyword_suffix}_{timestamp}.json"
ANALYSIS_RESULTS_FILENAME_PATTERN = "{subreddit}_analysis{keyword_suffix}_{timestamp}.json"
ANALYSIS_METADATA_FILENAME_PATTERN = "{subreddit}_analysis{keyword_suffix}_{timestamp}_metadata.json"

# ============================================================================
# ECHO CHAMBER METRICS
# ============================================================================

# Score ranges
MIN_METRIC_SCORE = 0
MAX_METRIC_SCORE = 10
MAX_ECHO_CHAMBER_SCORE = 50  # Sum of 5 metrics (0-10 each)

# Metric names (first 4 required, 5th optional for replies)
METRICS = [
    "argument_narrowness",
    "hostility",
    "suppression",
    "epistemic_closure",
    "argument_avoidance"
]

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

ANALYSIS_SYSTEM_PROMPT = "You are an expert social scientist analyzing online discourse for echo chamber dynamics. Respond only with valid JSON."

ANALYSIS_INSTRUCTIONS_PROMPT = """Analyze the following Reddit comment for echo chamber dynamics.

Step 1 — Classify the comment type(s) (one or more, list all that apply):
- argumentative (making or rebutting a claim)
- factual (sharing facts/resources without arguing)
- anecdotal (sharing personal story/experience)
- question (asking a question)
- humor_or_irony (joke, sarcasm, irony)
- other

Step 2 — Score metrics on 0-10 where HIGHER = more echo-chamber-like. If a metric does NOT apply to the comment type(s), return "N/A" for that metric.

1. Argument Narrowness (0-10, or N/A if non-argumentative)
   - 10: Ignores or denies other perspectives; false dichotomies
   - 5: Presents a single viewpoint without explicit dismissal
   - 0: Actively engages multiple perspectives

2. Hostility (0-10)
   - 10: Aggressive, insulting, dehumanizing
   - 5: Mild sarcasm/condescension
   - 0: Respectful and civil

3. Suppression (0-10, or N/A if non-argumentative)
   - 10: Discourages debate, ridicules dissent, “this is settled”
   - 5: Implies opposition is foolish
   - 0: Invites counter-arguments

4. Epistemic Closure (0-10, or N/A if non-argumentative)
   - 10: Rejects evidence, tribal certainty
   - 5: Accepts some evidence but cherry-picks
   - 0: Open to updating based on evidence

5. Argument Avoidance (0-10, replies only; N/A for top-level or non-argumentative)
   - 10: Ignores parent’s points, topic shifts, ad hominem
   - 5: Acknowledges parent but dodges specifics
   - 0: Directly engages specific points; steelman

Scoring rule: echo_chamber_score = sum of numeric metric values only (skip N/A). Do not normalize.

Return JSON only (no markdown, no code blocks):
{
  "comment_types": ["..."],
  "argument_narrowness": 0-10 or "N/A",
  "hostility": 0-10 or "N/A",
  "suppression": 0-10 or "N/A",
  "epistemic_closure": 0-10 or "N/A",
  "argument_avoidance": 0-10 or "N/A",
  "echo_chamber_score": 0-50 (sum of numeric metrics),
  "reasoning": "brief explanation"
}"""

# ============================================================================
# ANALYZE COMMENTS SCRIPT
# ============================================================================

ANALYZE_COMMENTS_SCRIPT = "python scripts/analyze_comments.py"
