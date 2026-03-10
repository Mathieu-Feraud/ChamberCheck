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
ANTHROPIC_HAIKU_MODEL = "claude-3-haiku-20240307"

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
# PREPROCESSING SELECTION
# ============================================================================

# Minimum discussion_score for a post to pass the preprocessing filter
PREPROCESSING_MIN_DISCUSSION_SCORE = 0.8

# Minimum number of posts sharing the same topic group (leaf preferred, mid
# fallback) for that group to qualify. Posts in groups below this threshold
# are excluded from the pre_process output.
PREPROCESSING_MIN_TOPIC_PEERS = 6

# ============================================================================
# COMMENT SCRAPING
# ============================================================================

# Only include comments posted within this many days of the original post
COMMENT_SCRAPING_WINDOW_DAYS = 3

# ============================================================================
# COMMENT PREPROCESSING
# ============================================================================

# Maximum number of comments to sample per post (trunk-based random sampling
# without replacement: whole trunk threads are added until this target is met
# or all trunks are exhausted — the final trunk is never split)
COMMENT_PREPROCESSING_MAX_SAMPLE = 100

# Comments (and their entire subtrees) shorter than this character count are
# removed before sampling
COMMENT_PREPROCESSING_MIN_CONTENT_LENGTH = 20

# ============================================================================
# COMMENT ANALYSIS PIPELINE
# ============================================================================

# Default provider and model for comment analysis
COMMENT_ANALYSIS_PROVIDER = "anthropic"
COMMENT_ANALYSIS_MODEL = "claude-haiku-4-5-20251001"

# Seconds to sleep between successive LLM calls (rate limit courtesy)
COMMENT_ANALYSIS_RATE_LIMIT_DELAY = 0.5

# Retry behaviour on transient API errors
COMMENT_ANALYSIS_MAX_RETRIES = 3
COMMENT_ANALYSIS_RETRY_WAIT_SECONDS = 5

# Sub-folder name inside scrape_XXX for comment analysis outputs
COMMENT_ANALYSIS_OUTPUT_BASE = "data/output"  # Base dir for all analysis outputs
COMMENT_ANALYSIS_FILE_PREFIX = "comment_analysis"  # Filename prefix (no trailing 's')

# ============================================================================
# POST TITLE ANALYSIS
# ============================================================================

# Retry behaviour for OpenAI API calls during post title analysis
POST_ANALYSIS_RETRY_LIMIT = 3
POST_ANALYSIS_RETRY_DELAY = 5  # seconds between retries

# User message template sent to the LLM for each post
POST_ANALYSIS_USER_TEMPLATE = 'Post title: "{title}"'

# Maximum number of secondary topic assignments allowed for post title analysis
POST_ANALYSIS_MAX_SECONDARY_TOPICS = 2

# Fixed taxonomy for post title analysis (leaf -> (mid, top))
POST_TOPIC_TAXONOMY = {
   # Politics / U.S.
   "trump_legal_cases": ("us_politics", "politics"),
   "trump_campaign_elections": ("us_politics", "politics"),
   "biden_admin_policy": ("us_politics", "politics"),
   "congress_legislation": ("us_politics", "politics"),
   "supreme_court_constitution": ("us_politics", "politics"),
   "election_integrity": ("us_elections", "politics"),
   "voting_access_rules": ("us_elections", "politics"),
   "media_bias_propaganda": ("political_communication", "politics"),
   "free_speech_censorship": ("civil_liberties", "politics"),
   "policing_crime_policy": ("domestic_policy", "politics"),
   "immigration_border_enforcement": ("domestic_policy", "politics"),
   "asylum_refugee_policy": ("domestic_policy", "politics"),
   "taxation_spending_policy": ("domestic_policy", "politics"),
   "welfare_social_programs": ("domestic_policy", "politics"),
   "inflation_cost_of_living_policy": ("domestic_policy", "politics"),

   # Politics / Geopolitics
   "israel_palestine_gaza": ("geopolitics_conflict", "politics"),
   "russia_ukraine": ("geopolitics_conflict", "politics"),
   "china_taiwan": ("geopolitics_conflict", "politics"),
   "nato_west_security": ("geopolitics_conflict", "politics"),
   "sanctions_energy_war": ("geopolitics_conflict", "politics"),

   # Economics
   "labor_rights": ("labor_and_work", "economics"),
   "wages_and_cost_of_living": ("labor_and_work", "economics"),
   "capitalism_critique": ("economic_systems", "economics"),
   "socialism_communism": ("economic_systems", "economics"),
   "inequality_and_wealth": ("distribution", "economics"),

   # Religion
   "christian_doctrine": ("christianity", "religion"),
   "atheism_theism_debate": ("belief_conflict", "religion"),
   "religion_in_public_life": ("public_life", "religion"),

   # Technology / Science
   "ai_ethics_and_risk": ("ai", "technology"),
   "platform_moderation_policy": ("internet_governance", "technology"),
   "privacy_surveillance": ("digital_rights", "technology"),
   "bioethics_health_claims": ("public_reasoning", "science"),
   "climate_science_policy": ("public_reasoning", "science"),

   # Society / Philosophy
   "moral_philosophy": ("ethics", "philosophy"),
   "political_philosophy": ("ideology", "philosophy"),
   "identity_politics": ("culture_war", "society"),
   "gender_family_values": ("social_values", "society"),
}

POST_TOPIC_ALLOWED_TOP = [
   "politics",
   "economics",
   "religion",
   "technology",
   "science",
   "philosophy",
   "society",
   "UNCLEAR",
]

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

ANALYSIS_SYSTEM_PROMPT = "You are an expert social scientist analyzing online discourse for echo chamber dynamics. Respond only with valid JSON."

PROMPT_POST_TITLE_ANALYSIS = """\
You are a research assistant analyzing Reddit post titles for discourse research.

You will be given only a post title.
Return ONLY valid JSON (no markdown, no prose).

OUTPUT SCHEMA:
{
   "topic": {
      "top": "<TOP>",
      "mid": "<MID>",
      "leaf": "<LEAF or null>"
   },
   "secondary_topics": [
      {
         "top": "<TOP>",
         "mid": "<MID>",
         "leaf": "<LEAF>"
      }
   ],
   "topic_confidence": <float 0.0-1.0>,
   "secondary_confidence": [<float 0.0-1.0>],
   "discussion_score": <float 0.0-1.0>,
   "discussion_reason": "<one short sentence>"
}

TASK A — TOPIC CLASSIFICATION (STRICT, PREDEFINED TAXONOMY)
- Use ONLY labels from this fixed taxonomy (leaf -> [mid, top]):
   trump_legal_cases -> [us_politics, politics]
   trump_campaign_elections -> [us_politics, politics]
   biden_admin_policy -> [us_politics, politics]
   congress_legislation -> [us_politics, politics]
   supreme_court_constitution -> [us_politics, politics]
   election_integrity -> [us_elections, politics]
   voting_access_rules -> [us_elections, politics]
   media_bias_propaganda -> [political_communication, politics]
   free_speech_censorship -> [civil_liberties, politics]
   policing_crime_policy -> [domestic_policy, politics]
   immigration_border_enforcement -> [domestic_policy, politics]
   asylum_refugee_policy -> [domestic_policy, politics]
   taxation_spending_policy -> [domestic_policy, politics]
   welfare_social_programs -> [domestic_policy, politics]
   inflation_cost_of_living_policy -> [domestic_policy, politics]
   israel_palestine_gaza -> [geopolitics_conflict, politics]
   russia_ukraine -> [geopolitics_conflict, politics]
   china_taiwan -> [geopolitics_conflict, politics]
   nato_west_security -> [geopolitics_conflict, politics]
   sanctions_energy_war -> [geopolitics_conflict, politics]
   labor_rights -> [labor_and_work, economics]
   wages_and_cost_of_living -> [labor_and_work, economics]
   capitalism_critique -> [economic_systems, economics]
   socialism_communism -> [economic_systems, economics]
   inequality_and_wealth -> [distribution, economics]
   christian_doctrine -> [christianity, religion]
   atheism_theism_debate -> [belief_conflict, religion]
   religion_in_public_life -> [public_life, religion]
   ai_ethics_and_risk -> [ai, technology]
   platform_moderation_policy -> [internet_governance, technology]
   privacy_surveillance -> [digital_rights, technology]
   bioethics_health_claims -> [public_reasoning, science]
   climate_science_policy -> [public_reasoning, science]
   moral_philosophy -> [ethics, philosophy]
   political_philosophy -> [ideology, philosophy]
   identity_politics -> [culture_war, society]
   gender_family_values -> [social_values, society]

- Allowed top labels: politics, economics, religion, technology, science, philosophy, society, UNCLEAR.
- Primary `topic` is required and should be the best single fit.
- `secondary_topics` is optional and can include 0 to 2 additional topics from the same taxonomy.
- Do not duplicate the primary topic in `secondary_topics`.
- If `secondary_topics` is empty, return an empty array and empty `secondary_confidence` array.
- If ambiguous, return:
   topic = {"top": "UNCLEAR", "mid": null, "leaf": null}
   topic_confidence <= 0.30
   secondary_topics = []
   secondary_confidence = []

TASK B — DISCUSSION SCORE (CALIBRATED)
Score expected substantive, opinion-driven disagreement:

0.00-0.20: Non-debate content (meme-like, personal update, simple announcement)
0.21-0.40: Mostly informational, weak disagreement potential
0.41-0.60: Some controversy, mixed discussion likelihood
0.61-0.80: Clear opposing viewpoints likely
0.81-1.00: Highly polarizing conflict likely

SCORING RULES
- Use the full range; do not default to high values.
- Reserve scores >0.80 for strongly polarizing moral/identity/political conflict.
- If title is primarily descriptive with no clear claim/conflict, usually score <=0.55.
- `discussion_reason` must be one concise sentence grounded in title wording.

QUALITY CHECK BEFORE RETURNING
- Validate JSON shape and numeric ranges.
- Ensure each topic triplet is consistent with the fixed taxonomy.
- Ensure `secondary_confidence` length exactly matches `secondary_topics` length.
- Return JSON only."""

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

COMMENT_ANALYSIS_PROMPT = """You will be given a single Reddit post with full prior thread context and relevant metadata.

The context includes the original post and any earlier comments in chronological thread order.
Use all prior context for interpretation, but score and analyze only the final comment in COMMENT TO ANALYZE.

Your task is to extract structured attributes used to measure discourse dynamics.

Unless a scoring rubric is provided for a specific attribute, use 0-10 scale where:
0 = completely absent
1-3 = weak or uncertain presence  
4-6 = moderate presence
7-9 = strong presence
10 = extreme presence
Some attributes have custom scales defined below.

If an attribute is not applicable, return "N/A", when in doubt, return "N/A".
Return valid JSON only.

DEFINITIONS:

comment_type (array of strings):
   Assign one or more types from this fixed list: question, information, opinion, sarcasm, meta-commentary, off-topic, advice, correction, anecdote, low effort, humour, other.

topic -> label (string):
   If given a parent topic label, decide whether the comment addresses the same substantive topic or introduces a new, meaningfully different topic. If it is the same topic, reuse the parent topic label verbatim. If it introduces a new topic, create a high-level topic label that can reasonably encompass future replies. Use the format "Global - Subcategory - Subcategory" (add more subcategories if needed). Example: "politics - law - abortion law - abortion loopholes".

topic -> stance -> value (number or N/A):
   Whether the author agrees or disagrees with the parent's position, argument, or claim, if parent expresses negative sentiment about X and commenter also expresses negative sentiment about X, this is AGREEMENT (positive stance)."
   -10 to -1 = extremely opposed to slightly opposed via Disagreement, contradiction, refutation, challenging the parent's point.
   -10 to -8 = strong, explicit disagreement with emotional language, direct attacks, or ridicule of the parent comment's point.
   -7 to -4 = respectful but clear disagreement, pointing out flaws or counter-evidence without personal attacks.
   -3 to -1 = mild or implicit disagreement with defensive language(see defensive section below).
   0 = Neither agreeing nor disagreeing, tangential response.
   1 to 10 = slightly supportive to extremely supportive via Agreement, support, affirmation, building on the parent's point.
   1 to 3 = mild or implicit agreement, possibly with hedging or uncertainty.
   4 to 7 = clear agreement, directly affirming or building on the parent's point.
   8 to 10 = strong, explicit and complete agreement with emotional language, emphasizing the importance of the parent's point.
   N/A = If the comment does not express a clear stance toward the parent comment or post, or if there is not enough context in the comment or parent comment or post or if the stance is ambiguous/unclear.

topic -> stance -> rationale (string):
   Brief explanation (<=40 words) of the reasoning behind the assigned stance score, citing specific language

epistemic_risk -> claim_strength (number or N/A):
   If the text is an opinion or question, mark as N/A.
   Measures how much evidence the claim should require (0 = trivial/common knowledge → 10 = extraordinary claim).

epistemic_risk -> evidence_quality (number or N/A):
   If the text is an opinion or question, mark as N/A.
   Measures the strength of empirical support provided (citations, data, studies, verifiable facts).
   0 = no evidence provided → 10 = robust, well-sourced evidence.

epistemic_risk -> reasoning_depth (number or N/A):
   Measures the depth of logical reasoning, explanation, and argumentation.
   Does NOT require empirical evidence—can be high for well-reasoned philosophical arguments, detailed explanations, or nuanced analysis.
   0 = no reasoning  → 10 = sophisticated, multi-layered argumentation.

epistemic_risk -> rationale (string):
   Brief explanation (<= 40 words) of the reasoning behind the assigned scores or N/A markings.

toxicity (number or N/A):
   Hostile or aggressive language directed at the parent comment/post or its author (insults, slurs, dehumanization, or threats toward the parent context).
   toxicity language toward a third party does not count towards toxicity, only towards the parent comment/post.
   if no discernible toxicity is apperent, keep as 0.

discrediting (number or N/A):
   Language that undermines the legitimacy, intelligence, motives, or moral standing of the parent comment/post (or its author) instead of substantively engaging with its argument.
   discrediting language toward a third party does not count towards discrediting, only towards the parent comment/post.
   if no discernible discrediting is apperent, keep as 0.
   Example: "You're an idiot" = high toxicity, moderate discrediting  
   Example: "This is the kind of naive take you'd expect from someone who's never worked in the field" = low toxicity, high discrediting

defensiveness (number or N/A):
   Language that anticipates social punishment or conflict with the parent comment/post and strategically softens or shields expression (e.g. disclaimers, fear of backlash, self-distancing, overly polite language directed at the parent context).
   defensiveness language toward a third party does not count towards defensiveness, only towards the parent comment/post.
   if no discernible defensiveness is apperent, keep as 0.
   Examples: "I know I'll get downvoted for this but...", "Not trying to be 
   rude, but...", "I'm probably wrong, but..."

civility (number or N/A):
   Surface-level civility and respectful phrasing toward the parent comment/post or its author (general profanity not aimed at the parent does not reduce politeness).
   civility language toward a third party does not count towards civility, only towards the parent comment/post.
   if no discernible civility is apperent, keep neutral as 5.
   0 = extremely aggressive, rude, or impolite.
   5 = not aggressive, rude, or impolite, neutral tone.
   10 = exceedingly polite or respectful.

emotion -> anger, anxiety, disgust (number or N/A):
   Strength of expressed emotional tone in the comment overall, not necessarily directed at the parent.
   if no discernible emotion is apperent, keep as 0.

GROUP CONTEXT:

{{GROUP_CONTEXT}}

PRIOR CONTEXT:

{{PARENT_TEXT}}

COMMENT TO ANALYZE:

{{TEXT}}

Output (return ONLY the JSON object — no markdown fences, no reasoning, no explanation):

{
   "parent_topic": "{{PARENT_TOPIC}}",
   "comment_type": ["string"],
   
   "topic": {
      "label": "string",
      "stance": {
         "value": "number | N/A",
         "rationale": "string"
      }
   },
   "epistemic_risk": {
      "claim_strength": "number | N/A",
      "evidence_quality": "number | N/A",
      "reasoning_depth": "number | N/A",
      "rationale": "string | N/A"
   },
   "toxicity": "number | N/A",
   "discrediting": "number | N/A",
   "defensiveness": "number | N/A",
   "civility": "number | N/A",
   "emotion": {
      "anger": "number | N/A",
      "anxiety": "number | N/A",
      "disgust": "number | N/A"
   }

}"""

# Static (cacheable) portion of COMMENT_ANALYSIS_PROMPT — task description +
# all metric definitions. Safe to cache in the Anthropic system block because
# it is identical across every comment call in a run.
COMMENT_ANALYSIS_STATIC_INSTRUCTIONS = COMMENT_ANALYSIS_PROMPT.split("GROUP CONTEXT:")[0].rstrip()

# Dynamic (per-call) portion — GROUP CONTEXT, PRIOR CONTEXT, COMMENT, Output.
# This is injected as the user message so the cached system block stays clean.
COMMENT_ANALYSIS_DYNAMIC_TEMPLATE = "GROUP CONTEXT:" + COMMENT_ANALYSIS_PROMPT.split("GROUP CONTEXT:", 1)[1]


def _remove_rationale_fields(prompt: str) -> str:
    """Strip the two rationale field definitions and their output schema entries.

    Removes:
    - ``topic -> stance -> rationale`` + description from the instructions block
    - ``epistemic_risk -> rationale`` + description from the instructions block
    - ``"rationale": ...`` lines from the JSON output template
    """
    result = []
    lines = prompt.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Skip rationale definition blocks in the instructions section
        if stripped in (
            "topic -> stance -> rationale (string):",
            "epistemic_risk -> rationale (string):",
        ):
            i += 1  # skip the header line
            # skip the following indented description line(s)
            while i < len(lines) and lines[i].startswith("   "):
                i += 1
            continue
        # Skip rationale value lines in the JSON output schema
        if stripped.startswith('"rationale":'):
            i += 1
            continue
        result.append(line)
        i += 1
    return "\n".join(result)


# No-rationale variants: same structure, but without the two rationale fields.
# Use these when config has include_rationale: false to reduce output tokens by
# roughly 25-30 %.
COMMENT_ANALYSIS_PROMPT_NO_RATIONALE = _remove_rationale_fields(COMMENT_ANALYSIS_PROMPT)
COMMENT_ANALYSIS_STATIC_INSTRUCTIONS_NO_RATIONALE = (
    COMMENT_ANALYSIS_PROMPT_NO_RATIONALE.split("GROUP CONTEXT:")[0].rstrip()
)
COMMENT_ANALYSIS_DYNAMIC_TEMPLATE_NO_RATIONALE = (
    "GROUP CONTEXT:" + COMMENT_ANALYSIS_PROMPT_NO_RATIONALE.split("GROUP CONTEXT:", 1)[1]
)

# ============================================================================
# ANALYZE COMMENTS SCRIPT
# ============================================================================

ANALYZE_COMMENTS_SCRIPT = "python scripts/analyze_comments.py"

# ============================================================================
# MEDIA PREPROCESSING DEFAULTS
# ============================================================================

# User agents for HTTP requests
MEDIA_USER_AGENT = "ChamberCheck/1.0"
MEDIA_BROWSER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Retry settings for LLM API calls
MEDIA_MAX_RETRIES = 3
MEDIA_INITIAL_RETRY_DELAY = 1  # seconds, will exponentially backoff

# Content limits
MEDIA_MAX_LINK_CONTENT_LENGTH = 5000  # Max chars to extract from external links
MEDIA_MAX_SUMMARY_WORDS = 300  # Target length for summarization
MEDIA_FALLBACK_TEXT_LENGTH = 1000  # Truncated length if summarization fails

# Request timeout
MEDIA_REQUEST_TIMEOUT = 10  # seconds

# Vision analysis prompt
MEDIA_VISION_ANALYSIS_PROMPT = """Analyze this media and provide structured extraction following this exact JSON schema.

Return ONLY valid JSON (no markdown, no code blocks). Use null for missing fields.

{
  "description": "string - brief 1-2 sentence description of what the media shows",
  "text_content": "string - all visible text extracted from the media, preserving structure/line breaks",
  "media_type": "string - one of: screenshot, photograph, graphic, chart, diagram, collage, video_thumbnail, other",
  "platform": "string - source platform if identifiable (e.g. 'YouTube', 'Twitter/X', 'Reddit', 'unknown') or null",
  "topic": "string - high-level topic label summarizing the main subject matter (2-6 words) or null",
  "extracted_data": {
    "author": "string - credited author/creator if visible or null",
    "title": "string - title or headline if present or null",
    "date_posted": "string - date visible in media or null",
    "key_elements": ["array of strings - main subjects/topics/objects visible"],
    "credentials": ["array - for quote images, list credentials of speakers"],
    "metadata": {
      "dimensions": "string - approximate size/resolution if visible or null",
      "has_watermark": "boolean",
      "is_edited": "boolean - appears to be manipulated/composite or null"
    }
  }
}

Key Guidelines:
- For screenshots: extract all visible text and identify the platform/application
- For photographs: describe composition, subjects, and any text/captions
- For graphics/charts: identify data type, axis labels, key numbers
- For collages: list all component elements
- For video thumbnails: describe the visual and extract any overlay text
- Use "key_elements" to list main topics (e.g., 'medical professionals', 'quotes about science')
- Keep description concise; put full text in text_content
- Do not interpret or fact-check content, only extract what is visible
"""

# ---------------------------------------------------------------------------
# V3 Echo Chamber Metrics (echo_chamber_metrics_V3.docx)
# ---------------------------------------------------------------------------

# Stance threshold: comments with |stance| <= this value are excluded from all
# stance-based classifications (CSS, SBI, MSDG, RDB, uRDB, CSAD, TD).
V3_STANCE_THRESHOLD = 2

# Thread / topic majority guard: the majority group must be at least this
# fraction of the classified comments, otherwise the thread/topic is excluded
# from CSS and RDB estimates.
V3_MAJORITY_MIN_FRACTION = 0.60

# MSDG: minimum minority-group comments required per topic.
V3_MSDG_MIN_MINORITY_PER_TOPIC = 10

# MSDG: topics with SBI > this value are excluded (minority/majority
# classification unstable when groups are near-equal).
V3_MSDG_MAX_SBI = 0.4

# uRDB: minimum replies a user must have in a thread to be counted.
V3_URDB_MIN_REPLIES_PER_USER = 4

# EAS: minimum comments required per topic for a topic-level EAS computation.
V3_EAS_TOPIC_MIN_COMMENTS = 50

# EAS: bootstrap iterations for confidence intervals.
V3_EAS_BOOTSTRAP_ITERS = 1000

# Per-topic breakdown: minimum comments in a topic (across all stances) for the
# topic to be included in compute_all_by_subreddit_topic().
V3_TOPIC_MIN_COMMENTS = 30
