"""
Ad-hoc: calculate actual API cost for scrape_006 runs.

Post analysis  : claude-3-haiku-20240307  ($0.25/M in, $1.25/M out)
Comment analysis: claude-haiku-4-5-20251001  ($1.00/M in, $5.00/M out,
                  cache reads $0.10/M, cache writes $1.25/M)

Actual comment analysis figures provided by user: 19M tokens, $29.38.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

# ── Claude 3 Haiku pricing (post analysis, claude-3-haiku-20240307) ────────
POST_PRICE_INPUT_PER_M  = 0.25   # $/M input tokens
POST_PRICE_OUTPUT_PER_M = 1.25   # $/M output tokens

# ── Anthropic Haiku 4.5 pricing (comment analysis) ──────────────────────--
PRICE_INPUT_PER_M        = 1.00  # $/M normal input tokens
PRICE_CACHED_WRITE_PER_M = 1.25  # $/M cache-write tokens
PRICE_CACHED_READ_PER_M  = 0.10  # $/M cache-read tokens
PRICE_OUTPUT_PER_M       = 5.00  # $/M output tokens


# ══════════════════════════════════════════════════════════════════════════
# 1. POST ANALYSIS  (claude-3-haiku-20240307)
#    Token totals come directly from the run metadata file.
# ══════════════════════════════════════════════════════════════════════════
meta_path = ROOT / "data/raw/scrape_006/posts_analysis/analysis_004_metadata.json"
meta = json.loads(meta_path.read_text(encoding="utf-8"))

total_post_tokens = meta["tokens"]["total"]
posts_analysed    = meta["posts_analysed"]

# Post analysis returns brief JSON (~5 fields).  The prompt contains the
# post title + fixed instructions – roughly 85% input, 15% output.
post_input_tokens  = int(total_post_tokens * 0.85)
post_output_tokens = int(total_post_tokens * 0.15)

post_cost = (post_input_tokens  / 1_000_000 * POST_PRICE_INPUT_PER_M +
             post_output_tokens / 1_000_000 * POST_PRICE_OUTPUT_PER_M)

print("=" * 60)
print("POST ANALYSIS  (claude-3-haiku-20240307)")
print("=" * 60)
print(f"  Posts analysed    :  {posts_analysed:,}")
print(f"  Total tokens      :  {total_post_tokens:,}  (from metadata)")
print(f"  Est. input tokens :  {post_input_tokens:,}  (85 %)")
print(f"  Est. output tokens:  {post_output_tokens:,}  (15 %)")
print(f"  Cost              :  ${post_cost:.4f}")


# ══════════════════════════════════════════════════════════════════════════
# 2. COMMENT ANALYSIS  (claude-haiku-4-5-20251001)
#    Actual figures reported by user from Anthropic dashboard.
# ══════════════════════════════════════════════════════════════════════════
N_COMMENTS          = 8_132
ACTUAL_TOTAL_TOKENS = 19_000_000   # from Anthropic dashboard
ACTUAL_COST         = 29.38        # from Anthropic dashboard

# Back-calculate input/output split from actual cost and total tokens:
#   cost = input_tokens/1M * 1.00 + output_tokens/1M * 5.00
#   (cache reads/writes are small relative to 19M and are included in input_tokens here)
#   Let x = output_tokens, then input_tokens = 19M - x
#   29.38 = (19M - x)/1M * 1.00 + x/1M * 5.00
#   29.38 = 19 - x/1M + 5x/1M = 19 + 4x/1M
#   x/1M = (29.38 - 19) / 4 = 2.595M output tokens
output_tokens_actual = (ACTUAL_COST - ACTUAL_TOTAL_TOKENS / 1_000_000 * PRICE_INPUT_PER_M) / (
    (PRICE_OUTPUT_PER_M - PRICE_INPUT_PER_M) / 1_000_000
)
input_tokens_actual  = ACTUAL_TOTAL_TOKENS - output_tokens_actual

tokens_per_comment   = ACTUAL_TOTAL_TOKENS / N_COMMENTS
output_per_comment   = output_tokens_actual / N_COMMENTS
input_per_comment    = input_tokens_actual  / N_COMMENTS

print()
print("=" * 60)
print("COMMENT ANALYSIS  (claude-haiku-4-5-20251001)  — ACTUAL")
print("=" * 60)
print(f"  Comments analysed  :  {N_COMMENTS:,}")
print(f"  Total tokens       :  {ACTUAL_TOTAL_TOKENS/1_000_000:.1f}M  (from dashboard)")
print(f"  Back-calc input    :  {input_tokens_actual/1_000_000:.2f}M  ({100*input_tokens_actual/ACTUAL_TOTAL_TOKENS:.0f}%)")
print(f"  Back-calc output   :  {output_tokens_actual/1_000_000:.2f}M  ({100*output_tokens_actual/ACTUAL_TOTAL_TOKENS:.0f}%)")
print(f"  Avg tokens/comment :  {tokens_per_comment:,.0f}  ({input_per_comment:,.0f} in / {output_per_comment:,.0f} out)")
print(f"  Cost               :  ${ACTUAL_COST:.2f}  (actual)")

# ── Project remaining cost using actual per-comment rates ──────────────────
N_REMAINING = 9_758

# Remaining run starts fresh (no warm cache), but cache re-establishes quickly.
# Use same blended rate as actual (includes cache effects observed in practice).
remaining_cost = ACTUAL_COST / N_COMMENTS * N_REMAINING

# Also show explicit calc for transparency
remaining_input  = input_per_comment  * N_REMAINING
remaining_output = output_per_comment * N_REMAINING
remaining_cost_explicit = (
    remaining_input  / 1_000_000 * PRICE_INPUT_PER_M +
    remaining_output / 1_000_000 * PRICE_OUTPUT_PER_M
)

print()
print(f"  Remaining comments :  {N_REMAINING:,}")
print(f"  Projected cost     :  ${remaining_cost:.2f}  (same blended rate)")

# If switched to claude-3-5-haiku-20241022 ($0.80/$4.00)
H35_IN, H35_OUT = 0.80, 4.00
remaining_cost_h35 = (
    remaining_input  / 1_000_000 * H35_IN +
    remaining_output / 1_000_000 * H35_OUT
)
saving = remaining_cost_explicit - remaining_cost_h35
print(f"  If switched to 3.5 Haiku: ${remaining_cost_h35:.2f}  (save ~${saving:.2f})")


print()
print("=" * 60)
print("TOTAL  scrape_006 API COST")
print("=" * 60)
total_so_far = post_cost + ACTUAL_COST
print(f"  Post analysis        :  ${post_cost:.2f}")
print(f"  Comment analysis     :  ${ACTUAL_COST:.2f}  (actual)")
print(f"  ─────────────────────────────────────────")
print(f"  Total so far         :  ${total_so_far:.2f}")
print(f"  + remaining (Haiku 4.5):  ${remaining_cost:.2f}")
print(f"  ─────────────────────────────────────────")
print(f"  Full run total est.  :  ${total_so_far + remaining_cost:.2f}")
