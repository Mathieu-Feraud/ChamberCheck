"""
Scoring modules for computing echo chamber metrics.

NOTE: Actual metric scoring is done by the LLM (see ANALYSIS_INSTRUCTIONS_PROMPT in constants.py).
The LLM analyzes comments and returns:
  - argument_narrowness (0-10): Ignores other perspectives vs. engages multiple viewpoints
  - hostility (0-10): Aggressive/insulting vs. respectful/civil
  - suppression (0-10): Discourages debate vs. invites counter-arguments
  - epistemic_closure (0-10): Rejects evidence vs. open to updating based on evidence
  - argument_avoidance (0-10, replies only): Ignores parent points vs. directly engages

Echo chamber score is the sum of numeric metric values (0-50 max).
Non-applicable metrics (marked "N/A") are excluded from the sum.
"""

__all__ = []
