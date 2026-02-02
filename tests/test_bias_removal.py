#!/usr/bin/env python3
"""Quick test to verify comment_id and score are removed from prompts."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ChamberCheck.analysis.comment_analyzer import CommentAnalyzer
from ChamberCheck.analysis.llm_provider import LLMProvider

class MockProvider(LLMProvider):
    """Mock provider that doesn't require API key."""
    def __init__(self):
        super().__init__("mock-key")
    
    def analyze_comment(self, prompt):
        return {
            "argument_narrowness": 5,
            "echo_chamber_score": 20
        }

def test_prompt_removal():
    """Verify that comment_id and score are NOT in the prompt sent to LLM."""
    
    # Create a sample comment
    sample_comment = {
        "comment_id": "abc123xyz",
        "author": "test_user",
        "content": "This is a test comment about the topic.",
        "depth": 1,
        "metadata": {
            "score": 42,  # Should NOT appear in prompt
            "created_utc": 1234567890
        }
    }
    
    # Create analyzer with mock provider
    provider = MockProvider()
    analyzer = CommentAnalyzer(provider, "philosophy")
    
    # Build prompt
    prompt = analyzer.build_prompt(sample_comment)
    
    print("=" * 80)
    print("TESTING: Comment ID and Score Removal from Prompts")
    print("=" * 80)
    
    # Check that problematic fields are NOT in the prompt
    issues = []
    
    if "abc123xyz" in prompt:
        issues.append("❌ FAILED: Comment ID 'abc123xyz' found in prompt!")
    else:
        print("✓ Comment ID correctly removed from prompt")
    
    if "Comment ID:" in prompt:
        issues.append("❌ FAILED: 'Comment ID:' label found in prompt!")
    else:
        print("✓ 'Comment ID:' label not in prompt")
    
    if "Comment score:" in prompt:
        issues.append("❌ FAILED: 'Comment score:' label found in prompt!")
    else:
        print("✓ 'Comment score:' label removed")
    
    if "42" in prompt:
        issues.append("⚠️  WARNING: Number '42' found in prompt (could be score or coincidence)")
    
    # Verify comment text is still there
    if "This is a test comment about the topic." in prompt:
        print("✓ Comment text is included")
    else:
        issues.append("❌ FAILED: Comment text missing from prompt!")
    
    # Verify JSON template doesn't have comment_id
    if '"comment_id"' in prompt:
        issues.append("❌ FAILED: 'comment_id' field in JSON template!")
    else:
        print("✓ 'comment_id' removed from JSON template")
    
    # Verify metrics are in template
    metrics = ["argument_narrowness", "hostility", "suppression", "epistemic_closure", "argument_avoidance"]
    for metric in metrics:
        if f'"{metric}"' in prompt:
            print(f"✓ Metric '{metric}' in template")
        else:
            issues.append(f"❌ FAILED: Metric '{metric}' missing!")
    
    print("\n" + "=" * 80)
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ ALL CHECKS PASSED")
        print("Comment ID and score are correctly removed from LLM prompts.")
        print("Comment ID will be added back programmatically after LLM analysis.")
        return True

if __name__ == "__main__":
    success = test_prompt_removal()
    sys.exit(0 if success else 1)
