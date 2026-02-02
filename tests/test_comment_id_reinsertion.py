#!/usr/bin/env python3
"""Test that comment_id is added back to results after LLM analysis."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ChamberCheck.analysis.comment_analyzer import CommentAnalyzer
from ChamberCheck.analysis.llm_provider import LLMProvider

class MockProviderWithResponse(LLMProvider):
    """Mock provider that returns proper LLM response."""
    def __init__(self):
        super().__init__("mock-key")
    
    def analyze_comment(self, prompt):
        """Return a mock LLM response (note: no comment_id field)."""
        return {
            "comment_types": ["argumentative"],
            "argument_narrowness": 7,
            "hostility": 3,
            "suppression": 5,
            "epistemic_closure": 6,
            "argument_avoidance": "N/A",
            "echo_chamber_score": 21,
            "reasoning": "Shows tunnel vision on the issue"
        }

def test_comment_id_reinsertion():
    """Verify that comment_id is added back to the result after LLM analysis."""
    
    print("=" * 80)
    print("TESTING: Comment ID Reinsertion After LLM Analysis")
    print("=" * 80)
    
    # Create a sample comment
    sample_comment = {
        "comment_id": "xyz789abc",  # This should NOT be in the prompt, but SHOULD be in the result
        "author": "test_user",
        "content": "This is a test comment.",
        "depth": 1,
        "metadata": {
            "score": 42,
            "created_utc": 1234567890
        }
    }
    
    # Create analyzer
    provider = MockProviderWithResponse()
    analyzer = CommentAnalyzer(provider, "philosophy")
    
    # Analyze comment
    result = analyzer.analyze(sample_comment)
    
    print("\nLLM Result (before comment_id reinsertion):")
    llm_response = {k: v for k, v in result.items() if k != 'comment_id'}
    print(json.dumps(llm_response, indent=2))
    
    print("\n" + "-" * 80)
    
    # Check that comment_id is in the result
    if 'comment_id' in result:
        print(f"✓ Comment ID is in the result: '{result['comment_id']}'")
        if result['comment_id'] == sample_comment['comment_id']:
            print(f"✓ Comment ID matches original: '{sample_comment['comment_id']}'")
        else:
            print(f"❌ Comment ID mismatch! Expected '{sample_comment['comment_id']}', got '{result['comment_id']}'")
            return False
    else:
        print("❌ FAILED: Comment ID not in result!")
        return False
    
    # Verify metrics are still there
    metrics = ["argument_narrowness", "hostility", "suppression", "epistemic_closure", "echo_chamber_score"]
    all_present = True
    for metric in metrics:
        if metric in result:
            print(f"✓ Metric '{metric}' present in result")
        else:
            print(f"❌ Metric '{metric}' missing!")
            all_present = False
    
    print("\n" + "=" * 80)
    if all_present:
        print("✅ ALL CHECKS PASSED")
        print("Comment ID is correctly added back to results after LLM analysis.")
        return True
    else:
        return False

if __name__ == "__main__":
    success = test_comment_id_reinsertion()
    sys.exit(0 if success else 1)
