"""
Data model for analysis results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class AnalysisResult:
    """
    Represents the results of discourse analysis for a community.
    
    Attributes:
        community: Community identifier
        platform: Platform name
        analysis_date: When the analysis was performed
        time_range_start: Start of analyzed time period
        time_range_end: End of analyzed time period
        num_posts: Number of posts analyzed
        num_comments: Number of comments analyzed
        base_metrics: Dictionary of base metric scores
        composite_scores: Dictionary of composite scores
        echo_chamber_score: Final echo chamber score
        topic_scores: Topic-specific scores (if applicable)
        metadata: Additional analysis metadata
    """
    
    community: str
    platform: str
    analysis_date: datetime
    time_range_start: datetime
    time_range_end: datetime
    num_posts: int
    num_comments: int
    base_metrics: Dict[str, float] = field(default_factory=dict)
    composite_scores: Dict[str, float] = field(default_factory=dict)
    echo_chamber_score: Optional[float] = None
    topic_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to dictionary."""
        return {
            'community': self.community,
            'platform': self.platform,
            'analysis_date': self.analysis_date.isoformat(),
            'time_range_start': self.time_range_start.isoformat(),
            'time_range_end': self.time_range_end.isoformat(),
            'num_posts': self.num_posts,
            'num_comments': self.num_comments,
            'base_metrics': self.base_metrics,
            'composite_scores': self.composite_scores,
            'echo_chamber_score': self.echo_chamber_score,
            'topic_scores': self.topic_scores,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create AnalysisResult from dictionary."""
        data['analysis_date'] = datetime.fromisoformat(data['analysis_date'])
        data['time_range_start'] = datetime.fromisoformat(data['time_range_start'])
        data['time_range_end'] = datetime.fromisoformat(data['time_range_end'])
        return cls(**data)
    
    def add_base_metric(self, metric_name: str, score: float):
        """Add a base metric score."""
        self.base_metrics[metric_name] = score
    
    def add_composite_score(self, score_name: str, score: float):
        """Add a composite score."""
        self.composite_scores[score_name] = score
    
    def add_topic_score(self, topic: str, scores: Dict[str, float]):
        """Add topic-specific scores."""
        self.topic_scores[topic] = scores
    
    def get_summary(self) -> str:
        """Get human-readable summary of results."""
        summary = [
            f"Analysis for {self.community} ({self.platform})",
            f"Period: {self.time_range_start.date()} to {self.time_range_end.date()}",
            f"Posts analyzed: {self.num_posts}",
            f"Comments analyzed: {self.num_comments}",
            "",
            "Base Metrics:"
        ]
        
        for metric, score in self.base_metrics.items():
            summary.append(f"  {metric}: {score:.2f}")
        
        summary.append("\nComposite Scores:")
        for score_name, score in self.composite_scores.items():
            summary.append(f"  {score_name}: {score:.2f}")
        
        if self.echo_chamber_score is not None:
            summary.append(f"\nEcho Chamber Score: {self.echo_chamber_score:.2f}")
        
        return "\n".join(summary)
