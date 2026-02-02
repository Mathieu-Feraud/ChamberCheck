"""
Test script for ChamberCheck using mock Reddit data.

This allows you to test the scraper infrastructure without Reddit API credentials.
"""

from datetime import datetime, timedelta
from chambercheck.models import Post, Comment
import json

def generate_mock_posts(subreddit: str, num_posts: int = 20):
    """Generate mock Reddit posts for testing."""
    posts = []
    base_time = datetime.now() - timedelta(days=30)
    
    topics = [
        ("Discussion about echo chambers in social media", "I've noticed that..."),
        ("Why this subreddit is becoming an echo chamber", "Let me explain..."),
        ("Diversity of opinions is important", "We need to consider..."),
        ("I disagree with the consensus here", "Unpopular opinion but..."),
        ("Another post supporting the majority view", "I completely agree that..."),
    ]
    
    for i in range(num_posts):
        topic_idx = i % len(topics)
        title, content_start = topics[topic_idx]
        
        post = Post(
            post_id=f"mock_{i}",
            platform="reddit",
            community=subreddit,
            title=f"{title} - Post {i+1}",
            content=f"{content_start} [mock content for testing]",
            author=f"user_{i % 10}",
            created_at=base_time + timedelta(days=i),
            upvotes=100 - (i * 3),
            downvotes=10 + i,
            num_comments=50 - i,
            url=f"https://reddit.com/r/{subreddit}/comments/mock_{i}",
            metadata={
                'upvote_ratio': 0.9 - (i * 0.01),
                'is_self': True,
                'flair': 'Discussion' if i % 2 == 0 else 'Meta'
            }
        )
        posts.append(post)
    
    return posts


def generate_mock_comments(post_ids: list, num_comments_per_post: int = 10):
    """Generate mock comments for testing."""
    comments = []
    
    comment_templates = [
        "I completely agree with this point.",
        "This is wrong and here's why...",
        "You clearly don't understand the issue.",
        "Great point, I never thought of it that way.",
        "This is just propaganda.",
        "Source?",
        "Can you elaborate on this?",
        "I respectfully disagree because...",
        "Stop being so naive.",
        "Interesting perspective, but consider..."
    ]
    
    for post_id in post_ids:
        for i in range(num_comments_per_post):
            comment = Comment(
                comment_id=f"c_{post_id}_{i}",
                post_id=post_id,
                platform="reddit",
                content=comment_templates[i % len(comment_templates)],
                author=f"commenter_{i % 15}",
                created_at=datetime.now() - timedelta(days=25, hours=i),
                upvotes=50 - i * 2,
                downvotes=5 + i if i > 5 else 0,
                parent_id=None if i < 5 else f"c_{post_id}_{i-1}",
                depth=0 if i < 5 else 1,
                metadata={
                    'is_submitter': i == 0,
                    'stickied': False
                }
            )
            comments.append(comment)
    
    return comments


def main():
    """Run mock scraping test."""
    print("=" * 60)
    print("ChamberCheck - Mock Reddit Scraper Test")
    print("=" * 60)
    
    # Generate mock data
    subreddit = "test_politics"
    print(f"\n📥 Generating mock data for r/{subreddit}...")
    
    posts = generate_mock_posts(subreddit, num_posts=20)
    print(f"✓ Generated {len(posts)} mock posts")
    
    # Get top 5 post IDs
    top_post_ids = [p.post_id for p in sorted(posts, key=lambda x: x.upvotes, reverse=True)[:5]]
    
    comments = generate_mock_comments(top_post_ids, num_comments_per_post=10)
    print(f"✓ Generated {len(comments)} mock comments")
    
    # Display statistics
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    print(f"\nSubreddit: r/{subreddit}")
    print(f"Total posts: {len(posts)}")
    print(f"Total comments: {len(comments)}")
    
    avg_upvotes = sum(p.upvotes for p in posts) / len(posts)
    avg_comments = sum(p.num_comments for p in posts) / len(posts)
    
    print(f"Average upvotes per post: {avg_upvotes:.1f}")
    print(f"Average comments per post: {avg_comments:.1f}")
    
    # Show top posts
    print("\n" + "-" * 60)
    print("TOP 5 POSTS BY UPVOTES")
    print("-" * 60)
    
    top_posts = sorted(posts, key=lambda x: x.upvotes, reverse=True)[:5]
    for i, post in enumerate(top_posts, 1):
        print(f"\n{i}. {post.title}")
        print(f"   Upvotes: {post.upvotes} | Comments: {post.num_comments}")
        print(f"   Author: {post.author}")
    
    # Show comment distribution
    print("\n" + "-" * 60)
    print("COMMENT ANALYSIS")
    print("-" * 60)
    
    top_level = sum(1 for c in comments if c.depth == 0)
    replies = sum(1 for c in comments if c.depth > 0)
    
    print(f"Top-level comments: {top_level}")
    print(f"Replies: {replies}")
    
    avg_comment_score = sum(c.upvotes for c in comments) / len(comments)
    print(f"Average comment upvotes: {avg_comment_score:.1f}")
    
    # Save to files
    print("\n" + "=" * 60)
    print("SAVING DATA")
    print("=" * 60)
    
    posts_data = [p.to_dict() for p in posts]
    comments_data = [c.to_dict() for c in comments]
    
    with open('data/raw/mock_posts.json', 'w') as f:
        json.dump(posts_data, f, indent=2, default=str)
    print("✓ Saved posts to: data/raw/mock_posts.json")
    
    with open('data/raw/mock_comments.json', 'w') as f:
        json.dump(comments_data, f, indent=2, default=str)
    print("✓ Saved comments to: data/raw/mock_comments.json")
    
    print("\n" + "=" * 60)
    print("✅ Mock scraping test completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the generated data in data/raw/")
    print("2. Test preprocessing pipeline with this data")
    print("3. Implement analysis and scoring modules")
    print("4. Set up Reddit API access when ready")


if __name__ == '__main__':
    main()
