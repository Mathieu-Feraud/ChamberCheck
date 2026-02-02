"""
Example: Basic Reddit Scraping

This notebook demonstrates how to use ChamberCheck to scrape data from Reddit.
"""

# %% [markdown]
# # ChamberCheck: Reddit Scraping Example
# 
# This notebook shows how to:
# 1. Set up authentication
# 2. Scrape posts from a subreddit
# 3. Fetch comments
# 4. Save data for analysis

# %% [markdown]
# ## Setup

# %%
import sys
sys.path.append('../src')

from datetime import datetime, timedelta
from chambercheck import Config
from chambercheck.scrapers import RedditScraper
import pandas as pd

# %% [markdown]
# ## Configure Credentials
# 
# Make sure you have set up your `.env` file or `config.json` with Reddit credentials.

# %%
# Load configuration
config = Config()

# Check if Reddit config is valid
if config.validate_scraper_config('reddit'):
    print("✓ Reddit configuration is valid")
else:
    print("✗ Reddit configuration is missing or invalid")
    print("Please set up your credentials in .env or config.json")

# %% [markdown]
# ## Initialize Scraper

# %%
# Create scraper instance
scraper = RedditScraper(config.get_scraper_config('reddit'))

# Authenticate
if scraper.authenticate():
    print("✓ Successfully authenticated with Reddit")
else:
    print("✗ Authentication failed")

# %% [markdown]
# ## Fetch Posts
# 
# Let's fetch the top posts from a subreddit over the last 30 days.

# %%
# Define time range
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Fetch posts
subreddit = 'politics'  # Change to your target subreddit
posts = scraper.fetch_posts_by_engagement(
    community=subreddit,
    start_date=start_date,
    end_date=end_date,
    sort_by='top',
    time_filter='month',
    limit=50
)

print(f"Fetched {len(posts)} posts from r/{subreddit}")

# %% [markdown]
# ## Explore the Data

# %%
# Convert to DataFrame for easier analysis
posts_df = pd.DataFrame([p.to_dict() for p in posts])
posts_df.head()

# %%
# Basic statistics
print(f"Total posts: {len(posts_df)}")
print(f"Date range: {posts_df['created_at'].min()} to {posts_df['created_at'].max()}")
print(f"Average upvotes: {posts_df['upvotes'].mean():.1f}")
print(f"Average comments: {posts_df['num_comments'].mean():.1f}")

# %%
# Top 5 posts by engagement
top_posts = posts_df.nlargest(5, 'upvotes')[['title', 'upvotes', 'num_comments']]
print("Top 5 posts by upvotes:")
print(top_posts)

# %% [markdown]
# ## Fetch Comments
# 
# Now let's fetch comments for the top posts.

# %%
# Get IDs of top 5 posts
top_post_ids = posts_df.nlargest(5, 'upvotes')['post_id'].tolist()

# Fetch comments
comments = scraper.fetch_comments(top_post_ids, limit=100)
print(f"Fetched {len(comments)} comments")

# %%
# Convert comments to DataFrame
comments_df = pd.DataFrame([c.to_dict() for c in comments])
comments_df.head()

# %%
# Comment statistics
print(f"Total comments: {len(comments_df)}")
print(f"Average upvotes per comment: {comments_df['upvotes'].mean():.1f}")
print(f"Top-level comments: {comments_df['depth'].eq(0).sum()}")
print(f"Replies: {comments_df['depth'].gt(0).sum()}")

# %% [markdown]
# ## Save Data
# 
# Save the scraped data for later analysis.

# %%
# Save to CSV
posts_df.to_csv('../data/raw/posts.csv', index=False)
comments_df.to_csv('../data/raw/comments.csv', index=False)

print("✓ Data saved to data/raw/")

# %% [markdown]
# ## Next Steps
# 
# - Preprocess the text data
# - Run topic modeling to categorize discussions
# - Analyze sentiment and hostility
# - Compute echo chamber metrics
