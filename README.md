# ChamberCheck

**An LLM-powered tool for analyzing echo chamber dynamics in online communities**

ChamberCheck quantifies discourse patterns in online communities (Reddit, Facebook, etc.) by measuring argument diversity, hostility, deviation suppression, and epistemic openness. The tool provides both individual metrics and a composite "Echo Chamber Score" to help researchers and community moderators understand discourse health.

## Installation

### Prerequisites

- Python 3.9 or higher
- Reddit API credentials (for Reddit scraping)
- LLM API key (OpenAI and/or Anthropic)

### User Guide

1. Install the package:
```bash
pip install chambercheck
```

2. **Configure API credentials:**

#### Option A: Environment Variables (Recommended)

Set environment variables before running ChamberCheck:

```powershell
# Windows PowerShell
$env:REDDIT_CLIENT_ID = "your_reddit_client_id"
$env:REDDIT_CLIENT_SECRET = "your_reddit_client_secret"
$env:REDDIT_USER_AGENT = "ChamberCheck/0.1"
$env:OPENAI_API_KEY = "your_openai_api_key"
$env:ANTHROPIC_API_KEY = "your_anthropic_api_key"
```

```bash
# Linux/Mac
export REDDIT_CLIENT_ID="your_reddit_client_id"
export REDDIT_CLIENT_SECRET="your_reddit_client_secret"
export REDDIT_USER_AGENT="ChamberCheck/0.1"
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

#### Option B: Local `.env` File (Development Only)

Create a `.env` file in the project root (already `.gitignore`d):

```env
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=ChamberCheck/0.1
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

#### Option C: User-Level Config (Reusable Across Projects)

Create `~/.chambercheck/config.yaml`:

```yaml
reddit:
  client_id: your_reddit_client_id
  client_secret: your_reddit_client_secret
  user_agent: ChamberCheck/0.1
post_analysis:
  provider: openai  # or anthropic
  model: gpt-4o    # or claude-3-5-sonnet-20241022
comment_analysis:
  provider: anthropic
  model: claude-haiku-4-5-20251001
```

## Documentation

For deeper technical documentation:

- [ARCHITECTURE_RULES.md](ARCHITECTURE_RULES.md) — Coding standards and project conventions
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — Detailed module documentation
- [PYPI_PUBLISH.md](PYPI_PUBLISH.md) — Package release and publishing workflow
- [ChamberCheck_Paper_v4.pdf](ChamberCheck_Paper_v4.pdf) — Paper draft containing the full supplementary material

## Data Access

Processed datasets from published analyses are available for download:

- [Processed Data (OneDrive)](https://1drv.ms/f/c/d5c2c787ee7f6d0f/IgDoJzL5o2AnRIRQsc02Q3UCAdBoIiIG4eM62kPH4NgA1w0?e=urhaDL) — Pre-analyzed Reddit data, metrics, and visualizations

These datasets are optional and not required to use ChamberCheck, but they are useful for:  
- Reproducing published findings
- Understanding output formats from the 8-stage pipeline
- Learning by example before running your own analyses

## Usage: The 8-Stage Pipeline

ChamberCheck implements a modular 8-stage pipeline for analyzing discourse:

### Overview

| Stage | Name | Input | Output | Module |
|-------|------|-------|--------|--------|
| 1 | **Scrape Posts** | Subreddit config | `posts.json` | `scrapers.batch_scrape_posts_only()` |
| 2 | **Analyse Post Titles** | Posts | `analysis_NNN.json` | `analysis.analyze_posts()` |
| 3 | **Preprocess Posts** | Posts + Analysis | `pre_process_NNN.json` | `preprocessing.preprocess_posts()` |
| 4 | **Scrape Comments** | Filtered posts | `comments_*.json` | `scrapers.scrape_comments()` |
| 5 | **Preprocess Comments** | Comments | `comments_filtered_*.json` | `preprocessing.preprocess_comments()` |
| 6 | **Analyse Comments** | Filtered comments | `comment_analysis_*.json` | `analysis.run_comment_analysis()` |
| 7 | **Compute Metrics** | Analyzed comments | `v3_metrics_*.json` | `CC_derived_metrics.V3Metrics` |
| 8 | **Visualize** | Metrics | PNG plots | Ad-hoc plotting scripts |

### Running the Full Pipeline

The simplest way to run all 8 stages:

```bash
python test_scripts/workflow.py
```

Edit `test_scripts/workflow.py` to configure:
- `CONFIG` — path to YAML config file (default: `config/config.test.yaml`)
- `SCRAPE_DIR` — skip scraping by pointing to existing data folder
- Comment/uncomment stages to run selectively

### Running Individual Stages

#### Stage 1: Scrape Posts
```bash
python test_scripts/run_scraper_posts.py
```

#### Stage 2: Analyse Post Titles
```bash
python test_scripts/run_analyze_posts.py
```

#### Stage 3: Preprocess Posts
```bash
python test_scripts/run_preprocess_posts.py
```

#### Stage 4: Scrape Comments
```bash
python test_scripts/run_scrape_comments.py
```

#### Stage 5: Preprocess Comments
```bash
python test_scripts/run_preprocess_comments.py
```

#### Stage 6: Analyse Comments
```bash
python test_scripts/run_analyze_comments.py
```

#### Stage 7: Compute Metrics
```bash
python test_scripts/run_v3_metrics.py
```

#### Stage 8: Generate Plots
```bash
python test_scripts/ad-hoc/plot_v3_metrics.py
```

### Configuration

All pipeline parameters are in YAML config files under `config/`:

```yaml
scraping:
  subreddits: ["politics", "atheism", "philosophy"]  # Which communities to scrape
  num_posts: 100                                       # Posts per subreddit
  sort_method: "top"                                  # Sorting: top, new, hot
  time_filter: "all"                                  # Time range: all, year, month, week, day

post_analysis:
  provider: "anthropic"                              # LLM: anthropic or openai
  model: "claude-3-haiku-20240307"                   # Model name
  temperature: 0.1                                   # Creativity: 0=deterministic, 1=random
  max_tokens: 500                                    # Output token limit

preprocessing:
  min_comments: 10                                   # Posts must have ≥10 comments
  top_n_per_subreddit: 30                            # Select top-30 posts per community
  min_discussion_score: 0.6                          # Discussion quality threshold
  min_topic_peers: 1                                 # Minimum peer comments for topic validation

comment_scraping:
  max_comments_per_post: 500                         # Max comments to fetch per post
```

### Example: Custom Analysis

```python
from chambercheck import Config
from chambercheck.CC_derived_metrics import V3Metrics
import json

# Load configuration
config = Config(config_path="config/config.test.yaml")

# Load computed metrics
metrics = V3Metrics.from_files(
    metric_file="data/output/scrape_001/v3_metrics_001.json"
)

# Inspect echo chamber score per subreddit
for subreddit, scores in metrics.by_subreddit.items():
    print(f"{subreddit}: Echo Argument Score = {scores['echo_argument_score']:.3f}")
```

For detailed methodology and metric definitions, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) and [ChamberCheck_Paper_v4.pdf](ChamberCheck_Paper_v4.pdf).

### A/B/n Testing Different LLM Models

ChamberCheck supports A/B/n testing to compare how different LLM models or configurations analyze the same comments. This is useful for evaluating model performance on discourse analysis tasks.

#### Workflow

1. **Generate A/B/n test set**: Select 50 representative comments with their reply chains from scraped data
2. **Export prompts**: Create consistent prompt files for comparison
3. **Run multiple analyses**: Test different models (e.g., gpt-4o vs claude-3.5-sonnet) on the same prompts
4. **Compare results**: Analyze differences in metric scores across models

#### Step 1: Generate A/B/n Test Set

```python
from chambercheck.analysis import generate_abn_test_set

result = generate_abn_test_set(
    raw_folder_path="data/raw/scrape_001",
    num_comments=50,  # Target total comments (including replies)
    random_seed=42,   # Optional: for reproducibility
)

# Result includes:
# - prompt_file: abn_test_prompts_001.txt
# - metadata_file: abn_test_prompts_metadata_001.json
# - random_seed: seed used for reproducibility
```

Or use the test script:
```bash
python test_scripts/run_abn_test_builder.py
```

#### Understanding the A/B/n Test Structure

**Comment Selection Algorithm:**
- Randomly selects ~50 top-level comments from the specified post
- For each selected comment, follows the reply chain by taking the most upvoted reply at each level
- Creates chains until no more replies exist (no branching, just linear chains)
- Stores comment IDs and chain structure in metadata

**Output Files:**

1. **abn_test_prompts_001.txt** - Raw text file containing all prompts exactly as LLMs will see them
   - Each prompt separated by `=` line
   - Shows parent comment context followed by the target comment
   
2. **abn_test_prompts_metadata_001.json** - Metadata file containing:
   - `random_seed`: Seed used for this run (for reproducibility)
   - `selected_comment_ids`: List of all 50 comment IDs in order
   - `comment_positions`: Position of each comment in its chain (0 = top-level)
   - `source_file`: Original raw data file
   - `chains_breakdown`: How many comments per chain

#### Step 2: Run Analysis on A/B Test Set

After generating the A/B test set with model A, switch models and run the same analysis:

```python
from chambercheck.analysis import batch_analyze_comments

# Run analysis with model A (e.g., gpt-4o)
metadata_a = batch_analyze_comments(
    comment_files=["path/to/extracted/from/abn_test.json"],
    limit=50,
    model_name="gpt-4o",
)
# Output: data/output/abn_test/abn_test_analysis_001_001.json

# Switch to model B and run again
metadata_b = batch_analyze_comments(
    comment_files=["path/to/extracted/from/abn_test.json"],
    limit=50,
    model_name="claude-3-5-sonnet-20241022",
)
# Output: data/output/abn_test/abn_test_analysis_001_002.json
```

#### File Naming Convention

- **Prompts**: `abn_test_prompts_XXX.txt` + `abn_test_prompts_metadata_XXX.json`
  - `XXX`: Prompt set number (001, 002, etc.)
  
- **Analysis**: `abn_test_analysis_XXX_YYY.json` + `abn_test_analysis_metadata_XXX_YYY.json`
  - `XXX`: Prompt set number (which prompts were used)
  - `YYY`: Analysis run number for that prompt set (001, 002, etc.)

Example: `abn_test_analysis_001_002.json` = Second analysis run using prompt set 001

#### Comparison Example

```python
import json

# Load results from different models
with open("data/output/abn_test/abn_test_analysis_001_001.json") as f:
    results_gpt4o = json.load(f)

with open("data/output/abn_test/abn_test_analysis_001_002.json") as f:
    results_claude = json.load(f)

# Compare average scores
for comment_gpt4, comment_claude in zip(results_gpt4o, results_claude):
    gpt_stance = comment_gpt4["topic"]["stance"]["value"]
    claude_stance = comment_claude["topic"]["stance"]["value"]
    
    if gpt_stance != claude_stance:
        print(f"Stance difference for {comment_gpt4['comment_id']}: "
              f"GPT4o={gpt_stance}, Claude={claude_stance}")
```


## Methodology

ChamberCheck decomposes echo chambers into measurable dimensions:

### Base Metrics
- **Argument Diversity**: Semantic clustering and novelty of arguments
- **Deviation Suppression**: Treatment of dissenting viewpoints
- **Hostility**: Personal attacks and ad hominem density
- **Epistemic Openness**: How evidence and external sources are handled
- **Identity Policing**: Boundary enforcement based on group membership

### Composite Scores
- **Civility Score**: Combines hostility and ad hominem metrics
- **Epistemic Health Score**: Measures openness to evidence
- **Discourse Pluralism Score**: Evaluates argument diversity and tolerance

### Echo Chamber Score
Weighted composite of low pluralism, high suppression, high hostility, and low epistemic openness.

## Research & Citations

This tool is based on academic research in social epistemology, political psychology, and computational social science. Key concepts include:

- Echo chamber operationalization (Cota et al., 2019)
- Embedding-based discourse metrics (Alatawi et al., 2023)
- Hostility and intergroup dynamics (Efstratiou et al., 2022)
- Topic-conditional analysis (various sources)

## Ethics & Limitations

- **Privacy**: User identities are anonymized; no longitudinal tracking
- **Bias**: LLM-based metrics require validation against human annotations
- **Interpretation**: Scores reflect observable discourse, not internal beliefs
- **Comparison**: Results should be interpreted comparatively, not as absolute judgments


## Project Structure

```
ChamberCheck/
├── src/
│   └── ChamberCheck/
│       ├── scrapers/                # Platform-specific data collection
│       ├── analysis/                # LLM-powered discourse analysis
│       ├── preprocessing/           # Data cleaning and filtering
│       ├── CC_derived_metrics/      # Echo chamber metrics computation
│       ├── model_analysis/          # A/B/n testing framework
│       ├── reporting/               # Report generation
│       ├── models/                  # Data models (Post, Comment, etc.)
│       ├── scoring/                 # Metric aggregation
│       ├── utils/                   # Utilities and logging
│       ├── config.py                # Configuration management
│       └── constants.py             # Project-wide constants
├── config/                          # YAML configuration files
│   ├── config.yaml                  # Production config
│   ├── config.test.yaml             # Test/dev config
│   └── config.intellectual.yaml     # Custom config template
├── data/
│   ├── raw/                         # Raw scraped data
│   ├── processed/                   # Cleaned data
│   └── output/                      # Analysis results and plots
├── test_scripts/                    # Pipeline execution scripts
│   ├── workflow.py                  # Full 8-stage pipeline runner
│   ├── run_*.py                     # Individual stage executors
│   └── ad-hoc/                      # Exploratory and diagnostic scripts
├── tests/                           # Unit and integration tests
├── pyproject.toml                   # Package metadata and dependencies
├── ARCHITECTURE_RULES.md            # Coding guidelines
└── README.md                        # This file
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests (run `pytest` before submitting)
4. Follow code style: `black src/` (line length 100)
5. Submit a pull request

See [ARCHITECTURE_RULES.md](ARCHITECTURE_RULES.md) for detailed guidelines.

## License

MIT License - see LICENSE file for details

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

---

**Note**: This project is research-grade software under active development. API signatures and data formats may change between releases.

**License**: MIT — See [LICENSE](LICENSE) file for details.
