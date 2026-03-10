# ChamberCheck

**An LLM-powered tool for analyzing echo chamber dynamics in online communities**

ChamberCheck quantifies discourse patterns in online communities (Reddit, Facebook, etc.) by measuring argument diversity, hostility, deviation suppression, and epistemic openness. The tool provides both individual metrics and a composite "Echo Chamber Score" to help researchers and community moderators understand discourse health.

## Features

- 🔍 **Multi-platform scraping** - Modular architecture supports Reddit, Facebook, and other platforms
- 📊 **Comprehensive metrics** - Measures argument diversity, hostility, suppression, and epistemic closure
- 🧠 **LLM-powered analysis** - Uses large language models for nuanced discourse understanding
- 📈 **Topic-conditional scoring** - Evaluates echo chamber behavior across different subject domains
- 🎯 **Research-grade methodology** - Based on social epistemology and political psychology literature

## Project Structure

```
ChamberCheck/
├── src/
│   └── chambercheck/
│       ├── scrapers/          # Platform-specific scrapers
│       ├── preprocessing/     # Text cleaning and parsing
│       ├── analysis/          # Topic modeling and sentiment analysis
│       ├── scoring/           # Metric computation
│       ├── models/            # Data models
│       └── utils/             # Utilities and helpers
├── data/
│   ├── raw/                   # Raw scraped data
│   ├── processed/             # Cleaned and processed data
│   └── output/                # Analysis results
├── notebooks/                 # Jupyter notebooks for exploration
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.9 or higher
- Reddit API credentials (for Reddit scraping)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ChamberCheck.git
cd ChamberCheck
```

2. Create a virtual environment:
```powershell
# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install the package with dependencies:
```bash
# Install in editable mode with all dependencies
pip install -e .

# Or install with development tools
pip install -e ".[dev]"
```

4. Configure credentials:

Create a `.env` file in the project root:
```env
# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=ChamberCheck/0.1

# LLM API (optional)
LLM_PROVIDER=openai
LLM_API_KEY=your_api_key
LLM_MODEL=gpt-4
```

Or create a `config.json` file:
```json
{
  "reddit": {
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "user_agent": "ChamberCheck/0.1"
  },
  "llm": {
    "provider": "openai",
    "api_key": "your_api_key",
    "model": "gpt-4"
  }
}
```

## Usage

### Basic Example: Scraping Reddit

```python
from datetime import datetime, timedelta
from chambercheck import Config
from chambercheck.scrapers import RedditScraper

# Initialize configuration
config = Config()

# Create Reddit scraper
scraper = RedditScraper(config.get_scraper_config('reddit'))
scraper.authenticate()

# Fetch posts from last 30 days
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

posts = scraper.fetch_posts_by_engagement(
    community='politics',
    start_date=start_date,
    end_date=end_date,
    sort_by='top',
    limit=100
)

print(f"Fetched {len(posts)} posts")
```

### Analyzing Echo Chamber Metrics

```python
# Coming soon: Full analysis pipeline
from chambercheck.analysis import TopicAnalyzer, ArgumentAnalyzer
from chambercheck.scoring import EchoChamberScore

# Analyze topics
topic_analyzer = TopicAnalyzer()
topics = topic_analyzer.analyze(posts)

# Score echo chamber dynamics
scorer = EchoChamberScore()
result = scorer.compute(posts, comments, topics)

print(result.get_summary())
```

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

## Topic-Conditional Analysis

ChamberCheck evaluates discourse across different subject domains, recognizing that communities may be open on some topics while defensive on others (e.g., sports vs. politics).

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
flake8 src/
mypy src/
```

## Roadmap

- [x] Core project structure
- [x] Reddit scraper implementation
- [ ] Preprocessing pipeline
- [ ] Topic modeling with LLMs
- [ ] Base metric implementations
- [ ] Composite scoring system
- [ ] Facebook scraper
- [ ] Web dashboard for visualization
- [ ] API endpoint for external integration

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

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

---

**Note**: This project is under active development. APIs may change.
