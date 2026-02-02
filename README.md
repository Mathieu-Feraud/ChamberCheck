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
