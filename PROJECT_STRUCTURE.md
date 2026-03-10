# Project Structure Guide

This guide explains the ChamberCheck project layout and how each component fits together.

## Directory Structure

```
ChamberCheck/
├── ARCHITECTURE_RULES.md          # Architecture specification (5-pipeline model)
├── REFACTORING_COMPLETE.md        # Refactoring completion summary
├── README.md                       # Main project readme
├── pyproject.toml                  # Package configuration & dependencies
├── setup.py                        # Setup script
│
├── config/                         # Configuration files
│   ├── config.example.json         # Template for local configuration
│   ├── config.json                 # Local configuration (git-ignored)
│   ├── config.analyze.json         # Analysis-specific settings
│   ├── scraper_config.json         # Production scraper config
│   └── scraper_config_test_run.json # Test scraper config
│
├── data/                           # Data directory (git-ignored)
│   ├── raw/                        # Raw scraped data
│   │   ├── scrape_001/             # First batch
│   │   │   ├── samharris.json      # Raw posts
│   │   │   ├── samharris_scraper_metadata.json
│   │   │   └── samharris_preprocess_metadata_001.json
│   │   ├── scrape_002/             # Second batch
│   │   └── OLD/                    # Legacy data
│   ├── processed/                  # Analyzed data
│   │   ├── samharris_analysis_001.json
│   │   └── samharris_analysis_metadata.json
│   └── output/                     # Reports and visualizations
│       ├── samharris_plot.png
│       └── samharris_summary.txt
│
├── src/ChamberCheck/               # Main package
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   ├── constants.py                # All editable constants
│   │
│   ├── scrapers/                   # Stage 1: Scraping
│   │   ├── base_scraper.py         # Abstract base class
│   │   ├── reddit_json_scraper.py  # Reddit implementation
│   │   ├── batch_scraper.py        # Orchestrator (business logic)
│   │   └── __init__.py
│   │
│   ├── preprocessing/              # Stage 2: Media extraction
│   │   ├── media_processor.py      # Orchestrator (business logic)
│   │   └── __init__.py
│   │
│   ├── analysis/                   # Stage 3: LLM analysis
│   │   ├── batch_analyzer.py       # Orchestrator (business logic)
│   │   ├── comment_analyzer.py     # Core analysis logic
│   │   ├── llm_provider.py         # LLM abstraction
│   │   ├── openai_provider.py      # OpenAI implementation
│   │   └── __init__.py
│   │
│   ├── reporting/                  # Stage 5: Report generation
│   │   ├── report_generator.py     # Orchestrator (business logic)
│   │   └── __init__.py
│   │
│   ├── scoring/                    # Stage 4: Metrics (placeholder)
│   │   └── __init__.py
│   │
│   ├── models/                     # Data models
│   │   ├── comment.py              # Comment dataclass
│   │   ├── post.py                 # Post dataclass
│   │   ├── analysis_result.py      # Analysis result dataclass
│   │   └── __init__.py
│   │
│   └── utils/                      # Utilities
│       ├── logger.py               # Logging setup
│       └── __init__.py
│
├── test_scripts/                   # CLI wrappers for testing (minimal)
│   ├── run_scraper.py              # → batch_scrape()
│   ├── run_preprocess.py           # → process_folder()
│   ├── run_media_processor.py      # → process_posts()
│   ├── run_analyze.py              # → batch_analyze_comments()
│   ├── run_report.py               # → generate_subreddit_report()
│   ├── run_full_pipeline.py        # Complete orchestrator
│   └── README.md                   # Test scripts documentation
│
├── tests/                          # Unit tests
│   ├── conftest.py                 # pytest fixtures
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_scraper.py
│   └── ...
│
└── .github/
    └── copilot-instructions.md     # AI coding guidelines
```

## Architecture Layers

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Scraper                                                │
│ Collects posts/comments from Reddit via RedditJSONScraper      │
│ Output: data/raw/scrape_NNN/subreddit.json                     │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│ Stage 2: Preprocessor                                           │
│ Extracts media (images, videos, links) from posts             │
│ Output: Added extracted_media field to posts                   │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│ Stage 3: LLM Analysis                                           │
│ Analyzes comments with vision/text LLM metrics                │
│ Output: data/processed/subreddit_analysis_NNN.json            │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│ Stage 4: Scoring                                                │
│ Computes echo chamber metrics (placeholder)                     │
│ Output: Scoring results (not yet implemented)                   │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│ Stage 5: Reporting                                              │
│ Generates plots, summaries, and visualizations                 │
│ Output: data/output/subreddit_plot.png, _summary.txt           │
└─────────────────────────────────────────────────────────────────┘
```

## Code Organization Pattern

Each pipeline stage follows this pattern:

### Scrapers Example
```python
# src/ChamberCheck/scrapers/batch_scraper.py (orchestrator)
def batch_scrape(config_path, output_folder=None):
    """Orchestrator function - business logic"""
    # Implementation...

# test_scripts/run_scraper.py (thin wrapper)
from ChamberCheck.scrapers import batch_scrape
batch_scrape("config/scraper_config_test_run.json")
```

### Preprocessing Example
```python
# src/ChamberCheck/preprocessing/media_processor.py
def process_folder(folder_path, post_ids=None):
    """Orchestrator function - business logic"""
    # Implementation...

# test_scripts/run_preprocess.py (thin wrapper)
from ChamberCheck.preprocessing import process_folder
process_folder("data/raw/scrape_001")
```

## Configuration Management

### Constants (src/ChamberCheck/constants.py)
All hardcoded values programmers might want to edit:
```python
# Media settings
MEDIA_MAX_RETRIES = 3
MEDIA_REQUEST_TIMEOUT = 30

# Analysis prompts
VISION_ANALYSIS_PROMPT = "..."

# Data directories
PROCESSED_DATA_DIR = "data/processed"
```

### Config (src/ChamberCheck/config.py)
Runtime configuration from environment or config.json:
```python
Config().get('openai.api_key')
Config().get('reddit.client_id')
```

### Config Files (config/)
- `config.example.json` - Template (always keep in sync)
- `config.json` - Local overrides (git-ignored)
- `scraper_config_test_run.json` - Test scraping settings

## Data Models (src/ChamberCheck/models/)

All entities use Python 3.9+ dataclasses:

```python
@dataclass
class Post:
    post_id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Serialize to JSON"""
```

Pattern: All dataclasses have `.to_dict()` method and can be loaded from dict via `@classmethod from_dict()`.

## Testing Strategy

### Unit Tests (tests/)
```bash
pytest tests/
pytest --cov=chambercheck  # With coverage
```

### Integration Tests (test_scripts/)
Minimal CLI wrappers that test entire pipeline stages:
```bash
python test_scripts/run_preprocess.py
python test_scripts/run_full_pipeline.py
```

## Import Usage Examples

### Importing for custom workflows
```python
# Single-stage operations
from ChamberCheck.scrapers import scrape_subreddit
from ChamberCheck.preprocessing import process_posts, process_folder
from ChamberCheck.analysis import batch_analyze_comments
from ChamberCheck.reporting import generate_subreddit_report

# Custom pipeline
for subreddit in subreddits:
    scrape_subreddit(subreddit)  # Stage 1
    
process_folder("data/raw/scrape_001")  # Stage 2

batch_analyze_comments(input_file, output_folder)  # Stage 3

generate_subreddit_report(subreddit)  # Stage 5
```

## File Naming Conventions

### Metadata Files
- `subreddit_scraper_metadata.json` - Scraper run metadata
- `subreddit_preprocess_metadata_001.json` - Preprocessing run metadata (auto-numbered)
- `subreddit_analysis_metadata.json` - Analysis run metadata

### Data Files
- `subreddit.json` - Posts/data (no number suffix)
- `subreddit_analysis_001.json` - Analysis results (auto-numbered)

### Output Files
- `subreddit_plot.png` - Report visualization
- `subreddit_summary.txt` - Report summary

## Development Workflow

### Local Development
```bash
# Create venv
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows

# Install in editable mode
pip install -e .
pip install -e ".[dev]"

# Code formatting
black src/ tests/

# Type checking
mypy src/

# Testing
pytest tests/
```

### Adding New Features

1. **New data source:** Create `scrapers/new_platform_scraper.py`
2. **New metric:** Add to `scoring/` module
3. **New dependency:** Update `pyproject.toml` [project.dependencies]
4. **New test:** Add to `tests/test_*.py`

## Key Files for Customization

Edit these files to customize behavior:

1. **Constants** → `src/ChamberCheck/constants.py` (all editable values)
2. **Config** → `config/config.json` (API keys, paths)
3. **Architecture** → `ARCHITECTURE_RULES.md` (design decisions)
4. **Tests** → `test_scripts/` (test workflows)

## Summary

ChamberCheck follows a **modular, stage-based pipeline** architecture:
- Business logic in `src/ChamberCheck/` (importable modules)
- CLI wrappers in `test_scripts/` (minimal, thin)
- Configuration centralized in `config.py` and `constants.py`
- Data flows through 5 clearly-defined stages
- Each stage can be tested independently or combined

This structure enables both CLI usage and direct Python imports for custom workflows.
