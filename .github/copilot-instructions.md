# ChamberCheck AI Coding Guidelines

## Project Overview

ChamberCheck is an LLM-powered research tool that analyzes echo chamber dynamics in online communities by quantifying discourse patterns (argument diversity, hostility, suppression, epistemic closure). The architecture follows a modular pipeline: **scraping → preprocessing → analysis → scoring**.

## Architecture & Key Components

### Data Flow Pipeline
1. **Scrapers** ([src/ChamberCheck/scrapers/](src/ChamberCheck/scrapers/)): Platform-specific data collectors
   - Inherit from `BaseScraper` abstract class
   - Implement `authenticate()`, `fetch_posts()`, `fetch_comments()`
   - Current: RedditScraper (PRAW), stub for Facebook
   - Design: One scraper class per platform

2. **Data Models** ([src/ChamberCheck/models/](src/ChamberCheck/models/)): Dataclass-based entities
   - `Post`: Social media posts with platform-agnostic schema
   - `Comment`: Discussion responses
   - `AnalysisResult`: Scoring outputs
   - All use `.to_dict()` for serialization

3. **Preprocessing & Analysis** (stubs in [src/ChamberCheck/preprocessing/](src/ChamberCheck/preprocessing/) and [src/ChamberCheck/analysis/](src/ChamberCheck/analysis/)): Future modules for text cleaning and LLM-powered analysis

4. **Scoring Module** ([src/ChamberCheck/scoring/](src/ChamberCheck/scoring/)): Computes echo chamber metrics (not yet implemented—guides future work)

5. **Configuration** ([src/ChamberCheck/config.py](src/ChamberCheck/config.py)): Centralized settings via `.env` or `config.json`
   - Dot-notation access: `Config().get('reddit.client_id')`
   - Fallback to environment variables

### Plugin Architecture
- New platform scrapers: Create new class in `scrapers/`, inherit `BaseScraper`, register in config
- All platform data converges to uniform `Post`/`Comment` schema via dataclasses

## Development Workflows

### Setup & Testing
```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows

# Install in editable mode (required for local development)
pip install -e .
pip install -e ".[dev]"  # Include testing tools

# Run tests with coverage
pytest tests/
# Or with verbose coverage output configured in pyproject.toml
pytest --cov=chambercheck
```

### Code Quality
- **Black formatting**: `black src/ tests/` (line length: 100, targets Python 3.9+)
- **Type checking**: `mypy src/` (not strict, but enabled)
- **Linting**: `flake8 src/ tests/`
- All configured in [pyproject.toml](pyproject.toml)

### Configuration & Credentials
- **Local testing**: Use `config.example.json` as template, create `config.json` with test credentials
- **CI/Environment**: Load from environment variables (REDDIT_CLIENT_ID, etc.)
- **Data directories**: Raw/processed/output paths via `config.data_dir` (see [Config.get_scraper_config()](src/ChamberCheck/config.py))

## Project-Specific Conventions

### Dataclass-First Models
- All entities use Python 3.9+ `@dataclass` (see [Post](src/ChamberCheck/models/post.py), [Comment](src/ChamberCheck/models/comment.py))
- Metadata stored in `metadata: Dict[str, Any]` field for platform-specific extensions
- `.to_dict()` for serialization, `@classmethod` constructors for deserialization

### Error Handling Pattern
- Scrapers use try/except with logging (see [RedditScraper.authenticate()](src/ChamberCheck/scrapers/reddit_scraper.py#L50))
- Configuration validation in `validate_config()` methods
- Graceful failures preferred (return empty lists rather than crash)

### Test Fixtures
- Centralized in [tests/conftest.py](tests/conftest.py): `sample_reddit_config`, `sample_post_data`, `sample_date_range`
- Use fixtures in test methods to ensure consistency

### Logging
- Logger setup via `utils.logger.setup_logger("ModuleName")` 
- Import pattern: `from ..utils import setup_logger`

## Integration Points & Dependencies

### External APIs
- **Reddit PRAW**: Requires credentials; wrapped in RedditScraper
- **LLM APIs**: OpenAI/Anthropic configured but not yet integrated (see [pyproject.toml](pyproject.toml) dependencies)

### Key Dependencies
- **PRAW 7.7+**: Reddit data access (handle `ImportError` gracefully if missing)
- **Pandas/NumPy**: Data manipulation in preprocessing pipeline
- **Transformers/Sentence-Transformers**: Embeddings for analysis (not yet integrated)
- **pytest 7.4+**: Test runner with coverage tracking

### Package Entry Point
- CLI via `[project.scripts]` section (see [pyproject.toml](pyproject.toml)): `chambercheck` command (not yet implemented)

## When Adding Features

1. **New Data Source**: Extend `BaseScraper`, add to `scrapers/` directory, update Config
2. **New Metric**: Implement in scoring module, register in `AnalysisResult` dataclass
3. **New Dependency**: Add to `[project.dependencies]` or `[project.optional-dependencies]` in pyproject.toml
4. **Tests**: Add to `tests/test_*.py`, use fixtures from conftest, run `pytest --cov`

## Red Flags & Gotchas

- **Config access**: Always use `Config().get()` with dot notation (not direct dict keys)
- **Platform-agnostic**: Ensure new features work across scrapers, not just Reddit
- **Type hints**: Not strictly enforced but expected in new code (mypy warnings are acceptable)
- **Python 3.9+ only**: Do not use Python 3.10+ syntax (e.g., match statements) without compatibility layer
