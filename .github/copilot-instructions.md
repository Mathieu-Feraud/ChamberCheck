# ChamberCheck AI Coding Guidelines

## Coding Assistant Behaviour

- **Before writing any non-trivial code, make sure you fully understand the request.** If anything is ambiguous — scope, output format, file location, filtering logic, etc. — ask a clarifying question before proceeding. Do not guess and implement something that may need to be redone.
- Prefer implementing changes over describing them.
- Keep scripts self-contained and runnable from the project root.
- Never create summary or documentation markdown files unless explicitly asked.

### Alert Flags

Use the following block whenever an alert is needed — it must be visible and not buried in prose:

```
##########################
######    ALERT     ######
##########################
```

**When to raise an ALERT:**
1. The user gives an instruction that conflicts with something in these coding guidelines — flag it before proceeding.
2. The work being done expands beyond what is documented here (new data layout, new script conventions, new pipeline stage, etc.) — flag it so the user can decide whether to update this file.

---

## Project Overview

ChamberCheck is an LLM-powered research tool that analyzes echo chamber dynamics in online communities by quantifying discourse patterns (argument diversity, hostility, suppression, epistemic closure). The architecture follows a modular pipeline: **scraping → post analysis → preprocessing → comment scraping → comment analysis → scoring**.

---

## Architecture & Key Components

### Data Flow Pipeline

1. **Scrapers** (`src/ChamberCheck/scrapers/`): Platform-specific data collectors
   - `batch_scrape_posts_only(config_path, output_folder)` — fetches posts only (no comments), saves all subreddits to a single `posts.json` + `posts_metadata.json`
   - `batch_scrape()` — full scrape including comments (legacy / future use)
   - Reddit scraper uses JSON API (no PRAW auth required for public data)
   - Design: one scraper class per platform, inherit `BaseScraper`

2. **Post Title Analysis** (`test_scripts/run_analyze_posts.py`):
   - Reads `data/raw/scrape_XXX/posts.json`
   - Calls OpenAI API per post title using `PROMPT_POST_TITLE_ANALYSIS` from `src/ChamberCheck/constants.py`
   - Outputs `data/raw/scrape_XXX/posts_analysis/analysis_NNN.json` (array of all results) + `analysis_NNN_metadata.json`
   - Run number auto-increments; metadata files excluded from counter glob
   - Response schema per post: `topic {top, mid, leaf}`, `topic_confidence`, `discussion_score`, `discussion_reason`

3. **Preprocessing** (`test_scripts/run_preprocess_posts.py`):
   - Joins `posts.json` with latest `analysis_NNN.json` on `post_id`
   - Filters: `num_comments >= MIN_COMMENTS` AND `topic.top != "UNCLEAR"`
   - Selects top-N posts per subreddit ranked by `discussion_score`
   - Outputs `data/raw/scrape_XXX/pre_process/pre_process_NNN.json` + `pre_process_NNN_metadata.json`

4. **Data Models** (`src/ChamberCheck/models/`): Dataclass-based entities
   - `Post`, `Comment`, `AnalysisResult` — all use `.to_dict()` for serialization
   - `metadata: Dict[str, Any]` field for platform-specific extensions

5. **Scoring / Derived Metrics** (`src/ChamberCheck/CC_derived_metrics/derived_metrics.py`):
   - EAS (Echo Argument Score) uses Spearman correlation (scipy.stats.spearmanr)
   - Fisher-z CI (always uses 1.96), bootstrap CI (500 iterations) for aggregate
   - BH FDR correction applied globally across all EAS p-values

6. **Configuration** (`src/ChamberCheck/config.py`):
   - Dot-notation access: `Config().get('reddit.client_id')`
   - Falls back to environment variables

### All Constants → `src/ChamberCheck/constants.py`

**Every hardcoded value belongs in `constants.py`** — no exceptions. This includes:
- LLM prompts and message templates
- Retry limits, delays, and timeout values
- Default model names, temperatures, token limits
- File naming patterns and directory paths
- Numeric thresholds, score ranges, and magic numbers

Never define constants inline in `src/` modules or `test_scripts/`. Always add named constants to `constants.py` first, then import them where needed.

---

## Data Layout

```
data/raw/scrape_XXX/
  posts.json                          # all posts, all subreddits (flat array under "posts" key)
  posts_metadata.json                 # scrape run metadata
  posts_analysis/
    analysis_NNN.json                 # LLM analysis results, one array per run
    analysis_NNN_metadata.json        # tokens, duration, model, subreddits
  pre_process/
    pre_process_NNN.json              # filtered + ranked posts ready for comment scraping
    pre_process_NNN_metadata.json     # filter params, counts, per-subreddit stats
  comments/                           # (future) fetched comments per selected post
```

Scrape folders are auto-incrementing (`scrape_001`, `scrape_002`, …). Scripts should auto-detect the latest relevant file (e.g. latest `analysis_NNN.json`) rather than hardcoding run numbers, except where a specific run is intentionally targeted.

---

## Credentials & Environment

- **All API keys are in `.env`** in the project root — never hardcode keys.
- Load with `from dotenv import load_dotenv; load_dotenv()` at the top of every script.
- Key names: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`

## Configuration Files

The project uses a **single unified YAML config** covering all pipeline stages, with two variants:

| File | Purpose |
|---|---|
| `config/config.yaml` | Production config |
| `config/config.test.yaml` | Test/development config (smaller subreddit lists, lower limits) |

**Sections in the config:**
```yaml
scraping:           # subreddits list, num_posts, sort_method, rate_limit_delay
post_analysis:      # model, temperature, max_tokens
preprocessing:      # min_comments, top_n_per_subreddit, exclude_topics
comment_scraping:   # max_comments_per_post
scoring:            # metric-specific parameters
```

- Scripts read config via `Config().get('section.key')` with dot notation
- Never hardcode config values inline in scripts — always read from config
- When adding a new pipeline stage, add its parameters as a new section in both config files

---

## Development Workflows

### Environment
```powershell
# Windows — activate venv
.\venv\Scripts\Activate.ps1

# Install in editable mode
pip install -e .
pip install -e ".[dev]"

# Run tests
pytest tests/
pytest --cov=chambercheck
```

### Running Scripts
All scripts are run from the project root:
```powershell
.\venv\Scripts\python.exe test_scripts/run_scraper_posts.py
.\venv\Scripts\python.exe test_scripts/run_analyze_posts.py
.\venv\Scripts\python.exe test_scripts/run_preprocess_posts.py
```

### Code Quality
- **Black**: `black src/ tests/` (line length: 100, Python 3.9+)
- **mypy**: `mypy src/` (not strict)
- **flake8**: `flake8 src/ tests/`

---

## Project-Specific Conventions

### Script Structure
- Scripts in `test_scripts/` are standalone entry points — **keep them minimal**. They should do little more than import from `src/` and call a function. Logic, processing, and configuration belong in `src/`, not in the script itself.
- Add `sys.path.insert(0, str(Path(__file__).parent.parent / "src"))` at top of every script
- Use `Path` (not `os.path`) for all file operations
- Always write a companion `*_metadata.json` alongside any output data file
- Auto-increment output file numbers; never overwrite previous runs

### Script Naming Convention (`test_scripts/`)

Named scripts follow this pattern:
```
run_<src_folder>_<main_file>_<function>.py
```
- `<src_folder>` — the `src/ChamberCheck/` subfolder containing the code being exercised (e.g. `scrapers`, `scoring`, `reporting`, `preprocessing`, `models`)
- `<main_file>` — the primary source file being called, without `.py` (e.g. `batch_scraper`, `report_generator`)
- `<function>` — the specific function or entry point being invoked (e.g. `batch_scrape_posts_only`)

Examples:
- `run_scrapers_batch_scraper_posts_only.py`
- `run_scoring_derived_metrics_eas.py`
- `run_reporting_report_generator_full.py`

If unsure which segment applies, ask before creating the file.

### Ad-hoc Scripts

One-off, exploratory, or diagnostic scripts go in `test_scripts/ad-hoc/` — not in the root of `test_scripts/`. If it is unclear whether a requested script is ad-hoc or a permanent pipeline script, ask.

### Error Handling
- Scrapers: try/except with logging, graceful failures (return empty lists rather than crash)
- LLM calls: retry loop (typically 3 attempts, 5s delay), return error dict on max retries
- Config: validate in `validate_config()` methods

### Logging
- `from ..utils import setup_logger` — `setup_logger("ModuleName")`

### Type Hints
- Expected in new code; mypy warnings acceptable but should not be introduced carelessly

### Python Version
- **3.9+ only** — no `match` statements, no `X | Y` union syntax without `from __future__ import annotations`

---

## Key Dependencies

- **openai** — post/comment LLM analysis (`gpt-4.1-mini` default for post analysis)
- **anthropic** — alternative LLM provider
- **python-dotenv** — `.env` loading
- **scipy** — Spearman correlation for EAS metric
- **matplotlib / numpy** — visualisation and data manipulation
- **python-docx** — report generation
- **requests** — Reddit JSON scraper (no auth needed for public endpoints)
- **pytest 7.4+** — test runner

---

## When Adding Features

1. **New scraper**: extend `BaseScraper`, add to `scrapers/`, update Config
2. **New LLM prompt or any hardcoded value**: add named constant to `constants.py`, import it — never define inline
3. **New metric**: implement in scoring module, register in `AnalysisResult`
4. **New dependency**: add to `[project.dependencies]` in `pyproject.toml`
5. **New test**: add to `tests/test_*.py`, use fixtures from `conftest.py`

---

## Red Flags & Gotchas

- **All constants in `constants.py`** — never define prompts, magic numbers, retry limits, templates, thresholds, or any other hardcoded value inline in `src/` modules or scripts; always add to `constants.py` first
- **Config access** — always use `Config().get()` with dot notation
- **Platform-agnostic** — features should work across scrapers, not just Reddit
- **`_metadata` glob exclusion** — when counting/auto-incrementing `analysis_NNN.json` or `pre_process_NNN.json` files, always exclude `*_metadata.json` from the glob
- **`communism` subreddit** — posts tend to be anniversary images with low comment counts and UNCLEAR topics; often drops out of filters
