# ChamberCheck Architecture Rules

## Core Principle
Separate **callable functions** from **CLI handling**. Users should be able to import and run any major operation independently.

## Pipeline Separation

The codebase is organized by **independent, composable stages**:

1. **Scraper** (`src/ChamberCheck/scrapers/`)
   - Entry point: `batch_scrape()` function
   - Handles: multi-platform data collection, incremental file management, metadata tracking
   - CLI wrapper: `scripts/scrape_reddit_noauth.py`

2. **Preprocessing** (`src/ChamberCheck/preprocessing/`)
   - Entry point: `process_posts()` function
   - Handles: data cleaning, format normalization, data validation
   - CLI wrapper: `scripts/preprocess_media.py`

3. **LLM Metrics** (`src/ChamberCheck/analysis/`)
   - Entry point: Individual analyzer classes (e.g., `CommentAnalyzer`) and orchestrators (e.g., `batch_analyze_comments()`)
   - Handles: LLM-powered analysis of individual items
   - CLI wrapper: `scripts/analyze_comments.py`

4. **Scoring** (`src/ChamberCheck/scoring/`)
   - Entry point: Aggregation and metric calculation functions
   - Handles: summarizing LLM analyses into echo chamber metrics
   - CLI wrapper: `scripts/calculate_scores.py` (or similar)

5. **Reporting** (`src/ChamberCheck/reporting/`)
   - Entry point: Report generation functions
   - Handles: output formatting, visualization prep, summary generation
   - CLI wrapper: `scripts/generate_report.py` (or similar)

## Code Organization Rules

### In `src/ChamberCheck/[module]/` (Business Logic)
- ✅ Implement core functionality
- ✅ Accept file paths, configs, and data structures as parameters
- ✅ Return dictionaries or dataclass objects
- ✅ Import only from `src/ChamberCheck/` and external packages
- ❌ Do NOT handle CLI argument parsing
- ❌ Do NOT perform `sys.path` manipulation
- ❌ Do NOT load `.env` files (caller handles this)

### In `scripts/[name].py` (CLI Wrappers)
- ✅ Parse CLI arguments
- ✅ Load environment variables and config files
- ✅ Call the corresponding orchestrator function from `src/ChamberCheck/`
- ✅ Handle result display and file output
- ❌ Do NOT contain business logic
- ❌ Do NOT be longer than ~50-60 lines
- Should follow pattern:
  ```python
  from ChamberCheck.[module].[submodule] import [orchestrator_function]
  
  args = parse_args()
  result = [orchestrator_function](args.input_file, **config)
  save_results(result)
  ```

## Constants Management

- **Location**: `src/ChamberCheck/constants.py`
- **Purpose**: All configurable values that a developer might want to edit
- **Import pattern**: `from ..constants import CONSTANT_NAME`
- **Naming**: 
  - `MODULE_SETTING_NAME` (e.g., `MEDIA_MAX_RETRIES`)
  - `MODULE_SETTING_VALUE` in `SCREAMING_SNAKE_CASE`
  - Group by module with section headers

## File Structure

```
src/ChamberCheck/
├── scrapers/
│   ├── __init__.py              # Export: batch_scrape
│   ├── base_scraper.py
│   ├── reddit_scraper.py
│   └── batch_scraper.py         # Orchestrator: batch_scrape()
│
├── preprocessing/
│   ├── __init__.py              # Export: process_posts
│   ├── media_processor.py       # Orchestrator: process_posts()
│   └── ...
│
├── analysis/
│   ├── __init__.py              # Export: batch_analyze_comments
│   ├── comment_analyzer.py      # Individual analysis
│   ├── batch_analyzer.py        # Orchestrator: batch_analyze_comments()
│   └── llm_provider.py
│
├── scoring/
│   ├── __init__.py              # Export: calculate_scores
│   ├── metric_calculator.py     # Core metrics
│   └── batch_scorer.py          # Orchestrator: calculate_scores()
│
├── reporting/
│   ├── __init__.py              # Export: generate_report
│   └── report_generator.py      # Orchestrator: generate_report()
│
└── constants.py                 # All config values

scripts/
├── scrape_reddit_noauth.py      # CLI wrapper (~40 lines)
├── preprocess_media.py          # CLI wrapper (~40 lines)
├── analyze_comments.py          # CLI wrapper (~40 lines)
├── calculate_scores.py          # CLI wrapper (~40 lines) [TODO]
├── generate_report.py           # CLI wrapper (~40 lines) [TODO]
└── OLD/                         # Backups of refactored scripts
```

## Import Guidelines

- **Within `src/ChamberCheck/`**: Use relative imports (`from ..module import func`)
- **In `scripts/`**: Use absolute imports from installed package (`from ChamberCheck.module import func`)
- **Test files**: Can import from both `src/` and `scripts/`

## Testing Rules

- Place tests in `tests/` mirroring source structure
- Import orchestrator functions, not just CLI scripts
- Use fixtures from `tests/conftest.py`
- Test that functions work when called directly (not just via CLI)

## When Adding Features

1. **New data source?** → Create scraper in `scrapers/`, add to `batch_scraper.py` orchestrator
2. **New analysis type?** → Create analyzer in `analysis/`, add to `batch_analyzer.py` orchestrator
3. **New metric?** → Create in `scoring/`, add to score calculation orchestrator
4. **Hardcoded value?** → Move to `constants.py` with descriptive name
5. **New CLI script?** → Create thin wrapper in `scripts/` that imports from `src/`

## Refactoring Checklist

- [ ] Extract business logic to `src/ChamberCheck/[module]/`
- [ ] Create orchestrator function named `batch_*` or `[action]_*`
- [ ] Export orchestrator from `__init__.py`
- [ ] Reduce CLI script to ~40 lines
- [ ] Move hardcoded values to `constants.py`
- [ ] Update imports throughout
- [ ] Test that the function works when imported directly
- [ ] Backup original script to `scripts/OLD/`
- [ ] Document changes in commit/summary

