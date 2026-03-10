"""
Tests for the ChamberCheck pipeline workflow.

Verifies that:
- Every pipeline-stage function is importable from its declared module.
- Function signatures match the expected parameters and defaults.
- Pure helper utilities (folder auto-increment, file-path resolution) behave
  correctly when given a mock filesystem via tmp_path.
- The workflow script (test_scripts/workflow.py) references each stage in the
  correct order.

No API calls or live disk writes are made; external I/O is either mocked or
exercised only through tmp_path fixtures.
"""

import inspect
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Stage 1 – Scrape posts
# ---------------------------------------------------------------------------

class TestStage1Imports:
    """Stage 1 functions are importable from the scrapers package."""

    def test_batch_scrape_posts_only_importable(self):
        from ChamberCheck.scrapers import batch_scrape_posts_only
        assert callable(batch_scrape_posts_only)

    def test_batch_scrape_importable(self):
        from ChamberCheck.scrapers import batch_scrape
        assert callable(batch_scrape)

    def test_scrape_subreddit_importable(self):
        from ChamberCheck.scrapers import scrape_subreddit
        assert callable(scrape_subreddit)


class TestStage1Signature:
    """batch_scrape_posts_only has the expected signature."""

    def test_params_exist(self):
        from ChamberCheck.scrapers import batch_scrape_posts_only
        sig = inspect.signature(batch_scrape_posts_only)
        params = sig.parameters
        assert "config_path" in params
        assert "output_folder" in params

    def test_output_folder_defaults_to_none(self):
        from ChamberCheck.scrapers import batch_scrape_posts_only
        sig = inspect.signature(batch_scrape_posts_only)
        assert sig.parameters["output_folder"].default is None


class TestStage1FolderAutoIncrement:
    """get_next_scrape_folder_number increments correctly."""

    def test_returns_one_when_no_folders_exist(self, tmp_path):
        from ChamberCheck.scrapers.batch_scraper import get_next_scrape_folder_number
        result = get_next_scrape_folder_number(data_dir=str(tmp_path))
        assert result == 1

    def test_returns_max_plus_one_with_existing_folders(self, tmp_path):
        from ChamberCheck.scrapers.batch_scraper import get_next_scrape_folder_number
        for name in ("scrape_001", "scrape_002", "scrape_005"):
            (tmp_path / name).mkdir()
        result = get_next_scrape_folder_number(data_dir=str(tmp_path))
        assert result == 6

    def test_ignores_non_scrape_folders(self, tmp_path):
        from ChamberCheck.scrapers.batch_scraper import get_next_scrape_folder_number
        (tmp_path / "other_folder").mkdir()
        (tmp_path / "scrape_003").mkdir()
        result = get_next_scrape_folder_number(data_dir=str(tmp_path))
        assert result == 4


# ---------------------------------------------------------------------------
# Stage 2 – Analyse post titles
# ---------------------------------------------------------------------------

class TestStage2Imports:
    """Stage 2 functions are importable from the analysis package."""

    def test_analyze_posts_importable(self):
        from ChamberCheck.analysis import analyze_posts
        assert callable(analyze_posts)


class TestStage2Signature:
    """analyze_posts has the expected signature."""

    def test_scrape_dir_defaults_to_none(self):
        from ChamberCheck.analysis import analyze_posts
        sig = inspect.signature(analyze_posts)
        assert sig.parameters["scrape_dir"].default is None

    def test_config_path_default(self):
        from ChamberCheck.analysis import analyze_posts
        sig = inspect.signature(analyze_posts)
        assert sig.parameters["config_path"].default == "config/config.yaml"


class TestStage2ResolveScrapeDir:
    """_resolve_scrape_dir picks the latest scrape_* folder."""

    def test_resolves_to_explicit_dir(self, tmp_path):
        from ChamberCheck.analysis.post_analyzer import _resolve_scrape_dir
        target = tmp_path / "scrape_007"
        target.mkdir()
        result = _resolve_scrape_dir(str(target))
        assert result == target

    def test_raises_if_explicit_dir_missing(self, tmp_path):
        from ChamberCheck.analysis.post_analyzer import _resolve_scrape_dir
        with pytest.raises(FileNotFoundError):
            _resolve_scrape_dir(str(tmp_path / "scrape_nonexistent"))

    def test_raises_when_no_scrape_dirs_present(self, tmp_path, monkeypatch):
        from ChamberCheck.analysis import post_analyzer
        monkeypatch.setattr(post_analyzer, "Path", lambda p: (tmp_path if p == "data/raw" else Path(p)))
        from ChamberCheck.analysis.post_analyzer import _resolve_scrape_dir
        # Patch the module-level Path so the glob over data/raw hits tmp_path
        with pytest.raises(FileNotFoundError):
            _resolve_scrape_dir(None)


# ---------------------------------------------------------------------------
# Stage 3 – Preprocess posts
# ---------------------------------------------------------------------------

class TestStage3Imports:
    """Stage 3 functions are importable from the preprocessing package."""

    def test_preprocess_posts_importable(self):
        from ChamberCheck.preprocessing import preprocess_posts
        assert callable(preprocess_posts)


class TestStage3Signature:
    """preprocess_posts has the expected signature."""

    def test_scrape_dir_required(self):
        from ChamberCheck.preprocessing import preprocess_posts
        sig = inspect.signature(preprocess_posts)
        p = sig.parameters["scrape_dir"]
        assert p.default is inspect.Parameter.empty

    def test_config_path_default(self):
        from ChamberCheck.preprocessing import preprocess_posts
        sig = inspect.signature(preprocess_posts)
        assert sig.parameters["config_path"].default == "config/config.yaml"


class TestStage3LatestAnalysis:
    """_latest_analysis picks the highest-numbered analysis file."""

    def test_picks_latest_file(self, tmp_path):
        from ChamberCheck.preprocessing.post_preprocessor import _latest_analysis
        for name in ("analysis_001.json", "analysis_002.json", "analysis_003.json"):
            (tmp_path / name).write_text("{}")
        result = _latest_analysis(tmp_path)
        assert result.name == "analysis_003.json"

    def test_excludes_metadata_files(self, tmp_path):
        from ChamberCheck.preprocessing.post_preprocessor import _latest_analysis
        (tmp_path / "analysis_001.json").write_text("{}")
        (tmp_path / "analysis_001_metadata.json").write_text("{}")
        result = _latest_analysis(tmp_path)
        assert result.name == "analysis_001.json"

    def test_raises_when_no_analysis_files(self, tmp_path):
        from ChamberCheck.preprocessing.post_preprocessor import _latest_analysis
        with pytest.raises(FileNotFoundError):
            _latest_analysis(tmp_path)


# ---------------------------------------------------------------------------
# Stage 4 – Scrape comments
# ---------------------------------------------------------------------------

class TestStage4Imports:
    """Stage 4 functions are importable from the scrapers package."""

    def test_scrape_comments_importable(self):
        from ChamberCheck.scrapers import scrape_comments
        assert callable(scrape_comments)


class TestStage4Signature:
    """scrape_comments has the expected signature."""

    def test_scrape_dir_defaults_to_none(self):
        from ChamberCheck.scrapers import scrape_comments
        sig = inspect.signature(scrape_comments)
        assert sig.parameters["scrape_dir"].default is None

    def test_config_path_param_exists(self):
        from ChamberCheck.scrapers import scrape_comments
        sig = inspect.signature(scrape_comments)
        assert "config_path" in sig.parameters


# ---------------------------------------------------------------------------
# Stage 5 – Preprocess comments
# ---------------------------------------------------------------------------

class TestStage5Imports:
    """Stage 5 functions are importable from the preprocessing package."""

    def test_preprocess_comments_importable(self):
        from ChamberCheck.preprocessing import preprocess_comments
        assert callable(preprocess_comments)


class TestStage5Signature:
    """preprocess_comments has the expected signature."""

    def test_config_path_param_exists(self):
        from ChamberCheck.preprocessing import preprocess_comments
        sig = inspect.signature(preprocess_comments)
        assert "config_path" in sig.parameters


# ---------------------------------------------------------------------------
# Stage 6 – Analyse comments
# ---------------------------------------------------------------------------

class TestStage6Imports:
    """Stage 6 functions are importable from the analysis package."""

    def test_run_comment_analysis_importable(self):
        from ChamberCheck.analysis import run_comment_analysis
        assert callable(run_comment_analysis)


class TestStage6Signature:
    """run_comment_analysis has the expected signature."""

    def test_scrape_dir_defaults_to_none(self):
        from ChamberCheck.analysis import run_comment_analysis
        sig = inspect.signature(run_comment_analysis)
        assert sig.parameters["scrape_dir"].default is None

    def test_config_path_param_exists(self):
        from ChamberCheck.analysis import run_comment_analysis
        sig = inspect.signature(run_comment_analysis)
        assert "config_path" in sig.parameters


# ---------------------------------------------------------------------------
# Stage 7 – V3 metrics
# ---------------------------------------------------------------------------

class TestStage7Imports:
    """Stage 7 classes are importable from the CC_derived_metrics package."""

    def test_v3metrics_importable(self):
        from ChamberCheck.CC_derived_metrics import V3Metrics
        assert V3Metrics is not None

    def test_metric_result_importable(self):
        from ChamberCheck.CC_derived_metrics import MetricResult
        assert MetricResult is not None


# ---------------------------------------------------------------------------
# Workflow script stage ordering
# ---------------------------------------------------------------------------

class TestWorkflowOrdering:
    """The workflow script references the eight pipeline stages in correct order."""

    _WORKFLOW = Path(__file__).parent.parent / "test_scripts" / "workflow.py"

    _EXPECTED_ORDER = [
        "batch_scrape_posts_only",   # 1 – scrape posts
        "analyze_posts",             # 2 – analyse post titles
        "preprocess_posts",          # 3 – preprocess posts
        "scrape_comments",           # 4 – scrape comments
        "preprocess_comments",       # 5 – preprocess comments
        "run_comment_analysis",      # 6 – analyse comments
        "run_v3_metrics",            # 7 – compute V3 metrics (via runpy)
        "plot_v3_metrics",           # 8 – plot metrics (via runpy)
    ]

    def test_workflow_file_exists(self):
        assert self._WORKFLOW.exists(), "test_scripts/workflow.py not found"

    def test_all_stages_present(self):
        source = self._WORKFLOW.read_text(encoding="utf-8")
        missing = [s for s in self._EXPECTED_ORDER if s not in source]
        assert not missing, f"Stages missing from workflow.py: {missing}"

    def test_stages_appear_in_order(self):
        source = self._WORKFLOW.read_text(encoding="utf-8")
        positions = [source.index(s) for s in self._EXPECTED_ORDER if s in source]
        assert positions == sorted(positions), (
            "Pipeline stages are not referenced in the expected order in workflow.py"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
