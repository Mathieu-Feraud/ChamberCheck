"""
Text preprocessing and data cleaning modules.

Handles text normalization, deduplication, structured data extraction, and media processing.
"""

from .post_preprocessor import preprocess_posts
from .comment_preprocessor import preprocess_comments

# media_processor imports are deferred to avoid a circular import chain:
#   preprocessing/__init__ → media_processor → analysis → prompt_builder → media_processor
def __getattr__(name):
    if name in ("process_posts", "process_folder", "enrich_posts_context"):
        from .media_processor import process_posts, process_folder, enrich_posts_context  # noqa: F401
        globals()["process_posts"] = process_posts
        globals()["process_folder"] = process_folder
        globals()["enrich_posts_context"] = enrich_posts_context
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["process_posts", "process_folder", "enrich_posts_context", "preprocess_posts", "preprocess_comments"]
