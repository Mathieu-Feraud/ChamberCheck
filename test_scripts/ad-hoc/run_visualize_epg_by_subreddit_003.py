import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


INPUT_DIR = Path("data/output/scrape_003/expressive_participation_gap_by_subreddit_001")
OUTPUT_DIR = INPUT_DIR / "plots"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _trim_topic(topic: str, max_len: int = 90) -> str:
    text = " ".join(topic.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _load_epg_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}")
    return data


def _subreddit_topic_values(payload: Dict[str, Any]) -> Dict[str, float]:
    by_topic = payload.get("by_topic", {})
    if not isinstance(by_topic, dict):
        return {}

    values: Dict[str, float] = {}
    for topic, metric_result in by_topic.items():
        if not isinstance(metric_result, dict):
            continue
        value = metric_result.get("value")
        if _is_number(value):
            values[topic] = float(value)
    return values


def _plot_subreddit(subreddit: str, values: Dict[str, float], output_path: Path) -> None:
    if not values:
        return

    topics = sorted(values.keys(), key=lambda key: values[key])
    y_labels = [_trim_topic(topic) for topic in topics]
    x_vals = [values[topic] for topic in topics]

    fig_h = max(6, len(topics) * 0.45)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(y_labels, x_vals)
    ax.set_title(f"Expressive Participation Gap by Topic: {subreddit}")
    ax.set_xlabel("EPG value")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for i, value in enumerate(x_vals):
        ax.text(min(0.995, value + 0.003), i, f"{value:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subreddit_to_values: Dict[str, Dict[str, float]] = {}

    for file_path in sorted(INPUT_DIR.glob("*_epg_by_topic.json")):
        payload = _load_epg_file(file_path)
        subreddit = payload.get("subreddit")
        if not isinstance(subreddit, str) or not subreddit:
            subreddit = file_path.stem.replace("_epg_by_topic", "")

        values = _subreddit_topic_values(payload)
        subreddit_to_values[subreddit] = values

        output_path = OUTPUT_DIR / f"{subreddit}_epg_by_topic.png"
        _plot_subreddit(subreddit=subreddit, values=values, output_path=output_path)

    aggregate_plot = OUTPUT_DIR / "epg_mean_by_subreddit.png"
    if aggregate_plot.exists():
        aggregate_plot.unlink()

    print(f"Saved EPG plots to: {OUTPUT_DIR}")
    print(f"Subreddits plotted: {len(subreddit_to_values)}")


if __name__ == "__main__":
    main()
