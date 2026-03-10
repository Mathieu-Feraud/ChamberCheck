import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


INPUT_PATH = Path("data/output/scrape_003/fake_llm_derived_metrics_by_subreddit_001.json")
OUTPUT_DIR = Path("data/output/scrape_003/fake_llm_visuals_by_subreddit_001")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _extract_subreddit_scalar(metric_payload: Any) -> Tuple[float, bool]:
    if isinstance(metric_payload, dict) and "value" in metric_payload:
        value = metric_payload.get("value")
        if _is_number(value):
            return float(value), True
        return float("nan"), False

    if isinstance(metric_payload, dict):
        vals: List[float] = []
        for _, topic_result in metric_payload.items():
            if not isinstance(topic_result, dict):
                continue
            value = topic_result.get("value")
            if _is_number(value):
                vals.append(float(value))
        if vals:
            return float(np.mean(vals)), True
        return float("nan"), False

    return float("nan"), False


def _extract_subreddit_scalar_with_ci(metric_payload: Any) -> Tuple[float, bool, float, float]:
    value, ok = _extract_subreddit_scalar(metric_payload)
    if not ok:
        return value, False, float("nan"), float("nan")

    if isinstance(metric_payload, dict) and "value" in metric_payload:
        ci_lower = metric_payload.get("ci_95_lower")
        ci_upper = metric_payload.get("ci_95_upper")
        if _is_number(ci_lower) and _is_number(ci_upper):
            v = float(value)
            lower_err = max(0.0, v - float(ci_lower))
            upper_err = max(0.0, float(ci_upper) - v)
            return v, True, lower_err, upper_err

    return float(value), True, float("nan"), float("nan")


def _draw_heatmap(matrix: np.ndarray, x_labels: List[str], y_labels: List[str], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(12, len(x_labels) * 1.1), max(6, len(y_labels) * 0.5)))
    valid = matrix[~np.isnan(matrix)]
    vmin = float(np.min(valid)) if valid.size else 0.0
    vmax = float(np.max(valid)) if valid.size else 1.0
    draw_matrix = np.nan_to_num(matrix, nan=0.0)

    image = ax.imshow(draw_matrix, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)

    for i in range(draw_matrix.shape[0]):
        for j in range(draw_matrix.shape[1]):
            cell_text = "NA" if np.isnan(matrix[i, j]) else f"{matrix[i, j]:.2f}"
            ax.text(j, i, cell_text, ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _draw_bar(
    values: List[float],
    labels: List[str],
    title: str,
    ylabel: str,
    output_path: Path,
    error_low: List[float] | None = None,
    error_high: List[float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.0), 6))
    yerr = None
    if error_low is not None and error_high is not None and len(error_low) == len(values) and len(error_high) == len(values):
        finite_count = sum(1 for lo, hi in zip(error_low, error_high) if np.isfinite(lo) and np.isfinite(hi))
        if finite_count > 0:
            yerr = np.array([
                [0.0 if not np.isfinite(lo) else lo for lo in error_low],
                [0.0 if not np.isfinite(hi) else hi for hi in error_high],
            ])

    ax.bar(labels, values, yerr=yerr, capsize=4 if yerr is not None else 0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    subreddits = sorted(payload.keys())
    metric_names = sorted(next(iter(payload.values())).keys())

    scalar_matrix = np.full((len(metric_names), len(subreddits)), np.nan)
    valid_count_matrix = np.zeros((len(metric_names), len(subreddits)), dtype=float)

    for i, metric_name in enumerate(metric_names):
        for j, subreddit in enumerate(subreddits):
            metric_payload = payload[subreddit][metric_name]
            value, ok = _extract_subreddit_scalar(metric_payload)
            if ok:
                scalar_matrix[i, j] = value
                valid_count_matrix[i, j] = 1.0
            else:
                valid_count_matrix[i, j] = 0.0

    _draw_heatmap(
        matrix=scalar_matrix,
        x_labels=subreddits,
        y_labels=metric_names,
        title="Derived metrics (subreddit-level view, scalar or topic-mean)",
        output_path=OUTPUT_DIR / "subreddit_metric_heatmap.png",
    )

    _draw_heatmap(
        matrix=valid_count_matrix,
        x_labels=subreddits,
        y_labels=metric_names,
        title="Derived metrics validity mask (1=has value, 0=NA)",
        output_path=OUTPUT_DIR / "subreddit_metric_validity.png",
    )

    key_metrics = [
        "Selective Engagement",
        "Discreditation Rate",
        "Dropout Rate",
    ]
    for metric_name in key_metrics:
        vals = []
        labs = []
        err_low = []
        err_high = []
        for subreddit in subreddits:
            metric_payload = payload[subreddit].get(metric_name)
            value, ok, low, high = _extract_subreddit_scalar_with_ci(metric_payload)
            if not ok or np.isnan(value):
                continue
            labs.append(subreddit)
            vals.append(value)
            err_low.append(low)
            err_high.append(high)

        if vals:
            _draw_bar(
                values=vals,
                labels=labs,
                title=f"{metric_name} by subreddit",
                ylabel="Value",
                output_path=OUTPUT_DIR / f"bar_{metric_name.replace('/', '-').replace(' ', '_')}.png",
                error_low=err_low,
                error_high=err_high,
            )

    print(f"Saved visuals to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
