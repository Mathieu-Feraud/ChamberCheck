import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


INPUT_PATH = Path("data/output/scrape_003/fake_llm_derived_metrics_001.json")
OUTPUT_DIR = Path("data/output/scrape_003/fake_llm_visuals_001")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _collect_scalar_values(
    payload: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
) -> Dict[str, Dict[str, List[float]]]:
    result: Dict[str, Dict[str, List[float]]] = {}
    for subreddit, metric_map in payload.items():
        for metric_name, topic_map in metric_map.items():
            for _, metric_result in topic_map.items():
                value = metric_result.get("value") if isinstance(metric_result, dict) else None
                if not _is_number(value):
                    continue
                result.setdefault(metric_name, {}).setdefault(subreddit, []).append(float(value))
    return result


def _collect_dict_values(
    payload: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    result: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for subreddit, metric_map in payload.items():
        for metric_name, topic_map in metric_map.items():
            for _, metric_result in topic_map.items():
                value = metric_result.get("value") if isinstance(metric_result, dict) else None
                if not isinstance(value, dict):
                    continue
                for component, comp_value in value.items():
                    if not _is_number(comp_value):
                        continue
                    result.setdefault(metric_name, {}).setdefault(component, {}).setdefault(subreddit, []).append(
                        float(comp_value)
                    )
    return result


def _collect_coverage(
    payload: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
) -> Tuple[List[str], List[str], np.ndarray]:
    subreddits = sorted(payload.keys())
    metric_names = sorted(next(iter(payload.values())).keys())

    matrix = np.zeros((len(metric_names), len(subreddits)), dtype=float)
    for i, metric_name in enumerate(metric_names):
        for j, subreddit in enumerate(subreddits):
            topic_map = payload[subreddit][metric_name]
            total = len(topic_map)
            valid = sum(1 for _, r in topic_map.items() if r.get("value") is not None)
            matrix[i, j] = (valid / total) if total else 0.0

    return subreddits, metric_names, matrix


def _draw_heatmap(
    matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    output_path: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(max(12, len(x_labels) * 1.1), max(6, len(y_labels) * 0.5)))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _draw_metric_boxplot(
    metric_name: str,
    by_subreddit: Dict[str, List[float]],
    output_path: Path,
) -> None:
    subreddits = sorted(by_subreddit.keys())
    data = [by_subreddit[s] for s in subreddits]

    fig, ax = plt.subplots(figsize=(max(12, len(subreddits) * 1.1), 6))
    ax.boxplot(data, tick_labels=subreddits, showmeans=True)
    ax.set_title(f"{metric_name}: distribution by subreddit")
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    subreddits, metric_names, coverage_matrix = _collect_coverage(payload)
    _draw_heatmap(
        matrix=coverage_matrix,
        x_labels=subreddits,
        y_labels=metric_names,
        title="Derived Metrics: valid-value coverage by subreddit",
        output_path=OUTPUT_DIR / "coverage_heatmap.png",
        vmin=0.0,
        vmax=1.0,
        cmap="magma",
    )

    scalar_values = _collect_scalar_values(payload)
    scalar_metrics = sorted(scalar_values.keys())

    mean_matrix = np.full((len(scalar_metrics), len(subreddits)), np.nan)
    for i, metric_name in enumerate(scalar_metrics):
        for j, subreddit in enumerate(subreddits):
            values = scalar_values.get(metric_name, {}).get(subreddit, [])
            if values:
                mean_matrix[i, j] = float(np.mean(values))

    filled = np.nan_to_num(mean_matrix, nan=0.0)
    _draw_heatmap(
        matrix=filled,
        x_labels=subreddits,
        y_labels=scalar_metrics,
        title="Derived Metrics: mean scalar values by subreddit",
        output_path=OUTPUT_DIR / "scalar_means_heatmap.png",
        vmin=float(np.nanmin(mean_matrix)) if np.any(~np.isnan(mean_matrix)) else 0.0,
        vmax=float(np.nanmax(mean_matrix)) if np.any(~np.isnan(mean_matrix)) else 1.0,
        cmap="coolwarm",
    )

    for metric_name in scalar_metrics:
        _draw_metric_boxplot(
            metric_name=metric_name,
            by_subreddit=scalar_values[metric_name],
            output_path=OUTPUT_DIR / f"boxplot_{metric_name.replace('/', '-').replace(' ', '_')}.png",
        )

    dict_values = _collect_dict_values(payload)
    for metric_name, component_map in dict_values.items():
        components = sorted(component_map.keys())
        matrix = np.zeros((len(components), len(subreddits)), dtype=float)
        for i, component in enumerate(components):
            for j, subreddit in enumerate(subreddits):
                values = component_map.get(component, {}).get(subreddit, [])
                matrix[i, j] = float(np.mean(values)) if values else 0.0

        file_name = f"components_{metric_name.replace('/', '-').replace(' ', '_')}.png"
        _draw_heatmap(
            matrix=matrix,
            x_labels=subreddits,
            y_labels=components,
            title=f"{metric_name}: mean component values by subreddit",
            output_path=OUTPUT_DIR / file_name,
            vmin=float(np.min(matrix)),
            vmax=float(np.max(matrix)),
            cmap="RdBu_r",
        )

    print(f"Saved visuals to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
