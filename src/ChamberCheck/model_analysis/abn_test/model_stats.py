"""Generate visual comparisons for ABN user and LLM runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ABN_DIR = PROJECT_ROOT / "data" / "output" / "scrape_003" / "abn_test"

NUMERIC_METRIC_PATHS: List[Tuple[str, Tuple[str, ...]]] = [
    ("stance_value", ("topic", "stance", "value")),
    ("epistemic_claim_strength", ("epistemic_risk", "claim_strength")),
    ("epistemic_evidence_quality", ("epistemic_risk", "evidence_quality")),
    ("epistemic_reasoning_depth", ("epistemic_risk", "reasoning_depth")),
    ("toxicity", ("toxicity",)),
    ("discrediting", ("discrediting",)),
    ("defensive", ("defensive",)),
    ("civility", ("civility",)),
    ("emotion_anger", ("emotion", "anger")),
    ("emotion_anxiety", ("emotion", "anxiety")),
    ("emotion_disgust", ("emotion", "disgust")),
]

EMPHASIS_METRICS = [
    "stance_value",
    "epistemic_claim_strength",
    "epistemic_evidence_quality",
    "epistemic_reasoning_depth",
]

MODEL_PRICING_USD_PER_MTOK: Dict[str, Dict[str, float]] = {
    "gpt-5.2": {"input": 1.75, "output": 14.0},
    "gpt-5-mini": {"input": 0.25, "output": 2.0},
    "gpt-5-nano": {"input": 0.05, "output": 0.4},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    "claude-3-5-haiku-20241022": {"input": 0.8, "output": 4.0},
    "claude-3-5-haiku-latest": {"input": 0.8, "output": 4.0},
}

PRICING_SOURCES = {
    "openai": "https://developers.openai.com/api/docs/pricing",
    "anthropic": "https://platform.claude.com/docs/en/about-claude/pricing",
}

METRIC_RELATIVE_DENOMINATOR: Dict[str, float] = {
    "stance_value": 20.0,
    "epistemic_claim_strength": 10.0,
    "epistemic_evidence_quality": 10.0,
    "epistemic_reasoning_depth": 10.0,
    "toxicity": 10.0,
    "discrediting": 10.0,
    "defensive": 10.0,
    "civility": 10.0,
    "emotion_anger": 10.0,
    "emotion_anxiety": 10.0,
    "emotion_disgust": 10.0,
}


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _safe_datetime(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.min


def _extract_run_number(path: Path) -> int:
    suffix = path.stem.split("_")[-1]
    try:
        return int(suffix)
    except ValueError:
        return -1


def _resolve_path(path_value: Optional[str], metadata_path: Path) -> Optional[Path]:
    if not path_value:
        return None

    normalized = path_value.replace("\\", "/")
    candidates = [
        Path(path_value),
        Path(normalized),
        PROJECT_ROOT / normalized,
        metadata_path.parent / Path(normalized).name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _numeric(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.upper() == "N/A":
            return np.nan
        try:
            return float(stripped)
        except ValueError:
            return np.nan
    return np.nan


def _nested_get(data: Dict[str, Any], path: Iterable[str]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _entries_to_dataframe(entries: List[Dict[str, Any]], source_label: str, source_type: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in entries:
        row: Dict[str, Any] = {
            "source_label": source_label,
            "source_type": source_type,
            "prompt_number": item.get("prompt_number"),
            "comment_id": item.get("comment_id"),
        }
        entry = item.get("entry") if isinstance(item, dict) else None
        if not isinstance(entry, dict):
            continue

        for metric_name, metric_path in NUMERIC_METRIC_PATHS:
            row[metric_name] = _numeric(_nested_get(entry, metric_path))

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["source_label", "source_type", "prompt_number", "comment_id"])
    return pd.DataFrame(rows)


def _discover_latest_user_metadata(abn_dir: Path) -> Optional[Path]:
    candidates = sorted(abn_dir.glob("abn_test_user_entries_metadata_*.json"))
    if not candidates:
        return None

    scored: List[Tuple[datetime, int, Path]] = []
    for path in candidates:
        metadata = _load_json(path)
        scored.append((_safe_datetime(metadata.get("run_timestamp")), _extract_run_number(path), path))

    scored.sort()
    return scored[-1][2]


def _discover_latest_llm_metadata_per_model(abn_dir: Path) -> Dict[str, Path]:
    candidates = sorted(abn_dir.glob("abn_test_llm_entries_metadata_*.json"))
    selected: Dict[str, Tuple[datetime, int, Path]] = {}

    for path in candidates:
        metadata = _load_json(path)
        model_key = metadata.get("model_requested") or metadata.get("model")
        if not model_key:
            continue

        score = (_safe_datetime(metadata.get("run_timestamp")), _extract_run_number(path), path)
        previous = selected.get(model_key)
        if previous is None or score > previous:
            selected[model_key] = score

    return {model: score[2] for model, score in selected.items()}


def _load_token_usage(metadata: Dict[str, Any], source_label: str) -> pd.DataFrame:
    usage_list = metadata.get("token_usage_per_prompt")
    if not isinstance(usage_list, list):
        return pd.DataFrame(columns=["source_label", "prompt_number", "status", "prompt_tokens", "completion_tokens", "total_tokens"])

    rows: List[Dict[str, Any]] = []
    for item in usage_list:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "source_label": source_label,
                "prompt_number": item.get("prompt_number"),
                "status": item.get("status"),
                "prompt_tokens": _numeric(item.get("prompt_tokens")),
                "completion_tokens": _numeric(item.get("completion_tokens")),
                "total_tokens": _numeric(item.get("total_tokens")),
            }
        )

    return pd.DataFrame(rows)


def _plot_all_metrics_heatmap(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    metric_columns = [name for name, _ in NUMERIC_METRIC_PATHS]
    means = metrics_df.groupby("source_label")[metric_columns].mean(numeric_only=True)
    plt.figure(figsize=(14, 6))
    sns.heatmap(means, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5)
    plt.title("ABN Numeric Metric Means by Source")
    plt.tight_layout()
    plt.savefig(output_dir / "all_metrics_mean_heatmap.png", dpi=200)
    plt.close()


def _plot_all_metrics_distribution(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    metric_columns = [name for name, _ in NUMERIC_METRIC_PATHS]
    melted = metrics_df.melt(id_vars=["source_label"], value_vars=metric_columns, var_name="metric", value_name="value")
    melted = melted.dropna(subset=["value"])

    plt.figure(figsize=(18, 7))
    sns.boxplot(data=melted, x="metric", y="value", hue="source_label", showfliers=False)
    plt.xticks(rotation=35, ha="right")
    plt.title("ABN Metric Distributions by Source")
    plt.tight_layout()
    plt.savefig(output_dir / "all_metrics_distribution.png", dpi=200)
    plt.close()


def _build_delta_dataframe(metrics_df: pd.DataFrame, user_label: str) -> pd.DataFrame:
    metric_columns = [name for name, _ in NUMERIC_METRIC_PATHS]
    user_df = metrics_df[metrics_df["source_label"] == user_label][["prompt_number"] + metric_columns].copy()
    user_df = user_df.rename(columns={metric: f"{metric}_user" for metric in metric_columns})

    rows: List[Dict[str, Any]] = []
    model_labels = sorted(label for label in metrics_df["source_label"].unique() if label != user_label)

    for label in model_labels:
        model_df = metrics_df[metrics_df["source_label"] == label][["prompt_number"] + metric_columns].copy()
        merged = user_df.merge(model_df, on="prompt_number", how="inner")
        if merged.empty:
            continue

        for _, row in merged.iterrows():
            prompt_number = row["prompt_number"]
            for metric in metric_columns:
                user_value = row.get(f"{metric}_user")
                model_value = row.get(metric)
                if pd.isna(user_value) or pd.isna(model_value):
                    continue
                delta = float(model_value - user_value)
                denominator = METRIC_RELATIVE_DENOMINATOR.get(metric, 10.0)
                relative_delta = delta / denominator if denominator else np.nan
                rows.append(
                    {
                        "source_label": label,
                        "prompt_number": int(prompt_number),
                        "metric": metric,
                        "user_value": float(user_value),
                        "model_value": float(model_value),
                        "delta": delta,
                        "relative_delta": relative_delta,
                        "abs_relative_delta": abs(relative_delta),
                    }
                )

    return pd.DataFrame(rows)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "model"


def _plot_per_model_prompt_delta_heatmaps(delta_df: pd.DataFrame, output_dir: Path) -> None:
    if delta_df.empty:
        return

    metric_order = [name for name, _ in NUMERIC_METRIC_PATHS]
    for label in sorted(delta_df["source_label"].unique()):
        model_delta = delta_df[delta_df["source_label"] == label].copy()
        if model_delta.empty:
            continue

        pivot = model_delta.pivot_table(
            index="prompt_number",
            columns="metric",
            values="delta",
            aggfunc="mean",
        )
        columns = [metric for metric in metric_order if metric in pivot.columns]
        pivot = pivot.reindex(columns=columns).sort_index()

        if pivot.empty:
            continue

        plt.figure(figsize=(16, 8))
        sns.heatmap(pivot, cmap="coolwarm", center=0.0, annot=True, fmt=".1f", linewidths=0.3)
        plt.title(f"Prompt-Level Delta vs User ({label})\n(delta = model - user)")
        plt.ylabel("prompt_number")
        plt.xlabel("metric")
        plt.tight_layout()
        plt.savefig(output_dir / f"prompt_delta_heatmap_{_slugify(label)}.png", dpi=200)
        plt.close()


def _plot_prompt_level_emphasis_deltas(delta_df: pd.DataFrame, output_dir: Path) -> None:
    focus_df = delta_df[delta_df["metric"].isin(EMPHASIS_METRICS)].copy()
    if focus_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    axes_flat = axes.flatten()

    for index, metric in enumerate(EMPHASIS_METRICS):
        axis = axes_flat[index]
        metric_df = focus_df[focus_df["metric"] == metric].sort_values(["source_label", "prompt_number"])
        if metric_df.empty:
            axis.set_visible(False)
            continue

        sns.lineplot(
            data=metric_df,
            x="prompt_number",
            y="delta",
            hue="source_label",
            marker="o",
            ax=axis,
        )
        axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.set_title(f"{metric} delta by prompt")
        axis.set_ylabel("model - user")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    for axis in axes_flat:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()

    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / "prompt_level_stance_epistemic_deltas.png", dpi=200)
    plt.close()


def _plot_prompt_level_total_abs_delta(delta_df: pd.DataFrame, output_dir: Path) -> None:
    if delta_df.empty:
        return

    prompt_disagreement = (
        delta_df.groupby(["source_label", "prompt_number"])["abs_delta"]
        .mean()
        .reset_index(name="mean_abs_delta")
    )
    plt.figure(figsize=(16, 7))
    sns.lineplot(
        data=prompt_disagreement.sort_values(["source_label", "prompt_number"]),
        x="prompt_number",
        y="mean_abs_delta",
        hue="source_label",
        marker="o",
    )
    plt.title("Prompt-Level Mean Absolute Delta vs User Across All Metrics")
    plt.ylabel("mean absolute delta")
    plt.tight_layout()
    plt.savefig(output_dir / "prompt_level_mean_abs_delta.png", dpi=200)
    plt.close()


def _plot_metric_values_vs_user(metrics_df: pd.DataFrame, user_label: str, output_dir: Path) -> None:
    if metrics_df.empty:
        return

    metric_plot_dir = output_dir / "metric_divergence"
    metric_plot_dir.mkdir(parents=True, exist_ok=True)

    metric_order = [name for name, _ in NUMERIC_METRIC_PATHS]
    for metric in metric_order:
        subset = metrics_df[["source_label", "prompt_number", metric]].copy()
        subset = subset.dropna(subset=[metric])
        if subset.empty:
            continue

        plt.figure(figsize=(16, 7))
        labels = sorted(subset["source_label"].unique())
        if user_label in labels:
            labels = [user_label] + [label for label in labels if label != user_label]

        for label in labels:
            label_df = subset[subset["source_label"] == label].sort_values("prompt_number")
            marker_size = 150 if label == user_label else 55
            edge_width = 2.4 if label == user_label else 0.8
            alpha = 0.95 if label == user_label else 0.75
            plt.scatter(
                label_df["prompt_number"],
                label_df[metric],
                s=marker_size,
                alpha=alpha,
                linewidths=edge_width,
                edgecolors="black",
                label=label,
                zorder=1 if label == user_label else 2,
            )

        plt.title(f"{metric}: Actual Values by Prompt (Scatter, User Highlighted)")
        plt.xlabel("prompt_number")
        plt.ylabel("value")
        prompt_ticks = list(range(1, 21))
        plt.xticks(prompt_ticks)
        plt.xlim(0.5, 20.5)
        plt.grid(axis="x", alpha=0.35)
        plt.grid(axis="y", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(metric_plot_dir / f"{metric}_values_vs_user.png", dpi=200)
        plt.close()


def _plot_metric_divergence_panels(
    delta_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    user_label: str,
    output_dir: Path,
) -> None:
    if delta_df.empty:
        return

    metric_plot_dir = output_dir / "metric_divergence"
    metric_plot_dir.mkdir(parents=True, exist_ok=True)

    metric_order = [name for name, _ in NUMERIC_METRIC_PATHS]
    for metric in metric_order:
        metric_df = delta_df[delta_df["metric"] == metric].copy()
        if metric_df.empty:
            continue

        metric_df = metric_df.sort_values(["source_label", "prompt_number"])
        metric_df["abs_divergence"] = metric_df["delta"].abs()
        overall = (
            metric_df.groupby("source_label")["abs_divergence"]
            .mean()
            .reset_index(name="mean_abs_divergence")
            .sort_values("mean_abs_divergence", ascending=False)
        )

        fig, axes = plt.subplots(3, 1, figsize=(16, 14), height_ratios=[2, 1, 1])

        for label in sorted(metric_df["source_label"].unique()):
            label_df = metric_df[metric_df["source_label"] == label]
            axes[0].scatter(
                label_df["prompt_number"],
                label_df["delta"],
                s=50,
                alpha=0.75,
                edgecolors="black",
                linewidths=0.6,
                label=label,
            )

        axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[0].set_title(f"{metric}: Prompt-Level Divergence vs User (model - user)")
        axes[0].set_xlabel("prompt_number")
        axes[0].set_ylabel("directional divergence")
        axes[0].set_xticks(list(range(1, 21)))
        axes[0].set_xlim(0.5, 20.5)
        axes[0].grid(axis="x", alpha=0.35)
        axes[0].grid(axis="y", alpha=0.2)
        axes[0].legend()

        sns.barplot(
            data=overall,
            x="source_label",
            y="mean_abs_divergence",
            ax=axes[1],
            color="#4C72B0",
        )
        axes[1].set_title("Overall Model Divergence (Mean Absolute Divergence)")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("mean |model - user|")
        axes[1].tick_params(axis="x", rotation=25)

        user_metric = (
            metrics_df[metrics_df["source_label"] == user_label][["prompt_number", metric]]
            .rename(columns={metric: "user_value"})
            .copy()
        )

        overlap_rows: List[Dict[str, Any]] = []
        model_labels = sorted(label for label in metrics_df["source_label"].unique() if label != user_label)
        for model_label in model_labels:
            model_metric = (
                metrics_df[metrics_df["source_label"] == model_label][["prompt_number", metric]]
                .rename(columns={metric: "model_value"})
                .copy()
            )

            merged = user_metric.merge(model_metric, on="prompt_number", how="outer")
            user_has = ~merged["user_value"].isna()
            model_has = ~merged["model_value"].isna()

            overlap_rows.extend(
                [
                    {
                        "source_label": model_label,
                        "category": "both_numeric",
                        "count": int((user_has & model_has).sum()),
                    },
                    {
                        "source_label": model_label,
                        "category": "both_na",
                        "count": int((~user_has & ~model_has).sum()),
                    },
                    {
                        "source_label": model_label,
                        "category": "model_numeric_user_na",
                        "count": int((~user_has & model_has).sum()),
                    },
                    {
                        "source_label": model_label,
                        "category": "model_na_user_numeric",
                        "count": int((user_has & ~model_has).sum()),
                    },
                ]
            )

        overlap_df = pd.DataFrame(overlap_rows)
        if not overlap_df.empty:
            overlap_pivot = overlap_df.pivot(
                index="source_label",
                columns="category",
                values="count",
            ).fillna(0)

            category_order = [
                "both_numeric",
                "both_na",
                "model_numeric_user_na",
                "model_na_user_numeric",
            ]
            overlap_pivot = overlap_pivot.reindex(columns=category_order, fill_value=0)

            bottom = np.zeros(len(overlap_pivot))
            colors = {
                "both_numeric": "#2E8B57",
                "both_na": "#808080",
                "model_numeric_user_na": "#E69F00",
                "model_na_user_numeric": "#CC79A7",
            }

            for category in category_order:
                values = overlap_pivot[category].values
                axes[2].bar(
                    overlap_pivot.index,
                    values,
                    bottom=bottom,
                    label=category,
                    color=colors.get(category),
                )
                bottom += values

            axes[2].set_title("Numeric/N/A Overlap Counts vs User by Model")
            axes[2].set_xlabel("")
            axes[2].set_ylabel("prompt count")
            axes[2].tick_params(axis="x", rotation=25)
            axes[2].legend()
        else:
            axes[2].set_visible(False)

        plt.tight_layout()
        plt.savefig(metric_plot_dir / f"{metric}_divergence.png", dpi=200)
        plt.close()


def _compute_mae_vs_user(metrics_df: pd.DataFrame, user_label: str, metrics: List[str]) -> pd.DataFrame:
    user_df = metrics_df[metrics_df["source_label"] == user_label][["prompt_number"] + metrics].copy()
    user_df = user_df.rename(columns={metric: f"{metric}_user" for metric in metrics})

    rows: List[Dict[str, Any]] = []
    model_labels = sorted(label for label in metrics_df["source_label"].unique() if label != user_label)
    for label in model_labels:
        model_df = metrics_df[metrics_df["source_label"] == label][["prompt_number"] + metrics].copy()
        merged = user_df.merge(model_df, on="prompt_number", how="inner")
        if merged.empty:
            continue

        for metric in metrics:
            lhs = merged[f"{metric}_user"]
            rhs = merged[metric]
            valid = ~(lhs.isna() | rhs.isna())
            if not valid.any():
                continue
            mae = float((lhs[valid] - rhs[valid]).abs().mean())
            rows.append({"source_label": label, "metric": metric, "mae_vs_user": mae})

    return pd.DataFrame(rows)


def _plot_stance_epistemic_focus(metrics_df: pd.DataFrame, user_label: str, output_dir: Path) -> None:
    focus = metrics_df[["source_label", "prompt_number"] + EMPHASIS_METRICS].copy()
    melted = focus.melt(id_vars=["source_label", "prompt_number"], value_vars=EMPHASIS_METRICS, var_name="metric", value_name="value")
    melted = melted.dropna(subset=["value"])

    mae_df = _compute_mae_vs_user(metrics_df, user_label=user_label, metrics=EMPHASIS_METRICS)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    sns.boxplot(data=melted, x="metric", y="value", hue="source_label", showfliers=False, ax=axes[0])
    axes[0].set_title("Stance and Epistemic Metric Distributions")
    axes[0].tick_params(axis="x", rotation=25)

    if not mae_df.empty:
        sns.barplot(data=mae_df, x="metric", y="mae_vs_user", hue="source_label", ax=axes[1])
        axes[1].set_title("MAE vs User Annotations (Lower is Better)")
        axes[1].tick_params(axis="x", rotation=25)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "stance_epistemic_focus.png", dpi=200)
    plt.close()


def _plot_metric_directional_error_scatter(delta_df: pd.DataFrame, output_dir: Path) -> None:
    if delta_df.empty:
        return

    summary = (
        delta_df.groupby(["metric", "source_label"])
        .agg(
            mean_directional_error=("delta", "mean"),
            mae=("delta", lambda series: series.abs().mean()),
            count=("delta", "size"),
        )
        .reset_index()
    )
    if summary.empty:
        return

    metric_order = [name for name, _ in NUMERIC_METRIC_PATHS]
    summary["metric"] = pd.Categorical(summary["metric"], categories=metric_order, ordered=True)
    summary = summary.sort_values(["metric", "source_label"])

    plt.figure(figsize=(18, 8))
    sns.scatterplot(
        data=summary,
        x="metric",
        y="mean_directional_error",
        hue="source_label",
        s=130,
        edgecolor="black",
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Mean Directional Error vs User by Metric and Model (model - user)")
    plt.xlabel("metric")
    plt.ylabel("mean directional error")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "metric_mean_directional_error_scatter.png", dpi=200)
    plt.close()

    summary.to_csv(output_dir / "metric_directional_error_summary_by_model.csv", index=False)


def _plot_token_costs(token_df: pd.DataFrame, output_dir: Path) -> None:
    valid = token_df[token_df["status"] == "ok"].copy()
    if valid.empty:
        return

    grouped = (
        valid.groupby("source_label")[["prompt_tokens", "completion_tokens", "total_tokens"]]
        .sum(numeric_only=True)
        .reset_index()
        .sort_values("total_tokens", ascending=False)
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    sns.barplot(data=grouped, x="source_label", y="total_tokens", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Overall Token Cost per LLM Run")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(grouped["source_label"], grouped["prompt_tokens"], label="prompt_tokens")
    axes[1].bar(
        grouped["source_label"],
        grouped["completion_tokens"],
        bottom=grouped["prompt_tokens"],
        label="completion_tokens",
    )
    axes[1].set_title("Prompt vs Completion Token Totals")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "model_token_costs.png", dpi=200)
    plt.close()


def _resolve_price_key(model_name: Optional[str]) -> Optional[str]:
    if not isinstance(model_name, str) or not model_name.strip():
        return None

    value = model_name.strip().lower()
    if value in MODEL_PRICING_USD_PER_MTOK:
        return value

    if value.startswith("gpt-5.2"):
        return "gpt-5.2"
    if value.startswith("gpt-5-mini"):
        return "gpt-5-mini"
    if value.startswith("gpt-5-nano"):
        return "gpt-5-nano"
    if value.startswith("claude-sonnet-4-6"):
        return "claude-sonnet-4-6"
    if value.startswith("claude-haiku-4-5"):
        return "claude-haiku-4-5-20251001"
    if value.startswith("claude-3-5-haiku"):
        return "claude-3-5-haiku-20241022"

    return None


def _estimate_model_costs(
    token_df: pd.DataFrame,
    source_to_model: Dict[str, Optional[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if token_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    valid = token_df[token_df["status"] == "ok"].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, row in valid.iterrows():
        source_label = row.get("source_label")
        model_name = source_to_model.get(source_label)
        price_key = _resolve_price_key(model_name) or _resolve_price_key(source_label)
        pricing = MODEL_PRICING_USD_PER_MTOK.get(price_key) if price_key else None
        if not pricing:
            continue

        prompt_tokens = _numeric(row.get("prompt_tokens"))
        completion_tokens = _numeric(row.get("completion_tokens"))
        if np.isnan(prompt_tokens) or np.isnan(completion_tokens):
            continue

        input_usd = (prompt_tokens / 1_000_000.0) * pricing["input"]
        output_usd = (completion_tokens / 1_000_000.0) * pricing["output"]
        total_usd = input_usd + output_usd

        rows.append(
            {
                "source_label": source_label,
                "model_used": model_name,
                "price_key": price_key,
                "prompt_number": row.get("prompt_number"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "input_cost_usd": input_usd,
                "output_cost_usd": output_usd,
                "total_cost_usd": total_usd,
            }
        )

    per_prompt_df = pd.DataFrame(rows)
    if per_prompt_df.empty:
        return per_prompt_df, pd.DataFrame()

    summary_df = (
        per_prompt_df.groupby(["source_label", "model_used", "price_key"], as_index=False)
        .agg(
            calls_count=("prompt_number", "count"),
            input_cost_usd=("input_cost_usd", "sum"),
            output_cost_usd=("output_cost_usd", "sum"),
            total_cost_usd=("total_cost_usd", "sum"),
            prompt_tokens=("prompt_tokens", "sum"),
            completion_tokens=("completion_tokens", "sum"),
        )
    )

    scale = 1000.0 / summary_df["calls_count"].clip(lower=1)
    summary_df["input_cost_usd_per_1000_calls"] = summary_df["input_cost_usd"] * scale
    summary_df["output_cost_usd_per_1000_calls"] = summary_df["output_cost_usd"] * scale
    summary_df["total_cost_usd_per_1000_calls"] = summary_df["total_cost_usd"] * scale

    summary_df = summary_df.sort_values("total_cost_usd_per_1000_calls", ascending=False)

    return per_prompt_df, summary_df


def _plot_model_cost_usd(cost_summary_df: pd.DataFrame, output_dir: Path) -> None:
    if cost_summary_df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    sns.barplot(
        data=cost_summary_df,
        x="source_label",
        y="total_cost_usd_per_1000_calls",
        ax=axes[0],
        color="#2E8B57",
    )
    axes[0].set_title("Estimated Total API Cost (USD) per 1000 Calls by Model")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("USD")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(
        cost_summary_df["source_label"],
        cost_summary_df["input_cost_usd_per_1000_calls"],
        label="input_cost_usd_per_1000_calls",
    )
    axes[1].bar(
        cost_summary_df["source_label"],
        cost_summary_df["output_cost_usd_per_1000_calls"],
        bottom=cost_summary_df["input_cost_usd_per_1000_calls"],
        label="output_cost_usd_per_1000_calls",
    )
    axes[1].set_title("Estimated USD Cost per 1000 Calls (Input vs Output)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("USD")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "model_cost_usd.png", dpi=200)
    plt.close()


def _write_pricing_snapshot(output_dir: Path, source_to_model: Dict[str, Optional[str]]) -> None:
    used_price_keys = set()
    source_to_price_key: Dict[str, Optional[str]] = {}

    for source_label, model_name in source_to_model.items():
        price_key = _resolve_price_key(model_name) or _resolve_price_key(source_label)
        source_to_price_key[source_label] = price_key
        if price_key:
            used_price_keys.add(price_key)

    pricing_snapshot = {
        "snapshot_timestamp": datetime.now().isoformat(),
        "pricing_sources": PRICING_SOURCES,
        "source_to_model": source_to_model,
        "source_to_price_key": source_to_price_key,
        "pricing_used_usd_per_mtok": {
            key: MODEL_PRICING_USD_PER_MTOK[key]
            for key in sorted(used_price_keys)
            if key in MODEL_PRICING_USD_PER_MTOK
        },
    }

    stamp = datetime.now().strftime("%Y%m%d")
    snapshot_path = output_dir / f"pricing_snapshot_{stamp}.json"
    with open(snapshot_path, "w", encoding="utf-8") as file:
        json.dump(pricing_snapshot, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ABN model comparison visuals.")
    parser.add_argument("--abn-dir", default=str(DEFAULT_ABN_DIR), help="Directory containing ABN run artifacts.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated visuals.")
    parser.add_argument("--user-metadata", default=None, help="Specific abn_test_user_entries_metadata_XXX.json file.")
    parser.add_argument(
        "--llm-metadata",
        nargs="*",
        default=None,
        help="Optional list of specific abn_test_llm_entries_metadata_XXX.json files.",
    )
    args = parser.parse_args()

    abn_dir = Path(args.abn_dir)
    if not abn_dir.exists():
        raise FileNotFoundError(f"ABN directory not found: {abn_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else (abn_dir / "model_stats")
    output_dir.mkdir(parents=True, exist_ok=True)

    user_metadata_path = Path(args.user_metadata) if args.user_metadata else _discover_latest_user_metadata(abn_dir)
    if user_metadata_path is None or not user_metadata_path.exists():
        raise FileNotFoundError("No user metadata file found. Provide --user-metadata.")

    if args.llm_metadata:
        llm_metadata_paths = [Path(path) for path in args.llm_metadata]
    else:
        llm_metadata_paths = list(_discover_latest_llm_metadata_per_model(abn_dir).values())

    if not llm_metadata_paths:
        raise FileNotFoundError("No LLM metadata files found. Provide --llm-metadata.")

    run_manifest: Dict[str, Any] = {"user_metadata": str(user_metadata_path), "llm_metadata": []}

    user_metadata = _load_json(user_metadata_path)
    user_entries_path = _resolve_path(user_metadata.get("output_file"), user_metadata_path)
    if user_entries_path is None:
        raise FileNotFoundError(f"Could not resolve user entries path from {user_metadata_path}")

    user_entries = _load_json(user_entries_path)
    user_label = "user_annotations"
    metrics_frames = [_entries_to_dataframe(user_entries, source_label=user_label, source_type="user")]
    token_frames: List[pd.DataFrame] = []
    source_to_model: Dict[str, Optional[str]] = {}

    for metadata_path in llm_metadata_paths:
        metadata = _load_json(metadata_path)
        entries_path = _resolve_path(metadata.get("output_file"), metadata_path)
        if entries_path is None:
            continue

        source_label = str(metadata.get("model_requested") or metadata.get("model") or metadata_path.stem)
        source_to_model[source_label] = metadata.get("model_used") or metadata.get("model_requested") or metadata.get("model")
        run_manifest["llm_metadata"].append({
            "source_label": source_label,
            "metadata_file": str(metadata_path),
            "entries_file": str(entries_path),
            "model_used": metadata.get("model_used"),
        })

        llm_entries = _load_json(entries_path)
        metrics_frames.append(_entries_to_dataframe(llm_entries, source_label=source_label, source_type="llm"))
        token_frames.append(_load_token_usage(metadata, source_label=source_label))

    metrics_df = pd.concat(metrics_frames, ignore_index=True)
    token_df = pd.concat(token_frames, ignore_index=True) if token_frames else pd.DataFrame()
    delta_df = _build_delta_dataframe(metrics_df, user_label=user_label)
    per_prompt_cost_df, cost_summary_df = _estimate_model_costs(token_df, source_to_model=source_to_model)

    metrics_df.to_csv(output_dir / "metrics_long.csv", index=False)
    if not delta_df.empty:
        delta_df.to_csv(output_dir / "metrics_delta_vs_user_long.csv", index=False)
    if not token_df.empty:
        token_df.to_csv(output_dir / "token_usage_long.csv", index=False)
    if not per_prompt_cost_df.empty:
        per_prompt_cost_df.to_csv(output_dir / "model_cost_per_prompt_usd.csv", index=False)
    if not cost_summary_df.empty:
        cost_summary_df.to_csv(output_dir / "model_cost_summary_usd.csv", index=False)

    _plot_all_metrics_heatmap(metrics_df, output_dir)
    _plot_all_metrics_distribution(metrics_df, output_dir)
    _plot_stance_epistemic_focus(metrics_df, user_label=user_label, output_dir=output_dir)
    _plot_metric_directional_error_scatter(delta_df, output_dir=output_dir)
    _plot_metric_divergence_panels(
        delta_df,
        metrics_df=metrics_df,
        user_label=user_label,
        output_dir=output_dir,
    )
    _plot_metric_values_vs_user(metrics_df, user_label=user_label, output_dir=output_dir)
    if not token_df.empty:
        _plot_token_costs(token_df, output_dir)
    if not cost_summary_df.empty:
        _plot_model_cost_usd(cost_summary_df, output_dir)

    with open(output_dir / "selected_runs.json", "w", encoding="utf-8") as file:
        json.dump(run_manifest, file, indent=2)

    with open(output_dir / "pricing_sources.json", "w", encoding="utf-8") as file:
        json.dump(PRICING_SOURCES, file, indent=2)

    _write_pricing_snapshot(output_dir, source_to_model=source_to_model)

    print(f"Saved model stats artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
