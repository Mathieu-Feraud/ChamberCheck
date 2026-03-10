import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from ChamberCheck.CC_derived_metrics.derived_metrics import CC_Metrics


TOPIC_INPUT = Path("data/output/scrape_003/fake_llm_derived_metrics_001.json")
SUBREDDIT_INPUT = Path("data/output/scrape_003/fake_llm_derived_metrics_by_subreddit_001.json")
EAS_METADATA_INPUT = Path("data/output/scrape_003/fake_llm_entries_metadata_001.json")
OUTPUT_ROOT = Path("data/output/scrape_003/requested_metric_plots_001")

METRIC_CONFIG: Dict[str, Dict[str, Any]] = {
    "Expressive Participation Gap": {"slug": "epg", "type": "scalar", "aggregate": False},
    "Linguistic Self-Protection Rate (LSPR)": {"slug": "lspr", "type": "scalar", "aggregate": True},
    "Counter-Evidence Exposure Rate (CER)": {"slug": "cer", "type": "scalar", "aggregate": True},
    "Constructive Counter-View Engagement (CCVE)": {"slug": "ccve", "type": "scalar", "aggregate": True},
    "Counter-Evidence Sentiment Shift (CESS)": {"slug": "cess", "type": "scalar", "aggregate": True},
    "Engagement Asymmetry Index (EAI)": {"slug": "eai", "type": "scalar", "aggregate": True},
    "Cross-Stance Interaction Rate (CSIR)": {"slug": "csir", "type": "scalar", "aggregate": True},
    "Low-Support Claim Amplification (LSCA)": {"slug": "lsca", "type": "scalar", "aggregate": True},
    "Emotional Amplification Score (EAS)": {"slug": "eas", "type": "dict", "components": ["anger", "anxiety", "disgust"], "aggregate": True},
    "Visible Opinion Compression (VOC)": {"slug": "voc", "type": "dict", "components": ["voc_visible", "voc_full", "delta_voc"], "aggregate": False},
}

PROPORTION_TOPIC_METRICS = {
    "Counter-Evidence Exposure Rate (CER)",
    "Constructive Counter-View Engagement (CCVE)",
}

TOPIC_BOUNDED_RANGES: Dict[str, Tuple[float, float]] = {
    "Expressive Participation Gap": (0.0, 1.0),
    "Linguistic Self-Protection Rate (LSPR)": (0.0, 1.0),
    "Counter-Evidence Exposure Rate (CER)": (0.0, 1.0),
    "Constructive Counter-View Engagement (CCVE)": (0.0, 1.0),
    "Engagement Asymmetry Index (EAI)": (-1.0, 1.0),
    "Cross-Stance Interaction Rate (CSIR)": (-1.0, 1.0),
    "Counter-Evidence Sentiment Shift (CESS)": (-20.0, 20.0),
}

DICT_COMPONENT_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Emotional Amplification Score (EAS)": {
        "anger": (-1.0, 1.0),
        "anxiety": (-1.0, 1.0),
        "disgust": (-1.0, 1.0),
    },
    "Visible Opinion Compression (VOC)": {
        "voc_visible": (0.0, 1.0),
        "voc_full": (0.0, 1.0),
        "delta_voc": (-1.0, 1.0),
    },
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _trim_topic(topic: str, max_len: int = 90) -> str:
    text = " ".join(topic.split())
    return text if len(text) <= max_len else (text[: max_len - 3] + "...")


def _slugify(name: str) -> str:
    chars = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch.lower())
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}")
    return data


def _pvalue_marker(pvalue: float | None) -> str:
    """Return significance asterisk(s): '**' p<0.01, '*' p<0.05, '' otherwise."""
    if pvalue is None or not _is_number(pvalue):
        return ""
    if float(pvalue) < 0.01:
        return "**"
    if float(pvalue) < 0.05:
        return "*"
    return ""


def _topic_scalar_values(topic_map: Dict[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for topic, metric_result in topic_map.items():
        if not isinstance(metric_result, dict):
            continue
        value = metric_result.get("value")
        if _is_number(value):
            result[topic] = float(value)
    return result


def _topic_scalar_weights(topic_map: Dict[str, Any]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for topic, metric_result in topic_map.items():
        if not isinstance(metric_result, dict):
            continue
        sample_size = metric_result.get("sample_size")
        if isinstance(sample_size, int) and sample_size > 0:
            result[topic] = sample_size
    return result


def _topic_sample_sizes(topic_map: Dict[str, Any]) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    for topic, metric_result in topic_map.items():
        if not isinstance(metric_result, dict):
            continue
        sample_size = metric_result.get("sample_size")
        if isinstance(sample_size, int) and sample_size >= 0:
            sizes[topic] = sample_size
    return sizes


def _topic_dict_values(topic_map: Dict[str, Any], components: List[str]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for topic, metric_result in topic_map.items():
        if not isinstance(metric_result, dict):
            continue
        value = metric_result.get("value")
        if not isinstance(value, dict):
            continue
        comp_values: Dict[str, float] = {}
        for comp in components:
            raw = value.get(comp)
            if _is_number(raw):
                comp_values[comp] = float(raw)
                continue
            if isinstance(raw, dict):
                rho = raw.get("rho")
                if _is_number(rho):
                    comp_values[comp] = float(rho)
        if comp_values:
            result[topic] = comp_values
    return result


def _topic_dict_pvalues(
    topic_map: Dict[str, Any],
    components: List[str],
) -> Dict[str, Dict[str, float]]:
    """Extract per-component Spearman p-values from topic_map EAS entries."""
    result: Dict[str, Dict[str, float]] = {}
    for topic, metric_result in topic_map.items():
        if not isinstance(metric_result, dict):
            continue
        value = metric_result.get("value")
        if not isinstance(value, dict):
            continue
        comp_pvals: Dict[str, float] = {}
        for comp in components:
            raw = value.get(comp)
            if isinstance(raw, dict):
                pval = raw.get("pvalue")
                if _is_number(pval):
                    comp_pvals[comp] = float(pval)
        if comp_pvals:
            result[topic] = comp_pvals
    return result


def _topic_dict_ci(
    metric_name: str,
    topic_map: Dict[str, Any],
    components: List[str],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    ci_map: Dict[str, Dict[str, Tuple[float, float]]] = {}

    component_ranges = DICT_COMPONENT_RANGES.get(metric_name, {})
    for topic, metric_result in topic_map.items():
        if not isinstance(metric_result, dict):
            continue
        value = metric_result.get("value")
        if not isinstance(value, dict):
            continue

        sample_size = metric_result.get("sample_size")
        if not isinstance(sample_size, int) or sample_size <= 0:
            continue

        topic_ci: Dict[str, Tuple[float, float]] = {}
        for comp in components:
            comp_val = value.get(comp)
            rho_val = None
            pvalue_val = None
            n_val = sample_size

            if _is_number(comp_val):
                rho_val = float(comp_val)
            elif isinstance(comp_val, dict):
                rho = comp_val.get("rho")
                pvalue = comp_val.get("pvalue")
                n_comp = comp_val.get("n")
                if _is_number(rho):
                    rho_val = float(rho)
                if _is_number(pvalue):
                    pvalue_val = float(pvalue)
                if isinstance(n_comp, int) and n_comp > 0:
                    n_val = n_comp

            if rho_val is None:
                continue

            # EAS Spearman: comp_val is {rho, pvalue, n} → use Fisher-z CI (always 95%)
            if isinstance(comp_val, dict):
                n_eff = max(4, int(n_val))
                if abs(rho_val) < 1.0 and n_eff > 3:
                    fisher_z = np.arctanh(rho_val)
                    se = 1.0 / np.sqrt(n_eff - 3)
                    lo = np.tanh(fisher_z - 1.96 * se)
                    hi = np.tanh(fisher_z + 1.96 * se)
                    topic_ci[comp] = (float(lo), float(hi))
                continue

            # Other dict metrics: range-based fallback
            if comp not in component_ranges:
                continue
            low_bound, high_bound = component_ranges[comp]
            metric_range = high_bound - low_bound
            val_f = float(rho_val)
            half_width = 1.96 * (metric_range / 2.0) / float(np.sqrt(max(1, n_val)))
            ci_low = max(low_bound, val_f - half_width)
            ci_high = min(high_bound, val_f + half_width)
            topic_ci[comp] = (ci_low, ci_high)

        if topic_ci:
            ci_map[topic] = topic_ci

    return ci_map


def _wilson_ci(value: float, sample_size: int) -> Tuple[float, float] | None:
    if sample_size <= 0 or not (0.0 <= value <= 1.0):
        return None
    z = 1.96
    p = float(value)
    n = int(sample_size)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half_width = (z * np.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _topic_scalar_ci(metric_name: str, topic_map: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    ci_map: Dict[str, Tuple[float, float]] = {}
    for topic, metric_result in topic_map.items():
        if not isinstance(metric_result, dict):
            continue
        value = metric_result.get("value")
        if not _is_number(value):
            continue

        ci_low = metric_result.get("ci_95_lower")
        ci_high = metric_result.get("ci_95_upper")
        if _is_number(ci_low) and _is_number(ci_high):
            ci_map[topic] = (float(ci_low), float(ci_high))
            continue

        if metric_name in PROPORTION_TOPIC_METRICS:
            sample_size = metric_result.get("sample_size")
            if isinstance(sample_size, int) and sample_size > 0:
                ci = _wilson_ci(float(value), sample_size)
                if ci is not None:
                    ci_map[topic] = ci
                    continue

        sample_size = metric_result.get("sample_size")
        if not isinstance(sample_size, int) or sample_size <= 0:
            continue

        if metric_name in TOPIC_BOUNDED_RANGES:
            lower_bound, upper_bound = TOPIC_BOUNDED_RANGES[metric_name]
            value_f = float(value)
            metric_range = upper_bound - lower_bound
            half_width = 1.96 * (metric_range / 2.0) / float(np.sqrt(sample_size))
            ci_low = max(lower_bound, value_f - half_width)
            ci_high = min(upper_bound, value_f + half_width)
            ci_map[topic] = (ci_low, ci_high)
            continue

        if metric_name == "Low-Support Claim Amplification (LSCA)":
            value_f = float(value)
            scale = max(0.1, abs(value_f) * 0.5)
            half_width = 1.96 * (scale / float(np.sqrt(sample_size)))
            ci_low = max(0.0, value_f - half_width)
            ci_high = value_f + half_width
            ci_map[topic] = (ci_low, ci_high)

    return ci_map


def _topic_metric_key_in_subreddit_payload(metric_name: str) -> str:
    return f"{metric_name} (by_topic)"


def _bootstrap_mean_ci(values: List[float], iterations: int = 500, seed: int = 42) -> Tuple[float, float] | None:
    if len(values) < 2:
        return None
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(iterations, dtype=float)
    for idx in range(iterations):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boot[idx] = float(np.mean(sample))
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _bootstrap_weighted_mean_ci(
    values: List[float],
    weights: List[int],
    iterations: int = 500,
    seed: int = 42,
) -> Tuple[float, float] | None:
    if len(values) < 2 or len(values) != len(weights):
        return None

    v = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    positive = w > 0
    v = v[positive]
    w = w[positive]
    if v.size < 2:
        return None

    p = w / np.sum(w)
    n = v.size
    rng = np.random.default_rng(seed)
    boot = np.empty(iterations, dtype=float)
    idxs = np.arange(n)
    for idx in range(iterations):
        sampled = rng.choice(idxs, size=n, replace=True, p=p)
        sv = v[sampled]
        sw = w[sampled]
        boot[idx] = float(np.sum(sv * sw) / np.sum(sw))

    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate or candidate.upper() == "N/A":
            return None
        try:
            return float(candidate)
        except ValueError:
            return None
    return None


def _eas_component_rho(values: np.ndarray, upvotes: np.ndarray) -> float | None:
    if values.size == 0 or upvotes.size == 0 or values.size != upvotes.size:
        return None
    rho, pvalue = spearmanr(upvotes, values)
    if np.isnan(rho) or np.isnan(pvalue):
        return None
    return float(rho)


def _bootstrap_eas_component_ci(
    values: np.ndarray,
    upvotes: np.ndarray,
    iterations: int = 500,
    seed: int = 42,
) -> Tuple[float, float] | None:
    if values.size < 2 or upvotes.size < 2 or values.size != upvotes.size:
        return None

    rng = np.random.default_rng(seed)
    n = values.size
    boot = np.empty(iterations, dtype=float)
    filled = 0

    for _ in range(iterations):
        sampled_idx = rng.choice(n, size=n, replace=True)
        stat = _eas_component_rho(values[sampled_idx], upvotes[sampled_idx])
        if stat is None:
            continue
        boot[filled] = stat
        filled += 1

    if filled < 2:
        return None

    used = boot[:filled]
    return float(np.quantile(used, 0.025)), float(np.quantile(used, 0.975))


def _bh_correct(pvalues: List[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values in original order."""
    m = len(pvalues)
    if m == 0:
        return np.array([], dtype=float)
    arr = np.array(pvalues, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    adjusted = np.minimum(1.0, arr * m / ranks)
    # Enforce monotonicity: take cumulative minimum from largest rank down
    adj_sorted = adjusted[order]
    for i in range(m - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adjusted[order] = adj_sorted
    return adjusted


def _eas_aggregate_from_comments(
    cc: CC_Metrics,
    subreddit: str,
    components: List[str],
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]], Dict[str, int], Dict[str, float]]:
    rows_by_component: Dict[str, List[Tuple[float, float]]] = {comp: [] for comp in components}

    for item in cc.entries:
        if not isinstance(item, dict):
            continue
        comment_id = item.get("comment_id")
        entry = item.get("entry")
        if not isinstance(comment_id, str) or not isinstance(entry, dict):
            continue

        meta = cc.comment_index.get(comment_id, {})
        community = meta.get("community") if isinstance(meta, dict) else None
        if community != subreddit:
            continue

        upvotes = _to_float(meta.get("upvotes") if isinstance(meta, dict) else None)
        if upvotes is None:
            continue

        emotion = entry.get("emotion")
        if not isinstance(emotion, dict):
            continue

        for comp in components:
            val = _to_float(emotion.get(comp))
            if val is None:
                continue
            rows_by_component[comp].append((upvotes, val))

    means: Dict[str, float] = {}
    cis: Dict[str, Tuple[float, float]] = {}
    counts: Dict[str, int] = {}
    pvalues: Dict[str, float] = {}

    for comp in components:
        rows = rows_by_component.get(comp, [])
        if len(rows) == 0:
            continue

        up = np.array([r[0] for r in rows], dtype=float)
        vals = np.array([r[1] for r in rows], dtype=float)
        try:
            from scipy.stats import spearmanr as _spearmanr
            result_sr = _spearmanr(up, vals)
            rho = float(result_sr.statistic) if hasattr(result_sr, "statistic") else float(result_sr[0])
            pval = float(result_sr.pvalue) if hasattr(result_sr, "pvalue") else float(result_sr[1])
        except Exception:
            rho_raw = _eas_component_rho(vals, up)
            if rho_raw is None:
                continue
            rho = rho_raw
            pval = None

        means[comp] = rho
        counts[comp] = int(vals.size)
        if pval is not None:
            pvalues[comp] = pval
        ci = _bootstrap_eas_component_ci(
            vals,
            up,
            seed=(abs(hash(("eas_comment_level", subreddit, comp))) % (2**32)),
        )
        if ci is not None:
            cis[comp] = ci

    return means, cis, counts, pvalues


def _plot_scalar_topic_bars(
    subreddit: str,
    metric_name: str,
    topic_values: Dict[str, float],
    output_path: Path,
    topic_ci: Dict[str, Tuple[float, float]] | None = None,
    topic_n: Dict[str, int] | None = None,
) -> None:
    if not topic_values:
        return

    topics = sorted(topic_values.keys(), key=lambda key: topic_values[key])
    y_labels: List[str] = []
    for topic in topics:
        base = _trim_topic(topic)
        if isinstance(topic_n, dict) and topic in topic_n:
            y_labels.append(f"{base}  [n={topic_n[topic]}]")
        else:
            y_labels.append(base)
    x_vals = [topic_values[topic] for topic in topics]

    xerr = None
    if isinstance(topic_ci, dict):
        left_err: List[float] = []
        right_err: List[float] = []
        for topic, value in zip(topics, x_vals):
            ci = topic_ci.get(topic)
            if not ci or len(ci) != 2:
                left_err.append(np.nan)
                right_err.append(np.nan)
                continue
            low, high = ci
            left_err.append(max(0.0, float(value) - float(low)))
            right_err.append(max(0.0, float(high) - float(value)))

        finite = sum(1 for le, re in zip(left_err, right_err) if np.isfinite(le) and np.isfinite(re))
        if finite > 0:
            xerr = np.array([
                [0.0 if not np.isfinite(le) else le for le in left_err],
                [0.0 if not np.isfinite(re) else re for re in right_err],
            ])

    fig_h = max(6, len(topics) * 0.45)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(y_labels, x_vals, xerr=xerr, capsize=3 if xerr is not None else 0)
    ax.set_title(f"{metric_name} by Topic: {subreddit}")
    ax.set_xlabel("Metric value")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    min_v = min(x_vals)
    max_v = max(x_vals)
    span = max(1e-6, max_v - min_v)
    pad = span * 0.08
    ax.set_xlim(min_v - pad, max_v + pad)

    for i, value in enumerate(x_vals):
        offset = span * 0.02
        ax.text(value + offset, i, f"{value:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_dict_components(
    subreddit: str,
    metric_name: str,
    topic_values: Dict[str, Dict[str, float]],
    components: List[str],
    output_path: Path,
    topic_n: Dict[str, int] | None = None,
    topic_ci: Dict[str, Dict[str, Tuple[float, float]]] | None = None,
    topic_pvalues: Dict[str, Dict[str, float]] | None = None,
) -> None:
    if not topic_values:
        return

    topics = sorted(topic_values.keys())
    labels: List[str] = []
    for topic in topics:
        base = _trim_topic(topic, max_len=65)
        if isinstance(topic_n, dict) and topic in topic_n:
            labels.append(f"{base}\n[n={topic_n[topic]}]")
        else:
            labels.append(base)

    x = np.arange(len(topics), dtype=float)
    width = 0.24

    fig_w = max(12, len(topics) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))

    offsets = np.linspace(-width, width, num=len(components))
    for idx, comp in enumerate(components):
        vals = [topic_values[topic].get(comp, np.nan) for topic in topics]
        clean = [0.0 if np.isnan(v) else v for v in vals]

        yerr = None
        if isinstance(topic_ci, dict):
            low_err: List[float] = []
            high_err: List[float] = []
            for topic, value in zip(topics, vals):
                if np.isnan(value):
                    low_err.append(np.nan)
                    high_err.append(np.nan)
                    continue
                comp_ci = topic_ci.get(topic, {}).get(comp) if isinstance(topic_ci.get(topic), dict) else None
                if not comp_ci or len(comp_ci) != 2:
                    low_err.append(np.nan)
                    high_err.append(np.nan)
                    continue
                ci_low, ci_high = comp_ci
                low_err.append(max(0.0, float(value) - float(ci_low)))
                high_err.append(max(0.0, float(ci_high) - float(value)))

            finite = sum(1 for lo, hi in zip(low_err, high_err) if np.isfinite(lo) and np.isfinite(hi))
            if finite > 0:
                yerr = np.array([
                    [0.0 if not np.isfinite(lo) else lo for lo in low_err],
                    [0.0 if not np.isfinite(hi) else hi for hi in high_err],
                ])

        bars = ax.bar(x + offsets[idx], clean, width=width, label=comp, yerr=yerr, capsize=3 if yerr is not None else 0)

        if isinstance(topic_pvalues, dict):
            for bar_obj, topic, value in zip(bars, topics, vals):
                if np.isnan(value):
                    continue
                pval = topic_pvalues.get(topic, {}).get(comp)
                marker = _pvalue_marker(pval)
                if not marker:
                    continue
                bar_x = bar_obj.get_x() + bar_obj.get_width() / 2.0
                bar_top = bar_obj.get_height()
                if value < 0:
                    bar_top = value
                    va = "top"
                    y_offset = -0.01
                else:
                    va = "bottom"
                    y_offset = 0.01
                ax.text(
                    bar_x,
                    bar_top + y_offset,
                    marker,
                    ha="center",
                    va=va,
                    fontsize=9,
                    color="black",
                    fontweight="bold",
                )

    ax.set_title(f"{metric_name} by Topic: {subreddit}")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    ax.text(
        0.99, 0.99,
        "* p<0.05  ** p<0.01 (BH-corrected)",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_subreddit_means(
    metric_name: str,
    subreddit_values: Dict[str, float],
    output_path: Path,
    subreddit_ci: Dict[str, Tuple[float, float]] | None = None,
    subreddit_n: Dict[str, int] | None = None,
    ci_note: str | None = None,
) -> None:
    if not subreddit_values:
        return

    labels = sorted(subreddit_values.keys())
    vals = [subreddit_values[label] for label in labels]

    yerr = None
    if isinstance(subreddit_ci, dict):
        low_err: List[float] = []
        high_err: List[float] = []
        for label, value in zip(labels, vals):
            ci = subreddit_ci.get(label)
            if not ci or len(ci) != 2:
                low_err.append(np.nan)
                high_err.append(np.nan)
                continue
            low, high = ci
            low_err.append(max(0.0, float(value) - float(low)))
            high_err.append(max(0.0, float(high) - float(value)))

        finite = sum(1 for lo, hi in zip(low_err, high_err) if np.isfinite(lo) and np.isfinite(hi))
        if finite > 0:
            yerr = np.array([
                [0.0 if not np.isfinite(lo) else lo for lo in low_err],
                [0.0 if not np.isfinite(hi) else hi for hi in high_err],
            ])

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_labels: List[str] = []
    for label in labels:
        if isinstance(subreddit_n, dict) and label in subreddit_n:
            plot_labels.append(f"{label}\n(n={subreddit_n[label]})")
        else:
            plot_labels.append(label)

    ax.bar(plot_labels, vals, yerr=yerr, capsize=4 if yerr is not None else 0)
    ax.set_title(f"{metric_name}: mean by subreddit (pooled by topic sample size)")
    ax.set_ylabel("Mean value")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if ci_note:
        ax.text(
            0.01,
            0.99,
            ci_note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    for i, value in enumerate(vals):
        ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_eas_aggregate_components_by_subreddit(
    output_path: Path,
    component_means: Dict[str, Dict[str, float]],
    component_ci: Dict[str, Dict[str, Tuple[float, float]]],
    component_n: Dict[str, Dict[str, int]],
    component_pvalues: Dict[str, Dict[str, float]] | None = None,
) -> None:
    if not component_means:
        return

    components = ["anger", "anxiety", "disgust"]
    subreddits = sorted(component_means.keys())
    x = np.arange(len(subreddits), dtype=float)
    width = 0.24
    offsets = np.linspace(-width, width, num=len(components))

    fig, ax = plt.subplots(figsize=(max(12, len(subreddits) * 1.1), 6.5))

    for idx, comp in enumerate(components):
        vals: List[float] = []
        low_err: List[float] = []
        high_err: List[float] = []
        for subreddit in subreddits:
            comp_value = component_means.get(subreddit, {}).get(comp)
            vals.append(float(comp_value) if _is_number(comp_value) else 0.0)

            ci = component_ci.get(subreddit, {}).get(comp)
            if ci and _is_number(comp_value):
                lo, hi = ci
                low_err.append(max(0.0, float(comp_value) - float(lo)))
                high_err.append(max(0.0, float(hi) - float(comp_value)))
            else:
                low_err.append(np.nan)
                high_err.append(np.nan)

        yerr = None
        finite = sum(1 for lo, hi in zip(low_err, high_err) if np.isfinite(lo) and np.isfinite(hi))
        if finite > 0:
            yerr = np.array([
                [0.0 if not np.isfinite(lo) else lo for lo in low_err],
                [0.0 if not np.isfinite(hi) else hi for hi in high_err],
            ])

        bars = ax.bar(
            x + offsets[idx],
            vals,
            width=width,
            label=comp,
            yerr=yerr,
            capsize=3 if yerr is not None else 0,
        )

        if isinstance(component_pvalues, dict):
            for bar_obj, subreddit, value in zip(bars, subreddits, vals):
                pval = component_pvalues.get(subreddit, {}).get(comp)
                marker = _pvalue_marker(pval)
                if not marker:
                    continue
                bar_x = bar_obj.get_x() + bar_obj.get_width() / 2.0
                if value < 0:
                    va = "top"
                    y_pos = value - 0.02
                else:
                    va = "bottom"
                    y_pos = value + 0.02
                ax.text(bar_x, y_pos, marker, ha="center", va=va, fontsize=9, color="black", fontweight="bold")

    xticklabels: List[str] = []
    comp_label = {"anger": "ang", "anxiety": "anx", "disgust": "dis"}
    for subreddit in subreddits:
        n_map = component_n.get(subreddit, {})
        n_parts = [f"{comp_label.get(c, c)}={n_map.get(c, 0)}" for c in components]
        xticklabels.append(f"{subreddit}\n({' '.join(n_parts)})")

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_ylabel("EAS Spearman rho")
    ax.set_title("Emotional Amplification Score (EAS) aggregate by subreddit (components separate)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    ax.text(
        0.01,
        0.99,
        "CI: 95% bootstrap over comment-level Spearman rho (topic ignored)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    ax.text(
        0.99,
        0.99,
        "* p<0.05  ** p<0.01 (BH-corrected)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_dropout_by_subreddit(payload: Dict[str, Any], output_dir: Path) -> None:
    labels: List[str] = []
    vals: List[float] = []
    err_low: List[float] = []
    err_high: List[float] = []

    for subreddit in sorted(payload.keys()):
        row = payload.get(subreddit, {})
        if not isinstance(row, dict):
            continue
        metric = row.get("Dropout Rate")
        if not isinstance(metric, dict):
            continue

        value = metric.get("value")
        if not _is_number(value):
            continue

        v = float(value)
        low = metric.get("ci_95_lower")
        high = metric.get("ci_95_upper")
        lo_err = max(0.0, v - float(low)) if _is_number(low) else np.nan
        hi_err = max(0.0, float(high) - v) if _is_number(high) else np.nan

        labels.append(subreddit)
        vals.append(v)
        err_low.append(lo_err)
        err_high.append(hi_err)

    if not vals:
        return

    yerr = None
    finite = sum(1 for lo, hi in zip(err_low, err_high) if np.isfinite(lo) and np.isfinite(hi))
    if finite > 0:
        yerr = np.array([
            [0.0 if not np.isfinite(lo) else lo for lo in err_low],
            [0.0 if not np.isfinite(hi) else hi for hi in err_high],
        ])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, vals, yerr=yerr, capsize=4 if yerr is not None else 0)
    ax.set_title("Dropout Rate by Subreddit")
    ax.set_ylabel("Dropout Rate")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for i, value in enumerate(vals):
        ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "dropout_rate_by_subreddit.png", dpi=170)
    plt.close(fig)


def main() -> None:
    topic_payload = _load_json(TOPIC_INPUT)
    subreddit_payload = _load_json(SUBREDDIT_INPUT)
    cc = CC_Metrics.from_abn_llm_run_metadata(str(EAS_METADATA_INPUT))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for metric_name, config in METRIC_CONFIG.items():
        metric_slug = config["slug"]
        metric_dir = OUTPUT_ROOT / metric_slug
        metric_dir.mkdir(parents=True, exist_ok=True)

        subreddit_means: Dict[str, float] = {}
        subreddit_ci: Dict[str, Tuple[float, float]] = {}
        subreddit_n: Dict[str, int] = {}
        eas_component_means: Dict[str, Dict[str, float]] = {}
        eas_component_ci: Dict[str, Dict[str, Tuple[float, float]]] = {}
        eas_component_n: Dict[str, Dict[str, int]] = {}
        eas_component_pvalues: Dict[str, Dict[str, float]] = {}
        # Deferred per-subreddit EAS topic data — BH correction applied after full collection
        eas_topic_plot_data: Dict[str, Dict] = {}

        for subreddit in sorted(topic_payload.keys()):
            subreddit_map = topic_payload.get(subreddit, {})
            if not isinstance(subreddit_map, dict):
                continue
            topic_map = subreddit_map.get(metric_name)
            if not isinstance(topic_map, dict):
                continue

            subreddit_slug = _slugify(subreddit)

            if config["type"] == "scalar":
                values = _topic_scalar_values(topic_map)
                topic_n = _topic_sample_sizes(topic_map)
                ci_source_map = topic_map
                subreddit_metric_map = subreddit_payload.get(subreddit, {})
                if isinstance(subreddit_metric_map, dict):
                    candidate = subreddit_metric_map.get(_topic_metric_key_in_subreddit_payload(metric_name))
                    if isinstance(candidate, dict):
                        ci_source_map = candidate

                ci_map = _topic_scalar_ci(metric_name=metric_name, topic_map=ci_source_map)
                _plot_scalar_topic_bars(
                    subreddit=subreddit,
                    metric_name=metric_name,
                    topic_values=values,
                    output_path=metric_dir / f"{subreddit_slug}_{metric_slug}_by_topic.png",
                    topic_ci=ci_map,
                    topic_n=topic_n,
                )
                if values and bool(config.get("aggregate", False)):
                    topic_vals = list(values.values())
                    subreddit_means[subreddit] = float(np.mean(topic_vals))
                    subreddit_n[subreddit] = len(topic_vals)
                    ci = _bootstrap_mean_ci(
                        topic_vals,
                        seed=(abs(hash((metric_name, subreddit))) % (2**32)),
                    )
                    if ci is not None:
                        subreddit_ci[subreddit] = ci

            elif config["type"] == "dict":
                components = list(config.get("components", []))
                values = _topic_dict_values(topic_map, components=components)
                topic_n = _topic_sample_sizes(topic_map)
                dict_ci = _topic_dict_ci(metric_name=metric_name, topic_map=topic_map, components=components)
                if metric_name == "Emotional Amplification Score (EAS)":
                    # Defer EAS topic plots — need all p-values for BH correction
                    eas_topic_plot_data[subreddit] = {
                        "values": values,
                        "n": topic_n,
                        "ci": dict_ci,
                        "pvalues_raw": _topic_dict_pvalues(topic_map, components=components),
                        "output_path": metric_dir / f"{subreddit_slug}_{metric_slug}_by_topic.png",
                        "components": components,
                    }
                else:
                    _plot_dict_components(
                        subreddit=subreddit,
                        metric_name=metric_name,
                        topic_values=values,
                        components=components,
                        output_path=metric_dir / f"{subreddit_slug}_{metric_slug}_by_topic.png",
                        topic_n=topic_n,
                        topic_ci=dict_ci,
                    )

                if values and bool(config.get("aggregate", False)) and metric_name == "Emotional Amplification Score (EAS)":
                    comp_means, comp_cis, comp_ns, comp_pvals = _eas_aggregate_from_comments(
                        cc=cc,
                        subreddit=subreddit,
                        components=components,
                    )

                    if comp_means:
                        eas_component_means[subreddit] = comp_means
                        eas_component_ci[subreddit] = comp_cis
                        eas_component_n[subreddit] = comp_ns
                        eas_component_pvalues[subreddit] = comp_pvals

        if bool(config.get("aggregate", False)) and subreddit_means:
            ci_note = "CI: 95% bootstrap over per-topic values (n = topic count)"
            _plot_subreddit_means(
                metric_name=metric_name,
                subreddit_values=subreddit_means,
                output_path=metric_dir / f"{metric_slug}_aggregate_by_subreddit.png",
                subreddit_ci=subreddit_ci,
                subreddit_n=subreddit_n,
                ci_note=ci_note,
            )

        if metric_name == "Emotional Amplification Score (EAS)" and (eas_topic_plot_data or eas_component_means):
            # ---- BH FDR correction across ALL EAS p-values (topic + aggregate) ----
            pvalue_keys: List[Any] = []
            pvalue_vals_flat: List[float] = []

            for sr, data in eas_topic_plot_data.items():
                for topic, comp_pvals in data["pvalues_raw"].items():
                    for comp, pv in comp_pvals.items():
                        pvalue_keys.append(("topic", sr, topic, comp))
                        pvalue_vals_flat.append(float(pv))

            for sr, comp_pvals in eas_component_pvalues.items():
                for comp, pv in comp_pvals.items():
                    pvalue_keys.append(("agg", sr, comp))
                    pvalue_vals_flat.append(float(pv))

            corrected_arr = _bh_correct(pvalue_vals_flat)
            corrected_map = {key: float(corrected_arr[i]) for i, key in enumerate(pvalue_keys)}

            # Generate deferred EAS topic plots with BH-corrected p-values
            for sr, data in eas_topic_plot_data.items():
                corrected_topic_pvals: Dict[str, Dict[str, float]] = {}
                for topic, comp_pvals in data["pvalues_raw"].items():
                    corrected_topic_pvals[topic] = {
                        comp: corrected_map[("topic", sr, topic, comp)]
                        for comp in comp_pvals
                    }
                _plot_dict_components(
                    subreddit=sr,
                    metric_name=metric_name,
                    topic_values=data["values"],
                    components=data["components"],
                    output_path=data["output_path"],
                    topic_n=data["n"],
                    topic_ci=data["ci"],
                    topic_pvalues=corrected_topic_pvals,
                )

            # Generate EAS aggregate plot with BH-corrected p-values
            if eas_component_means:
                corrected_agg_pvalues: Dict[str, Dict[str, float]] = {
                    sr: {
                        comp: corrected_map[("agg", sr, comp)]
                        for comp in comp_pvals
                    }
                    for sr, comp_pvals in eas_component_pvalues.items()
                }
                _plot_eas_aggregate_components_by_subreddit(
                    output_path=metric_dir / "eas_aggregate_by_subreddit.png",
                    component_means=eas_component_means,
                    component_ci=eas_component_ci,
                    component_n=eas_component_n,
                    component_pvalues=corrected_agg_pvalues,
                )

    dropout_dir = OUTPUT_ROOT / "dropout_rate"
    dropout_dir.mkdir(parents=True, exist_ok=True)
    _plot_dropout_by_subreddit(subreddit_payload, output_dir=dropout_dir)

    print(f"Saved requested metric plots to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
